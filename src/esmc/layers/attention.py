import functools
import os

import einops
import torch
import torch.nn.functional as F
from torch import nn

from esmc.layers.rotary import RotaryEmbedding

try:
    # FA4 (cute-DSL, Hopper/Blackwell). Public API moved from `flash_attn.*`
    # into `flash_attn.cute.interface`; signatures match FA2 for our arg set.
    # Note: `flash_attn.bert_padding` is gone in FA4 — we implement unpad/pad
    # locally below. First call triggers Triton JIT compile (~30-60s, cached).
    from flash_attn.cute.interface import flash_attn_func, flash_attn_varlen_func

    _FLASH_AVAILABLE = True
except ImportError:
    _FLASH_AVAILABLE = False


def _unpad_input(hidden_states: torch.Tensor, attention_mask: torch.Tensor):
    """Pack non-pad tokens of (B, L, ...) into (total_nnz, ...).

    Replaces `flash_attn.bert_padding.unpad_input` (removed in FA4).
    Returns (packed, indices, cu_seqlens_int32, max_seqlen).
    """
    seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
    max_seqlen = int(seqlens.max().item())
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    cu_seqlens = torch.nn.functional.pad(
        torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0)
    )
    packed = hidden_states.flatten(0, 1)[indices]
    return packed, indices, cu_seqlens, max_seqlen


def _pad_input(packed: torch.Tensor, indices: torch.Tensor, B: int, L: int) -> torch.Tensor:
    """Scatter (total_nnz, ...) back to (B, L, ...) with zeros at pad positions."""
    output = torch.zeros(
        B * L, *packed.shape[1:], dtype=packed.dtype, device=packed.device
    )
    output[indices] = packed
    return output.view(B, L, *packed.shape[1:])


def _use_flash() -> bool:
    """Runtime toggle. True only when env var is set AND package is importable."""
    return _FLASH_AVAILABLE and bool(int(os.environ.get("USE_FLASH_ATTN", "0")))


class MultiHeadAttention(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, bias: bool = False, qk_layernorm: bool = True
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads

        self.d_head = self.d_model // self.n_heads
        self.layernorm_qkv = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model * 3, bias=bias)
        )
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        if qk_layernorm:
            self.q_ln = nn.LayerNorm(d_model, bias=bias)
            self.k_ln = nn.LayerNorm(d_model, bias=bias)
        else:
            self.q_ln = nn.Identity()
            self.k_ln = nn.Identity()

        self.rotary = RotaryEmbedding(d_model // n_heads)

    def _apply_rotary(self, q: torch.Tensor, k: torch.Tensor):
        q = q.unflatten(-1, (self.n_heads, self.d_head))
        k = k.unflatten(-1, (self.n_heads, self.d_head))
        q, k = self.rotary(q, k)
        q = q.flatten(-2, -1)
        k = k.flatten(-2, -1)
        return q, k

    def _forward_sdpa(self, x, seq_id):
        qkv_BLD3 = self.layernorm_qkv(x)
        query_BLD, key_BLD, value_BLD = torch.chunk(qkv_BLD3, 3, dim=-1)
        query_BLD, key_BLD = (
            self.q_ln(query_BLD).to(query_BLD.dtype),
            self.k_ln(key_BLD).to(query_BLD.dtype),
        )
        query_BLD, key_BLD = self._apply_rotary(query_BLD, key_BLD)

        reshaper = functools.partial(
            einops.rearrange, pattern="b s (h d) -> b h s d", h=self.n_heads
        )

        query_BHLD, key_BHLD, value_BHLD = map(
            reshaper, (query_BLD, key_BLD, value_BLD)
        )

        if seq_id is not None:
            mask_BLL = seq_id.unsqueeze(-1) == seq_id.unsqueeze(-2)
            mask_BHLL = mask_BLL.unsqueeze(1)
            context_BHLD = F.scaled_dot_product_attention(
                query_BHLD, key_BHLD, value_BHLD, mask_BHLL
            )
        else:
            context_BHLD = F.scaled_dot_product_attention(
                query_BHLD, key_BHLD, value_BHLD
            )

        context_BLD = einops.rearrange(context_BHLD, "b h s d -> b s (h d)")
        return self.out_proj(context_BLD)

    def _forward_flash(self, x, seq_id):
        # x: (B, L, D), seq_id: (B, L) bool — True = non-pad.
        # Two paths:
        #   - No padding in this batch (seq_id.all()): use flash_attn_func dense.
        #     Skips unpad/pad bookkeeping entirely; rotary applies normally on (B,L,D).
        #   - Some padding present: use flash_attn_varlen_func with unpad/pad.
        #     Apply rotary AFTER unpadding so positions are per-sample [0..len-1].
        B, L, D = x.shape
        qkv_BLD3 = self.layernorm_qkv(x)
        query_BLD, key_BLD, value_BLD = torch.chunk(qkv_BLD3, 3, dim=-1)
        query_BLD, key_BLD = (
            self.q_ln(query_BLD).to(query_BLD.dtype),
            self.k_ln(key_BLD).to(query_BLD.dtype),
        )

        # flash_attn requires fp16/bf16. Lightning autocast(bf16) wraps PyTorch
        # ops but not external CUDA kernels — under DDP+bf16-mixed (params fp32)
        # the q/k/v tensors arrive as fp32 and flash crashes. Cast explicitly.
        # Pick bf16 (project standard); skip the cast if already low-precision.
        flash_dtype = torch.bfloat16 if query_BLD.dtype == torch.float32 else query_BLD.dtype

        if seq_id is None or seq_id.all():
            # Fast path: no pad. Rotary on padded BLD, reshape, dense flash.
            query_BLD, key_BLD = self._apply_rotary(query_BLD, key_BLD)
            q_BLHD = query_BLD.unflatten(-1, (self.n_heads, self.d_head)).to(flash_dtype)
            k_BLHD = key_BLD.unflatten(-1, (self.n_heads, self.d_head)).to(flash_dtype)
            v_BLHD = value_BLD.unflatten(-1, (self.n_heads, self.d_head)).to(flash_dtype)
            # FA4 returns (output, lse_or_None); we don't ask for LSE.
            out_BLHD = flash_attn_func(q_BLHD, k_BLHD, v_BLHD)[0]
            out_BLD = out_BLHD.flatten(-2, -1).to(query_BLD.dtype)
            return self.out_proj(out_BLD)

        # Padded path: rotary on padded form so pad positions get spurious rotation,
        # but they're unpadded out next so it doesn't reach the kernel.
        # Apply rotary BEFORE unpad to keep position computation simple — moving it
        # after unpad would need per-sample [0..len-1] reindexing (next optimization
        # opportunity if this turns out to dominate).
        query_BLD, key_BLD = self._apply_rotary(query_BLD, key_BLD)

        q_packed, indices, cu_seqlens, max_seqlen = _unpad_input(query_BLD, seq_id)
        k_packed, _, _, _ = _unpad_input(key_BLD, seq_id)
        v_packed, _, _, _ = _unpad_input(value_BLD, seq_id)

        q_packed = q_packed.unflatten(-1, (self.n_heads, self.d_head)).to(flash_dtype)
        k_packed = k_packed.unflatten(-1, (self.n_heads, self.d_head)).to(flash_dtype)
        v_packed = v_packed.unflatten(-1, (self.n_heads, self.d_head)).to(flash_dtype)

        # FA4 returns (output, lse_or_None); we don't ask for LSE.
        out_packed = flash_attn_varlen_func(
            q_packed, k_packed, v_packed,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
        )[0]

        out_packed = out_packed.flatten(-2, -1).to(query_BLD.dtype)
        out_BLD = _pad_input(out_packed, indices, B, L)
        return self.out_proj(out_BLD)

    def forward(self, x, seq_id):
        if _use_flash():
            return self._forward_flash(x, seq_id)
        return self._forward_sdpa(x, seq_id)
