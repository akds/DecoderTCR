"""Packed-sequence FlashAttention path for ESMC, no changes to the model code.

What this is
============

The ESMC transformer stack runs in (B, L, D) padded form. Our
`_forward_flash` in `src/esmc/layers/attention.py` does an unpad→flash→pad
*per attention layer*, which costs ~6 extra GPU kernel launches per layer.
At 80 layers × bs=8 (6B), that bookkeeping dominates the kernel-level FA4
win at our L≈472.

This module implements the standard "packed sequence training" pattern:
unpad ONCE after the input embedding, run all transformer layers on packed
`(nnz, D)`, pad ONCE before the output head. Per-token ops (LayerNorm, FFN,
linear projections, residuals, q_ln/k_ln) work fine on packed tensors. Only
attention itself + rotary need cu_seqlens-aware dispatch.

Contained entirely to this file + a small hook in `DecoderTCRC.__init__`
that replaces each block's `.attn` and wraps `ESMC.forward`. The `esmc/` tree is unchanged.

Toggle: pass `use_packed_flash=True` to `DecoderTCRC.__init__` (default off).

Requires: flash_attn_4 + quack.rotary (FA4's built-in varlen rotary helper).
"""

from __future__ import annotations

import contextvars
import types
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

try:
    from flash_attn.cute.interface import flash_attn_varlen_func as _flash_attn_varlen_func_raw
    from quack.rotary import apply_rotary_emb as _quack_apply_rotary_emb

    _PACKED_FLASH_AVAILABLE = True
except ImportError:
    _PACKED_FLASH_AVAILABLE = False
    _quack_apply_rotary_emb = None
    _flash_attn_varlen_func_raw = None


# torch._dynamo recompile-thrash mitigations.
#
# Two sites in the attention forward call into external CUTLASS-DSL-JIT kernels
# whose Python wrappers create a fresh closure (and therefore a fresh code-object
# id) on every invocation:
#
#   1. quack.rotary.apply_rotary_emb takes max_seqlen as a Python int. Dynamo
#      specializes on Python int values and our ctx.max_seqlen changes per batch
#      under concat_uniform (476, 477, 585, 589, ...). Even dynamic=True doesn't
#      help. Dynamo's symbolic-shape machinery only covers Tensor shapes.
#   2. flash_attn.cute.interface.flash_attn_varlen_func, its CUTLASS DSL
#      "tvm_ffi_provider" wrapper, creates a new internal _kwargs_wrapper code
#      object per call. Dynamo's ___check_obj_id on that code object always
#      mismatches, triggering recompiles.
#
# Both can be made compatible with torch.compile by wrapping in @dynamo.disable.
# The calls stay in eager (the kernels are already JIT-compiled by Triton/CUTLASS,
# so eager-vs-compiled doesn't change them) but the rest of the transformer block
# (LayerNorm, QKV proj, FFN, residuals, q_ln/k_ln, reshapes) is freely compilable
# by Inductor.
def _apply_rotary_packed_eager(q, k, rotary, cu_seqlens, max_seqlen):
    """Eager rotary on packed q/k, kept out of Dynamo's traced graph.

    The cache update (rotary._update_cos_sin_cache) is also done inside this
    function so its integer attribute reads (self._seq_len_cached) don't
    trigger recompiles in the outer compile-traced region.
    """
    rotary._update_cos_sin_cache(max_seqlen, device=q.device, dtype=q.dtype)
    cos = rotary._cos_cached
    sin = rotary._sin_cached
    q = _quack_apply_rotary_emb(q, cos, sin, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
    k = _quack_apply_rotary_emb(k, cos, sin, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
    return q, k


def _flash_attn_varlen_packed_eager(q, k, v, cu_seqlens, max_seqlen):
    """Eager FA4 varlen call, kept out of Dynamo's traced graph."""
    out, _ = _flash_attn_varlen_func_raw(
        q, k, v,
        cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
    )
    return out


# Apply the dynamo.disable decorator after import, only when torch._dynamo is reachable.
try:
    import torch._dynamo
    _apply_rotary_packed_eager = torch._dynamo.disable(_apply_rotary_packed_eager)
    _flash_attn_varlen_packed_eager = torch._dynamo.disable(_flash_attn_varlen_packed_eager)
except ImportError:
    pass


# Thread-local (well, contextvar) store for the (cu_seqlens, max_seqlen, indices, B, L)
# tuple set by the wrapping forward. Read by every PackedFlashAttention.forward
# call without changing the model's signature `self.attn(x, sequence_id)`.
@dataclass
class PackedCtx:
    cu_seqlens: torch.Tensor   # int32, shape (B+1,)
    max_seqlen: int
    indices: torch.Tensor      # int64, shape (nnz,), for repad
    B: int
    L: int


_PACKED_CTX: contextvars.ContextVar[Optional[PackedCtx]] = contextvars.ContextVar(
    "packed_ctx", default=None
)


def _unpad_input(hidden_states: torch.Tensor, attention_mask: torch.Tensor):
    """(B, L, ...) → (nnz, ...) packed. Returns indices + cu_seqlens + max_seqlen."""
    seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
    max_seqlen = int(seqlens.max().item())
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    cu_seqlens = torch.nn.functional.pad(
        torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0)
    )
    packed = hidden_states.flatten(0, 1)[indices]
    return packed, indices, cu_seqlens, max_seqlen


def _pad_input(packed: torch.Tensor, indices: torch.Tensor, B: int, L: int) -> torch.Tensor:
    """(nnz, ...) → (B, L, ...), zeros where pad."""
    output = torch.zeros(B * L, *packed.shape[1:], dtype=packed.dtype, device=packed.device)
    output[indices] = packed
    return output.view(B, L, *packed.shape[1:])


class PackedFlashAttention(nn.Module):
    """Drop-in replacement for `esmc.layers.attention.MultiHeadAttention`.

    Receives packed (nnz, D) input (the wrapping ESMC.forward unpadded once at
    the top). Reads cu_seqlens / max_seqlen from `_PACKED_CTX`. Calls FA4's
    `flash_attn_varlen_func` directly with quack's varlen-aware rotary helper.

    Steals sub-modules from the original `MultiHeadAttention` (shares weights,
    no parameter copy). The original module is replaced in-place after model
    construction.

    Signature compatibility: `forward(x, sequence_id)` matches the ESMC
    `UnifiedTransformerBlock.forward`'s `self.attn(x, sequence_id)` call.
    `sequence_id` is ignored (we use packed cu_seqlens from the contextvar).
    """

    def __init__(self, original_attn):
        super().__init__()
        # Steal sub-modules, share weights with the original
        self.layernorm_qkv = original_attn.layernorm_qkv
        self.q_ln = original_attn.q_ln
        self.k_ln = original_attn.k_ln
        self.rotary = original_attn.rotary  # RotaryEmbedding (cos/sin cache)
        self.out_proj = original_attn.out_proj
        self.n_heads = original_attn.n_heads
        self.d_head = original_attn.d_head
        self.d_model = original_attn.d_model

    def forward(self, x_packed: torch.Tensor, _sequence_id_unused) -> torch.Tensor:
        ctx = _PACKED_CTX.get()
        if ctx is None:
            raise RuntimeError(
                "PackedFlashAttention.forward called outside packed context. "
                "Did the ESMC.forward wrapper set _PACKED_CTX?"
            )

        # 1. QKV proj + qk-LN on packed form
        qkv = self.layernorm_qkv(x_packed)            # (nnz, 3D)
        q, k, v = qkv.chunk(3, dim=-1)                 # each (nnz, D)
        q_dtype = q.dtype
        q = self.q_ln(q).to(q_dtype)
        k = self.k_ln(k).to(q_dtype)

        # 2. Reshape to (nnz, H, D_h)
        q = q.unflatten(-1, (self.n_heads, self.d_head))
        k = k.unflatten(-1, (self.n_heads, self.d_head))
        v = v.unflatten(-1, (self.n_heads, self.d_head))

        # 3. FA4-aligned built-in rotary via quack.rotary.apply_rotary_emb.
        # Both the cache refresh and the apply call run inside the
        # dynamo.disable wrapper so torch.compile doesn't recompile-thrash on
        # max_seqlen (Python int) or self._seq_len_cached (nn.Module int attr).
        q, k = _apply_rotary_packed_eager(q, k, self.rotary, ctx.cu_seqlens, ctx.max_seqlen)

        # 4. FA4 varlen, wrapped in dynamo.disable so torch.compile doesn't
        # recompile-thrash on CUTLASS DSL's per-call _kwargs_wrapper.__code__ id.
        out = _flash_attn_varlen_packed_eager(q, k, v, ctx.cu_seqlens, ctx.max_seqlen)
        # (nnz, H, D_h)

        # 5. Reshape and out_proj
        out = out.flatten(-2, -1)                        # (nnz, D)
        return self.out_proj(out)


def install_packed_flash(decoder_tcrc_module):
    """Swap each transformer block's .attn and wrap ESMC.forward in-place.

    Call from DecoderTCRC.__init__ after the inner ESMC has been constructed
    and weights loaded. Idempotent: safe to call multiple times (will use the
    already-installed PackedFlashAttention).
    """
    if not _PACKED_FLASH_AVAILABLE:
        raise RuntimeError(
            "Packed flash attention requires flash_attn (FA4) + quack.rotary. "
            "Install with `uv pip install <flash_attn_4 wheel URL>`."
        )

    esmc = decoder_tcrc_module.model

    # 1. Replace every block's attention with the packed variant
    for block in esmc.transformer.blocks:
        if isinstance(block.attn, PackedFlashAttention):
            continue  # already installed
        block.attn = PackedFlashAttention(block.attn)

    # 2. Wrap ESMC.forward with the packing logic, only once
    if getattr(esmc, "_packed_flash_installed", False):
        return
    esmc._packed_flash_installed = True

    # Capture references the wrapped forward needs, mirroring ESMC.forward
    # semantics (src/esmc/models/esmc.py:55-79).
    pad_token_id = esmc.tokenizer.pad_token_id
    from esmc.models.esmc import ESMCOutput

    def packed_forward(self_esmc, sequence_tokens, sequence_id=None):
        # Build pad mask (matches the ESMC default)
        if sequence_id is None:
            sequence_id = sequence_tokens != pad_token_id

        # Embed → (B, L, D)
        x = self_esmc.embed(sequence_tokens)
        B, L, _D = x.shape

        # Unpad ONCE → (nnz, D)
        x_packed, indices, cu_seqlens, max_seqlen = _unpad_input(x, sequence_id)

        # Set context so PackedFlashAttention can read cu_seqlens
        token = _PACKED_CTX.set(PackedCtx(cu_seqlens, max_seqlen, indices, B, L))
        try:
            # All transformer blocks run on packed form. The ESMC
            # TransformerStack.forward calls block(x, sequence_id) for each
            # block. sequence_id is ignored by PackedFlashAttention. LayerNorm,
            # residuals, FFN are per-token and work fine on packed input.
            x_packed, _, hiddens = self_esmc.transformer(x_packed, sequence_id=sequence_id)
        finally:
            _PACKED_CTX.reset(token)

        # Pad ONCE back to (B, L, D)
        x = _pad_input(x_packed, indices, B, L)

        sequence_logits = self_esmc.sequence_head(x)
        return ESMCOutput(
            sequence_logits=sequence_logits, embeddings=x, hidden_states=hiddens
        )

    esmc.forward = types.MethodType(packed_forward, esmc)
