"""DecoderTCR-ESMC: TCR-pMHC masked-LM backbone built on Chan Zuckerberg Biohub ESMC (github.com/Biohub/esm).

`DecoderTCRC` wraps an `ESMC`. `DecoderTCRCModel` is the Lightning module
whose `.ckpt` files are released. Its `forward` returns the SAME dict contract as
the ESM2 sibling so `DecoderTCR.utils.scoring` runs unchanged against either backbone.

    model = DecoderTCRCModel.load_from_checkpoint(path, use_packed_flash=False)
    wrapper = model.model            # the DecoderTCRC wrapper (dict-returning forward)

Notable differences from the ESM2-based DecoderTCR:
- ESMC has NO `token_dropout`. The model sees `<mask>` embeddings directly.
- ESMC has NO contact head. `get_contact` is removed.
- ESMC's `sequence_logits` last dim is 64 (not the alphabet size 33). Masked-peptide
  PLL indexes amino-acid token ids in [0, 32], which line up across both heads, so the
  extra logit channels are simply never scored.
- At inference, always load with `use_packed_flash=False`: the packed-flash (FA4) path
  only accepts bf16/fp16, but inference runs fp32 by default. Weights are kernel-independent.
"""

import logging
import math

import lightning as pl
import torch

from esmc.models.esmc import ESMC
from esmc.tokenization import get_esmc_model_tokenizers
from DecoderTCR.constants import DECODERTCRC_ARCH

logger = logging.getLogger(__name__)


def _load_weights(model, path):
    """Load weights into an ESMC model.

    Supports two formats:
    - Plain state_dict (the Chan Zuckerberg Biohub ESMC .pth/.pt files at
      checkpoints/base_models/ESMC_*.pt): keys match `model.state_dict()` directly.
    - Lightning .ckpt: keys carry a "model.model." prefix from the wrapper hierarchy
      (DecoderTCRCModel.model = DecoderTCRC, and DecoderTCRC.model = ESMC).
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = {}
        for k, v in ckpt["state_dict"].items():
            if k.startswith("model.model."):
                state_dict[k.replace("model.model.", "")] = v
    else:
        # Plain state_dict
        state_dict = ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: missing keys in checkpoint: {missing}")
    if unexpected:
        print(f"Warning: unexpected keys in checkpoint: {unexpected}")


class DecoderTCRC(torch.nn.Module):
    """TCR-pMHC masked-LM backbone on Chan Zuckerberg Biohub ESMC (see module docstring)."""

    def __init__(
        self,
        base_model: str,
        init_weights: str | None = None,
        use_packed_flash: bool = False,
    ):
        super().__init__()
        if base_model not in DECODERTCRC_ARCH:
            raise ValueError(
                f"Unknown base_model '{base_model}'. "
                f"Choose from: {list(DECODERTCRC_ARCH.keys())}"
            )
        arch = DECODERTCRC_ARCH[base_model]
        # ESMC: SDPA-only (flash_attn stripped). PyTorch SDPA on H100
        # auto-dispatches to FlashAttention-2/3 kernels.
        self.model = ESMC(
            d_model=arch["d_model"],
            n_heads=arch["n_heads"],
            n_layers=arch["n_layers"],
            tokenizer=get_esmc_model_tokenizers(),
        )

        if init_weights is not None:
            logger.info("Loading weights from %s", init_weights)
            _load_weights(self.model, init_weights)
        else:
            logger.info("Constructed %s with random init (weights restored separately "
                        "when loading from a checkpoint)", base_model)

        # Optional packed-sequence FA4 path (training only, requires flash_attn + quack).
        # Never enabled at inference. See module docstring.
        if use_packed_flash:
            from DecoderTCR.model.packed_flash_attention import install_packed_flash
            print(f"Installing packed flash attention path ({arch['n_layers']} blocks rewired)")
            install_packed_flash(self)

    def forward(self, data: torch.Tensor, repr_layers=None, return_contacts: bool = False) -> dict:
        """Forward pass returning the same dict contract as the ESM2 wrapper.

        Args:
            data: int64 token tensor of shape (B, L), with CLS/EOS already in place.
            repr_layers, return_contacts: ignored, accepted only for ESM2 API parity
                so scoring code (DecoderTCR.utils.scoring) works against both backbones.

        Returns:
            {"logits": (B, L, 64), "representations": {-1: (B, L, d_model)}}
        """
        out = self.model(sequence_tokens=data)
        return {
            "logits": out.sequence_logits,
            "representations": {-1: out.embeddings},
        }

    def get_embeddings(self, data, repr_layers=None, return_contacts=False):
        """Last-layer per-residue representations, (B, L, d_model)."""
        return self.forward(data)["representations"][-1]


class DecoderTCRCModel(pl.LightningModule):
    def __init__(self, base_model, learning_rate=None, betas=(0.9, 0.999), eps=None,
                 weight_decay=None, warmup_steps=1000, init_weights=None,
                 training_config=None, lr_schedule="constant", max_steps=None,
                 cosine_min_lr_ratio=0.0, use_packed_flash=False):
        super().__init__()
        self.save_hyperparameters(ignore=["training_config"])

        self.model = DecoderTCRC(
            base_model=base_model,
            init_weights=init_weights,
            use_packed_flash=use_packed_flash,
        )
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        self.loss_fct_unreduced = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        self.epitope_weight_map = {}  # set by train.py after datamodule.setup()
        self.learning_rate = learning_rate
        self.betas = tuple(betas) if isinstance(betas, (list, tuple)) else tuple(map(float, betas.strip("()").split(",")))
        self.eps = eps
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.lr_schedule = lr_schedule
        self.max_steps = max_steps
        self.cosine_min_lr_ratio = cosine_min_lr_ratio
        self.base_model_name = base_model
        self._training_config = training_config
        self.train_dataset_size = None
        self.effective_batch_size = None

    def _masked_accuracy(self, logits, target):
        """Top-1 accuracy over masked tokens only (where target != -100)."""
        mask = target != -100
        preds = logits.argmax(dim=-1)
        return (preds[mask] == target[mask]).float().mean()

    def forward(self, data):
        return self.model(data)

    def get_embeddings(self, data):
        return self.model.get_embeddings(data)

    def training_step(self, batch, batch_idx):
        out = self.model(batch['masked_input'])
        logits = out['logits']
        bs, seq_len, vocab = logits.shape
        target = batch['masked_target']

        if self.epitope_weight_map:
            # Per-sample weighted loss
            token_loss = self.loss_fct_unreduced(logits.view(-1, vocab), target.view(-1))
            token_loss = token_loss.view(bs, seq_len)
            mask = target != -100
            counts = mask.sum(dim=1).clamp(min=1)
            per_sample_loss = (token_loss * mask).sum(dim=1) / counts
            peptides = [sp["peptide"] for sp in batch["seq_profile"]]
            weights = torch.tensor(
                [self.epitope_weight_map.get(p, 1.0) for p in peptides],
                device=per_sample_loss.device, dtype=per_sample_loss.dtype,
            )
            loss = (per_sample_loss * weights).sum() / weights.sum()
            self.log('train/epitope_weight_max', weights.max(), sync_dist=False)
            self.log('train/epitope_weight_min', weights.min(), sync_dist=False)
            self.log('train/epitope_weight_mean', weights.mean(), sync_dist=False)
        else:
            logits = logits.view(-1, vocab)
            target = target.view(-1)
            loss = self.loss_fct(logits, target)

        logits_flat = out['logits'].view(-1, vocab)
        target_flat = batch['masked_target'].view(-1)
        self.log('train/loss', loss, prog_bar=True, sync_dist=True)
        self.log('train/perplexity', torch.exp(loss), sync_dist=True)
        self.log('train/accuracy', self._masked_accuracy(logits_flat, target_flat), sync_dist=True)
        if self.train_dataset_size and self.effective_batch_size:
            equiv_epoch = self.global_step * self.effective_batch_size / self.train_dataset_size
            self.log('train/equiv_epoch', equiv_epoch, sync_dist=False)
        return loss

    def validation_step(self, batch, batch_idx):
        try:
            out = self.model(batch['masked_input'])
            logits = out['logits'].view(-1, out['logits'].size()[-1])
            target = batch['masked_target'].view(-1)
            loss = self.loss_fct(logits, target)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  OOM at validation batch {batch_idx}, skipping")
                torch.cuda.empty_cache()
                return None
            raise

        self.log('val/loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('val/perplexity', torch.exp(loss), on_epoch=True, sync_dist=True)
        self.log('val/accuracy', self._masked_accuracy(logits, target), on_epoch=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        """Run test-set evaluation (generalization tracking).

        Contact eval is omitted because ESMC has no contact head.
        """
        dm = getattr(self.trainer, "datamodule", None)
        if dm is None or not hasattr(dm, 'get_test_val_dataloader'):
            return
        test_dl = dm.get_test_val_dataloader()
        if test_dl is None:
            return

        total_loss = 0.0
        total_correct = 0
        total_masked = 0
        n_batches = 0

        for batch in test_dl:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            try:
                with torch.no_grad():
                    out = self.model(batch['masked_input'])
                    logits = out['logits'].view(-1, out['logits'].size(-1))
                    target = batch['masked_target'].view(-1)
                    loss = self.loss_fct(logits, target)

                    mask = target != -100
                    preds = logits.argmax(dim=-1)
                    total_correct += (preds[mask] == target[mask]).sum().item()
                    total_masked += mask.sum().item()
                    total_loss += loss.item()
                    n_batches += 1
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    continue
                raise

        if n_batches > 0:
            avg_loss = total_loss / n_batches
            self.log('test/loss', avg_loss, sync_dist=True)
            self.log('test/perplexity', math.exp(avg_loss), sync_dist=True)
            self.log('test/accuracy', total_correct / max(total_masked, 1), sync_dist=True)

    def on_train_start(self):
        dm = self.trainer.datamodule
        if dm is not None and hasattr(dm, 'train_dataset_size'):
            self.train_dataset_size = dm.train_dataset_size
        if hasattr(self.trainer, 'accumulate_grad_batches'):
            num_gpus = self.trainer.num_devices * self.trainer.num_nodes
            bs = self.trainer.datamodule.batch_size if dm else 8
            self.effective_batch_size = bs * num_gpus * self.trainer.accumulate_grad_batches

    def on_train_epoch_start(self):
        torch.cuda.empty_cache()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
        )

        warmup_steps = self.warmup_steps
        schedule = self.lr_schedule
        max_steps = self.max_steps
        min_ratio = self.cosine_min_lr_ratio

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            if schedule == "cosine" and max_steps is not None:
                progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
                cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
                return min_ratio + (1.0 - min_ratio) * cosine
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def on_save_checkpoint(self, checkpoint):
        if self._training_config is not None:
            checkpoint["training_config"] = self._training_config.to_dict()
        else:
            checkpoint["training_config"] = {
                "base_model": self.base_model_name,
                "learning_rate": self.learning_rate,
                "warmup_steps": self.warmup_steps,
                "betas": self.betas,
                "eps": self.eps,
                "weight_decay": self.weight_decay,
            }
