"""Model registry + unified loader for both DecoderTCR backbones.

V0.3 IS the ESMC **DecoderTCR-ESMC** line (updated training + updated backbone). It is the
default line. The original ESM2 V0.1 weights are kept under their original names
(`DecoderTCR_650M`, `DecoderTCR_3B`) for paper reproduction and backward compatibility.

    from DecoderTCR.utils.model_zoo import load
    model, n_layers = load()                      # default = DecoderTCR-ESMC_600M (common GPUs)
    model, n_layers = load("DecoderTCR-ESMC_6B", device="cuda:0")   # larger variant (80 GB GPU)

`load()` returns `(module, num_layers)` where `module.forward(tokens, ...)` yields
`{"logits": ...}` for BOTH backbones, exactly what `run_pll_benchmark` expects.
Checkpoints are populated by `scripts/download_weights.py` into the paths below.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from DecoderTCR.constants import ESM2_ARCH, DECODERTCRC_ARCH

# repo root: src/DecoderTCR/utils/model_zoo.py -> parents[3]
REPO_ROOT = Path(__file__).resolve().parents[3]

# Default when no model is specified: the 600M ESMC model. Strong across the
# benchmarks and runs on common GPUs (≤24 GB). The 6B is a larger variant that needs an 80 GB
# GPU (it only edges out 600M on PRP prioritization), so it is opt-in.
DEFAULT_MODEL = "DecoderTCR-ESMC_600M"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    backbone: str       # "esm2" | "esmc"
    arch: str           # key into ESM2_ARCH (esm2) or DECODERTCRC_ARCH (esmc)
    ckpt: str           # checkpoint path relative to repo root
    version: str        # "V0.3" | "V0.1"
    note: str = ""

    @property
    def ckpt_path(self) -> Path:
        return REPO_ROOT / self.ckpt


MODEL_ZOO: dict[str, ModelSpec] = {
    # --- V0.3 ESMC (DecoderTCR-ESMC): DecoderTCR-ESMC_600M is the default ---
    "DecoderTCR-ESMC_6B":   ModelSpec("DecoderTCR-ESMC_6B",   "esmc", "DecoderTCRC_6B",
                                   "checkpoints/DecoderTCR-ESMC-V0.3/6B.ckpt",   "V0.3", "larger variant (80 GB GPU), edges out 600M on PRP prioritization"),
    "DecoderTCR-ESMC_600M": ModelSpec("DecoderTCR-ESMC_600M", "esmc", "DecoderTCRC_600M",
                                   "checkpoints/DecoderTCR-ESMC-V0.3/600M.ckpt", "V0.3", "default, runs on ≤24 GB GPUs"),
    "DecoderTCR-ESMC_300M": ModelSpec("DecoderTCR-ESMC_300M", "esmc", "DecoderTCRC_300M",
                                   "checkpoints/DecoderTCR-ESMC-V0.3/300M.ckpt", "V0.3", "lightest ESMC"),
    # --- V0.1 ESM2, original public weights, kept for paper reproduction ---
    # (original names preserved so existing V0.1 code keeps working unchanged)
    "DecoderTCR_3B":   ModelSpec("DecoderTCR_3B",   "esm2", "ESM2_3B",
                                 "checkpoints/DecoderTCR-ESM2-V0.1/3B_DecoderTCR.ckpt",   "V0.1", "paper reproduction"),
    "DecoderTCR_650M": ModelSpec("DecoderTCR_650M", "esm2", "ESM2_650M",
                                 "checkpoints/DecoderTCR-ESM2-V0.1/650M_DecoderTCR.ckpt", "V0.1", "paper reproduction"),
}


def num_layers(spec: ModelSpec) -> int:
    if spec.backbone == "esm2":
        return ESM2_ARCH[spec.arch]["num_layers"]
    return DECODERTCRC_ARCH[spec.arch]["n_layers"]


def resolve(name: str) -> ModelSpec:
    """Look up a friendly name in the registry."""
    if name not in MODEL_ZOO:
        raise KeyError(
            f"Unknown model '{name}'. Available: {list(MODEL_ZOO)}. "
            f"Or pass a checkpoint path with backbone+arch overrides."
        )
    return MODEL_ZOO[name]


def load(
    name: str | None = None,
    device: str | torch.device = "cuda",
    *,
    checkpoint: str | Path | None = None,
    backbone: str | None = None,
    arch: str | None = None,
) -> tuple[torch.nn.Module, int]:
    """Load a model and return (module, num_layers).

    With no arguments, loads the V0.3 default (`DEFAULT_MODEL`). Pass a registry `name`,
    or an explicit `checkpoint` + `backbone` ("esm2"|"esmc") + `arch` to load an
    arbitrary checkpoint.

    The returned module has a forward returning {"logits": ...} for both backbones:
      - esm2: the raw ESM2 (model.model.model)
      - esmc: the DecoderTCRC wrapper (model.model)
    """
    if checkpoint is not None:
        if backbone is None or arch is None:
            raise ValueError("When passing `checkpoint`, also pass `backbone` and `arch`.")
        spec = ModelSpec(name or Path(checkpoint).stem, backbone, arch, str(checkpoint), "custom")
        ckpt_path = Path(checkpoint)
    else:
        spec = resolve(name or DEFAULT_MODEL)
        ckpt_path = spec.ckpt_path

    if not Path(ckpt_path).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Fetch weights first:  uv run python scripts/download_weights.py --models {spec.name}\n"
            f"(see checkpoints/README.md)."
        )

    device = torch.device(device)
    n = num_layers(spec)

    # init_weights=None overrides the saved hyperparameter (which points at the
    # original Phase-1 pretrain checkpoint, absent from a release). The released
    # .ckpt already holds the full fine-tuned weights, loaded by load_from_checkpoint
    # over the random init. The Phase-1 path is never needed.
    if spec.backbone == "esm2":
        from DecoderTCR.model.DecoderTCR import DecoderTCRModel
        lit = DecoderTCRModel.load_from_checkpoint(
            str(ckpt_path), base_model=spec.arch, init_weights=None, map_location=device,
        )
        lit.eval()
        lit.to(device)
        if device.type == "cpu":
            lit.float()                    # CPU runs fp32 (half-precision matmul is slow/unsupported on CPU)
        return lit.model.model, n          # raw ESM2

    elif spec.backbone == "esmc":
        from DecoderTCR.model.DecoderTCRC import DecoderTCRCModel
        # Force standard SDPA at inference: the packed-flash (FA4) kernel only
        # accepts bf16/fp16, but inference runs fp32. Weights are kernel-independent.
        lit = DecoderTCRCModel.load_from_checkpoint(
            str(ckpt_path), init_weights=None, map_location=device, use_packed_flash=False,
        )
        lit.eval()
        lit.to(device)
        if device.type == "cpu":
            lit.float()                    # CPU runs fp32 (half-precision matmul is slow/unsupported on CPU)
        return lit.model, n                # DecoderTCRC wrapper (dict-returning forward)

    raise ValueError(f"Unknown backbone '{spec.backbone}' (expected 'esm2' or 'esmc').")
