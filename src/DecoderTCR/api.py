"""In-process scoring and embedding APIs. The programmatic counterpart to the CLIs.

    import DecoderTCR as dt

    # score TCR-pMHC pairs inline, no CSV or shelling out:
    df  = dt.score([{"HLA_a": HLA, "Peptide": "YLQPRTFLL", "TCR_a": TCRA, "TCR_b": TCRB}])
    pll = dt.score_one("YLQPRTFLL", hla_a=HLA, tcr_a=TCRA, tcr_b=TCRB)

    # embeddings (per-complex, per-region, or per-residue):
    vecs    = dt.embed(df)                    # (N, d) mean-pooled
    regions = dt.embed(df, pool="regions")    # [{"Peptide": (d,), "TCR_b": (d,), ...}, ...]

All three reuse the same load() + tokenizer + forward as the predict CLIs, and run against
either backbone (ESM2 DecoderTCR or ESMC DecoderTCR-ESMC). Pass `model` as a registry name, a
module already returned by load() (then also pass `num_layers`), or None for the default.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from DecoderTCR.model.tokenizer import TCRpMHCTokenizer
from DecoderTCR.utils._predict_common import build_entries
from DecoderTCR.utils.model_zoo import DEFAULT_MODEL, load
from DecoderTCR.utils.scoring import run_pll_benchmark


def _resolve_device(device) -> torch.device:
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        dev = torch.device("cpu")
    return dev


def _resolve_model(model, num_layers, device, checkpoint, backbone, arch):
    """Return (module, num_layers, device, col_name, backbone).

    `model` may be a loaded module (then `num_layers` is required, and the module's own
    device is used, so the `device` argument is ignored), a registry name, or None (default).
    """
    if isinstance(model, torch.nn.Module):
        if num_layers is None:
            raise ValueError(
                "When passing a loaded module as `model`, also pass `num_layers` "
                "(the second value returned by DecoderTCR.load())."
            )
        p = next(model.parameters(), None)
        dev = p.device if p is not None else _resolve_device(device)
        bb = "esmc" if type(model).__name__ == "DecoderTCRC" else "esm2"
        return model, num_layers, dev, "model", bb
    dev = _resolve_device(device)
    if checkpoint is not None:
        mdl, n = load(device=dev, checkpoint=checkpoint, backbone=backbone, arch=arch)
        return mdl, n, dev, (model or Path(checkpoint).stem), backbone
    from DecoderTCR.utils.model_zoo import resolve
    bb = resolve(model or DEFAULT_MODEL).backbone
    mdl, n = load(model, device=dev)
    return mdl, n, dev, (model or DEFAULT_MODEL), bb


def _to_entries(data, with_tcr):
    """Normalize DataFrame / list-of-dicts / single dict into (frame, entries)."""
    if isinstance(data, pd.DataFrame):
        df = data.reset_index(drop=True)
        return df, build_entries(df, with_tcr)
    if isinstance(data, dict):
        data = [data]
    data = list(data)
    if data and isinstance(data[0], dict) and "sequences" in data[0]:
        rows = [{**e.get("sequences", {}), **e.get("meta_data", {})} for e in data]
        return pd.DataFrame(rows), data
    df = pd.DataFrame(data)
    return df, build_entries(df, with_tcr)


def score(data, model=None, *, num_layers=None, device="cuda", with_tcr=True,
          mask_region="peptide", name=None,
          checkpoint=None, backbone=None, arch=None, return_dataframe=True):
    """Score TCR-pMHC (or pMHC-only) pairs in-process.

    `data`: a DataFrame (case-insensitive columns HLA_a/hla, HLA_b/b2m, Peptide, TCR_a, TCR_b,
    plus optional label/epitope/allele/clone), a list of such flat dicts, a list of pre-built
    entry dicts, or a single dict.
    `model`: a registry name, a module already returned by load() (then pass `num_layers`), or
    None for the default (DecoderTCR-ESMC_600M).
    `with_tcr`: set False for pMHC-only scoring (ignores any TCR columns).

    Returns the input as a DataFrame with an added `pll_<model>` column, or a numpy array of
    PLLs if `return_dataframe=False`. The score is NaN where the peptide is empty.
    """
    df, entries = _to_entries(data, with_tcr)
    mdl, n, dev, col, _ = _resolve_model(model, num_layers, device, checkpoint, backbone, arch)
    col = name or col
    scores = run_pll_benchmark(mdl, n, entries, mask_region, dev)
    if not return_dataframe:
        return scores
    out = df.copy()
    out[f"pll_{col}"] = scores
    return out


def score_one(peptide, hla_a, *, hla_b="", tcr_a="", tcr_b="",
              model=None, num_layers=None, device="cuda", mask_region="peptide",
              checkpoint=None, backbone=None, arch=None):
    """Score a single pair and return the PLL float. Leave tcr_a/tcr_b blank for pMHC-only."""
    entry = {"sequences": {"HLA_a": hla_a, "HLA_b": hla_b, "Peptide": peptide,
                           "TCR_a": tcr_a, "TCR_b": tcr_b},
             "pocket_idx": {}, "meta_data": {}}
    arr = score([entry], model, num_layers=num_layers, device=device, mask_region=mask_region,
                checkpoint=checkpoint, backbone=backbone, arch=arch, return_dataframe=False)
    return float(arr[0])


@torch.no_grad()
def embed(data, model=None, *, num_layers=None, device="cuda", with_tcr=True,
          pool="mean", layer=None, checkpoint=None, backbone=None, arch=None):
    """Backbone embeddings for TCR-pMHC (or pMHC-only) complexes.

    `data`: same inputs accepted by `score`.
    `pool`:
      "mean"    -> np.ndarray (N, d): per-complex mean over residues (excludes CLS/EOS). [default]
      "regions" -> list[dict]: {region: (d,) np.ndarray} mean per present region. Keys are the
                   present components: HLA_a, HLA_b (the B2M chain), Peptide, TCR_a, TCR_b.
      None      -> list[np.ndarray]: per-residue (L_i, d) embeddings (excludes CLS/EOS).
    `layer`: transformer layer for the ESM2 backbone (0..num_layers, with None or -1 = last). The
    ESMC backbone exposes only its final layer and ignores `layer`.

    On empty input the pool="mean" return is an array of shape (0, 0). Feature width is only
    known once at least one complex is scored.
    """
    if pool not in ("mean", "regions", None):
        raise ValueError(f"pool must be 'mean', 'regions', or None, got {pool!r}")
    _, entries = _to_entries(data, with_tcr)
    mdl, n, dev, _, bb = _resolve_model(model, num_layers, device, checkpoint, backbone, arch)
    req_layer = n if layer in (None, -1) else layer          # None / -1 -> last layer
    if bb == "esm2" and not (0 <= req_layer <= n):
        raise ValueError(f"layer must be in [0, {n}] for the ESM2 backbone, got {layer}")
    if bb == "esmc" and layer not in (None, -1, n):
        import warnings
        warnings.warn("The ESMC backbone exposes only its final layer. `layer` is ignored.",
                      stacklevel=2)

    results: list = []
    for entry in entries:
        tok = TCRpMHCTokenizer(entry, mask_probs=None, use_sep=False)  # unmasked
        tokens = tok.original_ids.unsqueeze(0).to(dev)
        out = mdl(tokens, repr_layers=[req_layer], return_contacts=False)
        reps = out["representations"]
        rep = reps[req_layer] if req_layer in reps else reps[-1]  # ESM2 keyed by layer, ESMC by -1
        emb = rep[0].float().cpu().numpy()                        # (L+2, d)
        full_len = len(tok.full_seq)
        if full_len == 0:                                         # fully-empty complex
            d = emb.shape[1]
            results.append(np.full(d, np.nan, dtype=np.float32) if pool == "mean"
                           else emb[0:0] if pool is None else {})
            continue
        res = emb[1:1 + full_len]                                 # drop CLS (0) and trailing EOS
        if pool == "mean":
            results.append(res.mean(axis=0))
        elif pool is None:
            results.append(res)
        else:  # regions (region_slices are sequence-space, +1 for the CLS token)
            results.append({r: emb[sl.start + 1: sl.stop + 1].mean(axis=0)
                            for r, sl in tok.region_slices.items()})

    if pool == "mean":
        return np.stack(results) if results else np.empty((0, 0), dtype=np.float32)
    return results
