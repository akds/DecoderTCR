"""Shared logic for the predict_TpM / predict_pMHC CLIs.

Reads a CSV of TCR-pMHC (or pMHC-only) rows, computes masked-peptide PLL with a
chosen model, and writes the input back out with an added `pll_<model>` column.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from DecoderTCR.utils.model_zoo import MODEL_ZOO, DEFAULT_MODEL, load
from DecoderTCR.utils.scoring import run_pll_benchmark


def _col(df: pd.DataFrame, *names: str, default: str = "") -> pd.Series:
    """Fetch the first present column among case-insensitive `names`."""
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return df[lower[n.lower()]].fillna("").astype(str)
    return pd.Series([default] * len(df), index=df.index)


def build_entries(df: pd.DataFrame, with_tcr: bool) -> list[dict]:
    """Build scoring-entry dicts from a dataframe.

    Recognised columns (case-insensitive):
      HLA_a / hla / hla_chain   -> HLA heavy chain (required)
      HLA_b / b2m               -> B2M / MHC light chain (optional)
      Peptide / peptide         -> peptide to score (required)
      TCR_a, TCR_b              -> TCR alpha/beta (TpM only)
      label, epitope, allele, clone_id, clone -> carried into meta_data
    """
    hla_a = _col(df, "HLA_a", "hla", "hla_chain", "hla_seq")
    hla_b = _col(df, "HLA_b", "b2m", "beta2m")
    pep = _col(df, "Peptide", "peptide")
    tcr_a = _col(df, "TCR_a", "tcra") if with_tcr else pd.Series([""] * len(df))
    tcr_b = _col(df, "TCR_b", "tcrb") if with_tcr else pd.Series([""] * len(df))
    label = _col(df, "label", default="0")
    epitope = _col(df, "epitope")
    allele = _col(df, "allele", "hla_allele")
    clone = _col(df, "clone_id", "clone")

    entries = []
    for i in range(len(df)):
        ep = epitope.iloc[i] or pep.iloc[i]  # default epitope to the peptide itself
        try:
            lab = int(float(label.iloc[i])) if label.iloc[i] != "" else 0
        except ValueError:
            lab = 0
        entries.append({
            "sequences": {
                "HLA_a": hla_a.iloc[i],
                "HLA_b": hla_b.iloc[i],
                "Peptide": pep.iloc[i],
                "TCR_a": tcr_a.iloc[i],
                "TCR_b": tcr_b.iloc[i],
            },
            "pocket_idx": {},  # not needed for peptide masking
            "meta_data": {
                "label": lab,
                "epitope": ep,
                "hla_allele": allele.iloc[i],
                "clone_id": clone.iloc[i],
            },
        })
    return entries


def run_predict(args: argparse.Namespace, with_tcr: bool) -> None:
    device = torch.device(args.device)
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = torch.device("cpu")

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")
    entries = build_entries(df, with_tcr=with_tcr)

    if args.checkpoint:
        if not (args.backbone and args.arch):
            raise SystemExit("--checkpoint requires --backbone and --arch")
        model, n_layers = load(device=device, checkpoint=args.checkpoint,
                               backbone=args.backbone, arch=args.arch)
        col_name = args.model or Path(args.checkpoint).stem
    else:
        col_name = args.model or DEFAULT_MODEL
        model, n_layers = load(col_name, device=device)

    scores = run_pll_benchmark(model, n_layers, entries, "peptide", device)

    out = df.copy()
    out[f"pll_{col_name}"] = scores
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    n_valid = int(np.sum(~np.isnan(scores)))
    print(f"Scored {n_valid}/{len(scores)} rows  "
          f"(PLL range [{np.nanmin(scores):.4f}, {np.nanmax(scores):.4f}])")
    print(f"Wrote {out_path}")


def add_common_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("-i", "--input", required=True, type=Path, help="Input CSV")
    p.add_argument("-o", "--output", required=True, type=Path, help="Output CSV")
    p.add_argument("-m", "--model", default=None,
                   help=f"Registry model name. One of: {list(MODEL_ZOO)}")
    p.add_argument("-c", "--checkpoint", default=None,
                   help="Explicit checkpoint path (requires --backbone and --arch)")
    p.add_argument("--backbone", choices=["esm2", "esmc"], default=None,
                   help="Backbone for --checkpoint")
    p.add_argument("--arch", default=None,
                   help="Arch key for --checkpoint (e.g. ESM2_3B or DecoderTCRC_6B)")
    p.add_argument("-d", "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p
