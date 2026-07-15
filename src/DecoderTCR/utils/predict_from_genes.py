#!/usr/bin/env python3
"""Score TCR-pMHC pairs from V/J genes + CDR3 + HLA allele + peptide (no full sequences).

Supply the TCR's V/J genes and CDR3s, the HLA allele, and the peptide. DecoderTCR reconstructs
the full complex (stitchr/thimble for the TCR, HLA looked up by allele from the training
reference) and scores masked-peptide PLL.

Reconstruction (stitchr) is installed by default. Fetch IMGT germline data once:
    uv run stitchrdl -s human

Single pair (flags):
    python -m DecoderTCR.utils.predict_from_genes \\
        --trav TRAV21 --traj TRAJ6 --cdr3a CAVRPGGAGPFFVVF \\
        --trbv TRBV7-9 --trbj TRBJ2-7 --cdr3b CASSLGQAYEQYF \\
        --hla 'HLA-B*27:05' --peptide LRVMMLAPF -d cuda:0

Batch (a CSV in, a scored CSV out):
    python -m DecoderTCR.utils.predict_from_genes -i genes.csv -o out.csv -d cuda:0

CSV columns / flags (case-insensitive, aliases accepted): trav, traj, cdr3a, trbv, trbj, cdr3b,
hla, peptide (optional name, label). Output = input + reconstructed HLA_a/HLA_b/TCR_a/TCR_b +
ok / *_reason flags + pll_<model>.
"""

import argparse
from pathlib import Path

import pandas as pd
import torch

from DecoderTCR.utils.model_zoo import MODEL_ZOO

_FIELDS = ("trav", "traj", "cdr3a", "trbv", "trbj", "cdr3b", "hla", "peptide")


def main():
    p = argparse.ArgumentParser(
        description="DecoderTCR: score from V/J genes + CDR3 + HLA allele + peptide")
    p.add_argument("-i", "--input", type=Path, default=None, help="input CSV of pairs (batch mode)")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="output CSV (batch mode, optional in single-pair mode)")
    g = p.add_argument_group("single pair (use instead of -i)")
    for f in _FIELDS:
        g.add_argument(f"--{f}", default=None, help=f"{f} for a single pair")
    g.add_argument("--name", default=None, help="optional name for the single pair")
    p.add_argument("-m", "--model", default=None, help=f"registry model name, one of: {list(MODEL_ZOO)}")
    p.add_argument("-c", "--checkpoint", default=None, help="explicit checkpoint path (requires --backbone and --arch)")
    p.add_argument("--backbone", choices=["esm2", "esmc"], default=None, help="backbone for --checkpoint")
    p.add_argument("--arch", default=None, help="arch key for --checkpoint")
    p.add_argument("-d", "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    from DecoderTCR.reconstruct import score_from_components

    single = {f: getattr(args, f) for f in _FIELDS}
    if any(v is not None for v in single.values()):
        missing = [f for f, v in single.items() if not v]
        if missing:
            raise SystemExit(f"single-pair mode needs all of {list(_FIELDS)} (missing: {missing})")
        if args.name:
            single["name"] = args.name
        out = score_from_components(single, model=args.model, device=args.device,
                                    checkpoint=args.checkpoint, backbone=args.backbone, arch=args.arch)
        col = next(c for c in out.columns if c.startswith("pll_"))
        r = out.iloc[0]
        if bool(r["ok"]):
            print(f"PLL = {r[col]:.4f}   ({col})")
        else:
            print(f"reconstruction failed: {r.get('tcr_reason') or r.get('hla_reason') or 'unknown'}")
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(args.output, index=False)
            print(f"Wrote {args.output}")
        return

    if args.input is None or args.output is None:
        raise SystemExit("batch mode needs -i INPUT.csv and -o OUTPUT.csv "
                         "(or pass single-pair flags like --trav ... --peptide ...)")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")
    out = score_from_components(df, model=args.model, device=args.device,
                                checkpoint=args.checkpoint, backbone=args.backbone, arch=args.arch)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    n_ok = int(out["ok"].sum())
    n_fail = len(out) - n_ok
    print(f"Reconstructed + scored {n_ok}/{len(out)} pairs.")
    if n_fail:
        print(f"  {n_fail} failed reconstruction:")
        for _, r in out[~out["ok"]].iterrows():
            reason = r.get("tcr_reason") or r.get("hla_reason") or "unknown"
            print(f"    {r.get('name', '?')}: {str(reason)[:120]}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
