"""Per-model load + score smoke test for the DecoderTCR V0.3 release.

For each registry model: reconstruct and score the gene-input sample (genes_pairs.csv) with
masked-peptide PLL, assert the scores are finite, and report per-epitope binder/non-binder
separation. The 6B model needs an 80 GB GPU.

    uv run python scripts/smoke_test.py                 # all 5 models
    uv run python scripts/smoke_test.py DecoderTCR_650M DecoderTCR-ESMC_300M
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
import DecoderTCR as dt                                       # noqa: E402
from DecoderTCR.utils.model_zoo import MODEL_ZOO              # noqa: E402


def main():
    names = sys.argv[1:] or list(MODEL_ZOO)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  models={names}\n")
    pairs = pd.read_csv(ROOT / "Demo/sample_data/genes_pairs.csv")

    ok = True
    for name in names:
        spec = MODEL_ZOO[name]
        print(f"=== {name}  [{spec.backbone} / {spec.arch}] ===")
        try:
            scored = dt.score_from_components(pairs, model=name, device=device)
            valid = scored[scored.ok]
            s = valid[f"pll_{name}"].to_numpy()
            assert len(s) and np.isfinite(s).all(), "empty or non-finite scores"
            sep = []
            for pep, g in valid.groupby("peptide"):
                if g.label.nunique() == 2:
                    sep.append(f"{pep} pos={g[g.label == 1][f'pll_{name}'].mean():.2f} "
                               f"neg={g[g.label == 0][f'pll_{name}'].mean():.2f}")
            print(f"    {len(valid)}/{len(scored)} ok | PLL [{s.min():.3f},{s.max():.3f}] | "
                  + " | ".join(sep))
        except Exception as e:
            ok = False
            print(f"    FAILED: {type(e).__name__}: {e}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print()

    print("SMOKE OK" if ok else "SMOKE FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
