#!/usr/bin/env python3
"""Score pMHC binding with masked-peptide PLL (no TCR).

Input CSV columns (case-insensitive): peptide, hla (the HLA heavy chain). Optional:
b2m (HLA light chain / B2M), label, epitope, allele. TCR chains are left blank.
Output is the input CSV plus a `pll_<model>` column. Higher PLL = more binder-like.

Usage:
    python -m DecoderTCR.utils.predict_pMHC \\
        -i input.csv -o output.csv -m DecoderTCR-ESMC_300M -d cuda:0
"""

import argparse

from DecoderTCR.utils._predict_common import add_common_args, run_predict


def main():
    p = argparse.ArgumentParser(description="DecoderTCR: pMHC binding scoring")
    add_common_args(p)
    args = p.parse_args()
    run_predict(args, with_tcr=False)


if __name__ == "__main__":
    main()
