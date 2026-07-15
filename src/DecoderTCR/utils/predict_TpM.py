#!/usr/bin/env python3
"""Score TCR-pMHC interactions with masked-peptide PLL.

Input CSV columns (case-insensitive): HLA_a, HLA_b, Peptide, TCR_a, TCR_b
(optional: label, epitope, allele, clone_id). Output is the input CSV plus a
`pll_<model>` column. Higher PLL means more binder-like.

Usage:
    python -m DecoderTCR.utils.predict_TpM \\
        -i input.csv -o output.csv -m DecoderTCR_650M -d cuda:0

    # arbitrary checkpoint:
    python -m DecoderTCR.utils.predict_TpM \\
        -i input.csv -o output.csv \\
        -c checkpoints/DecoderTCR-ESMC-V0.3/6B.ckpt \\
        --backbone esmc --arch DecoderTCRC_6B -d cuda:0
"""

import argparse

from DecoderTCR.utils._predict_common import add_common_args, run_predict


def main():
    p = argparse.ArgumentParser(description="DecoderTCR: TCR-pMHC interaction scoring")
    add_common_args(p)
    args = p.parse_args()
    run_predict(args, with_tcr=True)


if __name__ == "__main__":
    main()
