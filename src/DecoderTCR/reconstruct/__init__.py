"""User-friendly reconstruction: V/J genes + CDR3 + HLA allele + peptide -> score.

Instead of supplying full HLA_a/HLA_b/Peptide/TCR_a/TCR_b sequences, supply the
components and let DecoderTCR reconstruct the full complex:

  - TCR alpha/beta full chains: stitched from (V gene, J gene, CDR3) via stitchr/thimble
    (same invocation used to build the benchmark sets), including the IMGT leader.
  - HLA_a + HLA_b: looked up by allele from the training reference (byte-identical to
    what the model was trained on), so the HLA/B2M form matches the training distribution.

Installed by default. Fetch IMGT germline data once with `uv run stitchrdl -s human`.

    from DecoderTCR.reconstruct import score_from_components
    df = score_from_components([
        {"trav": "TRAV21", "traj": "TRAJ6", "cdr3a": "CAVRPGGAGPFFVVF",
         "trbv": "TRBV7-9", "trbj": "TRBJ2-7", "cdr3b": "CASSLGQAYEQYF",
         "hla": "HLA-B*27:05", "peptide": "LRVMMLAPF"}],
        model="DecoderTCR-ESMC_6B", device="cuda:0")
"""

from DecoderTCR.reconstruct.hla import lookup_hla, list_alleles, normalize_allele_key
from DecoderTCR.reconstruct.tcr import stitch_tcrs, normalize_gene
from DecoderTCR.reconstruct.components import (
    score_from_components,
    reconstruct_components,
)

__all__ = [
    "score_from_components",
    "reconstruct_components",
    "lookup_hla",
    "list_alleles",
    "normalize_allele_key",
    "stitch_tcrs",
    "normalize_gene",
]
