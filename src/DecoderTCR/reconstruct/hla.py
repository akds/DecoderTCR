"""HLA allele -> (HLA_a, HLA_b) lookup, byte-identical to the training distribution.

The reference (data/hla_reference.json) was extracted from the V1 training data
(VDJdb_full.json): one (HLA_a, HLA_b) per allele key. Keys look like 'B2705__B2M'.
For class-I alleles HLA_b is the 100-aa B2M. For class-II it is the MHC-II beta chain
(as used in training). Using these guarantees the HLA/B2M form matches what the model
saw at train time.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

_REF_PATH = Path(__file__).resolve().parent / "data" / "hla_reference.json"


@lru_cache(maxsize=1)
def _reference() -> dict[str, dict[str, str]]:
    with open(_REF_PATH) as f:
        return json.load(f)


def normalize_allele_key(allele: str) -> str:
    """Normalize a user allele string to the reference key.

    Accepts e.g. 'HLA-B*27:05', 'B*27:05', 'B27:05', 'B2705', 'B2705__B2M'
    -> 'B2705__B2M'. Strips the optional 'HLA-' prefix, '*' and ':', uppercases.
    """
    a = allele.strip().upper()
    a = re.sub(r"^HLA[-_]?", "", a)        # drop optional HLA- / HLA_ prefix
    if a.endswith("__B2M"):
        a = a[:-5]
    a = a.replace("*", "").replace(":", "").replace("-", "")
    return f"{a}__B2M"


def lookup_hla(allele: str) -> tuple[str, str]:
    """Return (HLA_a, HLA_b) for an allele. Raises KeyError with guidance if absent."""
    key = normalize_allele_key(allele)
    ref = _reference()
    if key not in ref:
        # offer near matches on the same locus to help the user
        locus = key[0]
        near = sorted(k for k in ref if k.startswith(locus))[:12]
        raise KeyError(
            f"HLA allele '{allele}' (key '{key}') is not in the training reference "
            f"({len(ref)} alleles). Scoring requires an HLA sequence matching the "
            f"training distribution. Same-locus alleles available: {near} ... "
            f"(see DecoderTCR.reconstruct.list_alleles())."
        )
    v = ref[key]
    return v["HLA_a"], v["HLA_b"]


def list_alleles() -> list[str]:
    """All available allele keys (e.g. 'B2705__B2M')."""
    return sorted(_reference())
