"""Unified tokenizer for TCR-pMHC structured entries.

Accepts a raw JSON entry dict, builds the concatenated sequence,
computes region indices, applies masking at string level, and
tokenizes via ESM's BatchConverter.

All region indices are in **sequence space** (no CLS offset).
Token-level outputs include CLS/EOS automatically.

Usage:
    tok = TCRpMHCTokenizer(entry, mask_probs={'bg': 0.15, 'peptide': 0.5})
    tok.input_ids       # masked token tensor (1D)
    tok.labels          # target tensor (-100 for unmasked)
    tok.original_ids    # unmasked token tensor (1D)
    tok.get_region_seq('peptide')   # "HPNGYKSLSTL"
    tok.get_region_ids('peptide')   # tensor of original token IDs
"""

from __future__ import annotations

import numpy as np
import torch

from DecoderTCR.constants import ALPHABET, BATCH_CONVERTER, MASK_IDX, SEP_PLACEHOLDER, SEP_IDX

SEQUENCE_ORDER = ["HLA_a", "HLA_b", "Peptide", "TCR_a", "TCR_b"]


def build_sequence(
    sequences: dict[str, str],
    use_sep: bool = True,
) -> tuple[str, dict[str, slice], list[int]]:
    """Build concatenated sequence with optional chain separator placeholders.

    Args:
        sequences: {component_name: amino_acid_string}
        use_sep: Whether to insert separator placeholders between components

    Returns:
        full_seq: Concatenated string (with placeholder chars if use_sep)
        region_slices: {name: slice(start, end)} in sequence-space
        sep_positions: List of separator positions in sequence-space
    """
    parts = []
    region_slices = {}
    sep_positions = []
    offset = 0

    present_keys = [k for k in SEQUENCE_ORDER if sequences.get(k, "")]

    for i, key in enumerate(present_keys):
        seq = sequences[key]
        start = offset
        end = offset + len(seq)
        region_slices[key] = slice(start, end)
        parts.append(seq)
        offset = end

        if use_sep and i < len(present_keys) - 1:
            parts.append(SEP_PLACEHOLDER)
            sep_positions.append(offset)
            offset += 1

    return "".join(parts), region_slices, sep_positions

DEFAULT_MASK_PROBS = {
    "bg": 0.15,
    "pocket": 0.15,
    "peptide": 0.15,
    "cdr": 0.15,
}


class TCRpMHCTokenizer:
    """Tokenizer + masking for a single TCR-pMHC entry.

    Args:
        entry: dict with 'sequences' and 'pocket_idx' keys.
        mask_probs: dict with keys bg/pocket/peptide/cdr. None = no masking.
        label: sequence label for batch_converter.
    """

    def __init__(
        self,
        entry: dict,
        mask_probs: dict | None = None,
        mask_strategy: str = "default",
        label: str = "seq",
        use_sep: bool = False,
    ):
        seqs = entry["sequences"]
        pocket_info = entry.get("pocket_idx", {})

        # --- Build concatenated sequence and region maps ---
        self._full_seq, self._region_slices, self._sep_indices = build_sequence(
            seqs, use_sep=use_sep
        )

        # --- Compute per-region annotation indices ---
        self._pocket_indices = []
        self._peptide_indices = []
        self._cdr_indices = []

        for key, sl in self._region_slices.items():
            # Pocket indices (HLA_a, HLA_b)
            if key in ("HLA_a", "HLA_b"):
                pi = pocket_info.get(key, [])
                if pi:
                    self._pocket_indices.extend(
                        range(pi[0] + sl.start, pi[1] + sl.start)
                    )

            # Peptide indices
            if key == "Peptide":
                self._peptide_indices.extend(range(sl.start, sl.stop))

            # CDR indices (TCR_a → CDRa, TCR_b → CDRb)
            cdr_key = {"TCR_a": "CDRa", "TCR_b": "CDRb"}.get(key)
            if cdr_key:
                for cdr_range in pocket_info.get(cdr_key, []):
                    self._cdr_indices.extend(
                        range(cdr_range[0] + sl.start, cdr_range[1] + sl.start)
                    )

        self._pocket_indices = np.array(self._pocket_indices, dtype=np.intp)
        self._peptide_indices = np.array(self._peptide_indices, dtype=np.intp)
        self._cdr_indices = np.array(self._cdr_indices, dtype=np.intp)
        self._sep_indices = np.array(self._sep_indices, dtype=np.intp)

        # CDR3 indices (3rd range per chain) for random_segment masking
        self._cdr3a_indices = np.array([], dtype=np.intp)
        self._cdr3b_indices = np.array([], dtype=np.intp)
        for key, sl in self._region_slices.items():
            cdr_key = {"TCR_a": "CDRa", "TCR_b": "CDRb"}.get(key)
            if cdr_key:
                ranges = pocket_info.get(cdr_key, [])
                if len(ranges) >= 3:  # CDR3 is the 3rd range
                    r = ranges[2]
                    indices = np.array(
                        range(r[0] + sl.start, r[1] + sl.start), dtype=np.intp
                    )
                    if key == "TCR_a":
                        self._cdr3a_indices = indices
                    else:
                        self._cdr3b_indices = indices

        # --- Metadata from entry ---
        meta = entry.get("meta_data", {})
        self.hla_class = meta.get("seq_type", "")
        self.hla_allele = meta.get("allele", meta.get("allele_id", ""))
        self.seq_profile = {
            "HLA_a": seqs.get("HLA_a", ""),
            "HLA_b": seqs.get("HLA_b", ""),
            "peptide": seqs.get("Peptide", ""),
            "TCR_a": seqs.get("TCR_a", ""),
            "TCR_b": seqs.get("TCR_b", ""),
        }

        # --- Apply masking at string level ---
        if mask_probs is not None:
            probs = {**DEFAULT_MASK_PROBS, **mask_probs}
            if mask_strategy == "random_segment":
                self._masked_seq = self._apply_random_segment_masking(probs)
            else:
                self._masked_seq = self._apply_masking(probs)
        else:
            self._masked_seq = self._full_seq  # no masking

        # --- Tokenize ---
        _, _, orig = BATCH_CONVERTER([(label, self._full_seq)])
        _, _, masked = BATCH_CONVERTER([(label, self._masked_seq)])
        self._original_ids = orig[0]
        self._input_ids = masked[0]

        # Replace placeholder tokens with EOS at separator positions
        if len(self._sep_indices) > 0:
            self._apply_sep()

        # Labels: original token where masked, -100 elsewhere
        self._labels = self._original_ids.clone()
        self._labels[self._input_ids != MASK_IDX] = -100

    def _apply_sep(self):
        """Replace placeholder token IDs with EOS_IDX at separator positions."""
        # +1 for CLS offset: sequence-space → token-space
        token_positions = self._sep_indices + 1
        self._original_ids[token_positions] = SEP_IDX
        self._input_ids[token_positions] = SEP_IDX

    def _apply_masking(self, probs: dict) -> str:
        """Replace amino acids with <mask> based on per-region probabilities."""
        seq_list = list(self._full_seq)
        mask_prob = torch.full((len(self._full_seq),), probs["bg"], dtype=torch.float)

        if len(self._pocket_indices) > 0:
            mask_prob[self._pocket_indices] = probs["pocket"]
        if len(self._cdr_indices) > 0:
            mask_prob[self._cdr_indices] = probs["cdr"]
        if len(self._peptide_indices) > 0:
            mask_prob[self._peptide_indices] = probs["peptide"]

        # Never mask separator tokens
        if len(self._sep_indices) > 0:
            mask_prob[self._sep_indices] = 0.0

        masked_positions = torch.bernoulli(mask_prob).bool()
        for j in masked_positions.nonzero().squeeze(-1).tolist():
            seq_list[j] = "<mask>"

        return "".join(seq_list)

    def _apply_random_segment_masking(self, probs: dict) -> str:
        """Randomly select one segment from {CDR3a, CDR3b, peptide} and mask it fully."""
        # Collect non-empty segments
        segments = {}
        if len(self._cdr3a_indices) > 0:
            segments["cdr3a"] = self._cdr3a_indices
        if len(self._cdr3b_indices) > 0:
            segments["cdr3b"] = self._cdr3b_indices
        if len(self._peptide_indices) > 0:
            segments["peptide"] = self._peptide_indices

        if not segments:
            return self._apply_masking(probs)  # fallback

        # Pick one at random
        keys = list(segments.keys())
        chosen = keys[torch.randint(len(keys), (1,)).item()]

        # Build mask: bg everywhere, 1.0 for chosen segment
        seq_list = list(self._full_seq)
        mask_prob = torch.full((len(self._full_seq),), probs["bg"], dtype=torch.float)
        mask_prob[segments[chosen]] = 1.0

        # Never mask separator tokens
        if len(self._sep_indices) > 0:
            mask_prob[self._sep_indices] = 0.0

        masked_positions = torch.bernoulli(mask_prob).bool()
        for j in masked_positions.nonzero().squeeze(-1).tolist():
            seq_list[j] = "<mask>"

        return "".join(seq_list)

    # --- Token outputs ---

    @property
    def input_ids(self) -> torch.Tensor:
        """Masked token IDs (1D, includes CLS/EOS)."""
        return self._input_ids

    @property
    def labels(self) -> torch.Tensor:
        """Target token IDs: original where masked, -100 elsewhere."""
        return self._labels

    @property
    def original_ids(self) -> torch.Tensor:
        """Unmasked token IDs (1D, includes CLS/EOS)."""
        return self._original_ids

    # --- Sequence outputs ---

    @property
    def full_seq(self) -> str:
        """Concatenated amino acid string (no special tokens)."""
        return self._full_seq

    @property
    def masked_seq(self) -> str:
        """Sequence string with <mask> replacements."""
        return self._masked_seq

    # --- Region accessors (sequence-level indices, no CLS offset) ---

    @property
    def region_slices(self) -> dict[str, slice]:
        """Dict mapping region name → slice in sequence space."""
        return self._region_slices

    @property
    def pocket_indices(self) -> np.ndarray:
        return self._pocket_indices

    @property
    def peptide_indices(self) -> np.ndarray:
        return self._peptide_indices

    @property
    def cdr_indices(self) -> np.ndarray:
        return self._cdr_indices

    @property
    def sep_indices(self) -> np.ndarray:
        return self._sep_indices

    def get_region_seq(self, name: str) -> str:
        """Extract amino acid substring for a region."""
        if name == "pocket":
            return "".join(self._full_seq[i] for i in self._pocket_indices)
        if name == "peptide":
            return "".join(self._full_seq[i] for i in self._peptide_indices)
        if name == "cdr":
            return "".join(self._full_seq[i] for i in self._cdr_indices)
        s = self._region_slices.get(name)
        if s is None:
            raise KeyError(f"Unknown region: {name}")
        return self._full_seq[s]

    def get_region_ids(self, name: str) -> torch.Tensor:
        """Extract original (unmasked) token IDs for a region.

        Returns token IDs from the original_ids tensor, mapped from
        sequence-level indices (+1 for CLS offset applied internally).
        """
        if name == "pocket":
            indices = self._pocket_indices
        elif name == "peptide":
            indices = self._peptide_indices
        elif name == "cdr":
            indices = self._cdr_indices
        else:
            s = self._region_slices.get(name)
            if s is None:
                raise KeyError(f"Unknown region: {name}")
            indices = np.arange(s.start, s.stop)

        if len(indices) == 0:
            return torch.tensor([], dtype=torch.long)
        # +1 for CLS token in token space
        return self._original_ids[indices + 1]


def collate_tokenized(
    batch: list[TCRpMHCTokenizer],
) -> dict[str, torch.Tensor | tuple]:
    """Collate a list of TCRpMHCTokenizer instances into a training batch.

    Pads input_ids, labels, and original_ids to the max length in the batch.
    """
    max_len = max(len(tok.input_ids) for tok in batch)
    pad_idx = ALPHABET.padding_idx
    bs = len(batch)

    input_ids = torch.full((bs, max_len), pad_idx, dtype=torch.long)
    labels = torch.full((bs, max_len), -100, dtype=torch.long)
    original_ids = torch.full((bs, max_len), pad_idx, dtype=torch.long)

    for i, tok in enumerate(batch):
        L = len(tok.input_ids)
        input_ids[i, :L] = tok.input_ids
        labels[i, :L] = tok.labels
        original_ids[i, :L] = tok.original_ids

    return {
        "masked_input": input_ids,
        "masked_target": labels,
        "input": original_ids,
        "seqs": tuple(tok.full_seq for tok in batch),
        "seq_profile": tuple(tok.seq_profile for tok in batch),
        "HLA_class": tuple(tok.hla_class for tok in batch),
        "HLA_allele": tuple(tok.hla_allele for tok in batch),
        "aa_to_idx": ALPHABET.to_dict(),
    }
