"""Region-masked pseudo-log-likelihood (PLL) scoring + per-epitope metrics.

Masks a target region (peptide, CDR, pocket, or background) and computes the
normalized log-likelihood of the true amino acids at the masked positions. Higher
PLL = the model finds the true residues more probable = more binder-like.

Backbone-agnostic: this runs against either the ESM2 `DecoderTCR` (33-wide logit
head) or the ESMC `DecoderTCRC` (64-wide logit head). `compute_pll` indexes the
true amino-acid token ids (Meta ESM-1b ids in [0, 32]). Those channels line up
across both heads, so the extra ESMC logit channels are simply never scored.

Peptide masking is deterministic given the non-peptide context, so logits are
cached per (HLA_a, HLA_b, TCR_a, TCR_b, peptide-length) tuple.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from DecoderTCR.model.tokenizer import TCRpMHCTokenizer
from DecoderTCR.constants import MASK_IDX

REGION_TO_MASK_PROBS = {
    "peptide": {"bg": 0.0, "pocket": 0.0, "peptide": 1.0, "cdr": 0.0},
    "cdr": {"bg": 0.0, "pocket": 0.0, "peptide": 0.0, "cdr": 1.0},
    "cdr3": {"bg": 0.0, "pocket": 0.0, "peptide": 0.0, "cdr": 0.0},  # CDR3 masking handled separately
    "pocket": {"bg": 0.0, "pocket": 1.0, "peptide": 0.0, "cdr": 0.0},
    "bg": {"bg": 1.0, "pocket": 0.0, "peptide": 0.0, "cdr": 0.0},
}


def compute_pll(true_token_ids: torch.Tensor, logits: torch.Tensor) -> float:
    """Compute normalized pseudo-log-likelihood.

    Args:
        true_token_ids: (N,) token indices of the true amino acids.
        logits: (N, vocab_size) logits at the masked positions.

    Returns:
        PLL score: sum(log P(true_aa)) / N.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs[torch.arange(len(true_token_ids)), true_token_ids]
    return token_log_probs.sum().item() / len(true_token_ids)


def _get_target_indices(tok: TCRpMHCTokenizer, mask_region: str) -> np.ndarray:
    """Get sequence-space indices for the target region."""
    if mask_region == "peptide":
        return tok.peptide_indices
    elif mask_region == "cdr":
        return tok.cdr_indices
    elif mask_region == "cdr3":
        return np.concatenate([tok._cdr3a_indices, tok._cdr3b_indices])
    elif mask_region == "pocket":
        return tok.pocket_indices
    elif mask_region == "bg":
        # Background = everything not in peptide, cdr, or pocket
        all_special = set(tok.peptide_indices) | set(tok.cdr_indices) | set(tok.pocket_indices)
        seq_len = len(tok._full_seq)
        bg = np.array([i for i in range(seq_len) if i not in all_special], dtype=np.intp)
        return bg
    else:
        raise ValueError(f"Unknown mask_region: {mask_region}")


def _cache_key(entry: dict, pep_len: int) -> tuple:
    """Build cache key from non-masked context (full sequences).

    Entries with identical non-peptide sequences and same peptide length
    produce identical logits when the peptide is fully masked, so their
    logits can be shared.
    """
    seqs = entry["sequences"]
    return (
        seqs.get("HLA_a", ""),
        seqs.get("HLA_b", ""),
        seqs.get("TCR_a", ""),
        seqs.get("TCR_b", ""),
        pep_len,
    )


@torch.no_grad()
def run_pll_benchmark(
    model: torch.nn.Module,
    num_layers: int,
    entries: list[dict],
    mask_region: str = "peptide",
    device: torch.device | str = "cpu",
) -> np.ndarray:
    """Compute PLL scores for all entries with a given mask region.

    Args:
        model: A backbone module whose forward returns {"logits": (B, L, V), ...},
            i.e. the raw ESM2 (DecoderTCR) or the DecoderTCRC wrapper.
        num_layers: Number of layers (passed as repr_layers for ESM2, ignored by ESMC).
        entries: List of entry dicts with 'sequences' and 'pocket_idx'.
        mask_region: Which region to mask ('peptide', 'cdr', 'cdr3', 'pocket', 'bg').
        device: Torch device.

    Returns:
        Array of PLL scores, one per entry (NaN where the target region is empty).
    """
    if mask_region not in REGION_TO_MASK_PROBS:
        raise ValueError(f"Unknown mask_region '{mask_region}'. Choose from: {list(REGION_TO_MASK_PROBS)}")

    mask_probs = REGION_TO_MASK_PROBS[mask_region]
    use_cache = mask_region == "peptide"
    cache: dict[tuple, torch.Tensor] = {}

    scores = []
    for entry in tqdm(entries, desc=f"PLL ({mask_region})"):
        tok = TCRpMHCTokenizer(entry, mask_probs=mask_probs, use_sep=False)
        target_indices = _get_target_indices(tok, mask_region)

        if len(target_indices) == 0:
            scores.append(float("nan"))
            continue

        # For cdr3: manually mask CDR3 positions (tokenizer left them unmasked)
        if mask_region == "cdr3":
            token_indices_tmp = target_indices + 1  # CLS offset
            tok._input_ids[token_indices_tmp] = MASK_IDX
            tok._labels = tok._original_ids.clone()
            tok._labels[tok._input_ids != MASK_IDX] = -100

        # Token-space indices (sequence-space + 1 for CLS)
        token_indices = target_indices + 1

        # Check cache (only for peptide masking)
        pep_len = len(entry["sequences"].get("Peptide", ""))
        ck = _cache_key(entry, pep_len) if use_cache else None

        if use_cache and ck in cache:
            logits = cache[ck]
        else:
            tokens = tok.input_ids.unsqueeze(0).to(device)
            out = model(tokens, repr_layers=[num_layers], return_contacts=False)
            logits = out["logits"][0].cpu()
            if use_cache:
                cache[ck] = logits

        target_logits = logits[token_indices]
        true_ids = tok.original_ids[token_indices]
        pll = compute_pll(true_ids, target_logits)
        scores.append(pll)

    return np.array(scores)


def auroc_per_epitope(
    labels: np.ndarray,
    scores: np.ndarray,
    epitopes: np.ndarray,
) -> dict[str, float]:
    """Compute AUROC per epitope and macro-average.

    Args:
        labels: Binary labels (1 = binder, 0 = non-binder).
        scores: Model scores (higher = more likely binder).
        epitopes: Epitope string for each entry.

    Returns:
        Dict with per-epitope AUROC and 'macro_avg'.
        Epitopes with only one class are skipped.
    """
    results = {}
    valid_aucs = []

    for ep in sorted(set(epitopes)):
        mask = epitopes == ep
        ep_labels = labels[mask]
        ep_scores = scores[mask]

        if len(np.unique(ep_labels)) < 2:
            continue

        auc = roc_auc_score(ep_labels, ep_scores)
        n_pos = int(ep_labels.sum())
        n_neg = int(mask.sum()) - n_pos
        results[ep] = {"auroc": auc, "n": int(mask.sum()), "n_pos": n_pos, "n_neg": n_neg}
        valid_aucs.append(auc)

    results["macro_avg"] = float(np.mean(valid_aucs)) if valid_aucs else float("nan")
    results["n_epitopes"] = len(valid_aucs)
    return results
