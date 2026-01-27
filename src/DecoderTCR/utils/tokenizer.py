import os
import torch
import esm

# Set torch hub cache directory before loading ESM
# Must be set via TORCH_HUB_DIR environment variable
HUB_DIR = os.environ.get('TORCH_HUB_DIR')
if HUB_DIR:
    os.makedirs(HUB_DIR, exist_ok=True)
    torch.hub.set_dir(HUB_DIR)

# Initialize ESM alphabet and batch converter
_, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
batch_converter = alphabet.get_batch_converter()
cls_idx = alphabet.cls_idx
eos_idx = alphabet.eos_idx
pad_idx = alphabet.padding_idx
mask_idx = alphabet.mask_idx


def tokenize_tcr_pmhc(in_dict):
    """
    Tokenize HLA + peptide + TCR sequence with peptide masking.
    
    Args:
        in_dict: Dictionary with keys:
            - 'HLA_seq': HLA sequence
            - 'epitope': Peptide/epitope sequence
            - 'TCR_seq': TCR sequence
    
    Returns:
        masked_token: Tokenized sequence with peptide positions masked
        pep_idx: Tuple of (start, end) indices for peptide in sequence
        full_seq: Full concatenated sequence string
        mask_pos: Boolean tensor indicating masked positions
    """
    HLA_seq = in_dict['HLA_seq']
    peptide = in_dict['epitope']
    TCR_seq = in_dict['TCR_seq']
    
    pep_idx = (len(HLA_seq), len(HLA_seq) + len(peptide))
    
    _, seq_strs, seq_tokens = batch_converter([('', HLA_seq + peptide + TCR_seq)])
    special_tokens_mask = (seq_tokens[0] == cls_idx) | (seq_tokens[0] == eos_idx) | (seq_tokens[0] == pad_idx)
    mask_prob = torch.full(seq_tokens[0].size(), 0, device='cpu', dtype=torch.float)

    # Mask peptide region (+1 offset for CLS token)
    mask_prob[pep_idx[0] + 1:pep_idx[1] + 1] = 1

    mask_prob.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(mask_prob).bool()

    out_token = seq_tokens[0].clone()
    out_token[~masked_indices] = -100
    masked_token = seq_tokens[0].clone()
    masked_token[masked_indices] = mask_idx
    
    mask_pos = masked_indices
    full_seq = HLA_seq + peptide + TCR_seq
    
    return masked_token, pep_idx, full_seq, mask_pos


def tokenize_pmhc(in_dict):
    """
    Tokenize HLA + peptide sequence with peptide masking (no TCR).
    
    Args:
        in_dict: Dictionary with keys:
            - 'HLA_seq': HLA sequence
            - 'epitope': Peptide/epitope sequence
    
    Returns:
        masked_token: Tokenized sequence with peptide positions masked
        pep_idx: Tuple of (start, end) indices for peptide in sequence
        full_seq: Full concatenated sequence string (HLA + peptide only)
        mask_pos: Boolean tensor indicating masked positions
    """
    HLA_seq = in_dict['HLA_seq']
    peptide = in_dict['epitope']
    
    pep_idx = (len(HLA_seq), len(HLA_seq) + len(peptide))
    
    _, seq_strs, seq_tokens = batch_converter([('', HLA_seq + peptide)])
    special_tokens_mask = (seq_tokens[0] == cls_idx) | (seq_tokens[0] == eos_idx) | (seq_tokens[0] == pad_idx)
    mask_prob = torch.full(seq_tokens[0].size(), 0, device='cpu', dtype=torch.float)

    # Mask peptide region (+1 offset for CLS token)
    mask_prob[pep_idx[0] + 1:pep_idx[1] + 1] = 1

    mask_prob.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(mask_prob).bool()

    out_token = seq_tokens[0].clone()
    out_token[~masked_indices] = -100
    masked_token = seq_tokens[0].clone()
    masked_token[masked_indices] = mask_idx
    
    mask_pos = masked_indices
    full_seq = HLA_seq + peptide

    return masked_token, pep_idx, full_seq, mask_pos


def tokenize(sequence):
    """
    Simple tokenization without masking.
    
    Args:
        sequence: Amino acid sequence string
    
    Returns:
        tokens: Tokenized sequence tensor (1, seq_len)
    """
    _, _, tokens = batch_converter([('', sequence)])
    return tokens


def tokenize_batch(sequences):
    """
    Tokenize a batch of sequences without masking.
    
    Args:
        sequences: List of amino acid sequence strings
    
    Returns:
        tokens: Batch of tokenized sequences (batch_size, max_seq_len), padded
    """
    data = [('', seq) for seq in sequences]
    _, _, tokens = batch_converter(data)
    return tokens
