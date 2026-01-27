import torch
import torch.nn.functional as F
from DecoderTCR.utils.tokenizer import alphabet

# Amino acid to index mapping from ESM alphabet
aa2idx = alphabet.to_dict()


def interaction_score(logits1, logits2, seqs):
    """
    Calculate interaction score between two sets of logits.
    
    Computes the average log-probability difference for each amino acid
    in the sequences between two model outputs.
    
    Args:
        logits1: Logits from model 1, shape (batch, seq_len, vocab_size)
        logits2: Logits from model 2, shape (batch, seq_len, vocab_size)
        seqs: List of amino acid sequences
    
    Returns:
        score: Average log-probability difference across all positions
    """
    score = 0
    sl = 0
    
    for i, seq in enumerate(seqs):
        p1 = F.softmax(logits1[i], dim=1)
        p2 = F.softmax(logits2[i], dim=1)
        
        for j, aa in enumerate(seq):
            logit_idx = aa2idx[aa]
            score += (torch.log(p1[j, logit_idx]).item() - torch.log(p2[j, logit_idx]).item())
            sl += 1
    
    return score / sl


def span_pseudolikelihood(logits, seqs):
    """
    Calculate span pseudo-likelihood score.
    
    Computes the average log-probability of each amino acid in the 
    masked span given the surrounding context.
    
    Args:
        logits: Model logits for masked positions, shape (seq_len, vocab_size)
        seqs: Amino acid sequence string for the masked span
    
    Returns:
        score: Average log-probability across the span
    """
    score = 0
    sl = 0
    
    probs = F.softmax(logits, dim=1)
    
    for j, aa in enumerate(seqs):
        logit_idx = aa2idx[aa]
        score += torch.log(probs[j, logit_idx]).item()
        sl += 1
    
    return score / sl if sl > 0 else 0.0
