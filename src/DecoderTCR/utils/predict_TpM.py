#!/usr/bin/env python
"""
Prediction utilities for TCR-pMHC interaction scoring.

Usage as CLI:
    python -m DecoderTCR.utils.predict_TpM -i input.csv -o output.csv -c /path/to/checkpoint.ckpt
"""

import argparse
import torch
import pandas as pd
from tqdm import tqdm

from DecoderTCR.model.DecoderTCR import DecoderTCRModel
from DecoderTCR.utils.tokenizer import tokenize_tcr_pmhc, tokenize_pmhc
from DecoderTCR.utils.scoring import interaction_score


def load_model(checkpoint_path, device='cuda:0'):
    """
    Load DecoderTCR model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint (required)
        device: Device to load model on
    
    Returns:
        model: Loaded DecoderTCRModel instance
    """
    model_size = checkpoint_path.split('/')[-1].split('_')[0]
    model = DecoderTCRModel.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        base_model=f'ESM2_{model_size}'
    )
    model.eval()
    model.to(device)
    
    return model


def predict_single(model, sample_in, device='cuda:0'):
    """
    Predict interaction score for a single sample.
    
    Args:
        model: DecoderTCRModel instance
        sample_in: Dictionary with 'HLA_seq', 'epitope', 'TCR_seq'
        device: Device to run inference on
    
    Returns:
        score: Interaction score
    """
    # Tokenize full TCR-pMHC sequence
    masked_token_full, pep_idx_full, full_seq, _ = tokenize_tcr_pmhc(sample_in)
    results_full = model(masked_token_full.unsqueeze(0).to(device), repr_layers=[33], return_contacts=False)
    logits_full = results_full['logits'][0, 1:-1].detach().cpu()
    pep_logits_full = logits_full[pep_idx_full[0]:pep_idx_full[1], :]
    pep_seq_full = full_seq[pep_idx_full[0]:pep_idx_full[1]]

    # Tokenize pMHC only (no TCR)
    masked_token_pMHC, pep_idx_pMHC, pMHC_seq, _ = tokenize_pmhc(sample_in)
    results_pMHC = model(masked_token_pMHC.unsqueeze(0).to(device), repr_layers=[33], return_contacts=False)
    logits_pMHC = results_pMHC['logits'][0, 1:-1].detach().cpu()
    pep_logits_pMHC = logits_pMHC[pep_idx_pMHC[0]:pep_idx_pMHC[1], :]

    # Calculate interaction score
    score = interaction_score([pep_logits_full], [pep_logits_pMHC], [pep_seq_full])
    
    return score


def predict_batch(model, df, device='cuda:0'):
    """
    Predict interaction scores for a DataFrame.
    
    Args:
        model: DecoderTCRModel instance
        df: DataFrame with columns 'HLA_seq', 'epitope', 'TCRa_seq', 'TCRb_seq'
        device: Device to run inference on
    
    Returns:
        scores: List of interaction scores
    """
    scores = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
        sample_in = {
            'HLA_seq': row['HLA_seq'],
            'epitope': row['epitope'],
            'TCR_seq': row['TCRa_seq'] + row['TCRb_seq'],
        }
        
        with torch.no_grad():
            score = predict_single(model, sample_in, device)
        
        scores.append(score)
    
    return scores


def predict_csv(model, input_path, output_path, device='cuda:0'):
    """
    Predict interaction scores from a CSV file and save results.
    
    Args:
        model: DecoderTCRModel instance
        input_path: Path to input CSV file
        output_path: Path to output CSV file
        device: Device to run inference on
    
    Returns:
        df: DataFrame with predictions
    """
    # Load data
    df = pd.read_csv(input_path)
    
    # Validate columns
    required_cols = ['HLA_seq', 'epitope', 'TCRa_seq', 'TCRb_seq']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Predict
    scores = predict_batch(model, df, device=device)
    
    # Save results
    df['prediction_score'] = scores
    df.to_csv(output_path, index=False)
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Predict TCR-pMHC interaction scores',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    python -m DecoderTCR.utils.predict_TpM -i data.csv -o predictions.csv -c /path/to/model.ckpt

    Input CSV should have columns:
    - HLA_seq: HLA sequence
    - epitope: Peptide/epitope sequence  
    - TCRa_seq: TCR alpha chain sequence
    - TCRb_seq: TCR beta chain sequence
        """
    )
    parser.add_argument('--input', '-i', type=str, required=True, help='Input CSV file')
    parser.add_argument('--output', '-o', type=str, required=True, help='Output CSV file')
    parser.add_argument('--checkpoint', '-c', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--device', '-d', type=str, default='cuda:0', help='Device (default: cuda:0)')
    args = parser.parse_args()

    # Load model
    print(f"Checkpoint: {args.checkpoint}")
    model = load_model(
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    # Predict
    print(f"Input: {args.input}")
    print("Running predictions...")
    df = predict_csv(model, args.input, args.output, device=args.device)
    print(f"Output: {args.output}")
    print(f"Processed {len(df)} samples")


if __name__ == '__main__':
    main()
