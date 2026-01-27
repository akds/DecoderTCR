#!/usr/bin/env python
"""
Prediction utilities for pMHC binding scoring using span pseudo-likelihood.

Usage as CLI:
    python -m DecoderTCR.utils.predict_pMHC -i input.csv -o output.csv -c /path/to/checkpoint.ckpt
"""

import argparse
import torch
import pandas as pd
from tqdm import tqdm

from DecoderTCR.model.DecoderTCR import DecoderTCRModel
from DecoderTCR.utils.tokenizer import tokenize_pmhc
from DecoderTCR.utils.scoring import span_pseudolikelihood


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
    Predict pMHC binding score for a single sample using span pseudo-likelihood.
    
    Args:
        model: DecoderTCRModel instance
        sample_in: Dictionary with 'HLA_seq', 'epitope'
        device: Device to run inference on
    
    Returns:
        score: Span pseudo-likelihood score for the epitope
    """
    # Tokenize pMHC sequence with epitope masked
    masked_token, pep_idx, pMHC_seq, _ = tokenize_pmhc(sample_in)
    
    # Run model
    results = model(masked_token.unsqueeze(0).to(device), repr_layers=[33], return_contacts=False)
    logits = results['logits'][0, 1:-1].detach().cpu()
    
    # Extract logits for peptide region
    pep_logits = logits[pep_idx[0]:pep_idx[1], :]
    pep_seq = pMHC_seq[pep_idx[0]:pep_idx[1]]
    
    # Calculate span pseudo-likelihood
    score = span_pseudolikelihood(pep_logits, pep_seq)
    
    return score


def predict_batch(model, df, device='cuda:0'):
    """
    Predict pMHC binding scores for a DataFrame.
    
    Args:
        model: DecoderTCRModel instance
        df: DataFrame with columns 'HLA_seq', 'epitope'
        device: Device to run inference on
    
    Returns:
        scores: List of pseudo-likelihood scores
    """
    scores = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
        sample_in = {
            'HLA_seq': row['HLA_seq'],
            'epitope': row['epitope'],
        }
        
        with torch.no_grad():
            score = predict_single(model, sample_in, device)
        
        scores.append(score)
    
    return scores


def predict_csv(model, input_path, output_path, device='cuda:0'):
    """
    Predict pMHC binding scores from a CSV file and save results.
    
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
    required_cols = ['HLA_seq', 'epitope']
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
        description='Predict pMHC binding scores using span pseudo-likelihood',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    python -m DecoderTCR.utils.predict_pMHC -i data.csv -o predictions.csv -c /path/to/model.ckpt

    Input CSV should have columns:
    - HLA_seq: HLA sequence
    - epitope: Peptide/epitope sequence
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
