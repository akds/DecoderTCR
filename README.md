# DecoderTCR

A protein language model for TCR-pMHC modeling.

## Installation

```bash
cd <your_path>/DecoderTCR
pip install -e .
```

## Quick Start

### TCR-pMHC Interaction Prediction

Predict TCR-pMHC binding using interaction scores (comparing TCR+pMHC vs pMHC alone):

```python
from DecoderTCR.utils.predict_TpM import load_model, predict_single

# Load model from checkpoint
model = load_model(checkpoint_path='/path/to/checkpoint.ckpt', device='cuda:0')

# Predict single sample
sample = {
    'HLA_seq': 'GSHSMRYF...',
    'epitope': 'GILGFVFTL',
    'TCR_seq': 'CASSFSTCSANYGYT...'
}
score = predict_single(model, sample, device='cuda:0')
```

### pMHC Binding Prediction

Score epitope-HLA binding using span pseudo-likelihood:

```python
from DecoderTCR.utils.predict_pMHC import load_model, predict_single

# Load model from checkpoint
model = load_model(checkpoint_path='/path/to/checkpoint.ckpt', device='cuda:0')

# Predict single sample
sample = {
    'HLA_seq': 'GSHSMRYF...',
    'epitope': 'GILGFVFTL'
}
score = predict_single(model, sample, device='cuda:0')
```

## Command Line Interface

### TCR-pMHC Prediction

Example:

```bash
python -m DecoderTCR.utils.predict_TpM \
    -i Demo/sample_data/YLQ_validated.csv \
    -o ./YLQ_validated_pred.csv \
    -c <path to checkpoint> \
    -d cuda:0
```

**Input CSV format:**

| HLA_seq | epitope | TCRa_seq | TCRb_seq |
|---------|---------|----------|----------|
| GSHSMRYF... | GILGFVFTL | CAVS... | CASSF... |

### pMHC Binding Prediction

Example:

```bash
python -m DecoderTCR.utils.predict_pMHC \
    -i Demo/sample_data/A0252_heldout.csv \
    -o ./A0252_heldout_pred.csv \
    -c <path to checkpoint> \
    -d cuda:0
```

**Input CSV format:**

| HLA_seq | epitope |
|---------|---------|
| GSHSMRYF... | GILGFVFTL |

## Available Models

| Model | Parameters |
|-------|------------|
| DecoderTCR_650M | 650M |
| DecoderTCR_3B   | 3B   |
## API Reference

### Tokenization

```python
from DecoderTCR.utils import tokenize_tcr_pmhc, tokenize_pmhc

# TCR + pMHC tokenization with epitope masking
masked_token, pep_idx, full_seq, mask_pos = tokenize_tcr_pmhc(sample_dict)

# pMHC only tokenization with epitope masking
masked_token, pep_idx, pMHC_seq, mask_pos = tokenize_pmhc(sample_dict)
```

### Scoring Functions

```python
from DecoderTCR.utils.scoring import interaction_score, span_pseudolikelihood

# Interaction score (difference between two conditions)
score = interaction_score(logits1, logits2, sequences)

# Span pseudo-likelihood (average log-prob of masked region)
score = span_pseudolikelihood(logits, sequence)
```

## Project Structure

```
DecoderTCR/
├── pyproject.toml
├── README.md
├── LICENSE
├── Demo/
|   ├── quick_start.ipynb
└── src/DecoderTCR/
    ├── __init__.py
    ├── model/
    │   ├── __init__.py
    │   └── DecoderTCR.py        # Model classes
    └── utils/
        ├── __init__.py
        ├── model_zoo.py          # Base model loader
        ├── tokenizer.py          # Sequence tokenization
        ├── scoring.py            # Scoring functions
        ├── predict_TpM.py        # TCR-pMHC prediction
        └── predict_pMHC.py       # pMHC binding prediction
```

## Environment Variables

Set `TORCH_HUB_DIR` to customize where base models are cached:

```bash
export TORCH_HUB_DIR=/path/to/cache
```

