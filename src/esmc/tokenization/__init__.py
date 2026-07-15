"""Tokenization helpers (ESM-C subset).

The runtime code path uses the project's Meta Alphabet (`DecoderTCR.constants.ALPHABET`).
This HF tokenizer is kept for: (1) test_tokenizer_equivalence proving id parity,
and (2) being passed into `ESMC(tokenizer=...)` for the model to remember its
own pad/mask/cls ids (used when `sequence_id` is auto-built from `tokens != pad_id`).
"""

from esmc.tokenization.sequence_tokenizer import EsmSequenceTokenizer


def get_esmc_model_tokenizers() -> EsmSequenceTokenizer:
    return EsmSequenceTokenizer()
