# Checkpoints

Model weights are not committed (tens of GB). Fetch them from the HuggingFace release into the
paths the model registry expects with the download script:

```bash
uv run python scripts/download_weights.py                      # all models
uv run python scripts/download_weights.py -m DecoderTCR-ESMC_600M  # just the default
```

This populates:

```
checkpoints/
├── DecoderTCR-ESMC-V0.3/{300M,600M,6B}.ckpt             # ESMC V0.3 (600M is the default)
└── DecoderTCR-ESM2-V0.1/{650M,3B}_DecoderTCR.ckpt    # ESM2 V0.1, paper reproduction
```

The exact filenames live in [`scripts/download_weights.py`](../scripts/download_weights.py), driven
by the registry so they cannot drift from the loader. The HuggingFace repo
([`biohub/DecoderTCR`](https://huggingface.co/biohub/DecoderTCR)) mirrors this same
`DecoderTCR-ESMC-V0.3/` and `DecoderTCR-ESM2-V0.1/` folder layout. All weights are released under the
MIT license (see [`../LICENSE.md`](../LICENSE.md)).
