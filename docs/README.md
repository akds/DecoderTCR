# docs

`overview.png` is the DecoderTCR schematic referenced at the top of the main
[README](../README.md). Two panels:

- **A.** Compositional continual pre-training: unpaired TCRs and unpaired pMHC (Stage 1) then
  paired TCR-pMHC (Stage 2) into the protein language model, used for CDR3 region design and
  zero-shot tasks.
- **B.** Iterative Entropy-Guided Refinement for CDR3 design: from all-masked initialization,
  resolve the high-confidence (low-entropy) anchor residues first, then refine the flexible
  center to the final CDR3.
