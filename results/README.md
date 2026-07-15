# V0.3 benchmark results

V0.3 results across four TCR-pMHC evaluations: **[TCRvdb](https://github.com/schumacherlab/TCRvdb)**, **IMMREP23**, **Viral**
(ePytope-TCR), and **PRP** (Peptide Recognition Profiling, HLA-B\*27:05). Models score a
pair by masked-peptide PLL (higher means more binder-like), and metrics are per epitope or
clone, macro-averaged. Figures are rendered by
`eval/scripts/{plot_macro_comparisons,plot_prp}.py` in the research project.

## Summary

[`figures/summary_dotplot_scaling.png`](figures/summary_dotplot_scaling.png) is a Cleveland
dot plot of macro AUROC (one marker per method per benchmark) plus an ESMC scaling panel
(macro AUROC vs 300M, 600M, 6B), over the three balanced benchmarks (TCRvdb, IMMREP23,
Viral). **PRP is excluded**, because at ~0.4% prevalence AUROC is uninformative. PRP is
reported separately by **macro AUPRC and recall@K** (below).

## Per-benchmark figures (featured baselines)

| Benchmark | What it probes | Figure |
|---|---|---|
| **TCRvdb** (YLQ / GLC) | label-free probe on **seen** TCRs (pos/neg equal-frequency, no labels given). Data from [Messemaker et al., bioRxiv 2025](https://doi.org/10.1101/2025.04.28.651095) | [tcrvdb ROC](figures/tcrvdb_roc.png) (per-epitope) |
| **IMMREP23** | per-epitope AUROC over near-novel TCRs (~0.3% train overlap), from the IMMREP23 challenge ([Nielsen et al., *ImmunoInformatics* 2024](https://doi.org/10.1016/j.immuno.2024.100045)) | [immrep23](figures/immrep23_macro_comparison_with_esmc.png) |
| **Viral** (ePytope-TCR) | viral-epitope TCR specificity, baselines from the ePytope-TCR benchmark (Drost et al., *Cell Genomics* 2025) | [viral](figures/viral_macro_comparison_with_esmc.png) |
| **PRP** HLA-B\*27:05 | 16-clone peptide-library screen (**macro AUPRC**, AUROC uninformative) | [prp](figures/prp_macro_auprc.png) |

> IMMREP25 was dropped from the V0.3 results (redundant with IMMREP23 for this comparison).

## PRP (Peptide Recognition Profiling), headline

From [Deep peptide recognition profiling decodes TCR specificity and enables
disease-associated antigen discovery](https://www.nature.com/articles/s41587-026-03128-x)
(*Nature Biotechnology*, 2026): 16 HLA-B\*27:05-restricted TCR clones screened against an
anchor-fixed peptide library (R@P2, P@P8). A retrieval task at ~0.4% prevalence.

The default scalar is **macro AUPRC** (per-clone average precision, chance equals the ~0.004
prevalence). DecoderTCR-ESMC scores 6B 0.391, 600M 0.351, 300M 0.303, against ≤0.02 for V0.1,
the untrained backbones, and all third-party tools. AUROC is not reported, because at this
prevalence it is uninformative (a near-random top still scores about 0.7). The operational
view is **recall@K**, the fraction of a TCR's true binders recovered in its top-K ranked
peptides. We use recall@K rather than precision@K, which is capped by n_pos/K and makes even
a perfect ranker look bad at large K.

- **macro AUPRC by method** (the default scalar): [prp_macro_auprc](figures/prp_macro_auprc.png).
  DecoderTCR-ESMC 6B 0.391, 600M 0.351, 300M 0.303. V0.1, the untrained backbones, and all
  third-party tools sit ≤0.02, just above the 0.004 prevalence line.
- **recall@K vs K** (mean over 16 clones), all models and baselines: [prp_topk_recall](figures/prp_topk_recall.png).
  DecoderTCR-ESMC **6B recall@500 ≈ 0.62, recall@100 ≈ 0.35** (against ~0.01 and 0.003 at
  random), 600M close behind, 300M lower. DecoderTCR V0.1, the untrained ESMC and ESM2
  backbones, and all six third-party tools (NetTCR-2.2, ERGO-II, pMTnet, TULIP, DLpTCR,
  PanPep) track the random line. Epitope-specific tools (MixTCRpred, TITAN, NetTCR-cat) do
  not apply to a novel-peptide library.
- **per-clonotype recall@100 by method**: [prp_per_clone_recall](figures/prp_per_clone_recall.png).
  Rows are DecoderTCR-ESMC V0.3, DecoderTCR V0.1, the untrained ESMC and ESM2 backbones, and
  third-party. Columns are the 16 clones, with seen • and held-out ○.

Full-set macro AUPRC is **6B 0.391 > 600M 0.351 > 300M 0.303**, with V0.1, base backbones,
and third-party all ≤0.02. These use the released V0.3 checkpoints.

Each figure also has a `.pdf` next to the `.png`.

## Notes

- **TCRvdb** and **PRP** are seen-TCR evaluations (clones appear in training-data VDJdb).
  **IMMREP23** and **Viral** are near-novel-TCR. Read scaling and headline claims with that
  distinction in mind. The PRP recall@K is the most decision-relevant view.
- The PRP result reflects the corrected TCR reconstruction (leader and constant-region
  repair). After the fix, ESMC scaling is monotonic (6B > 600M > 300M) and the 6B model
  recovers the large majority of true binders in each TCR's top-ranked peptides.
