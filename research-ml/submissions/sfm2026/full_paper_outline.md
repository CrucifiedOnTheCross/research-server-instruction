# SFM 2026 proceedings paper outline

## Working title

Feature-Space Quality Does Not Guarantee Utility of Synthetic Dermoscopic Images for Long-Tailed Classification

## Central claim

Visual plausibility, diversity, and proximity in one foundation-model embedding are not sufficient conditions for useful synthetic augmentation. Utility must be demonstrated against source- and count-matched real replay using ranking, threshold, calibration, and lesion-group uncertainty analyses.

## Proposed structure

1. Motivation: long-tailed dermoscopy and weak synthetic-data controls.
2. HAM10000 lesion-group split, leakage audit, and locked-test policy.
3. ConvNeXt-S baseline qualification and equal-budget source-matched replay.
4. Geometry-stratified selection in DINOv2 and task-specific feature spaces.
5. Confirmatory synthetic-versus-replay comparison across three paired seeds.
6. PRDC, Vendi diversity, representation mismatch, frequency, and ranking-tail diagnostics.
7. Causal decomposition of crop, VAE round-trip, and one-step denoising.
8. Generator qualification as a negative gate, not downstream evidence.
9. Limitations: one dataset for the primary claim, no clinician realism study, SD1.5-specific pipeline, locked test not yet opened.
10. Conclusion and preregistered next validation.

## Main figures

1. Leakage-safe experimental protocol.
2. Stage 10 geometry forest plot.
3. Stage 15A causal decomposition.
4. Task-space failure diagnostics from Stage 12.

## Supplementary figures

- Generator qualification trade-off and internal contact sheets.
- Per-seed paired effects and per-class AUPRC/AUROC.
- ISIC 2019 baseline and generator-gate scale-out results.

## Freeze rule

The conference abstract is based on lesion-group validation. Any later locked-test opening must be documented before evaluation and cannot be used to change the method, checkpoint selection, endpoints, or abstract claim.
