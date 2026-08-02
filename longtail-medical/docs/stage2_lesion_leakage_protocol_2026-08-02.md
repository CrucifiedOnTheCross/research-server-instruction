# Stage 2: lesion leakage and strict generalization protocol

Date: 2026-08-02  
Status: Stage 2 completed
Test state: opened once for preregistered confirmatory inference; now
permanently closed. All further analyses use immutable prediction files.

## MLflow grouping metadata

Runs are grouped with stable MLflow tags: `project`, `stage`, `dataset`, `task`,
`experiment_arm`, `contamination_policy`, `protocol_version`, `seed`, `model`,
`checkpoint_policy`, `git_commit`, `run_signature`, and `test_evaluated`.
These tags are observability metadata only and do not change data, optimization,
random seeds, checkpoints, or metric computation. Existing runs can be updated
idempotently with `tools/backfill_mlflow_tags.py`; checkpoints are not uploaded.

The one-shot locked-test inference persisted all 24 prediction sets (12 runs,
`last.pt` and `best.pt`) before the original scalar bootstrap stalled. The test
was not reopened. Statistical analysis is resumed exclusively from those immutable
prediction CSV files. Bootstrap target metrics now use a confusion-matrix-only
implementation for MCC and balanced accuracy instead of recalculating unrelated
AUROC, AUPRC, NLL, Brier score, and ECE on every resample. A regression test checks
numerical equivalence with scikit-learn.

The generated strict protocol contains exactly 10,322/400/800 images and has
zero lesion overlap for train-validation, train-test, and validation-test.
For the retrospective controlled audit, 673 train images are removed. Of these, 620 can be
replaced by unique eligible images; the remaining 38 VASC and 15 DF exposures
are matched by repeated clean sampling. The test decomposition contains 383
leaked and 417 clean images.

The full overlap audit additionally found 44 lesion IDs present in train,
validation, and test simultaneously, affecting 86, 55, and 59 images in the
three splits. Removing test-overlapping lesions therefore also reduces part of
the train-validation overlap. This is measured explicitly and prevents a claim
that lesion contamination is the only changed factor.

## Motivation

Stage 1 found 328 lesion identifiers shared between MONICA IR100 train and
test, affecting 673 training images and 383 test images. Different photographs
of one lesion are not independent clinical cases. Therefore MONICA remains a
benchmark-comparability protocol, while lesion-disjoint evaluation becomes the
primary generalization protocol.

The magnitude and direction of bias are not assumed. They will be estimated.
The leaked and clean test subsets can differ in class mix and difficulty, so a
subgroup contrast alone is descriptive rather than causal.

## Experiment A: descriptive MONICA test decomposition

After all models and checkpoints are frozen, divide the unchanged MONICA test:

- `test_leaked`: lesion ID occurs in the original MONICA train;
- `test_clean`: lesion ID does not occur in the original MONICA train.

Report full-test, leaked-subset, and clean-subset balanced accuracy, macro F1,
MCC, macro/per-class AUROC and AUPRC, calibration, and class support. Also use a
lesion-group bootstrap. The difference is labelled an association, not a pure
causal effect.

## Experiment B: retrospective controlled lesion-contamination audit

Compare two training arms on the same unchanged validation and test images:

1. Original MONICA train.
2. Remove every train image whose lesion occurs in MONICA test, then restore
   the exact per-class image counts using official ISIC 2019 images outside all
   MONICA IR100 splits whose lesions occur in neither validation nor test.

The restoration is only exact with unique images when the official residual
pool permits it. The rare classes use nearly the entire dataset: the initial
audit found that VASC needs 41 replacements but only 3 unique VASC images are
available outside the benchmark lists. The implementation therefore saves:

- an unmatched clean arm with no replacement;
- an exposure-matched arm that uses every eligible unique replacement first
  and fills only the remaining deficit by repeated sampling of clean images.

Repeated exposure is a documented secondary intervention and weakens a purely
causal interpretation. Results from both clean arms must be shown; the paper
must not describe the exposure-matched arm as containing all-unique images.
`replacement_balance_report.csv` records class, age, sex, anatomical site,
resolution, lesion multiplicity, and source for each removed/replacement
exposure. Source is not an official metadata column and is explicitly marked as
an inference from `HAM_*`/`BCN_*` lesion prefixes. Matching is guaranteed only
for class and exposure count, not source, device, demographics, resolution, or
clinical difficulty.

This keeps architecture, optimizer, augmentation, training duration, class
counts, validation set, test set, and seeds fixed. It changes the image content
and removes train-test lesion contamination. Residual train-validation overlap
from MONICA is retained in both arms and disclosed because removing it would
change two factors at once. Test predictions and metrics were not inspected
while constructing the protocol, but test lesion identifiers were used for
removal, replacement, and subgroup definition. Consequently this is not a
blind test or a pure causal estimate.

Run seeds 42, 43, and 44. Test is opened once only after all twelve checkpoints
are frozen. Primary paired effect:

`MCC(decontaminated_exposure_matched, last.pt) - MCC(original_monica, last.pt)`.

Balanced accuracy is the key secondary endpoint. Macro F1, macro/per-class
AUROC and AUPRC, melanoma sensitivity, per-class recall, NLL, Brier score, and
ECE are additional endpoints. The unmatched-minus-original contrast is a
robustness analysis. A leaked-minus-clean contrast is descriptive association.

The prior 11-run launch is marked as a pilot and excluded. The confirmatory
matrix contains 12 new runs: all four arms at seeds 42, 43, and 44. Every run
uses the same code commit, environment, ResNet-50 initialization policy,
augmentation, Adam settings, physical batch 256, and 50 epochs without early
stopping. `last.pt` is primary; validation-selected `best.pt` is secondary.

## Experiment C: primary lesion-disjoint protocol

Build ISIC-2019-LT IR100 from official ISIC 2019 data by treating every
`lesion_id` as an indivisible group. Per class, choose complete lesion groups
that total exactly 100 test images, 50 validation images, and the MONICA IR100
training target (5000, 2590, 1342, 695, 360, 187, 97, 51). No lesion may occur
in more than one split. Empty identifiers, if any, are isolated by image ID.

This protocol answers generalization to unseen lesions. It is not directly
comparable numerically with MONICA because its images differ. It receives a
separate experiment name and is the primary protocol for the future article.
MCC is its primary scientific metric and balanced accuracy the key secondary.
Metrics are reported both per image and after averaging probabilities within
each lesion. Image and lesion counts and their respective imbalance ratios are
saved because image-level IR=100 does not imply lesion-level IR=100.

## Checkpoints and uncertainty

All models run for the fixed 50 epochs. Primary results use `last.pt` only.
`best.pt`, selected by validation balanced accuracy, appears only in a separate
benchmark-sensitivity table. Results from the two checkpoint policies are never
mixed in one column.

Uncertainty uses 10,000 paired lesion-stratified bootstrap repetitions. Lesions
are sampled with replacement within class, all images of each selected lesion
are retained, and identical bootstrap draws are applied to compared models.
The report includes each seed's effect, mean and standard deviation across
seeds, and a 95% paired lesion-bootstrap interval. A t-test over three seeds is
not the primary inferential procedure.

## Locked-test policy

Test predictions are forbidden until:

1. Manifests, hashes, class counts, and zero lesion overlap are verified.
2. Seeds and model selection rules are frozen.
3. All checkpoints for a comparison are complete.
4. The evaluation script checks the preregistered manifest hashes.
5. Exactly 12 runs have matching signatures containing git commit, resolved
   configuration hash, train and validation hashes, protocol version,
   checkpoint policy, and epoch count.
6. Both `last.pt` and `best.pt` exist and carry the same run signature.

The one-shot evaluation creates a start marker before reading any test image
and refuses a second invocation. No hyperparameter changes may be made after
test inspection. Stage 1 and the stopped Stage 2 pilot have not loaded or
evaluated any test image.

The planned outputs are separated into four tables: MONICA benchmark results
from `best.pt`; retrospective contamination effects from `last.pt`; descriptive
full/leaked/clean MONICA-test results; and primary lesion-disjoint image- and
lesion-level results. The stopped pilot under `outputs/stage2` is never eligible
for these tables; confirmatory runs live under `outputs/stage2_confirmatory`.

## Evidence from primary sources

- The ISIC 2019 challenge provides 25,331 training images and metadata used for
  lesion identification: https://challenge.isic-archive.com/data/#2019
- HAM10000 describes 10,015 dermatoscopic images collected from multiple
  sources and includes multiple images of lesions:
  https://doi.org/10.1038/sdata.2018.161
- BCN20000 reports 18,946 images corresponding to 5,583 lesions and supplies a
  lesion identifier, demonstrating that image and lesion are distinct sampling
  units: https://doi.org/10.1038/s41597-024-03387-w
- MONICA defines its long-tail image-level benchmark and balanced evaluation
  subsets: https://arxiv.org/abs/2410.02010

The exact inflation cannot be borrowed from another imaging modality. It must
be measured on these manifests with paired seeds and lesion-group uncertainty.

## Confirmatory results

All 12 preregistered runs completed at the same training commit
`d0824eaebabd5f699f49236df8534a2864b61b49`. Readiness accepted exactly 12
runs with matching configuration, split, registry, and checkpoint signatures.
The locked test was opened once. Primary results use `last.pt`; `best.pt`
remains a separately named secondary benchmark.

### Primary lesion-disjoint baseline

| Unit | MCC, mean (SD) | Balanced accuracy | Macro F1 | Macro AUPRC | ECE |
|---|---:|---:|---:|---:|---:|
| Image | 0.428 (0.042) | 0.484 (0.041) | 0.448 (0.051) | 0.555 (0.035) | 0.413 (0.040) |
| Lesion | **0.489 (0.061)** | **0.513 (0.059)** | 0.489 (0.065) | 0.632 (0.054) | 0.304 (0.054) |

Across seeds 42/43/44, lesion-level MCC bootstrap means were 0.466, 0.443,
and 0.558. Their seed-specific 95% lesion-stratified intervals were
[0.419, 0.512], [0.394, 0.492], and [0.509, 0.606]. Split variability must
therefore be quantified in Stage 3B in addition to model-seed variability.

### Retrospective MONICA contamination audit

| Primary checkpoint (`last.pt`) | Image MCC | Balanced accuracy | Macro AUPRC | ECE |
|---|---:|---:|---:|---:|
| Original MONICA | 0.569 (0.004) | 0.611 (0.003) | 0.723 (0.007) | 0.295 (0.006) |
| Exposure-matched | 0.403 (0.013) | 0.461 (0.010) | 0.537 (0.010) | 0.437 (0.008) |
| Unmatched | 0.374 (0.013) | 0.436 (0.012) | 0.515 (0.019) | 0.457 (0.008) |

The preregistered exposure-matched-minus-original effect was MCC **-0.166**
(seed SD 0.012; paired lesion-bootstrap 95% CI [-0.191, -0.142]) and balanced
accuracy **-0.150** (95% CI [-0.171, -0.128]). The unmatched effect was larger:
MCC -0.196 (95% CI [-0.220, -0.173]). Thus reduced training exposure explains
part, but not all, of the original-to-cleaned difference.

Within Original MONICA, the descriptive leaked subset had MCC 0.684 and macro
AUPRC 0.842, versus MCC 0.461 and macro AUPRC 0.591 on the clean subset. This
is strong evidence of optimistic association, but it is not labelled a causal
estimate because test lesion identifiers informed the retrospective audit.

## Frozen decision for Stage 3

The sole development protocol is `ISIC 2019 LT IR=100 lesion-disjoint`.
Original MONICA is retained only for one final benchmark-comparability check;
exposure-matched is retained only as the principal retrospective audit arm;
unmatched is complete as a sensitivity analysis and is not scaled further.
No future loss, sampler, augmentation, architecture, or synthetic-data choice
may be selected on MONICA validation or test results.

MLflow contains all 12 runs with grouping tags and separate `test/*` primary
and `test_best/*` secondary metrics. Future live runs emit both legacy metric
keys and readable namespaces: `train/loss`, `val/*`, and `system/*`.
