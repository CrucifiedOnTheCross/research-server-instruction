# Stage 2: lesion leakage and strict generalization protocol

Date: 2026-08-02  
Status: preregistered before opening MONICA test  
Current test state: closed

The generated strict protocol contains exactly 10,322/400/800 images and has
zero lesion overlap for train-validation, train-test, and validation-test.
For the causal control, 673 train images are removed. Of these, 620 can be
replaced by unique eligible images; the remaining 38 VASC and 15 DF exposures
are matched by repeated clean sampling. The test decomposition contains 383
leaked and 417 clean images.

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

## Experiment B: matched contamination intervention

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

This keeps architecture, optimizer, augmentation, training duration, class
counts, validation set, test set, and seeds fixed. It changes the image content
and removes train-test lesion contamination. Residual train-validation overlap
from MONICA is retained in both arms and disclosed because removing it would
change two factors at once.

Run seeds 42, 43, and 44. Test is opened once only after all six checkpoints
are frozen. Primary paired effect:

`metric(decontaminated_matched) - metric(original_monica)`.

The validation-only training matrix additionally includes the unmatched clean
arm and the strict lesion-disjoint arm. It consists of 11 new runs: original
MONICA seeds 43-44 plus three seeds for each of unmatched clean,
exposure-matched clean, and lesion-disjoint protocols. Stage 1 seed 42 is the
original-MONICA anchor. Every run uses 50 epochs without early stopping.

## Experiment C: primary lesion-disjoint protocol

Build ISIC-2019-LT IR100 from official ISIC 2019 data by treating every
`lesion_id` as an indivisible group. Per class, choose complete lesion groups
that total exactly 100 test images, 50 validation images, and the MONICA IR100
training target (5000, 2590, 1342, 695, 360, 187, 97, 51). No lesion may occur
in more than one split. Empty identifiers, if any, are isolated by image ID.

This protocol answers generalization to unseen lesions. It is not directly
comparable numerically with MONICA because its images differ. It receives a
separate experiment name and is the primary protocol for the future article.

## Locked-test policy

Test predictions are forbidden until:

1. Manifests, hashes, class counts, and zero lesion overlap are verified.
2. Seeds and model selection rules are frozen.
3. All checkpoints for a comparison are complete.
4. The evaluation script checks the preregistered manifest hashes.

No hyperparameter changes may be made after test inspection. Stage 1 has not
loaded or evaluated any test image.

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
