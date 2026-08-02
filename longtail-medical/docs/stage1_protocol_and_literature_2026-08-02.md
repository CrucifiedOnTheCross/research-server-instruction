# Stage 1: qualification of ISIC-2019-LT and the MONICA baseline

Date: 2026-08-02  
Status: implementation and server qualification  
Locked test: closed (`test_evaluated=false`)

## Research question

Before testing frequency-dispersion reweighting, uncertainty-controlled mixing,
or deferred classifier calibration, establish whether the published benchmark,
class mapping, training recipe, and evaluation units can be reproduced without
hidden leakage. Stage 1 therefore estimates no novelty claim. It creates a
traceable reference point against which later methods can be judged.

## Protocol selected for direct comparison

The public MONICA revision is pinned to
`3dd808d6d578b9e0f9bf4ee1402727ff46d1c243`. Its ISIC-2019-LT class order is:

| Numeric label | Diagnosis |
|---:|---|
| 0 | NV |
| 1 | MEL |
| 2 | BCC |
| 3 | BKL |
| 4 | AK |
| 5 | SCC |
| 6 | VASC |
| 7 | DF |

This order is verified image by image against the official ISIC 2019 ground
truth. It differs from the order stated in the supplied research report and is
therefore treated as a critical reproducibility check.

For imbalance ratio 100, MONICA contains 10,322 training images. The class
counts are 5000, 2590, 1342, 695, 360, 187, 97, and 51. Validation contains 50
images per class and test contains 100 images per class.

## Audit findings that change the interpretation

The lists are image-disjoint but not lesion-disjoint. Using the official
`lesion_id` metadata, imbalance ratio 100 has:

| Pair | Shared lesions | Images affected in first split | Images affected in second split |
|---|---:|---:|---:|
| train / validation | 175 | 390 | 195 |
| train / test | 328 | 673 | 383 |
| validation / test | 77 | 93 | 103 |

Consequences:

1. MONICA scores remain useful only for direct comparison with published work.
2. They must not be described as lesion-independent clinical generalization.
3. Stage 2 must build a separate lesion-disjoint protocol and report it under a
   different name; its scores cannot be mixed into the MONICA leaderboard.
4. The official metadata contains no `patient_id`. A patient-disjoint claim is
   impossible without a separately validated source of patient identifiers.

The validation and test lists also change between imbalance ratios 100, 200,
and 500. Thus cross-ratio changes mix training imbalance with evaluation-cohort
changes. Later ratio experiments need both published-list results and a fixed
lesion-disjoint evaluation cohort.

## What the source papers actually trained

### MONICA

The paper and public configuration use an ImageNet-pretrained ResNet-50,
224-pixel input, Adam with learning rate `3e-4`, effective batch 256, and 50
epochs. Training augmentation resizes to 264 pixels, takes a random 224-pixel
crop, applies horizontal and vertical flips, rotation up to 10 degrees, and
color jitter of 0.2. Validation uses resize 264 and center crop 224. The authors
also note that 50 epochs may be suboptimal for several compared methods.

The RTX 5080 implementation starts with physical batch 128 and gradient
accumulation 2. This preserves effective batch size, not exact per-step batch
statistics or bitwise equivalence. A server memory smoke test may qualify
physical batch 256 before the definitive run.

### BPaCo

BPaCo is not a drop-in loss. It uses a momentum encoder, a feature queue,
across-batch class averaging, two forms of class centres, supervised contrastive
learning, and compensated classifier logits. The MICCAI paper reports ResNet-50,
RandAugment, 224-pixel input, batch 128, SGD with learning rate 0.01, momentum
0.999, weight decay `1e-4`, 128-dimensional features, and 1000 epochs on eight
RTX 3090 GPUs. Public parser defaults differ in batch and learning rate; later
reproduction must explicitly record whether paper or code settings are used.

### SPMix and TailBoost

Original SPMix uses a hybrid ResNet-50/ViT-S architecture with query and
momentum key encoders, saliency-guided patch-level feature mixing, and a
contrastive objective. Reported training uses two RTX 3090 GPUs, batch 64,
AdamW with learning rate `5e-6`, weight decay 0.1, and 500 epochs. It cannot be
represented scientifically as ordinary image MixUp added to the Stage 1
ResNet. TailBoost is a later method and must not be cited as the original SPMix.

## Stage 1 implementation contract

- Separate package: `longtail-medical`; no imports from `research-ml`.
- Pinned source arrays and SHA-256 verification.
- Official ground-truth cross-check for every image label.
- Automated image and lesion overlap reports.
- Natural-sampling cross-entropy ResNet-50 only.
- Balanced accuracy is the primary validation selection metric.
- Secondary metrics: macro F1, MCC, macro/per-class AUROC and AUPRC, worst-class
  recall, negative log-likelihood, Brier score, and expected calibration error.
- Structured artifacts are canonical; logs are diagnostic.
- Test manifest is generated but never instantiated by Stage 1 training.

## Preregistered Stage 1 gates

1. All pinned hashes and labels match.
2. No image identifier overlap exists within a MONICA ratio.
3. Lesion overlap is measured and disclosed rather than silently ignored.
4. All eight validation classes are present.
5. The GPU smoke test produces finite loss and finite probabilities.
6. Two repeated runs with the same seed reproduce the early trajectory within
   the tolerance expected from the CUDA stack; multi-seed confirmation follows.
7. No new weighting or mixing method proceeds until this baseline is complete.

## Planned next decision

After baseline qualification, run seeds 42, 43, and 44 and compare the result
with the MONICA reference. Stage 2 then freezes a lesion-disjoint protocol.
Only after both references are stable should the project implement strong
baselines and the proposed uncertainty-aware method. For rare classes, raw
within-class dispersion will require shrinkage or posterior uncertainty: a tiny
class can have deceptively low empirical variance because it is poorly sampled.

## Primary sources

1. MONICA, arXiv:2410.02010; official code: https://github.com/PyJulie/MONICA
2. BPaCo, MICCAI 2024, DOI: 10.1007/978-3-031-72378-0_36; official code:
   https://github.com/Davidczy/BPaCo
3. SPMix, arXiv:2406.10801; official code: https://github.com/Yancy10-1/SPMix
4. GCL, CVPR 2022; official code: https://github.com/Keke921/GCLLoss
5. TailBoost, Sensors 2026, DOI: 10.3390/s26113343
