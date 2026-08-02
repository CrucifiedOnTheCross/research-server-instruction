# Stage 3A: lesion-disjoint long-tail loss screening

Date: 2026-08-02  
Status: preregistered before Stage 3A training  
Test state: Stage 2 test was opened once and is now permanently closed  
Selection data: lesion-disjoint train and validation only

## Scientific question

Which established long-tail objective improves generalization to previously
unseen lesions when backbone, pretrained initialization, optimizer,
augmentation, split, batch, epoch count, and model seeds are held fixed?

This is a screening stage, not the final claim. Its result selects CE and at
most two methods for Stage 3B confirmation across new lesion-disjoint split
seeds 101, 202, and 303. No Stage 3A decision may use the Stage 2 test.

## Frozen matrix

Six objectives are trained for seeds 42, 43, and 44 (18 runs):

| Arm | Frozen implementation |
|---|---|
| CE | ordinary multiclass cross-entropy |
| Weighted CE | inverse class frequency, normalized to mean weight 1 |
| Focal | multiclass focal CE, gamma 2 |
| CB-Focal | effective-number weights, beta 0.9999, gamma 2 |
| Balanced Softmax | CE on `logits + log(class_count)` |
| LDAM-DRW | margin proportional to `n_c^-1/4`, max margin 0.5, cosine classifier, scale 30; effective-number reweighting only after epoch 40 |

Post-hoc Logit Adjustment with tau 1 is derived from CE validation
predictions and does not retrain a duplicate model. This is deliberate:
training-time logit adjustment with tau 1 has the same objective as Balanced
Softmax. The post-hoc variant uses `log(p_c) - log(pi_c)` and is recorded as a
derived validation-only method.

All training runs use ResNet-50 with ImageNet-1K v2 initialization, Adam with
learning rate 0.0003 and zero weight decay, physical batch 256, bfloat16,
224-pixel crops, the Stage 2 augmentations, exactly 50 epochs, and no early
stopping. `last.pt` is primary; `best.pt` is secondary and selected only by
lesion-level validation MCC.

LDAM's normalized classifier is an explicit component of its canonical
implementation. It is saved as `classifier_type=cosine_normed_linear`; other
arms use the Stage 2 linear classifier. Therefore any LDAM effect is attributed
to the canonical LDAM-DRW package, not to the margin in isolation.

## Endpoints and decision rule

Primary screening endpoint: lesion-level validation MCC from `last.pt`.

Key secondary endpoints: lesion-level balanced accuracy, macro AUPRC, MEL/SCC/AK
recall and AUPRC, worst-class recall, ECE, NLL, Brier score, image-level metrics,
seed sign stability, epoch trajectory, elapsed time, and peak VRAM.

Ranking is based on multi-seed mean, paired-seed effects versus CE, and lesion
bootstrap uncertainty. A method is not promoted solely because one seed is
best. CE and at most two methods advance. Test remains unavailable. Calibration
is reported but does not replace the primary ranking metric.

## Reproducibility artifacts

Every run must contain resolved YAML and JSON configuration, environment,
split hashes, run signature, class counts, model and loss initialization,
epoch CSV, last/best validation metrics and predictions, `last.pt`, `best.pt`,
and `summary.json` with `test_evaluated=false`. MLflow uses project, stage,
split, method, seed, protocol, checkpoint, and test-policy tags.

Readiness requires exactly 18 valid training runs from one commit and three
derived post-hoc runs. Configurations contain no `test_csv`; split artifacts
must state `test_loaded=false` and `test_evaluated=false`.

## Literature and implementation audit

- Lin et al., Focal Loss for Dense Object Detection, ICCV 2017:
  https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf
- Cui et al., Class-Balanced Loss Based on Effective Number of Samples, CVPR 2019:
  https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html
- Cao et al., Label-Distribution-Aware Margin Loss, NeurIPS 2019:
  https://proceedings.neurips.cc/paper/2019/hash/621461af90cadfdaf0e8d4cc25129f91-Abstract.html
- Ren et al., Balanced Meta-Softmax, NeurIPS 2020:
  https://proceedings.neurips.cc/paper/2020/hash/2ba61cc3a8f44143e1f2f13b2b729ab3-Abstract.html
- Menon et al., Long-tail learning via logit adjustment, ICLR 2021:
  https://research.google/pubs/long-tail-learning-via-logit-adjustment/
- Ju et al., MONICA medical long-tail benchmark:
  https://arxiv.org/abs/2410.02010

Official code snapshots inspected before implementation:

- MONICA commit `3dd808d6d578b9e0f9bf4ee1402727ff46d1c243`;
- LDAM-DRW commit `2536330f2afdaa65618323cb5a5850efccce762a`;
- Balanced Meta-Softmax commit `34a61e432881816c2da14d577d6ed63501288f5f`.

The audit confirmed Balanced Softmax adds log class counts to logits; the
effective-number implementations normalize weights to the number of classes;
canonical LDAM uses `n_c^-1/4`, maximum margin 0.5, scale 30, and deferred
reweighting. MONICA defers reweighting until 80% of its configured epochs.

## Stage 3B gate

After Stage 3A, freeze CE and no more than two promoted methods before creating
or inspecting any new split test. Stage 3B must separate split variability from
model-seed variability. Synthetic augmentation belongs to Stage 3C and must be
compared with exposure-matched oversampling and simpler augmentation controls.
