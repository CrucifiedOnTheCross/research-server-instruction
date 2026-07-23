# Stage 8: methodology repair and robust screening

## Purpose

Stage 8 replaces the confounded synthetic-data protocol used in Stages 2-7. The old synthetic pool is retained only as a documented negative pilot result and is not reused for new training.

The scientific question is:

> After removing acquisition artifacts and evaluation leakage, can independently selected synthetic dermoscopic images improve long-tail classification beyond strong real-only baselines?

## Critical issues corrected

1. **Black letterbox artifact**
   - The previous generator placed each 600x450 HAM10000 image on a black 512x512 canvas.
   - The median black-border share was 0.9805 for synthetic images and 0.0 for real images.
   - Stage 8 uses aspect-preserving resize plus center crop and runs a mandatory pixel-artifact audit before feature analysis.

2. **Circular feature selection**
   - Previous selection and downstream classification used features from the same task-trained ConvNeXt.
   - Stage 8 uses pretrained `vit_base_patch14_dinov2.lvd142m` for synthetic geometry and a separate `convnext_base.fb_in22k_ft_in1k_384` classifier.

3. **Incorrect prior correction**
   - Stage 7 combined weighted sampling with Logit Adjustment.
   - Stage 8 evaluates Logit Adjustment only with natural sampling and includes natural-CE and weighted-CE controls.

4. **Repeated test-set use**
   - A new group-aware split is created before generation.
   - Generation uses only `splits/stage8/train_real.csv`.
   - Validation and locked test contain no `lesion_id/group_id` used as an img2img source.
   - Screening sets `evaluation.run_test=false`; the locked test remains untouched until a final method is selected.

5. **Single-seed evidence**
   - Four screening arms run with seeds 42, 43, 44, 45, and 46.
   - The final two methods will be evaluated with group-aware cross-validation and one locked-test evaluation.

6. **Insufficient image resolution**
   - Classification is upgraded from ConvNeXt Tiny at 224 px to ConvNeXt Base at 384 px.
   - Evaluation uses resize 438 followed by center crop 384. The dataset code now rejects resize values smaller than the crop.

## Corrected candidate generation

- Base model: Stable Diffusion 1.5 img2img.
- Input: Stage 8 real training split only.
- Resolution: 512x512 PNG.
- Source preprocessing: aspect-preserving center crop; no padding.
- Strength sweep: 0.15, 0.30, 0.45.
- Classes: `mel`, `akiec`, `bkl`.
- Sources: up to 160 real lesions per class.
- Candidates: 1,440 total.
- Diffusion batch: 4 on RTX 5080.
- Every image records source image, source group, seed, strength, prompt, model, preprocessing, and format.

## Independent selection

DINOv2 features are used to calculate:

- class-conditional nearest-real distance;
- distance to clinically confusing classes;
- feature margin;
- PRDC precision, density, and coverage;
- synthetic nearest-neighbor distance;
- source diversity and duplicate rate.

The selector keeps up to 80 diverse candidates per target class. The ConvNeXt classifier is not used by the selector.

## Screening matrix

| Arm | Sampling | Loss | Synthetic data |
|---|---|---|---|
| Real CE natural | natural | cross-entropy | no |
| Real CE weighted | class-balanced sampler | cross-entropy | no |
| Real Logit Adjustment | natural | logit-adjusted CE, tau=1 | no |
| Corrected synthetic + DINO | class-balanced sampler | cross-entropy | selected mel/akiec/bkl, loss weight 0.5 |

Common settings:

- ConvNeXt Base IN22k to IN1k, 384 px;
- batch 32, BF16, channels-last;
- AdamW, learning rate 1e-4, cosine decay;
- 50 epochs, early stopping patience 10;
- deterministic algorithms enabled;
- five training seeds;
- all resolved configs, environment, predictions, checkpoints, and metrics are saved.

The RTX 5080 smoke test measured 11.97 GiB peak allocated and 12.51 GiB peak reserved at batch 32, leaving sufficient headroom within 16 GiB VRAM.

## Literature basis

- Farooq et al. **Derm-T2IM**: domain adaptation of Stable Diffusion with DreamBooth/LoRA, artifact rejection, and downstream validation. <https://arxiv.org/abs/2401.05159>
- Samuel et al. **SeedSelect**: reference-guided diffusion seed selection for rare concepts without generator fine-tuning. <https://arxiv.org/abs/2304.14530>
- Wang et al. **Improving the Effectiveness of Deep Generative Data**: downstream utility depends on content/domain gap, not visual fidelity alone. <https://openaccess.thecvf.com/content/WACV2024/html/Wang_Improving_the_Effectiveness_of_Deep_Generative_Data_WACV_2024_paper.html>
- Yamaguchi. **Analyzing Diffusion Models on Synthesizing Training Datasets**: diffusion reconstructions can help as local interpolation but do not reproduce the full real distribution. <https://proceedings.mlr.press/v260/yamaguchi25a.html>
- Kim et al. **Diffusion-based skin disease data augmentation with fine-grained detail preservation and interpolation for data diversity**: class conditioning, lesion masks, multi-level image features, and medical-detail preservation. <https://doi.org/10.1371/journal.pone.0331404>
- Menon et al. **Long-tail learning via logit adjustment**: prior correction is evaluated under the natural class distribution rather than combined with class-balanced resampling. <https://arxiv.org/abs/2007.07314>

## Decision rule

No locked-test result is inspected during screening.

Advance a synthetic policy only if, across five seeds, it:

- improves mean validation macro F1 and MCC over both real-only controls;
- does not reduce balanced accuracy or worst-class recall materially;
- shows a consistent seed-level effect rather than one exceptional run;
- passes the artifact audit and improves DINOv2 precision/coverage over the old pool.

If corrected off-the-shelf SD 1.5 still fails these criteria, the next generator will be a train-only dermoscopy LoRA with the same downstream protocol. This avoids confusing a better generator with a changed classifier or evaluation split.

## Research change log

### S8-H1: artifact-free train-only generation

- Status: running
- Date: 2026-07-23
- Code commits: `f33b52c`, `47ea4e8`
- Observation: the old synthetic pool had near-black letterbox borders that were absent from real images, and a real-vs-synthetic detector separated the domains almost perfectly.
- Hypothesis: removing padding artifacts and restricting img2img sources to the new training split will reduce the synthetic domain gap and make downstream utility measurable without source leakage.
- Rationale: visual fidelity alone is insufficient; useful synthetic samples must be compatible with the real feature distribution and evaluation protocol.
- Literature: Wang et al. (WACV 2024), Yamaguchi (MIDL 2025), Farooq et al. (Derm-T2IM), Samuel et al. (SeedSelect).
- Code/config changes: corrected center-crop preprocessing, batched generation, source-group provenance, pixel audit, fresh group-aware split and independent DINOv2 selector.
- Data and split: HAM10000 split by `lesion_id/group_id`; generation sources are restricted to `splits/stage8/train_real.csv`.
- Independent variable: old artifact-contaminated pool versus corrected train-only pool.
- Outcomes: artifact rate, real-vs-synthetic separability, DINOv2 precision/coverage and downstream validation macro F1/MCC.
- Controls: real-only natural CE, real-only weighted CE and natural-sampling Logit Adjustment.
- Seeds/folds: screening seeds 42-46; group-aware cross-validation is reserved for the two finalists.
- Leakage safeguards: validation and locked-test source groups cannot appear in generation; locked-test evaluation is disabled during screening.
- Predefined success criterion: the selected synthetic policy must improve five-seed validation macro F1 and MCC over both real-only controls without a material loss in balanced accuracy or worst-class recall.
- Structured artifacts: `outputs/reports/stage8_artifact_audit`, `outputs/reports/stage8_dino_geometry`, `outputs/reports/stage8_multiseed` and per-run resolved configs/predictions.
- Current result: pipeline launched; final scientific decision is pending structured multi-seed results.
- Decision: pending.
- Follow-up hypothesis: if corrected SD 1.5 remains ineffective, test a train-only dermoscopy LoRA while freezing the classifier and evaluation protocol.

#### Runtime verification: 2026-07-23 10:18

- Structured sources checked: `artifact_audit.json`,
  `stage6_feature_geometry_report.json`, `metrics.csv` and
  `val_metrics_best.json`. Training logs were not used as a metric source.
- Candidate generation completed with 1,440 images; the DINOv2 selector retained
  80 images for each of `mel`, `akiec` and `bkl`.
- The black-border audit passed. Median and p90 black-border share are zero for
  both real and synthetic images. The single-feature AUROC based on black-border
  share fell to `0.505`, compared with the almost perfectly separable old pool.
- A residual domain gap remains. DINOv2 PRDC precision is `0.125` for `mel`,
  `0.169` for `akiec` and `0.260` for `bkl`; coverage is `0.076`, `0.272` and
  `0.120`, respectively. Only 35, 36 and 63 candidates passed the strict geometry
  filter before the diversity/top-k fallback filled each class to 80.
- The first real-only natural-CE run (`seed=42`) reached epoch 3 without metric
  anomalies. Train loss changed `1.231 -> 0.627 -> 0.447`; validation macro F1
  changed `0.136 -> 0.570 -> 0.591`; MCC changed
  `0.078 -> 0.397 -> 0.520`. These early values are not used for model selection.
- Twelve consecutive GPU samples showed 100% SM utilization, 68-88% memory
  controller utilization, 13.7/16.3 GiB VRAM use and maximum 2,857 MHz SM clock.
  No thermal or power violation was reported; temperature remained 61-66 C.
- Host measurements showed 0% I/O wait, zero swap use, approximately 51 GiB
  available RAM and 828 GiB free SSD space. CPU remained mostly idle because the
  workload is GPU-bound and the 12 persistent DataLoader workers already keep the
  GPU continuously supplied.
- Decision: keep batch 32 and the current worker count. Increasing either during
  the run would reduce memory headroom and break comparability without addressing
  a measured bottleneck.
- Follow-up check: compare the selected-pool PRDC metrics with downstream
  five-seed utility. If utility remains absent, test whether strict-only selection
  performs better than filling every class to a fixed top-k of 80.

### Engineering changes

- ConvNeXt Base at 384 px replaces the 224 px Tiny screening backbone.
- Model-only best checkpoints are retained; optimizer state and redundant last checkpoints are disabled to keep multi-seed storage bounded.
- The physical SSD LVM was expanded online from 100 GiB to approximately 951 GiB before the Stage 8 run.
