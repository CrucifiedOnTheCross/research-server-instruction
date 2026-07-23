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
