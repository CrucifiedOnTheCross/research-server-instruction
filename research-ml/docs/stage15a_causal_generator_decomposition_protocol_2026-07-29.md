# Stage 15A: causal generator decomposition protocol

Дата preregistration: 2026-07-29

Статус: implementation complete, locked test закрыт.

## Причина этапа

Retrospective audit показал, что Stage 13B смешивал четыре воздействия:

1. offline center crop `600x450 -> 512x512`;
2. fixed сохранённый view вместо online stochastic view;
3. generic SD1.5 4-channel VAE round-trip;
4. примерно один UNet denoising step при `strength=0.05`.

Поэтому отрицательную downstream utility нельзя причинно приписать
генератору или отбору. Stage 15A разлагает composite effect при одинаковой
дозе, источниках, architecture и training recipe.

## Гипотезы

### H15A-1: offline-view hypothesis

Фиксированный offline center crop ухудшает utility относительно original
source replay из-за потери 25% поля изображения и уменьшения разнообразия
online views.

Сравнение: `B - A`.

### H15A-2: generic-VAE hypothesis

Generic SD1.5 4-channel VAE дополнительно ухудшает utility и frequency/detail
preservation даже без UNet denoising.

Сравнение: `C - B`.

### H15A-3: near-zero denoising hypothesis

Один img2img denoising step не добавляет полезной семантической вариативности
после VAE reconstruction.

Сравнение: `D - C`.

### H15A-4: composite reproduction

Строго `strength=0.05` img2img arm воспроизводит отрицательный или нулевой
эффект Stage 13B относительно replay.

Сравнение: `D - A`.

## Зафиксированная экспериментальная матрица

| Arm | Experiment | Воздействие |
|---|---|---|
| A | `stage15a_original_replay_convnext_small_384` | исходный real replay |
| B | `stage15a_offline_crop_convnext_small_384` | exact center crop, PNG |
| C | `stage15a_vae_roundtrip_convnext_small_384` | B + SD1.5 VAE mode reconstruction |
| D | `stage15a_img2img_strength05_convnext_small_384` | текущий SD1.5 img2img, строго strength 0.05 |

Общие условия:

- одни и те же 90 source images;
- 90 уникальных `source_group_id/lesion_id`;
- `mel`, `akiec`, `bkl`: по 30 дополнений;
- source/count matched;
- вес каждого дополнительного наблюдения `0.5`;
- ConvNeXt-Small `convnext_small.fb_in22k_ft_in1k_384`;
- ImageNet-22k -> ImageNet-1k initialization;
- `80` epochs максимум;
- batch `32`, bf16 inherited from base config;
- layer decay `0.925`;
- EMA `0.9999` с warmup;
- seeds `42, 43, 44`;
- early stopping patience `15`;
- locked test: `evaluation.run_test=false`.

## Контроль генератора

VAE:

- model: `stable-diffusion-v1-5/stable-diffusion-v1-5`;
- revision:
  `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`;
- subfolder: `vae`;
- posterior: deterministic `latent_dist.mode()`;
- decode без UNet и без latent sampling;
- float32 для исключения дополнительного fp16 reconstruction effect;
- output: lossless PNG.

Img2img arm:

- для каждого source выбирается существующий sample с `strength=0.05`;
- samples `strength=0.10` из Stage 13B исключены;
- source image и source lesion mapping должны точно совпадать со всеми arms.

## Endpoint и checkpoint policy

Primary endpoint:

- validation macro one-vs-rest AUPRC.

Checkpoint monitor:

- `val/auprc_ovr_macro`, mode `max`.

Это решение принято до запуска и устраняет Stage 11B/13B mismatch, где
checkpoint выбирался по macro F1, а ranking conclusions делались по AUPRC.

Secondary endpoints:

- macro F1;
- MCC;
- balanced accuracy;
- ECE;
- worst-class recall;
- macro/per-class AUROC и AUPRC;
- melanoma precision, recall, F1, AUPRC;
- fixed-specificity diagnostics;
- elapsed time, best epoch, GPU/VRAM.

## Статистический анализ

- paired seeds `42-44`;
- mean/std;
- paired deltas для `B-A`, `C-A`, `D-A`;
- производные causal contrasts `C-B` и `D-C`;
- 5000-repeat hierarchical lesion-group bootstrap;
- знак эффекта по seeds;
- ranking effects отделяются от threshold effects;
- locked test не открывается.

Три pairwise analysis директории:

- `outputs/reports/stage15a_analysis/offline_crop`;
- `outputs/reports/stage15a_analysis/vae_roundtrip`;
- `outputs/reports/stage15a_analysis/img2img_strength05`.

## Gate

До обучения `tools/check_stage15a_gate.py` проверяет:

- наличие всех images/CSV/manifest;
- ровно 90 additions на arm;
- баланс 30/30/30;
- 90 уникальных source images;
- 90 уникальных source lesions;
- идентичные source sets;
- идентичный real base multiset;
- идентичные total class counts;
- отсутствие source lesion overlap с validation/locked test;
- pinned SD1.5 revision;
- строго `strength=0.05` в D;
- вес `0.5`;
- одинаковый training recipe;
- monitor `val/auprc_ovr_macro`;
- `test_evaluated=false`.

## Decision rules для Stage 15B

1. Если `B-A < 0` устойчиво по seeds/bootstrap, следующий generator pilot
   обязан сохранять aspect ratio или использовать diverse online views.
2. Если `C-B < 0`, generic 4-channel VAE исключается; основной кандидат —
   domain-trained 8-channel VAE.
3. Если `D-C <= 0`, strength 0.05 img2img не считается генерацией полезной
   вариативности.
4. Если `D-C > 0`, но `D-A <= 0`, denoising частично компенсирует VAE/crop,
   но не превосходит простой replay.
5. Domain generator pilot запускается только после анализа всех 12 runs.

## Научные основания

- Kim et al., PLOS ONE 2025: 8-channel medical-domain VAE, lesion masks,
  multi-level embeddings; generic latent compression теряет детали.
- DiDGen, MICCAI 2025: structured clinical prompts и region-aware attention
  для dermoscopic image-mask generation.
- Bissoto et al., CVPRW 2021: необходимость controlled utility comparisons и
  отсутствие гарантии ID improvement от визуально качественной синтетики.
- When Pretty Isn't Useful, CVPR 2026: visual fidelity не гарантирует
  distribution coverage и training utility.
- Diffusion Curriculum, ICCV 2025: utility зависит от image-guidance strength,
  фиксированная сила генерации необязательно оптимальна.

Полный bibliographic audit:
`docs/stage15_generation_selection_failure_audit_2026-07-29.md`.

## Реализация

- `tools/prepare_stage15a_variants.py`;
- `tools/check_stage15a_gate.py`;
- `tests/test_stage15a_protocol.py`;
- `scripts/run_stage15a_causal_decomposition.sh`;
- `scripts/start_stage15a_container.sh`;
- `scripts/run_stage15a_analysis.sh`;
- четыре `configs/ham10000_stage15a_*.yaml`.

## Исправление запуска 2026-07-29

Первый run arm A seed 42 завершился валидно. При старте arm B gate не заметил,
что подготовленные addition rows содержали `sample_weight=0.5`, тогда как у
base rows после CSV concat образовался `NaN`. Dataset loader проверял только
`weight <= 0`, а `NaN` эту проверку обходит. В training loop веса всегда
умножались на `sample_weight`, поэтому loss стал `NaN` с первого batch.

Исправление:

- Stage 15A synthetic arms больше не записывают `sample_weight`; их вес
  задаётся только `training.synthetic_weight=0.5`;
- loader теперь fail-closed отклоняет любые non-finite sample weights;
- добавлен regression test;
- незавершённый arm B каталог помечен `invalid_nan_sample_weight` и никогда
  не включается в сводки;
- валидный arm A seed 42 не перезапускается.
