# Резюме этапов 1-2: HAM10000, дисбаланс классов и синтетические изображения

Дата среза: 2026-07-08.

Цель документа: передать исследовательской группе полный контекст первых двух этапов, фактические результаты, связь с литературой, текущие ограничения и план следующих экспериментов. Документ написан как рабочий research memo, а не как финальный текст статьи.

## 1. Короткий вывод

На первом этапе были получены сильные real-only baseline на HAM10000 с контролируемым split по `lesion_id`/`patient_id`, сохранением конфигов, логов, метрик и тестовых предсказаний. Лучший результат по `macro F1` дал `balanced_softmax_none` (`0.7989`), лучший результат по `balanced accuracy` и `worst-class recall` дал `cross_entropy_weighted` (`balanced accuracy=0.8279`, `worst recall=0.7305`).

На втором этапе была проверена гипотеза, что image-conditioned synthetic augmentation для проблемных классов `mel`, `akiec`, `bkl` улучшит качество сверх сильных baseline. Было сгенерировано 960 синтетических изображений Stable Diffusion v1.5 img2img, затем построены два режима отбора: строгий feature-space отбор (`43` изображения) и более широкий `rule_or_topk80` (`240` изображений). Оба режима технически отработали, но **ни один stage2-run не превзошел лучшие stage1-baseline по основным test-метрикам**.

Главный научный результат текущего среза: визуально правдоподобная или даже feature-filtered синтетика не гарантирует улучшение модели. Полезность синтетики нужно оценивать через распределительный сдвиг, положение в признаковом пространстве, влияние на конкретные границы классов, calibration и устойчивость на real-only test.

## 2. Данные и split

Использованный датасет: HAM10000, 10 015 дерматоскопических изображений, публично доступный через ISIC Archive. Оригинальная работа Tschandl et al. описывает HAM10000 как коллекцию 10 015 dermatoscopic images с диагностическими категориями pigmented skin lesions и разными источниками/модальностями сбора данных: <https://arxiv.org/abs/1803.10417>, <https://www.nature.com/articles/sdata2018161>.

Подготовленный split:

| Split | Images |
|---|---:|
| train | 7011 |
| val | 1502 |
| test | 1502 |

Ключевой принцип: split строился с учетом `lesion_id`/`patient_id` как `group_id`. Пересечение групп между `train`, `val`, `test` равно `0`. Это важно, потому что одно и то же поражение может иметь несколько снимков, и обычный random image split дал бы leakage.

Классы HAM10000 в коде:

| Label | Интерпретация |
|---|---|
| `mel` | melanoma |
| `nv` | melanocytic nevus |
| `bkl` | benign keratosis-like lesion |
| `bcc` | basal cell carcinoma |
| `akiec` | actinic keratoses / intraepithelial carcinoma family |
| `df` | dermatofibroma |
| `vasc` | vascular lesions |

## 3. Инфраструктура и воспроизводимость

Сервер:

- Host: `lab-bio`, `10.200.1.180`
- GPU: NVIDIA GeForce RTX 5080 16 GB
- Docker image: `local/research-cuda-notebook:latest`
- PyTorch: `2.12.1+cu130`
- Основной проект на сервере: `/srv/research/projects/default/research-ml`
- HAM10000 на сервере: `/srv/research/projects/default/ham10000`

Общие параметры stage1/stage2 training:

| Параметр | Значение |
|---|---|
| Model | `convnext_tiny.fb_in22k_ft_in1k` |
| Input train | 224 px |
| Val/test | resize 256, center crop 224 |
| Batch size | 96 |
| AMP | bf16 |
| Memory format | channels_last |
| DataLoader | `num_workers=12`, `pin_memory=true`, `prefetch=4`, `persistent_workers=true` |
| GPU utilization | около 100% |
| VRAM при обучении | около 7 GB |

Что сохраняется в каждом run:

- `config.resolved.yaml`
- `environment.json`
- `class_counts.json`
- `metrics.csv`
- `metrics.jsonl`
- `run.log`
- `best.pt`
- `last.pt`
- `val_metrics_best.json`
- `test_metrics.json`
- `val_predictions_best.csv`
- `test_predictions.csv`

Ограничение воспроизводимости, которое нужно исправить: серверная папка не является git-репозиторием, поэтому в `environment.json` сейчас `git_commit=None`. В следующем этапе нужно либо запускать обучение из git clone, либо принудительно передавать `CODE_VERSION`/commit hash в конфиг и сохранять checksum ключевых файлов.

## 4. Этап 1: real-only baseline matrix

Цель этапа 1: получить сильные baseline без синтетики, понять, какие loss/sampler работают лучше, и выявить проблемные границы классов.

Матрица:

| Experiment | Epochs | Best epoch | Test macro F1 | Balanced acc | MCC | ECE | Worst recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| `stage1_balanced_softmax_none` | 50 | 38 | **0.7989** | 0.7810 | **0.7760** | 0.0954 | 0.6347 |
| `stage1_balanced_softmax_weighted` | 42 | 30 | 0.7915 | 0.8177 | 0.7759 | 0.0954 | 0.6886 |
| `stage1_cross_entropy_none` | 30 | 18 | 0.7886 | 0.8034 | 0.7641 | 0.0924 | 0.6970 |
| `stage1_cross_entropy_weighted` | 32 | 20 | 0.7955 | **0.8279** | 0.7673 | 0.0931 | **0.7305** |
| `stage1_focal_none` | 24 | 12 | 0.7588 | 0.7202 | 0.7539 | **0.0203** | 0.4706 |
| `stage1_focal_weighted` | 39 | 27 | 0.7695 | 0.7546 | 0.7609 | 0.0512 | 0.5294 |

### 4.1. Интерпретация stage1

`cross_entropy_weighted` оказался лучшим для редких классов и balanced accuracy. Это хороший baseline для медицинской постановки, где важен recall меньшинства.

`balanced_softmax_none` оказался лучшим по `macro F1` и почти лучшим по MCC. Это значит, что корректировка логитов под class prior может давать хорошую общую геометрию без weighted sampler.

`focal loss` дал заметно лучшую calibration (`ECE=0.0203` без weighted sampler), но потерял recall редких/сложных классов. Его нельзя считать лучшим основным baseline, но он полезен как calibration reference.

### 4.2. Feature-space диагностика stage1

Для `cross_entropy_weighted`:

- cosine silhouette test: `0.4514`
- mean centroid margin: `0.3711`
- плохая margin-доля: `mel=29.3%`, `akiec=26.5%`, `bkl=24.8%`
- ближайшие центроиды: `mel-nv`, `bkl-nv`, `bkl-mel`, `bcc-mel`, `bcc-nv`, `bcc-bkl`, `akiec-bkl`
- top kNN confusions: `nv->mel`, `mel->nv`, `nv->bkl`, `bkl->mel`, `mel->bkl`, `bkl->nv`, `bkl->akiec`, `akiec->bkl`

Для `balanced_softmax_none`:

- cosine silhouette test: `0.5056`
- mean centroid margin: `0.6037`
- плохая margin-доля: `df=41.2%`, `mel=37.1%`, `akiec=28.6%`, `bkl=23.6%`
- top kNN confusions: `mel->nv`, `bkl->nv`, `mel->bkl`, `nv->mel`, `bkl->mel`, `nv->bkl`

Вывод: целевые границы для stage2 были выбраны не произвольно. Главные проблемные пары: `mel/nv`, `mel/bkl`, `akiec/bkl`, `akiec/bcc`.

## 5. Этап 2: генерация и отбор синтетики

Цель этапа 2: проверить, может ли targeted image-conditioned synthetic augmentation улучшить качество сверх сильных real-only baseline.

### 5.1. Генерация

Генератор:

| Параметр | Значение |
|---|---|
| Model | `stable-diffusion-v1-5/stable-diffusion-v1-5` |
| Mode | img2img |
| Target classes | `mel`, `akiec`, `bkl` |
| Real source per class | 160 |
| Synthetic per real | 2 |
| Total synthetic | 960 |
| Image size | 512 |
| Strength | 0.35 |
| Guidance scale | 6.0 |
| Inference steps | 30 |
| Seed | 20260708 |

Промпты были class-conditioned и dermoscopy-oriented. Для negative prompt использовались общие запреты (`cartoon`, `text`, `watermark`, `hair`, `non dermoscopic`, etc.) и confusing class descriptions. В логах был warning о превышении CLIP token limit для некоторых negative prompts. Это не ломало запуск, но часть negative prompt могла быть обрезана. В следующем этапе negative prompts нужно укоротить и явно логировать token length.

Артефакты:

- raw synthetic images: `/srv/research/projects/default/ham10000/synthetic/ham10000_mel_boundary_img2img_v1/`
- synthetic manifest: `/srv/research/projects/default/ham10000/synthetic/ham10000_mel_boundary_img2img_v1/synthetic_manifest.csv`
- визуальные галереи: `/srv/research/projects/default/ham10000/reports/synthetic_galleries/ham10000_mel_boundary_img2img_v1/`

В JupyterHub это открывается как:

`project-default/ham10000/reports/synthetic_galleries/ham10000_mel_boundary_img2img_v1/index.html`

### 5.2. Отбор синтетики

Мы проверили два режима.

Strict feature-space selection:

- критерий: nearest same-class distance не хуже real-real threshold (`same_quantile=0.95`) и positive margin к nearest confusing class (`min_margin=0.05`)
- результат: `43` synthetic images
- распределение: `bkl=28`, `mel=15`, `akiec=0`

Top-k fallback selection:

- режим: `rule_or_topk80`
- логика: если strict rule слишком консервативен, брать top-k по feature margin / same-class distance с diversity filter
- результат: `240` synthetic images
- распределение: `akiec=80`, `bkl=80`, `mel=80`

Сводка synthetic pool:

| Pool | Total | akiec | bkl | mel |
|---|---:|---:|---:|---:|
| Raw generated | 960 | 320 | 320 | 320 |
| Strict selected | 43 | 0 | 28 | 15 |
| Top-k selected | 240 | 80 | 80 | 80 |

Важный сигнал: strict selection не пропустил ни одного `akiec`. Это означает, что сгенерированная `akiec`-синтетика при текущем генераторе/промптах плохо попадает в область реального `akiec` в признаковом пространстве baseline-модели либо недостаточно отделена от confusing classes. Это не просто техническая деталь, а научно важный negative result.

## 6. Stage2 training results

Stage2 matrix:

| Experiment | Synthetic pool | Loss / sampler | Epochs | Best epoch | Test macro F1 | Balanced acc | MCC | ECE | Worst recall |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `stage2_synthetic_strict_ce_weighted` | strict 43 | CE + weighted sampler | 37 | 25 | 0.7851 | 0.7841 | **0.7797** | 0.0859 | 0.5882 |
| `stage2_synthetic_strict_balanced_softmax_none` | strict 43 | Balanced Softmax | 47 | 35 | 0.7526 | 0.7640 | 0.7482 | 0.1045 | 0.5808 |
| `stage2_synthetic_topk80_ce_weighted` | topk80 240 | CE + weighted sampler | 33 | 21 | **0.7874** | **0.8039** | 0.7637 | 0.0922 | **0.6826** |
| `stage2_synthetic_topk80_balanced_softmax_none` | topk80 240 | Balanced Softmax | 28 | 16 | 0.7649 | 0.7813 | 0.7693 | **0.0803** | 0.6407 |

### 6.1. Сравнение с stage1

Best stage1 по `macro F1`:

- `stage1_balanced_softmax_none`: `macro F1=0.7989`, `MCC=0.7760`

Best stage1 по `balanced accuracy` и `worst recall`:

- `stage1_cross_entropy_weighted`: `macro F1=0.7955`, `balanced accuracy=0.8279`, `worst recall=0.7305`

Best stage2:

- по `macro F1`: `topk80_ce_weighted`, `0.7874`
- по `balanced accuracy`: `topk80_ce_weighted`, `0.8039`
- по `MCC`: `strict_ce_weighted`, `0.7797`
- по `ECE`: `topk80_balanced_softmax_none`, `0.0803`

Итог: stage2 не превзошел stage1 по основным test-метрикам. Единственное заметное улучшение относительно stage1-best — `strict_ce_weighted` дал лучший MCC (`0.7797` против `0.7760`), но при этом сильно проиграл по balanced accuracy и worst recall.

### 6.2. Per-class recall

| Experiment | akiec | bcc | bkl | df | mel | nv | vasc |
|---|---:|---:|---:|---:|---:|---:|---:|
| `stage1_balanced_softmax_none` | 0.7143 | - | 0.7636 | - | 0.6347 | - | - |
| `stage1_cross_entropy_weighted` | 0.7347 | - | 0.7636 | - | **0.7305** | - | - |
| `stage2_strict_ce_weighted` | **0.8163** | 0.8571 | 0.6970 | 0.5882 | 0.7186 | 0.9543 | 0.8571 |
| `stage2_topk80_ce_weighted` | 0.7143 | **0.8961** | 0.7273 | **0.7647** | 0.6826 | 0.9374 | 0.9048 |
| `stage2_topk80_balanced_softmax_none` | 0.7347 | **0.8961** | 0.6848 | 0.6471 | 0.6407 | 0.9612 | 0.9048 |

Stage2 не дал общий прирост, но локально есть интересные эффекты:

- `strict_ce_weighted` поднял `akiec recall` до `0.8163`, но снизил `bkl`, `df`, `mel`.
- `topk80_ce_weighted` улучшил `df` и `bcc`, но не улучшил `mel`, а `mel` остается критическим классом.
- `topk80_balanced_softmax_none` дал лучшую calibration среди stage2, но слабые `macro F1` и `worst recall`.

### 6.3. Почему stage2 мог не улучшить test-score

1. **Synthetic-real gap**. Синтетика могла быть визуально правдоподобной, но находиться в другом подраспределении признаков. Strict filter частично это поймал: для `akiec` прошли `0/320`.

2. **Неправильный источник генерации**. Мы брали до 160 real images per class, но не обязательно hard/boundary examples. Возможно, генерировать нужно не из случайных `mel/akiec/bkl`, а из объектов около decision boundary.

3. **Слишком слабое медицинское условие генерации**. SD v1.5 не является dermoscopy-specific генератором. Даже img2img с `strength=0.35` может менять клинически значимые признаки не так, как нужно downstream-классификатору.

4. **Negative prompt truncation**. В текущем запуске warning говорит, что часть условий могла быть обрезана tokenizer limit. Это могло ухудшить inter-class separation.

5. **Синтетика добавлена как обычные изображения**. Модель не знает, какие samples synthetic. Работы про synthetic-aware обучение предполагают, что real/synthetic gap нужно моделировать, а не просто смешивать данные.

6. **Один seed и один generator setting**. Для выводов статьи нужно минимум 3 seeds и сетка по generation strength / CFG / ratio.

7. **Validation-test mismatch**. `topk80_ce_weighted` имел `val macro F1=0.8095`, но `test macro F1=0.7874`. Это похоже на selection bias или нестабильность split-level оценки.

## 7. На какие статьи опирались

Ниже не просто список литературы, а связь между статьей и нашим протоколом.

### 7.1. HAM10000 dataset

Tschandl et al., "The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions", Scientific Data 2018.

Ссылки: <https://arxiv.org/abs/1803.10417>, <https://www.nature.com/articles/sdata2018161>.

Как использовано:

- выбор HAM10000 как стартовой площадки;
- понимание многоисточниковости и неоднородности данных;
- необходимость аккуратного split по lesion/group id;
- real-only test как обязательная проверка.

### 7.2. Augmented Conditioning Is Enough For Effective Training Image Generation

Chen, Zhang, Romero-Soriano, 2025: <https://arxiv.org/abs/2502.04475>.

Ключевая идея: синтетические изображения для downstream training должны быть не только реалистичными, но и разнообразными внутри support целевого распределения. Авторы показывают важность conditioning on augmented real image + text prompt для in-domain generation.

Как использовано:

- мы выбрали img2img, а не class-label-only text-to-image;
- source real image сохраняется в manifest;
- каждая synthetic sample связана с реальным источником;
- галереи строятся как пары `real source -> synthetic`.

Что наши результаты добавляют:

- image conditioning само по себе недостаточно, если генератор не медицинский или если clinical/feature alignment слабый;
- нужно оценивать не только in-domain appearance, но и downstream feature margin.

### 7.3. Synthetic Data Generation for Long-Tail Medical Image Classification: Skin Lesions

Jiang, Subedar, Tickoo, 2026: <https://arxiv.org/abs/2605.03221>.

Ключевая идея: для long-tail medical image classification полезна diffusion-based augmentation, но важна post-selection, чтобы синтетика была разнообразной, реалистичной и клинически осмысленной.

Как использовано:

- мы добавили post-selection, а не обучали на всем raw synthetic pool;
- selection был feature-aware и confusing-class-aware;
- отдельно сохранили diagnostics CSV и selected manifest.

Что наши результаты добавляют:

- строгий отбор может показать, что синтетика для части классов непригодна (`akiec=0/320`);
- если post-selection слишком строгий, synthetic ratio становится слишком маленьким, и downstream effect может быть слабым;
- если post-selection смягчить, растет coverage, но появляется риск synthetic-real gap.

### 7.4. Synthetic Data Augmentation using Pre-trained Diffusion Models for Long-tailed Food Image Classification

Koh et al., 2025: <https://arxiv.org/abs/2506.01368>.

Ключевая идея: при long-tailed classification pretrained diffusion может смешивать похожие классы; нужно явно учитывать inter-class separation. Авторы используют positive prompts и visually similar confusing class как negative prompt.

Как использовано:

- мы задали confusing classes: `mel: [nv, bkl]`, `akiec: [bkl, bcc]`, `bkl: [mel, nv, akiec]`;
- feature-selection проверяет distance to same class и margin to confusing classes;
- stage2 сфокусирован на проблемных границах, а не на полном blind balancing.

Что наши результаты добавляют:

- negative prompts нужно контролировать tokenizer length;
- confusing prompt без специализированной guidance может быть слабым;
- лучше выбирать confusing class не вручную, а из feature/confusion matrix baseline.

### 7.5. When Does Synthetic Data Augmentation Improve Score-Based Imbalanced Classification?

Ma, Lyu, Zhang, 2026: <https://arxiv.org/abs/2606.26053>.

Ключевая идея: synthetic augmentation может менять effective class weighting и одновременно вносить synthetic distribution error. Улучшения могут быть немонотонными; при хорошо специфицированной модели population-level прирост не гарантирован.

Как использовано:

- мы сравнивали синтетику против сильных real-only baseline, а не против слабого baseline;
- оценивали `macro F1`, `balanced accuracy`, `worst recall`, `ECE`, `MCC`, а не только accuracy;
- проверили два synthetic selection режима, а не один.

Что наши результаты подтверждают:

- synthetic augmentation действительно дала немонотонные эффекты;
- локальное улучшение `akiec recall` не превратилось в общий прирост;
- synthetic distribution error вероятно перекрыл потенциальный gain.

### 7.6. Balanced Contrastive Learning for Long-Tailed Visual Recognition

Zhu et al., CVPR 2022: <https://arxiv.org/abs/2207.09052>.

Ключевая идея: long-tailed recognition — это не только bias classifier head, но и проблема геометрии признакового пространства. Обычный supervised contrastive learning на long-tail может формировать неправильную геометрию; BCL балансирует вклад классов.

Как использовано:

- мы анализировали centroid distances, silhouette, kNN confusions;
- feature-space diagnostics стали частью отбора синтетики;
- следующий этап должен перейти от data-only augmentation к representation-level objectives.

### 7.7. SAU: Dual-Branch Network for Long-Tailed Recognition via Generative Models

Li, Song, Zheng, 2024: <https://arxiv.org/abs/2408.16273>.

Ключевая идея: synthetic data может помочь long-tailed recognition, но существует real/synthetic discrepancy. SAU явно моделирует synthetic-aware и synthetic-unaware branches.

Как использовано:

- текущий stage2 не использует synthetic-aware branch, что теперь выглядит важным ограничением;
- результаты stage2 показывают, что простое смешивание real+synthetic недостаточно.

Следующий шаг:

- добавить domain/synthetic indicator;
- проверить synthetic-aware auxiliary loss или two-stage training: pretrain with synthetic, final real-only fine-tune.

### 7.8. When Generative Augmentation Hurts

Gupta, Brown, 2026: <https://arxiv.org/abs/2603.16134>.

Ключевая идея: генеративная аугментация может вредить, особенно при low-data и fine-grained classes. Feature embedding analysis может показать synthetic clusters outside real distribution.

Как использовано:

- мы сделали галереи и feature diagnostics;
- negative result stage2 не нужно скрывать: это ожидаемый и публикуемый тип результата при неправильной синтетике.

Что нужно добавить:

- real-vs-synthetic classifier;
- UMAP/t-SNE с raw synthetic / selected synthetic / real train / real test;
- nearest-neighbor visual audit.

### 7.9. Diffusion-Based Data Augmentation for Image Recognition: Systematic Analysis

Li et al., 2026: <https://arxiv.org/abs/2603.08364>.

Ключевая идея: Diffusion augmentation нужно раскладывать на этапы: generator tuning/conditioning, generation, utilization in classifier training.

Как использовано:

- код разделен на generation, selection, training matrix;
- artifacts каждого этапа сохраняются отдельно;
- галереи и manifests позволяют аудировать generation и selection.

### 7.10. The Learnability Gap in Medical Latent Diffusion

Dombrowski, Nützel, Kainz, 2026: <https://arxiv.org/abs/2605.17087>.

Ключевая идея: хорошая реконструкция/генерация медицинских изображений не обязательно означает полезное latent/feature representation для downstream classification.

Как связано с нашими результатами:

- stage2 показал именно такой риск: визуальная синтетика может не улучшать downstream classifier;
- нужно измерять learnability/downstream value, а не только image quality.

## 8. Что можно сделать дальше

### 8.1. Немедленный анализ уже полученных данных

1. Построить UMAP/t-SNE:
   - real train;
   - raw synthetic;
   - strict selected;
   - topk80 selected;
   - real test.

2. Обучить real-vs-synthetic detector:
   - общий detector;
   - per-class detector;
   - report AUROC/AUPRC.

3. Сделать nearest-neighbor audit:
   - для каждой synthetic sample найти ближайшие real same-class и confusing-class изображения;
   - сохранить HTML gallery `synthetic -> nearest same / nearest confusing`.

4. Посчитать distribution diagnostics:
   - same-class nearest distance distribution;
   - confusing-class margin distribution;
   - class-wise density ratio;
   - duplicate/near-duplicate detection.

5. Проанализировать confusion deltas:
   - stage1 CE weighted vs stage2 topk80 CE weighted;
   - какие ошибки уменьшились, какие выросли.

### 8.2. Следующие generation experiments

1. Grid по img2img strength:
   - `0.15`, `0.25`, `0.35`, `0.45`
   - гипотеза: `0.35` мог слишком сильно менять lesion features.

2. Grid по guidance:
   - `2.0`, `4.0`, `6.0`
   - высокая guidance может снижать diversity и усиливать prompt artifacts.

3. Укоротить negative prompts:
   - убрать длинные class descriptions;
   - сделать compact tokens: `nevus-like, melanoma-like, keratosis-like`;
   - логировать tokenized length.

4. Boundary-conditioned source selection:
   - генерировать не из случайных 160 images/class;
   - выбирать hard examples с плохой margin из stage1 feature analysis.

5. Compare generators:
   - SD v1.5 img2img;
   - SDXL img2img;
   - dermoscopy/medical-specific diffusion, если доступен;
   - LoRA на HAM10000 train-only, если хватит данных и есть четкий протокол.

6. Inpainting / lesion-centered generation:
   - если сделать masks, можно менять lesion texture при сохранении контекста;
   - это ближе к skin-lesion medical generation papers.

### 8.3. Следующие training experiments

1. Synthetic weight < 1:
   - synthetic samples не должны обязательно иметь тот же вес, что real;
   - проверить `synthetic_weight=0.25/0.5`.

2. Two-stage training:
   - train real+synthetic;
   - final fine-tune only real train.

3. Synthetic-aware branch/loss:
   - добавить `is_synthetic` indicator;
   - auxiliary domain classifier;
   - gradient reversal или SAU-like branch.

4. Representation learning:
   - supervised contrastive / balanced contrastive objective;
   - class-balanced prototypes;
   - margin-aware losses.

5. Calibration:
   - temperature scaling;
   - class-wise threshold tuning на val;
   - отдельно report sensitivity/specificity для `mel`.

6. More baselines:
   - LDAM/DRW;
   - logit adjustment;
   - class-balanced loss;
   - deferred reweighting;
   - mixup/cutmix with class-balanced sampling.

### 8.4. Дизайн для статьи

Минимум для публикационного протокола:

1. 3 random seeds для каждого ключевого run.
2. Один fixed lesion/group split + дополнительный repeated group split, если позволит время.
3. Real-only test всегда неизменный.
4. Отдельная внешняя проверка: ISIC2019/ISIC2020 или другой skin-lesion benchmark.
5. Статистическая проверка paired differences по test predictions.
6. Полный artifact registry:
   - config;
   - command;
   - git commit;
   - data manifest checksum;
   - synthetic manifest checksum;
   - selected manifest checksum;
   - environment.

## 9. Вопросы к коду и инженерной архитектуре

1. Нужно ли превратить серверную папку `/srv/research/projects/default/research-ml` в полноценный git clone, чтобы каждый run сохранял настоящий `git_commit`?

2. Нужно ли добавить общий `run_registry.csv/jsonl`, куда каждый запуск пишет:
   - command line;
   - config path;
   - overrides;
   - Docker image id;
   - git hash;
   - output dir;
   - status;
   - start/end time?

3. Нужно ли сделать единый CLI `research-ml`:
   - `research-ml prepare`;
   - `research-ml train`;
   - `research-ml generate`;
   - `research-ml select`;
   - `research-ml report`;
   - `research-ml analyze-features`?

4. Нужно ли добавить validation layer для CSV:
   - обязательные поля;
   - существование файлов;
   - отсутствие group leakage;
   - class distribution;
   - checksum?

5. Нужно ли вынести synthetic selection в конфиг YAML, чтобы selection-mode/top-k/thresholds сохранялись не только в командной строке?

6. Нужно ли добавить HTML reports автоматически после каждого этапа:
   - training dashboard;
   - confusion matrix;
   - per-class metrics;
   - synthetic gallery;
   - nearest-neighbor gallery;
   - feature-space plots?

7. Нужно ли хранить generated images immutable:
   - запрет перезаписи generation folder;
   - `generation_id`;
   - manifest checksum;
   - source config checksum?

8. Нужно ли добавить lightweight local analysis notebooks, которые запускаются без GPU на ноутбуке/ПК пользователя?

## 10. Научно-архитектурные вопросы

1. Как формализовать полезность синтетики: через визуальное качество, feature alignment, diversity, real/synthetic indistinguishability или downstream delta?

2. Должна ли синтетика приближать real class distribution или специально заполнять boundary/hard regions?

3. Что важнее для `mel`: увеличить recall любой ценой или удерживать precision/калибровку?

4. Можно ли получить выигрыш от синтетики без synthetic-aware training, или real/synthetic gap слишком силен?

5. Нужно ли перейти от image generation к counterfactual generation:
   - менять только clinically relevant features;
   - сохранять identity/context;
   - строить пары для contrastive learning?

6. Нужно ли делать generation не для всех tail classes, а только для конкретных confusion pairs?

7. Если strict filter отбрасывает почти всю синтетику, это значит:
   - генератор плох;
   - baseline embedding слишком biased;
   - критерий слишком строг;
   - или class labels/visual concepts слишком неоднозначны?

8. Как избежать того, чтобы synthetic augmentation просто меняла class prior и дублировала эффект weighted sampler?

9. Нужно ли переносить исследование с HAM10000 на более общий vision benchmark:
   - ImageNet-LT/CIFAR-LT для методологической чистоты;
   - industrial defects для rare-class setting;
   - remote sensing для domain shift;
   - skin lesions оставить как прикладной medical case study?

10. Какая кандидатская формулировка сильнее:
    - "управляемая генерация изображений для дисбаланса классов";
    - "оценка полезности синтетических изображений через структуру признакового пространства";
    - "synthetic-aware representation learning для long-tailed image classification";
    - "контрфактическая генерация для устойчивых признаков при редких классах"?

## 11. Рекомендуемый следующий этап

Я бы не запускал сразу новую большую генерацию. Сначала стоит сделать stage2-analysis:

1. Real-vs-synthetic detector.
2. UMAP/nearest-neighbor galleries.
3. Confusion-delta report.
4. Source-image hardness analysis: какие real images дали полезную/вредную синтетику.
5. Token-length audit prompt/negative prompt.

После этого запускать stage3:

1. Boundary-conditioned generation.
2. Short negative prompts.
3. Strength grid `0.15/0.25/0.35`.
4. Synthetic weight `0.25/0.5/1.0`.
5. Final real-only fine-tune.
6. 3 seeds только для лучших 2-3 configurations.

Рабочая гипотеза для stage3:

> Синтетика становится полезной не тогда, когда она просто визуально похожа на дерматоскопическое изображение и увеличивает число samples, а когда она локально улучшает покрытие real feature manifold в областях class-boundary confusion и при этом не образует отдельный synthetic domain.

