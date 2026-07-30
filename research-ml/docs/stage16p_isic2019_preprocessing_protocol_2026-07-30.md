# Stage 16P: ISIC 2019 preprocessing qualification

## Задача

До любых новых synthetic-vs-replay экспериментов проверить, не определяется
ли downstream utility способом приведения разнородных ISIC 2019 изображений
к входу 384x384.

Основной причинный вопрос:

> Улучшает ли сохранение полной геометрии кадра или удаление только тёмного
> внешнего поля macro AUPRC относительно текущего random-resized/center-crop
> protocol при неизменных данных, модели, loss и optimizer?

Locked test не открывается. Stage 16B Natural CE является контролем.

## Почему этот этап нужен

ISIC 2019 объединяет HAM10000, BCN20000 и MSK с разными разрешениями и
предшествующей обработкой. Победившая работа Gessert et al. отдельно
учитывала эти различия: удаляла внешние чёрные области, сохраняла aspect
ratio, сравнивала cropping strategies и resolutions.

При этом Bissoto et al. показали, что фон содержит смесь нежелательных
dataset biases и потенциально полезного контекста. Его полное удаление не
является автоматически правильным. Mahbod et al. также показали, что
жёсткое masking lesion могло ухудшать классификацию, тогда как dilated crop
иногда помогал.

Поэтому Stage 16P не использует lesion-only mask, hair removal и color
constancy одновременно. Каждый фактор должен проверяться отдельно.

## Preregistered arms

### A. Existing control

`stage16_isic2019_real_ce_natural_384`, seeds 42-44:

- train: RandomResizedCrop 384;
- validation: resize 438 + center crop 384;
- Natural CE.

### B. Full-frame aspect pad

`stage16p_isic2019_real_aspect_pad_natural_384`:

- train/eval: сохранить aspect ratio;
- полный кадр масштабировать до 384;
- дополнить до квадрата ImageNet-mean padding;
- flips, color jitter и RandAugment оставить без изменений.

### C. Dark-FOV crop + aspect pad

`stage16p_isic2019_real_dark_fov_pad_natural_384`:

- определить внешний тёмный фон по mean RGB > 8;
- анализировать thumbnail до 512 px, чтобы не декодировать многомегапиксельный
  mask в каждом DataLoader worker;
- построить bounding box поля зрения;
- добавить margin 2%;
- не применять crop, если удаляется менее 1% площади;
- не применять crop, если foreground меньше 25% кадра;
- после crop сохранить aspect ratio и дополнить до 384.

Arm C вдохновлён Gessert et al., но не является точной репликацией их
ellipse-moment heuristic. Это явно маркируется как simplified deterministic
dark-FOV transform.

## Зафиксированные параметры

Во всех arms одинаковы:

- immutable Stage 16A split policy `isic2019_connected_group_v2`;
- train/validation/locked-test IDs;
- ConvNeXt-S `fb_in22k_ft_in1k_384`;
- ImageNet initialization;
- batch 48, BF16, AdamW, layer decay;
- scheduler, warmup, label smoothing, EMA;
- Natural CE и natural sampling;
- monitor `val/auprc_ovr_macro`;
- `evaluation.run_test=false`.

## Screening и confirmation

Screening:

- существующий arm A seed 42;
- новые arms B/C seed 42;
- никакой model selection по locked test.

В confirmation seeds 43-44 переносится максимум один preprocessing arm.
Он должен выполнить:

1. macro AUPRC delta относительно A не ниже +0.005 на seed 42;
2. MCC не ниже A более чем на 0.005;
3. ECE не хуже A более чем на 0.02;
4. melanoma и SCC AUPRC не имеют material degradation более 0.01;
5. structured artifacts полны и test закрыт.

Если ни один arm не проходит все guardrails, текущая предобработка A
сохраняется. Post-hoc изменение порогов запрещено.

## Анализ

- macro AUPRC, AUROC, F1, MCC, balanced accuracy, ECE;
- worst-class recall;
- melanoma и SCC precision/recall/F1/AUPRC;
- per-class AUROC/AUPRC;
- fixed-specificity diagnostics;
- convergence и compute cost;
- paired lesion-group bootstrap после confirmation;
- source subgroup `ham10000` против `isic2019_non_ham`;
- artifact subgroup для dark-frame candidates.

Ranking effects анализируются отдельно от threshold effects.

## Реализация

- `src/datasets.py`: integration of preprocessing into train/eval;
- `src/image_transforms.py`: dependency-light `CropDarkFieldOfView`;
- `configs/isic2019_stage16p_real_aspect_pad_natural_384.yaml`;
- `configs/isic2019_stage16p_real_dark_fov_pad_natural_384.yaml`;
- `scripts/run_stage16p_preprocessing_screen.sh`;
- `scripts/start_stage16p_container.sh`;
- `scripts/queue_stage16p_after_confirmation.sh`;
- regression tests в `tests/test_stage16_isic2019_protocol.py`.

Stage 16P ставится в очередь после контейнера
`research-stage16b-confirmation`, чтобы не конкурировать за RTX 5080.

## Фактические результаты

Оба seed-42 screening runs завершены с полными structured artifacts,
`weights_source=ema` и `test_evaluated=false`. Baseline и кандидаты
оценивались на одних 3799 изображениях и 3210 lesion groups.

| Arm | Macro AUPRC | Macro F1 | MCC | Bal. acc. | ECE | Mel AUPRC | SCC AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Existing square crop | 0.76363 | 0.70009 | 0.73654 | 0.67727 | 0.04464 | 0.78505 | 0.53507 |
| Aspect pad | 0.76443 | 0.70710 | 0.72431 | 0.68555 | 0.05658 | 0.76917 | 0.56804 |
| Dark-FOV crop + pad | 0.76138 | 0.70192 | 0.72395 | 0.67984 | 0.06205 | 0.76275 | 0.55850 |

Относительно existing square crop:

| Arm | Macro AUPRC | MCC | ECE | Mel AUPRC | SCC AUPRC | Gate |
|---|---:|---:|---:|---:|---:|---|
| Aspect pad | +0.00080 | -0.01223 | +0.01194 | -0.01588 | +0.03298 | fail |
| Dark-FOV crop + pad | -0.00225 | -0.01259 | +0.01741 | -0.02230 | +0.02344 | fail |

Оба arms нарушают минимум три preregistered условия: недостаточный macro
AUPRC effect, MCC degradation более 0.005 и melanoma AUPRC degradation
более 0.01. Поэтому seeds 43-44 не запускались.

### Paired lesion-group bootstrap

5000 bootstrap repeats на seed 42:

| Arm и metric | Mean delta | 95% CI |
|---|---:|---:|
| Aspect pad, macro AUPRC | +0.00064 | [-0.01310; +0.01449] |
| Aspect pad, MCC | -0.01223 | [-0.02682; +0.00174] |
| Aspect pad, ECE | +0.01166 | [+0.00235; +0.02079] |
| Aspect pad, melanoma AUPRC | -0.01592 | [-0.03305; +0.00197] |
| Dark-FOV pad, macro AUPRC | -0.00219 | [-0.01620; +0.01189] |
| Dark-FOV pad, MCC | -0.01257 | [-0.02699; +0.00206] |
| Dark-FOV pad, ECE | +0.01725 | [+0.00769; +0.02690] |
| Dark-FOV pad, melanoma AUPRC | -0.02212 | [-0.04174; -0.00188] |

Ни один arm не показывает practically relevant общий ranking gain.
Ухудшение ECE устойчиво для обоих, а для dark-FOV arm также подтверждено
ухудшение melanoma AUPRC.

### Source и dark-frame audit

Dark-FOV transform фактически изменил crop только для 153 из 3799
validation images (4.0%). На этой группе:

- baseline macro AUPRC: 0.62231;
- aspect pad: 0.65270;
- dark-FOV crop + pad: 0.61775.

Следовательно, локальный выигрыш связан с сохранением полного кадра через
aspect pad, а не с применённой dark-border эвристикой. Упрощённый crop не
решает задачу даже в своей целевой artifact subgroup.

Обнаружена source interaction:

| Source | Existing crop | Aspect pad | Dark-FOV pad |
|---|---:|---:|---:|
| HAM10000, macro AUPRC | 0.82583 | 0.81482 | 0.81169 |
| ISIC 2019 non-HAM, macro AUPRC | 0.68551 | 0.71109 | 0.71255 |

Оба full-frame варианта улучшают более сложную non-HAM часть, но ухудшают
HAM10000 настолько, что общий preregistered endpoint не растёт. Это
научно интересный признак domain/source-dependent preprocessing, но он
получен в secondary subgroup analysis одного seed и не является основанием
для post-hoc выбора arm.

## Решение

1. Сохранить existing square-crop Natural CE как основной Stage 16 baseline.
2. Не запускать Stage 16P confirmation на seeds 43-44.
3. Не добавлять simplified dark-FOV crop в основной pipeline.
4. Зафиксировать source interaction как гипотезу для будущего external
   robustness исследования, а не оптимизировать preprocessing по текущей
   validation выборке.
5. Следующим основным этапом выбрать Stage 16C: equal-dose synthetic
   augmentation против source-matched real replay на ISIC 2019 с Natural
   CE baseline. Color constancy оставить отдельным real-only фактором,
   который нельзя смешивать со Stage 16C contrast.

Канонические результаты:
`outputs/reports/stage16p_analysis/analysis_summary.json`,
`screening_metrics.csv`, `paired_lesion_bootstrap.csv`,
`subgroup_metrics.csv`, `dark_fov_candidates.csv` и
`artifact_integrity.csv`.

## Зафиксированные дальнейшие факторы

Не входят в текущий screening:

1. Shades-of-Gray color constancy;
2. hair removal;
3. lesion-mask-guided dilated crop;
4. test-time multi-crop;
5. metadata fusion.

Color constancy является следующим кандидатом только после geometry
qualification. Barata et al. показали пользу на multisource dermoscopy, а
Gessert et al. применяли Shades of Gray `p=6` на ISIC 2019. Но одновременное
включение color normalization и crop сделало бы эффект неидентифицируемым.

## Литература

1. Gessert N, Nielsen M, Shaikh M, Werner R, Schlaefer A. Skin Lesion
   Classification Using Ensembles of Multi-Resolution EfficientNets with
   Meta Data. MethodsX, 2020. DOI: 10.1016/j.mex.2020.100864.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC7150512/
   Official code: https://github.com/ngessert/isic2019
2. Bissoto A, Valle E, Avila S. Debiasing Skin Lesion Datasets and Models?
   Not So Fast. CVPRW ISIC, 2020.
   https://openaccess.thecvf.com/content_CVPRW_2020/papers/w42/Bissoto_Debiasing_Skin_Lesion_Datasets_and_Models_Not_So_Fast_CVPRW_2020_paper.pdf
3. Mahbod A, Schaefer G, et al. The Effects of Skin Lesion Segmentation on
   the Performance of Dermatoscopic Image Classification, 2020.
   https://arxiv.org/abs/2008.12602
4. Barata C, Celebi ME, Marques JS. Improving dermoscopy image
   classification using color constancy. IEEE JBHI, 2015.
   DOI: 10.1109/JBHI.2014.2336473.
   https://pubmed.ncbi.nlm.nih.gov/25073179/
5. Bissoto A, Fornaciali M, Valle E, Avila S. (De)Constructing Bias on
   Skin Lesion Datasets. CVPRW ISIC, 2019.
   https://openaccess.thecvf.com/content_CVPRW_2019/html/ISIC/Bissoto_DeConstructing_Bias_on_Skin_Lesion_Datasets_CVPRW_2019_paper.html
