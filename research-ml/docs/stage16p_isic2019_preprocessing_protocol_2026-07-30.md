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
