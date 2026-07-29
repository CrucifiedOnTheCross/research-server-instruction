# Stage 14: retrospective audit and conditional next-stage plan

Дата фиксации: 2026-07-29, до получения результатов Stage 13B.

Статус: `branch_c_preprocessing_qualification`. Locked test закрыт.

## Цель

Проверить, не объясняются ли результаты Stage 1–13 технической ошибкой,
слабым baseline, неадекватной предобработкой или несостоятельным способом
оценки синтетики. После этого выбрать только одну следующую ветку по заранее
заданным критериям и не подгонять Stage 14 под увиденные validation-метрики.

Канонические источники аудита:

- `summary.json`, `config.resolved.yaml`, `sampling_plan.json`;
- `model_initialization.json`, `class_counts.json`;
- `val_metrics_best.json`, `val_predictions_best.csv`;
- calibration и lesion-group bootstrap CSV;
- generation/selection manifests.

Логи не использовались для извлечения результатов.

## Ретроспектива результатов

Аудит охватывает 95 завершённых runs и 43 experiment names до Stage 13B.
Пары `experiment × seed` уникальны. В Stage 9–13 locked test не открывался.
Пересчёт метрик из predictions в Stage 9–11B расходится с сохранёнными
метриками не более чем на `1.1e-7`.

| Этап | Главный результат | Интерпретация |
|---|---|---|
| Stage 8 | CE oversampling: macro F1 0.7681; synthetic+DINO: 0.7384 | Простое real oversampling сильнее ранней синтетики |
| Stage 9 | balanced softmax: 0.7513; undersampling: 0.6086 | Удаление real majority data явно вредно; replay обязателен как контроль |
| Stage 10 | `strict_id` synthetic − replay: macro F1 +0.0261, MCC +0.0662, macro AUPRC +0.0174; melanoma recall −0.0769 | Возможен boundary gain, но minority trade-off и широкие CI |
| Stage 11 | ConvNeXt-S: macro F1 0.7699, MCC 0.6610, macro AUPRC 0.8068 | Квалифицирован сильный и экономичный downstream baseline |
| Stage 11B | synthetic − replay: macro F1 +0.0115; macro AUPRC −0.0136; melanoma AUPRC −0.0447; ECE +0.0235 | Confirmatory utility criteria не выполнены |
| Stage 12 | PRDC precision/density/coverage хуже replay в 11/12; frequency gap 14/15 | Проблема не равна простому mode collapse или плохой картинке |
| Stage 13A3 | coverage +0.1547, precision +0.3083, density +0.4317; 90 уникальных sources | Геометрический gate пройден; downstream utility проверяет Stage 13B |

Сравнивать только macro F1 недостаточно. На Stage 11B argmax-метрики немного
выросли, но ranking melanoma ухудшился на всех трёх seeds. Для медицинской
задачи это содержательный отрицательный сигнал, а не косметическая разница.

## Аудит корректности

### Что подтверждено

1. `group_id` строится из `lesion_id`, затем patient ID, и только при их
   отсутствии из image ID. Split validator запрещает пересечение групп.
2. Synthetic sources и source-matched replay проверяются против validation и
   locked-test group IDs до запуска.
3. Взвешенный objective реализован как
   `sum(loss_i * weight_i) / sum(weight_i)`.
4. EMA-модель используется и для model selection, и для сохранённого best
   checkpoint; invalid pilot без EMA warmup исключён из анализа.
5. Softmax, macro F1, MCC, balanced accuracy, ECE, per-class AUROC/AUPRC
   пересчитываются из сохранённых вероятностей.
6. Paired comparisons используют одинаковые seeds; uncertainty считается
   hierarchical bootstrap по lesion groups.
7. Современные Stage 9+ runs сохраняют resolved config, sampling,
   initialization, class counts, predictions и checkpoint. Локальный
   облегчённый audit archive намеренно не содержит тяжёлые `best.pt`, поэтому
   checkpoint integrity проверяется на сервере.

### Исправленные дефекты

1. Отсутствовал `scripts/run_stage13b_analysis.sh`, хотя automation должна
   была его вызвать после шести runs. Добавлен идемпотентный calibration,
   paired analysis, 5000-repeat lesion bootstrap и MLflow sync.
2. Stage 11B analyzer был жёстко привязан к `strict_id` и старым experiment
   names. Он параметризован, сохраняя обратную совместимость.
3. Взвешенный train loss отображался с denominator `batch size`, хотя
   оптимизация использовала `sum(weights)`. Градиенты и validation не были
   затронуты; отчётная агрегация исправлена.
4. Добавлен `audit_experiment_history.py`, который fail-closed проверяет
   duplicate experiment/seed, locked-test flag и structured artifacts.

### Слабые места, не являющиеся доказанными bugs

| Риск | Почему важен | Решение |
|---|---|---|
| Square crop исходных 600×450 кадров | Center/random crop может удалить периферический контекст или часть крупной lesion | Real-only ablation: current crop против aspect-preserving pad и mask-guided crop |
| ImageNet normalization | Не гарантированно оптимальна для dermoscopy и разных acquisition sites | Проверить color constancy отдельно, не смешивая с hair removal |
| Волосы, линейки, тёмные углы | Могут стать shortcut или менять segmentation/generation | Subgroup audit; preprocessing только после paired ablation |
| Нет ROI conditioning | Общий SD1.5 меняет текстуру и class evidence вместе | Mask-conditioned inpainting/lesion-focused generator pilot |
| Один внутренний HAM validation split | Недостаточно для заявления о клинической generalization | External real evaluation на ISIC2017/другом совместимом наборе |
| Три confirmatory seeds | Достаточно для направления, мало для точной оценки малых эффектов | Lesion bootstrap + при promotion дополнительные seeds/external set |
| Validation используется для early stopping | Calibration после selection остаётся exploratory | Финальная calibration только на отдельной calibration partition |

Hair removal нельзя объявлять обязательным заранее. Авторы HAM10000 вручную
проверяли качество, исправляли цвет/яркость при необходимости и допускали
терминальные волосы; следовательно, автоматическое удаление может как убрать
shortcut, так и создать inpainting artifacts. Нужна отдельная абляция.

## Научная база

### Dataset и предобработка

**Tschandl, Rosendahl, Kittler, Scientific Data 2018.**
HAM10000 содержит 10 015 multi-source dermoscopic images, полученных разными
устройствами и методами. Оригинальная curation подтверждает acquisition
heterogeneity и необходимость lesion-level split и subgroup analysis.

https://www.nature.com/articles/sdata2018161

**Chu, Oh, Yang, ICCV Workshops 2025.**
На всех 10 015 HAM10000 images сравниваются MedSAM, SAM-Med2D и BiomedParse.
Обнаружены различия по artifact, цвету, anatomy и diagnosis. Это обосновывает
использование lesion masks и artifact subgroups, но не автоматическое
применение одного segmenter без проверки его subgroup bias.

https://openaccess.thecvf.com/content/ICCV2025W/BISCUIT/html/Chu_Evaluating_the_Trustworthiness_of_Foundation_Models_for_Skin_Lesion_Segmentation_ICCVW_2025_paper.html

**Perez et al., low-cost augmentation search, 2023.**
Работа на HAM10000 использует 5-fold evaluation и показывает, что обычные
augmentation policies остаются сильным baseline. Для нашей статьи новый
generator обязан превосходить не только no-augmentation, но и
replay/oversampling/qualified augmentation.

https://pmc.ncbi.nlm.nih.gov/articles/PMC10521644/

### Генерация

**Bissoto, Valle, Avila, CVPRW 2021.**
Критический controlled study показывает нестабильность GAN augmentation в
skin-lesion analysis и выделяет synthetic ratio и sampling как ключевые
факторы. Наш equal-dose source-matched replay непосредственно закрывает эту
проблему и должен сохраняться.

https://openaccess.thecvf.com/content/CVPR2021W/ISIC/html/Bissoto_GAN-Based_Data_Augmentation_and_Anonymization_for_Skin-Lesion_Analysis_A_Critical_CVPRW_2021_paper.html

**Farooq et al., Derm-T2IM, 2024.**
Domain-specific Stable Diffusion обучена для dermatoscopic synthesis и
проверена на двух skin-lesion datasets. Публичный checkpoint делает её
реалистичным кандидатом для inference на RTX 5080 16 GB. Ограничение:
text-to-image samples не имеют естественной source pairing, поэтому их нельзя
напрямую сравнивать с source replay без отдельного matching protocol.

https://arxiv.org/abs/2401.05159

**Sun et al., LF-VAR, MICCAI 2025.**
Lesion-focused VQ-VAE + VAR использует diagnosis, lesion masks и измерения,
обучается на HAM10000 при 512 px и проверяет cross-dataset generation.
Метод научно ближе к нашей frequency/ROI проблеме, чем увеличение размера
общего SD, но в статье основной выигрыш относится к FID; reviewers отдельно
отмечали недостаток downstream utility. Поэтому LF-VAR допустим сначала
только как feasibility/generator benchmark.

https://papers.miccai.org/miccai-2025/0183-Paper0807.html

Код: https://github.com/echosun1996/LF-VAR

**Mekala et al., 2024.**
StyleGAN latent factorization создаёт контролируемые semantic variations на
HAM10000. Это альтернативная гипотеза: полезность может исходить из
интерпретируемого изменения factors, а не из максимальной photorealism.
Однако перед downstream запуском требуются проверка lesion-group protocol и
privacy/source-neighbour diagnostics.

https://arxiv.org/abs/2410.05114

## Почему «более сильная модель» сама по себе не решение

Stage 12 показал, что синтетика может выглядеть разнообразной и иметь высокий
Vendi, но плохо покрывать полезную real support и искажать ranking tail.
Более крупный generic text-to-image model способен улучшить визуальную
реалистичность, одновременно усилив domain/frequency gap. Приоритет:

1. lesion-aware conditioning;
2. сохранение source provenance там, где это возможно;
3. frequency, PRDC, privacy и class-margin gate до classifier training;
4. equal-dose replay и real augmentation controls;
5. downstream melanoma AUPRC/fixed-specificity, а не FID как criterion.

## Предварительно зафиксированный Stage 14

Decision gate реализован в `configs/stage14_decision.yaml` и
`tools/check_stage14_readiness.py`.

## Фактическое решение после Stage 13B

Stage 13B завершился `null_or_negative`:

- macro F1 synthetic − replay `−0.0033`, 1/3 wins;
- MCC `−0.0022`, 2/3 wins;
- macro AUPRC `−0.0136`, 0/3 wins;
- melanoma AUPRC `−0.0198`, 0/3 wins;
- macro F1 lesion-bootstrap 95% CI `[−0.0323; +0.0236]`;
- ECE в среднем хуже на `+0.0178`.

Predeclared decision gate выбрал `C_preprocessing_qualification`. Более
крупный generator и дальнейший downstream synthetic training заблокированы
до квалификации входного real-only representation.

### Stage 14P-1: full-frame aspect-preserving policy

Первый запуск меняет один фактор:

- historical control: три завершённых Stage 11 ConvNeXt-S seeds с
  `RandomResizedCrop` train и resize/center-crop eval;
- candidate: те же data split, model, optimizer, schedule, augmentation
  после geometry step и seeds 42–44;
- train/eval сохраняют полный кадр с исходным aspect ratio;
- изображение вписывается в 384×384 и дополняется цветом ImageNet mean,
  который после normalization равен примерно нулю;
- random/center crop отключены;
- real-only, `evaluation.run_test=false`.

HAM10000 lesion masks на сервере отсутствуют. Поэтому lesion-guided crop не
смешивается с текущей проверкой: сначала потребуется frozen segmentation,
mask-quality и subgroup audit. Hair removal и color constancy также остаются
отдельными последующими факторами.

Gate проверяет:

- фактическое решение Branch C;
- полное совпадение model/training/imbalance recipe с квалифицированным
  ConvNeXt-S;
- три целых historical baseline runs;
- real-only train и lesion-group disjointness;
- отсутствие locked-test evaluation.

Promotion требует положительного paired mean по macro AUPRC и отсутствия
ухудшения melanoma AUPRC/recall. Macro F1 без ranking improvement не
достаточен.

### Branch A: strong positive Stage 13B

Условие:

- macro F1, MCC и macro AUPRC положительны;
- macro F1 и macro AUPRC выигрывают 3/3 seeds;
- lower lesion-bootstrap CI выше нуля для macro F1 или MCC;
- melanoma recall delta не ниже −0.05;
- melanoma AUPRC delta не ниже −0.02;
- ECE worsening не более +0.02.

Действие: не менять generator. Сначала external real validation и
preprocessing subgroup robustness. Это наиболее сильный путь к статье.

### Branch B: mixed positive Stage 13B

Условие: устойчивый argmax или ranking signal есть, но один из
inferential/minority/calibration guardrails не выполнен.

Stage 14A generator pilot, только train data:

- текущий SD1.5 img2img как frozen control;
- Derm-T2IM как domain-specific text-to-image;
- mask-conditioned lesion inpainting;
- LF-VAR только если checkpoint, masks и inference воспроизводимы.

Для каждого кандидата одинаковая class dose. До обучения считаются:

- PRDC/Vendi в DINOv2 и трёх real-only ConvNeXt-S spaces;
- source/privacy nearest-neighbour, где определён source;
- frequency and gradient gaps;
- lesion-mask area, boundary and radiomic plausibility;
- artifact subgroups и visual audit в FiftyOne.

Classifier training запрещён, пока кандидат не превосходит current Stage 13
subset по coverage без ухудшения precision/density/frequency/privacy gates.

### Branch C: null or negative Stage 13B

Сначала Stage 14P real-only preprocessing qualification:

- current square crop;
- aspect-preserving resize + pad;
- lesion-mask-guided crop;
- отдельно hair removal;
- отдельно color constancy.

Используются ConvNeXt-S, seeds 42–44, те же split/recipe и locked test.
Комбинированный preprocessing pipeline разрешён только если отдельный фактор
улучшает macro AUPRC и melanoma guardrails без subgroup regression. Новая
синтетика не обучается до квалификации входного representation.

## Критерии публикационной готовности

Для статьи в «Компьютерной оптике» недостаточно одного положительного
validation delta. Минимальный пакет:

1. сильный real-only baseline и простые imbalance controls;
2. equal-dose/source-matched replay;
3. 3+ paired seeds и lesion-group uncertainty;
4. macro F1/MCC/balanced accuracy вместе с AUROC/AUPRC/ECE;
5. melanoma endpoints и fixed-specificity sensitivity;
6. generator geometry/frequency/privacy diagnostics;
7. external real validation либо честно обозначенная внутренняя validation
   study;
8. locked test открывается один раз после заморозки метода.

## Открытые вопросы

1. Доступны ли совместимые lesion masks для всех train images или их нужно
   получить одним frozen foundation segmenter?
2. Можно ли использовать ISIC2017 как external set без пересечения source
   lesions с HAM10000?
3. Требуется ли clinically reviewed visual sample для итоговой статьи?
4. Допустимо ли ограничить генерацию тремя целевыми классами, или нужен
   negative-control class, например `nv`?
5. Какой privacy threshold считать достаточным для source-conditioned
   images: абсолютный nearest distance или отношение к real-real neighbours?
