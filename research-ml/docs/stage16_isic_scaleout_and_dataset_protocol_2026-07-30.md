# Stage 16: перенос на ISIC и протокол построения качественного датасета

Дата исследования и preregistration draft: 2026-07-30

Статус: планирование. GPU-запуск запрещён до завершения data audit, split
gate и фиксации экспериментальной матрицы.

## Научная цель

Проверить, переносятся ли выводы HAM10000 Stage 8-15A на более крупную,
мультицентровую и сильнее несбалансированную выборку:

1. визуальное качество синтетики не гарантирует downstream utility;
2. source-matched replay является обязательным контролем;
3. необратимое square-cropping может объяснять значительную часть потери
   utility;
4. generic VAE не обязательно является главным источником ошибки;
5. слабый img2img denoising может менять decision boundary без улучшения
   ranking;
6. feature-space filtering не исправляет плохую геометрию и нарушение
   клинической семантики;
7. lesion-aware mask-conditioned generation должна сравниваться с простыми
   методами балансировки при одинаковой дозе и optimization budget.

Целевой результат Stage 16 — не просто большой набор файлов, а
версионированный датасет с доказанной групповой независимостью,
прослеживаемым происхождением, контролем дубликатов, сохранённой геометрией и
измеренной downstream utility.

## Выбор релизов ISIC

### Основной датасет: ISIC 2019

ISIC 2019 является основной задачей переноса:

- 25 331 training images;
- 8 известных диагностических классов: `MEL`, `NV`, `BCC`, `AK`, `BKL`,
  `DF`, `VASC`, `SCC`;
- 8 238 official test images, включая неизвестный класс в challenge
  protocol;
- training metadata содержат common lesion identifier;
- источники включают HAM10000, BCN20000 и MSK;
- размер official training JPEG archive около 9.1 GB;
- лицензия aggregate dataset: CC-BY-NC.

ISIC 2019 нельзя называть внешней проверкой относительно HAM10000: это
строгий superset ISIC 2018/HAM10000. HAM images должны быть явно помечены
полем `source_dataset=HAM10000`, а все сравнения обязаны исключать
пересечение по ISIC ID, hash и duplicate cluster.

### Независимый patient-level стресс-тест: ISIC 2020

ISIC 2020 используется как отдельная бинарная melanoma-задача:

- 33 126 training images;
- 2 056 пациентов;
- 584 histopathology-confirmed melanoma;
- metadata v2 содержат `patient_id` и `lesion_id`;
- официальный список содержит 425 duplicate images;
- training JPEG archive около 23 GB;
- лицензия CC-BY-NC.

Основное внешнее сравнение: модель ISIC 2019 выдаёт probability `MEL` и без
дообучения оценивается на заранее закрытом patient-disjoint ISIC 2020
holdout. Отдельный ISIC 2020 fine-tuning protocol допустим только как
вторичная проверка переносимости результатов при экстремальном дисбалансе.

### ISIC 2024 SLICE-3D

SLICE-3D не включается в основной dermoscopy experiment:

- 401 059 lesion crops либо permissive subset 217 477;
- изображения являются 15 x 15 mm crops из 3D total-body photography, а не
  дерматоскопией;
- задача бинарная и имеет другой acquisition pipeline.

SLICE-3D пригоден для будущего modality-shift исследования, но его
объединение с ISIC 2019 как обычных дополнительных изображений сделает
интерпретацию синтетической аугментации некорректной.

## Критические ограничения текущего кода

1. `tools/prepare_ham10000.py` предназначен для семиклассовой HAM ontology.
   В нём `squamous cell carcinoma, nos` исторически сводится к `akiec`.
   В ISIC 2019 `SCC` — самостоятельный класс. Старый label mapper нельзя
   переиспользовать.
2. `group_id_from_row` выбирает первый доступный lesion/patient ID. Для
   ISIC 2020 split должен быть строго patient-level, даже если lesion ID
   доступен.
3. Текущий split учитывает majority label группы, но не оптимизирует
   одновременно class, source site и patient distribution.
4. Текущий manifest не хранит hash, duplicate cluster, acquisition source,
   confirmation method, исходный размер, quality flags и dataset version.
5. Простое продолжение HAM checkpoints на ISIC смешает эффект transfer
   learning с эффектом synthetic augmentation.

Для Stage 16 нужен новый loader/curator, не изменение HAM ontology на месте.

## Версионированный manifest

Предлагаемое имя внутреннего набора: `isic-lt-curated-v1`.

Обязательные поля одной строки:

| Группа | Поля |
|---|---|
| Идентичность | `isic_id`, `original_image_name`, `dataset_release` |
| Метка | `label_raw`, `label_canonical`, `benign_malignant`, `diagnosis_confirm_type` |
| Происхождение | `source_dataset`, `contributor_center`, `collection_id`, `license` |
| Группы | `patient_id`, `lesion_id`, `duplicate_cluster_id`, `split_group_id` |
| Файл | `image_path`, `sha256`, `file_size`, `width`, `height`, `aspect_ratio` |
| Near-duplicate | `phash`, `embedding_cluster_id`, `nearest_real_id`, `nearest_distance` |
| Качество | `decode_ok`, `dark_corner_score`, `hair_score`, `ruler_score`, `blur_score`, `mask_quality` |
| Split | `split`, `fold`, `split_seed`, `split_policy_version` |
| Synthetic lineage | `is_synthetic`, `generator_id`, `generator_revision`, `prompt_id`, `source_group_id`, `mask_id`, `seed`, `guidance`, `strength` |

Raw archives неизменяемы. Для каждого архива сохраняются URL, дата
скачивания, SHA256 и license text. Любая очистка создаёт новый manifest, а
не перезаписывает исходные данные.

## Дедупликация и защита от leakage

Порядок объединения изображений в группы:

1. официальный `patient_id`;
2. официальный `lesion_id`;
3. официальный duplicate list;
4. exact SHA256 cluster;
5. near-duplicate cluster по pHash и двум независимым image encoders;
6. отдельный image ID только при отсутствии связей.

`split_group_id` должен быть транзитивным connected component по всем
известным связям. Если два изображения имеют разные lesion ID, но являются
near-duplicates, они остаются в одной split group до ручного разрешения.

Gate должен прекращать pipeline при любом пересечении train/validation/test
по:

- patient ID;
- lesion ID;
- duplicate cluster;
- exact hash;
- synthetic `source_group_id`;
- real nearest neighbour ниже заранее заданного memorization threshold.

## Split protocol

### ISIC 2019 development

- Official 8 238-image test остаётся locked до полного выбора метода.
- 25 331 training images разбиваются на development train и validation.
- Grouping: patient, если он восстановим; иначе common lesion ID; затем
  duplicate cluster.
- Stratification одновременно по canonical class и source dataset.
- Rare classes не разрешается переносить между split через repeated views.
- Фиксируются три split seeds, но основной model-selection split выбирается
  до экспериментов, а не по полученным метрикам.

Дополнительный domain-shift endpoint:

- train на двух source families;
- validation на train sources;
- test на полностью отложенном source family;
- классы и support пересечения описываются явно.

### ISIC 2020

- Все splits строго patient-disjoint.
- 425 официальных duplicates объединяются до split.
- Для самостоятельного binary fine-tuning предпочтительны 5 patient-group
  folds из-за только 584 positive images.
- Для внешней оценки ISIC 2019 classifier весь ISIC 2020 holdout остаётся
  закрытым и не используется для threshold/model selection.

## Перенос моделей

### Основное сравнение

Все arms каждого сравнения начинают с одного и того же внешнего
pretraining:

- основной reproducible baseline:
  `convnext_small.fb_in22k_ft_in1k_384`;
- dermatology-specific baseline: PanDerm-Base linear probe и один
  full-fine-tuning protocol после license/checkpoint gate;
- независимый audit encoder: DINOv2-B/14.

ConvNeXt-S сохраняется, потому что он уже квалифицирован на HAM10000 и
помещается в 16 GB VRAM. PanDerm-Base важен как domain-specific контроль:
PanDerm был pretrained более чем на 2 млн dermatology images и имеет
официальные weights и fine-tuning code. Лицензия PanDerm
CC-BY-NC-ND допускает только некоммерческое академическое использование.

### Что делать с HAM checkpoints

HAM checkpoints не используются как initialization основного causal
comparison. Отдельная transfer ablation:

1. одинаковый ISIC 2019 training subset;
2. ImageNet-22k initialized ConvNeXt-S;
3. HAM Stage 15A replay checkpoint initialized ConvNeXt-S;
4. полный fine-tuning всех слоёв с одинаковым budget;
5. label head для 8 классов создаётся заново;
6. сравниваются convergence speed, low-data utility и OOD robustness.

Это отвечает на вопрос о повторном использовании модели, не загрязняя
главный вывод о синтетике.

## Baseline qualification до генерации

Минимальные real-only arms:

1. natural sampling + cross entropy;
2. class-weighted CE;
3. balanced softmax;
4. weighted sampling;
5. source-matched replay для будущей synthetic dose;
6. optional cRT только при подтверждении Stage 9 результата.

Primary endpoint: validation macro one-vs-rest AUPRC.

Обязательные secondary endpoints:

- macro F1, MCC, balanced accuracy и ECE;
- worst-class recall;
- per-class AUROC/AUPRC;
- melanoma precision/recall/F1/AUPRC;
- melanoma sensitivity при specificity 0.90 и 0.95;
- source-wise и center-wise metrics;
- calibration slope/intercept;
- training time, peak VRAM и energy proxy.

Image-only модель является основным endpoint. Metadata fusion проводится
отдельно: ISIC challenge winners показывают пользу diversity ансамбля, но
metadata может маскировать качество изображения и эффект генерации.

## Перенос результатов Stage 15A

На ISIC 2019 проводится компактная replication, а не повтор всех старых
поисковых этапов:

| Arm | Воздействие |
|---|---|
| A | original source-matched replay |
| B | fixed offline square crop |
| C | B + deterministic generic VAE round-trip |
| D | C + weak img2img denoising |

Условия:

- tail classes `DF`, `VASC`, `SCC`, `AK`;
- одинаковые source lesions и количество additions;
- checkpoint monitor `val/auprc_ovr_macro`;
- один screening seed, затем seeds 42-44 только для preregistered contrasts;
- locked official test закрыт;
- сравнения `B-A`, `C-B`, `D-C`, `D-A`.

Если знак Stage 15A не переносится, причиной может быть dataset scale или
source diversity. Если переносится, это сильное подтверждение механизма.

## Новый генератор

### Основной кандидат

DiDGen является предпочтительным воспроизводимым кандидатом:

- официальный код открыт;
- Stable Diffusion дообучается с structured clinical prompts;
- region-aware attention связывает lesion tokens с областью поражения;
- генерируются согласованные image-mask pairs;
- MICCAI 2025 работа расширена статьёй Medical Image Analysis 2026.

До использования обязателен server gate:

- зафиксированный git commit;
- доступные base-model revisions;
- license manifest;
- воспроизводимый 8-image smoke;
- отсутствие train/test path leakage;
- peak VRAM не выше 15.5 GB.

Если полный fine-tuning не помещается в RTX 5080 16 GB, разрешён отдельный
LoRA/gradient-checkpointing pilot. Он должен называться адаптацией DiDGen, а
не точным воспроизведением оригинального метода.

### Геометрия

Stage 15A запрещает необратимый square crop перед генерацией.

Pipeline:

1. исходное изображение приводится к 512 canvas через aspect-preserving pad;
2. padding mask хранится отдельно;
3. lesion mask задаёт область генерации/inpainting;
4. background и acquisition artifacts не заменяются без отдельной
   интервенции;
5. результат unpad возвращается к исходному aspect ratio;
6. classifier получает тот же online transform, что и replay control.

### Маски

- использовать official ISIC segmentation masks там, где они доступны;
- pseudo-masks строить только для training images;
- основной mask encoder: PanDerm либо отдельно квалифицированная
  segmentation model;
- сохранять uncertainty и mask-quality score;
- отклонять маски с неустойчивой границей, пустой lesion region или
  чрезмерным охватом кадра;
- вручную проверять стратифицированную выборку по классам и source sites.

### Conditioning

Structured prompt содержит только подтверждённые training metadata:

- diagnosis;
- lesion location;
- lesion geometry и mask-derived measurements;
- dermoscopic attributes, если они доступны или независимо подтверждены;
- acquisition/source token только в специальной source-control ablation.

Нельзя автоматически приписывать редкому классу признаки, которых нет в
metadata. LLM-generated descriptions сохраняются как отдельные, проверяемые
artifacts и не считаются ground truth.

## Построение synthetic pool

Сначала создаётся pool, в 5 раз превышающий максимальную downstream dose.
Отбор выполняется только после независимых gates:

1. **File gate:** decode, размеры, RGB, отсутствие NaN/empty files.
2. **Geometry gate:** lesion находится внутри valid unpadded frame,
   согласованность image-mask и сохранение заданной площади/формы.
3. **Memorization gate:** exact hash, pHash, LPIPS и independent encoder
   nearest neighbours; слишком близкие копии исключаются.
4. **Label gate:** согласие PanDerm и ConvNeXt используется как diagnostic,
   но не как единственный критерий, чтобы не выбирать только лёгкие samples.
5. **Distribution gate:** per-class precision, density, coverage, Vendi и
   source-conditioned coverage в двух независимых feature spaces.
6. **Frequency/artifact gate:** спектральный gap, VAE residual, dark
   corners, ruler, hair и JPEG artifacts.
7. **Clinical audit:** blinded review стратифицированной выборки до
   downstream training.

После hard gates выбирается репрезентативное подмножество facility-location
или k-center методом. Stage 13 показывает, что такой отбор не является
лечением плохого генератора: он допустим только после прохождения
geometry/clinical gates.

## Synthetic dose и контроли

Большой ISIC не следует балансировать синтетикой до размера `NV`: это
создаст synthetic-dominated training set.

Screening ratios:

- synthetic:real additions `0.25`;
- `0.50`;
- `1.00` только если первые два уровня не ухудшают validation ranking.

Для каждого уровня обязательны:

- source/count-matched real replay;
- обычный oversampling;
- одинаковое число optimizer steps;
- одинаковый class exposure;
- одинаковые seeds и checkpoint policy;
- synthetic probability cap в batch.

Primary confirmatory comparison:

`lesion-aware synthetic - source-matched replay`.

Вторичные contrasts:

- gated synthetic - ungated synthetic;
- geometry-matched generator - legacy generic img2img;
- synthetic - weighted sampler;
- fresh ImageNet initialization - HAM transfer initialization.

## Статистика

- screening: один seed без test;
- confirmation: seeds 42-44 минимум;
- для выбранного финального contrast желательно 5 seeds;
- hierarchical bootstrap: seed -> patient -> lesion -> image;
- 5 000 или 10 000 bootstrap repeats;
- paired differences и 95% CI;
- знак эффекта по seeds;
- ranking metrics отдельно от threshold metrics;
- fixed-specificity melanoma diagnostics;
- multiplicity control для per-class secondary hypotheses;
- effect size и compute cost, а не только p-value.

Критерий успешной синтетики:

1. macro AUPRC выше source-matched replay;
2. знак положителен минимум на 2/3 seeds и не зависит от одного source site;
3. MCC/balanced accuracy не имеют устойчивого отрицательного эффекта;
4. melanoma AUPRC либо fixed-specificity sensitivity не ухудшаются;
5. improvement переносится хотя бы на один external/source holdout;
6. synthetic-only geometry metrics не используются как доказательство
   utility.

## Этапы реализации

### Stage 16A: data audit

- официальный download и checksum;
- unified metadata;
- ontology;
- duplicate graph;
- patient/lesion/source audit;
- immutable manifests;
- split gate;
- FiftyOne dataset для ручного контроля.

GPU не нужен. Оценка: 4-10 часов после скачивания.

### Stage 16B: baseline qualification

- ConvNeXt-S real-only controls;
- PanDerm-Base linear probe;
- optional PanDerm-Base fine-tuning;
- HAM-transfer ablation;
- source-wise и group-bootstrap analysis.

Оценка на RTX 5080: 1-2 суток при последовательном выполнении
preregistered arms.

### Stage 16C: Stage 15A replication

- четыре causal arms;
- сначала seed 42;
- затем только заранее выбранные contrasts seeds 43-44.

Оценка: 1-2 суток.

### Stage 16D: lesion-aware generator pilot

- DiDGen reproduction/VRAM gate;
- train-only masks и prompts;
- LoRA или full fine-tuning по заранее зафиксированной ветке;
- candidate generation;
- geometry, memorization, coverage и clinical gates.

Оценка: 1-3 суток для LoRA pilot; full reproduction может потребовать
больше времени или GPU с большей VRAM.

### Stage 16E: downstream confirmation

- equal-dose synthetic/replay/oversampling;
- 3-5 seeds;
- internal validation, source holdout, затем один locked test opening;
- итоговая статистика и manuscript tables.

Оценка: 2-4 суток.

Полный минимальный цикл: примерно 5-10 суток непрерывной работы сервера без
учёта ручного клинического аудита.

## Ресурсы lab-bio

RTX 5080 16 GB достаточна для:

- ConvNeXt-S 384;
- PanDerm-Base с gradient accumulation;
- DINOv2-B feature extraction;
- SD1.5/SD2.1 LoRA с mixed precision, gradient checkpointing и memory
  efficient attention;
- последовательной генерации 512 px.

64 GB RAM позволяют хранить metadata, hash index и часть decoded cache, но
не нужно загружать весь image corpus как tensors. Для DataLoader:

- 12-16 workers как начальный benchmark;
- persistent workers;
- pinned memory;
- prefetch 3-4;
- локальный SSD cache;
- настройки фиксируются по фактическому throughput, а не по числу CPU cores.

Предварительный storage budget:

- official raw archives: 35-40 GB;
- immutable/extracted/cached data: 50-100 GB;
- masks и metadata: 10-30 GB;
- synthetic candidate pool: 30-100 GB;
- checkpoints, MLflow references и reports: 50-100 GB.

Рабочий резерв 250-350 GB достаточен; полный 1 TB SSD использовать без
необходимости не следует. Heavy checkpoints остаются на сервере, MLflow
хранит ссылки и лёгкие artifacts.

## Правила публикации датасета

- CC-BY-NC attribution сохраняется для ISIC 2019/2020.
- Не публиковать повторно official images.
- Публиковать код, versioned manifests, hashes, split IDs и инструкции
  воспроизведения.
- Synthetic images рассматривать как потенциально производные материалы до
  отдельной проверки лицензий base generator и source datasets.
- Не утверждать anonymization без membership/memorization audit.
- Dataset card должна описывать centres, demographics, label certainty,
  artifacts, duplicates, excluded rows и known limitations.

## Решение

Перенос рекомендуется. Основная статья становится сильнее, если Stage 16
проверяет не «даёт ли больше данных выше accuracy», а следующий причинный
вопрос:

> Переносится ли обнаруженный на HAM10000 разрыв между визуальным качеством,
> покрытием распределения и downstream utility на мультицентровый ISIC, и
> устраняет ли geometry-preserving lesion-aware generation этот разрыв
> относительно source-matched replay?

ISIC 2019 является основным multiclass replication dataset. ISIC 2020 —
patient-level external binary stress test. ISIC 2024 не смешивается с
дерматоскопией и остаётся отдельной будущей задачей.

## Реализация Stage 16A/16B

Подготовлен воспроизводимый pipeline:

- `tools/prepare_isic2019.py` загружает официальные архивы, проверяет ZIP,
  декодирование, размеры, SHA-256 и perceptual dHash;
- связанные по lesion ID или точному SHA-256 изображения
  объединяются до разбиения;
- группы с конфликтующими диагнозами помещаются в quarantine;
- SCC сохраняется отдельным восьмым классом и не отображается в HAM10000
  AKIEC;
- формируются immutable manifest, dataset summary, train/validation/locked
  test split и малые smoke splits;
- `tools/check_stage16a_gate.py` блокирует обучение при нарушении ontology,
  количества изображений, group disjointness, checksum или locked-test
  policy;
- `scripts/start_stage16a_data_container.sh` выполняет загрузку и gate в
  отдельном CPU-контейнере;
- `scripts/start_stage16_smoke_container.sh` проверяет полный GPU path на
  64 train и 64 validation изображениях без открытия test;
- `scripts/start_stage16b_container.sh` запускает GPU screening только после
  успешного data gate.

Stage 16B является real-only qualification, а не проверкой синтетики.
Фиксируются ConvNeXt-S 384, одинаковая ImageNet-инициализация, одинаковая
аугментация и три метода работы с дисбалансом:

1. natural sampling + cross entropy;
2. weighted sampling + cross entropy;
3. natural sampling + Balanced Softmax.

Первый screening выполняется на seed 42 и закрытой validation. После
проверки корректности артефактов и вычислительного бюджета лучшие методы
повторяются на seeds 43 и 44. Locked test остаётся закрытым
(`evaluation.run_test=false`). Основной критерий выбора:
`val/auprc_ovr_macro`; дополнительно анализируются macro F1, MCC, balanced
accuracy, ECE, worst-class recall, melanoma и SCC AUPRC.

### Исправление Stage 16A перед запуском

Первый data audit обнаружил `978` quarantined изображений в 48 конфликтных
компонентах. Крупнейшая компонента ошибочно содержала 429 изображений всех
восьми классов. Причиной было использование точного совпадения 64-bit dHash
как identity edge: коллизии образовали транзитивные цепочки между
несвязанными lesions.

До запуска GPU protocol исправлен до
`isic2019_connected_group_v2`:

- blocking identity edges: official lesion ID и точный SHA-256;
- dHash сохраняется в manifest как diagnostic candidate signal;
- dHash не используется для group assignment, quarantine или split
  disjointness без последующего image-level подтверждения;
- добавлен regression test, запрещающий объединение разных изображений
  только по совпавшему perceptual hash;
- dataset manifest, splits, summary и gate пересоздаются из official
  cached artifacts.

Это изменение исключает систематическое удаление трудных примеров и
искажение class distribution до обучения. Near-duplicate audit по
perceptual embeddings остаётся отдельным анализом и не должен автоматически
менять locked split.

### GPU smoke и фактический batch

Первый smoke с ConvNeXt-S 384 и physical batch 64 завершился ожидаемо
контролируемым OOM до создания epoch metrics: процесс использовал 15.40 GiB
из доступных 15.47 GiB и не смог выделить ещё 54 MiB. Невалидный smoke run
не включается в научные сводки.

Для RTX 5080 16 GB physical batch зафиксирован равным 48 для всех Stage 16B
arms. Добавлен `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
Сопоставимость arms сохраняется, потому что batch, scheduler и optimizer
одинаковы; изменение является hardware qualification, выполненным до
screening. Полный запуск разрешается только после успешного повторного smoke
с `summary.json`, `val_metrics_best.json`, checkpoint и
`test_evaluated=false`.

### Фактический запуск 2026-07-30

После исправления protocol v2 data gate открыт:

- official rows: 25 331;
- usable: 25 327;
- quarantine: 4 exact-duplicate/label-conflict rows;
- connected groups: 13 926;
- train/validation/locked test: 17 729 / 3 799 / 3 799;
- locked test не вычислялся;
- все 8 классов присутствуют во всех splits.

Повторный smoke с batch 48 завершился с exit code 0 за 5.7 s training
time. Созданы `summary.json`, `val_metrics_best.json`,
`val_predictions_best.csv`, `best.pt`, model/sampling/environment artifacts;
`weights_source=ema`, `test_evaluated=false`.

Stage 16B screening запущен в контейнере
`research-stage16b-isic2019`. Первый run:
`stage16_isic2019_real_ce_natural_384`, seed 42. При запуске наблюдались
98-99% GPU utilization, 14.6 GB VRAM и до 278 W. После двух эпох:

- train loss: 1.653 -> 1.194;
- validation macro AUPRC: 0.187 -> 0.395;
- validation macro AUROC: 0.627 -> 0.877;
- validation MCC: 0.131 -> 0.484;
- ECE: 0.049 -> 0.030.

Эти ранние значения используются только как execution sanity check, а не
как научный результат. Канонический вывод делается после завершения
заранее заданных arms и multi-seed подтверждения.

## Использованные источники

1. ISIC Challenge Datasets. Official releases, metadata, duplicate lists and
   licenses:
   https://challenge.isic-archive.com/data/
2. Hernández-Pérez C, Combalia M, et al. BCN20000: Dermoscopic lesions in
   the wild. Scientific Data, 2024.
3. Gessert N, Nielsen M, Shaikh M, Werner R, Schlaefer A. Skin Lesion
   Classification Using Ensembles of Multi-Resolution EfficientNets with
   Meta Data, 2019.
4. Rotemberg V, Kurtansky N, et al. A patient-centric dataset of images and
   metadata for identifying melanomas using clinical context. Scientific
   Data, 2021.
5. Ha Q, Liu B, Liu F. Identifying Melanoma Images using EfficientNet
   Ensemble: Winning Solution to the SIIM-ISIC Melanoma Classification
   Challenge, 2020.
6. Yan S, Yu Z, et al. A multimodal vision foundation model for clinical
   dermatology. Nature Medicine, 2025. Official code:
   https://github.com/SiyuanYan1/PanDerm
7. Shentu J, Watson M, Al Moubayed N. DiDGen: Diffusion-based Dual-task
   Synthesis for Dermoscopic Data Generation. MICCAI, 2025; extended Medical
   Image Analysis paper, 2026. Official code:
   https://github.com/junjie-shentu/DiDGen
8. Bissoto A, Valle E, Avila S. GAN-Based Data Augmentation and
   Anonymization for Skin-Lesion Analysis: A Critical Review. CVPRW, 2021.
9. Adamkiewicz K, Moser B, et al. When Pretty Isn't Useful: Investigating
   Why Modern Text-to-Image Models Fail as Reliable Training Data
   Generators. CVPR, 2026.
10. Tschandl P, Rosendahl C, Kittler H. The HAM10000 dataset. Scientific
    Data, 2018.
