# Stage 9: controlled synthetic utility protocol

Дата фиксации протокола: 2026-07-24.

## Научный вопрос

Добавляют ли отобранные синтетические дерматоскопические изображения
дискриминативную информацию сверх:

1. естественного обучения на реальных данных;
2. random oversampling реальных minority-классов;
3. random undersampling head-классов;
4. коррекции class prior в loss;
5. отдельной настройки classifier head;
6. повторного предъявления тех же реальных source images;
7. post-hoc calibration и изменения operating point?

Stage 9 не должен доказывать пользу генерации сравнением только с natural CE.
Главный matched control для текущей synthetic-ветки: source-matched real replay.

## Основание в литературе

### Decoupling Representation and Classifier for Long-Tailed Recognition

Источник: <https://arxiv.org/abs/1910.09217>.

Kang et al. показывают, что естественное instance-balanced обучение может
сформировать сильное представление, после чего достаточно отдельно настроить
classifier. Реализация cRT в Stage 9:

- encoder загружается из лучшего Stage 8 natural CE checkpoint того же seed;
- classifier переинициализируется;
- encoder полностью замораживается;
- classifier обучается с weighted random sampler;
- checkpoint, seed и список trainable parameters сохраняются.

### Long-tail learning via logit adjustment

Источник: <https://arxiv.org/abs/2007.07314>.

Logit Adjustment является prior-correction методом. Он сравнивается с natural
sampler и не объединяется с weighted sampler в одной основной ветке, чтобы не
смешивать два механизма.

### MONICA

Источник: <https://arxiv.org/abs/2410.02010>.

MONICA требует единого backbone, split, augmentation и protocol для более чем
одного класса long-tail методов. Поэтому Stage 9A сохраняет ConvNeXt Base 384,
Stage 8 train/validation/locked-test split и одинаковые seeds.

### Random oversampling and undersampling

Источник: <https://doi.org/10.1186/s40537-023-00857-7>.

Работа показывает, что resampling может менять calibration и threshold-specific
метрики без устойчивого выигрыша ranking. Поэтому сохраняются logits,
per-class AUROC/AUPRC, NLL, Brier и ECE; post-hoc calibration рассматривается
отдельно.

### DiffuLT

Источник:
<https://papers.nips.cc/paper_files/paper/2024/hash/de7858e3e7f9f0f7b2c7bfdc86f6d928-Abstract-Conference.html>.

Гипотеза об approximately-in-distribution samples переносится в Stage 9C.
В Stage 9A геометрию не меняем: сначала устанавливаем, превосходит ли текущая
синтетика простые controls.

## Stage 9A: матрица controls

Используются Stage 8 anchors для seeds 42-46:

- `stage8_real_ce_natural_384`;
- `stage8_real_ce_weighted_384`;
- `stage8_real_logit_adjust_natural_384`;
- `stage8_synthetic_dino_384`.

Новые screening-ветки:

| ID | Метод | Sampling | Данные | Назначение |
|---|---|---|---|---|
| S9A-U | Real CE undersampling | Равное число уникальных real samples/class без replacement | Только real | Проверка потери head diversity |
| S9A-BS | Real Balanced Softmax | Natural | Только real | Loss-level prior control |
| S9A-cRT | cRT classifier retraining | Weighted | Только real | Representation/classifier control |
| S9A-R | Source-matched replay | Weighted | Real + дубликаты source images | Контроль нового содержания |

Первые screening seeds: `42 43 44`. Stage 8 anchors позволяют сразу выполнить
paired comparison на этих seed. Расширение до seeds `45 46` выполняется только
для веток, прошедших критерий shortlist.

## Source-matched replay

Для каждого synthetic sample, выбранного в Stage 8:

- находится его `source_image_id`;
- в train добавляется отдельная replay-row с исходным real image;
- class, source group и число дополнительных rows совпадают с synthetic arm;
- replay-row получает sample weight `0.5`, как synthetic sample;
- weighted sampler видит одинаковые class-level добавления;
- validation и locked test остаются real-only.

Это не новая аугментация изображения. Обычные train transforms применяются
независимо при каждом предъявлении source image.

Интерпретация:

- synthetic лучше replay: возможна добавочная информация от изменения изображения;
- synthetic равно replay: эффект объясняется повторным предъявлением источников;
- synthetic хуже replay: generator/filter добавляет вредный distribution shift.

## Endpoints

Primary:

- macro F1;
- multiclass MCC.

Guardrails:

- balanced accuracy;
- worst-class recall;
- ECE после и до calibration;
- melanoma precision, recall и F1.

Ranking endpoints:

- macro OVR AUROC/AUPRC;
- per-class AUROC/AUPRC;
- melanoma sensitivity при заранее фиксированных specificity levels.

Locked test не вычисляется на screening.

## Критерий shortlist

Новая ветка проходит в confirmatory CV, если одновременно:

1. paired mean macro F1 не хуже weighted CE более чем на `0.01`;
2. paired mean MCC не хуже weighted CE более чем на `0.02`;
3. улучшение melanoma recall не сопровождается падением mel-F1 более `0.02`;
4. результат выигрывает минимум на двух из трех screening seeds по primary
   composite rank.

Эти значения являются engineering screening margins, не клинически
обоснованными equivalence margins и не используются для финального утверждения
эквивалентности.

## Stage 9B: calibration diagnostic

На Stage 9 fixed validation выполняется только exploratory split:

- целые lesion groups разделяются на calibration/evaluation subsets;
- temperature и melanoma logit offset подбираются только на calibration;
- метрики сообщаются на evaluation subset;
- ранняя остановка модели все еще использовала общую validation, поэтому
  результат не считается confirmatory.

Confirmatory calibration переносится внутрь outer group-aware CV.

Реализация:

- `tools/calibrate_stage9_predictions.py`;
- `scripts/run_stage9_calibration.sh`;
- scalar temperature минимизирует calibration NLL;
- melanoma logit offset выбирается отдельно по macro F1, MCC и mel-F1;
- operating points оценивают melanoma sensitivity при specificity `0.90` и
  `0.95`;
- calibration/evaluation assignments сохраняются по `group_id`.

## Stage 9C: geometry-dose experiment

После Stage 9A создаются равные по размеру synthetic strata:

- strict-ID;
- AID distance band;
- OOD;
- random eligible.

Запрещен top-k fallback между strata. Число samples, class composition,
synthetic weight, sampler и optimizer steps должны совпадать.

## Reproducibility artifacts

Каждый запуск обязан сохранять:

- `config.resolved.yaml`;
- `environment.json`;
- `class_counts.json`;
- `sampling_plan.json`;
- `model_initialization.json`;
- `trainable_parameters.json`;
- `metrics.csv` и `metrics.jsonl`;
- `val_metrics_best.json`;
- `val_predictions_best.csv` с probabilities и raw logits;
- `best.pt`;
- `summary.json` с best epoch и test lock status.

Результаты извлекаются из JSON/CSV. Логи используются только для диагностики
ошибки выполнения.

## Вычислительный бюджет

Сервер: RTX 5080 16 GB, 64 GB RAM, SSD 1 TB.

Stage 9A сохраняет Stage 8 throughput-конфигурацию:

- ConvNeXt Base 384;
- batch size 32;
- BF16;
- channels-last;
- 12 DataLoader workers;
- 24 GB shared memory контейнера;
- locked test disabled.

Undersampling имеет меньше optimizer steps на epoch и считается отдельным
data-budget baseline. Он не интерпретируется как compute-matched доказательство.
Время, epochs и фактическое число samples per epoch сохраняются для последующей
нормализации.

## Правило изменения протокола

Любое изменение sampler, loss, source replay, calibration, endpoint, seed set,
split или decision rule сначала добавляется в этот файл с датой и причиной, а
затем реализуется в коде.

## Журнал реализации и запуска

### 2026-07-24: Stage 9A controls

Коммит: `360a8c2`.

Реализованы и проверены:

- deterministic class-balanced undersampler без replacement;
- row-level sample weights;
- source-matched replay builder;
- checkpoint initialization и classifier-only cRT;
- переинициализация classifier head;
- raw logits и per-class AUROC/AUPRC;
- `sampling_plan.json`, `model_initialization.json`,
  `trainable_parameters.json`;
- семь серверных unit tests и CUDA cRT forward.

Replay audit:

- real rows: `7228`;
- replay rows: `240`;
- `80` rows для каждого `mel`, `akiec`, `bkl`;
- unique source images: `203`;
- maximum source reuse: `3`;
- replay sample weight: `0.5`.

### 2026-07-24: Stage 9B instrumentation

Коммит: `885c894`.

Добавлены group-held-out exploratory temperature scaling, melanoma offsets и
fixed-specificity operating points. Locked test этим инструментом не читается.

### Первый structured result

`stage9_real_ce_undersample_384`, seed `42`:

- best epoch: `10`;
- elapsed: `149.4 s`;
- macro F1: `0.5673`;
- balanced accuracy: `0.6899`;
- MCC: `0.4092`;
- ECE: `0.1188`;
- mel precision/recall/F1: `0.2413 / 0.6294 / 0.3488`;
- macro OVR AUROC/AUPRC: `0.9057 / 0.6678`;
- locked test: not evaluated.

Это промежуточный screening signal. Он не интерпретируется до завершения трех
seed. Низкие macro F1 и MCC согласуются с потерей head-class diversity: за
эпоху undersampling использует `567` уникальных изображений (`81` на класс)
вместо `7228` real rows.
