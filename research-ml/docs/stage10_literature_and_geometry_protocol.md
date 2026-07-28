# Stage 10: литература и протокол geometry-conditioned utility

Дата фиксации: 2026-07-27.

## Научная задача

Stage 8-9 показали, что текущая смесь synthetic samples:

- не превосходит random oversampling;
- практически совпадает с source-matched real replay;
- меняет operating point сильнее, чем ranking quality;
- содержит смесь strict-filtered и top-k fallback изображений.

Stage 10 проверяет более узкий причинный вопрос:

> Зависит ли добавочная downstream utility синтетического изображения от его
> положения относительно real class manifold, если число строк, классовая
> композиция, источники, веса и optimizer protocol контролируются?

Primary estimand для каждого stratum:

`utility(synthetic stratum) - utility(source-matched replay stratum)`.

Сравнение только между synthetic strata недостаточно: оно смешивает новое
содержание с повторным предъявлением конкретных source images.

## Изученная литература

### DiffuLT: Diffusion for Long-tail Recognition Without External Knowledge

Shao et al., NeurIPS 2024:
<https://proceedings.neurips.cc/paper_files/paper/2024/hash/de7858e3e7f9f0f7b2c7bfdc86f6d928-Abstract-Conference.html>.

Что использовано:

- авторы делят generated samples на ID, AID и OOD в feature space;
- центр класса вычисляется по real features;
- порог `d_f` задается как максимальное расстояние между двумя real samples;
- ID: `d <= d_f`, AID: `d_f < d <= 2d_f`, OOD: `d > 2d_f`;
- в их CIFAR100-LT ablation AID-only filtering дал более высокую accuracy, чем
  использование всех generated samples;
- synthetic loss weight по умолчанию равен `0.3`, для ImageNet-LT `0.5`;
- авторы отдельно фильтруют harmful OOD.

Что нельзя переносить буквально:

- их feature extractor является baseline classifier, у нас независимый DINOv2;
- Euclidean distance до centroid в CIFAR не является валидированной
  дерматоскопической мерой;
- работа балансирует классы до заданного `N_t`, а наша гипотеза требует
  небольшого равного dose;
- AID не означает клиническую корректность.

Решение Stage 10: повторить идею radial bands как controlled ablation, но не
называть расстояние доказательством качества и обязательно сравнить каждый band
с replay.

### Disentangling dataset size from synthetic diversity in tuberculosis chest X-ray classification

Pink and Sykes, PMLR 318, 2026:
<https://proceedings.mlr.press/v318/pink26a.html>.

Ключевой результат:

- latent diffusion лучше WGAN-GP по FID и Rad-DINO FID;
- classifier training фиксировался на 4000 optimizer steps;
- dataset size и model-selection criteria совпадали;
- synthetic augmentation не превзошел count-matched duplicate-real control;
- лучший synthetic режим имел малую долю `r=0.25`;
- увеличение synthetic ratio ухудшало downstream utility.

Это прямая методическая опора для source-matched replay. Generative metric может
ранжировать генераторы и при этом не предсказывать полезность классификатору.

### Augmenting medical image classifiers with synthetic data from latent diffusion models

Sagers et al., 2023:
<https://arxiv.org/abs/2308.12453>.

Что важно:

- сгенерировано 458 920 изображений заболеваний кожи;
- сравнивались inpainting, in-then-outpainting и text-to-image;
- анализировались CLIP embeddings и расстояния к source images;
- использовались пять повторов обучения;
- наибольший выигрыш наблюдался при 16-64 real images/class;
- при 228 real images/class эффекты были малы и интервалы включали ноль;
- dose-response насыщался выше synthetic:real `10:1`;
- добавление real data было полезнее synthetic data.

Следствие: utility зависит от режима дефицита и дозы. Один большой synthetic arm
не позволяет заявлять общий эффект.

### Data Augmentation as Feature Manipulation

Shen, Bubeck, Gunasekar, ICML 2022:
<https://proceedings.mlr.press/v162/shen22a.html>.

Статья показывает, что augmentation может менять относительную важность
признаков и динамику их усвоения, а не просто увеличивать dataset size.

Следствие для проекта: после downstream runs нужно анализировать не только
macro F1, но и:

- class-wise AUROC/AUPRC;
- confusion transitions;
- representation geometry;
- различие эффекта synthetic и replay.

### Дополнительные уже зафиксированные источники

- MONICA: одинаковый backbone/split/protocol для long-tail baselines;
- Kang et al. cRT: representation и classifier bias анализируются отдельно;
- Menon et al. Logit Adjustment: prior correction не смешивается с sampler;
- PRDC: fidelity и coverage являются разными осями;
- Yang et al.: resampling, discrimination и calibration разделяются.

Новые работы Pink and Sykes, Sagers et al. и Shen et al. добавлены в Google
Sheet проекта строками `64-66`. DiffuLT уже находился в строке `63`.

## Зафиксированные strata

Все distances берутся из сохраненного Stage 8 DINOv2 score table. Повторное
извлечение features не выполняется, поэтому strata полностью воспроизводимы.

Для каждого `mel`, `akiec`, `bkl` выбирается ровно 30 изображений.

### `strict_id`

- `passes_geometry_filter == 1`;
- приоритет: больший `geometry_score`, затем меньшая дистанция до real;
- все выбранные изображения находятся внутри PRDC real manifold.

### `aid_radial`

- `q50 < nearest_real_distance <= q75` внутри класса;
- strict-ID rows исключены;
- приоритет: близость к середине radial band, затем больший feature margin.

Это адаптация AID, а не буквальная реализация DiffuLT. В DINO space текущего
pool такие изображения преимущественно находятся вне локального PRDC manifold,
поэтому результат должен интерпретироваться как moderate radial deviation.

### `ood_far`

- sample находится вне PRDC real manifold;
- `nearest_real_distance > class q75`;
- приоритет: наибольшая дистанция.

Ветка намеренно содержит наиболее дальние кандидаты и служит negative control.

### `random_remaining`

- детерминированная случайная выборка после исключения выбранных изображений
  остальных strata;
- seed `20260727`;
- не является отдельным геометрическим типом.

## Реальные характеристики выбранных данных

Dry-run по 1440 кандидатам:

| Stratum | Rows | Unique sources | Mean real distance | Mean feature margin | Inside manifold |
|---|---:|---:|---:|---:|---:|
| strict-ID | 90 | 84 | 0.1394 | +0.0761 | 1.000 |
| AID radial | 90 | 85 | 0.2891 | -0.0165 | 0.044 |
| OOD far | 90 | 88 | 0.4917 | -0.0422 | 0.000 |
| Random remaining | 90 | 84 | 0.2604 | -0.0127 | 0.156 |

Максимальное повторение одного source внутри stratum равно двум. Изображения
между strata не пересекаются.

## Матрица эксперимента

Для каждого stratum:

1. synthetic arm: `7228 real + 90 synthetic`;
2. replay arm: `7228 real + 90 повторов соответствующих source images`.

Общее:

- `30` строк на каждый целевой класс;
- sample/synthetic weight `0.5`;
- weighted sampler;
- ConvNeXt Base `384`;
- batch size `32`;
- BF16, channels-last;
- одинаковые transforms, optimizer, scheduler и early stopping;
- seeds `42, 43, 44`;
- synthetic и replay запускаются последовательно внутри одного matched seed;
- validation real-only, 1280 images / 599 lesion groups;
- locked test disabled.

Всего: `4 strata x 2 arms x 3 seeds = 24 runs`.

## Endpoints

Primary:

- macro F1;
- multiclass MCC.

Secondary:

- balanced accuracy;
- worst-class recall;
- melanoma precision, recall, F1;
- macro and per-class AUROC/AUPRC;
- ECE, NLL, Brier после отдельной group-held-out calibration diagnostic.

Механистические:

- `delta synthetic - replay` внутри каждого seed/stratum;
- interaction contrast:
  `delta_AID - delta_strict`, `delta_AID - delta_OOD`,
  `delta_AID - delta_random`;
- изменение confusion `mel/nv/bkl/akiec`;
- связь mean feature distance/margin stratum с downstream delta.

## Screening decision rule

Stratum не переносится дальше, если:

1. mean synthetic-minus-replay macro F1 `<= 0`;
2. или mean synthetic-minus-replay MCC `< -0.02`;
3. или synthetic уступает Stage 8 real oversampling по macro F1 более `0.01`;
4. или рост melanoma recall сопровождается снижением mel-F1 более `0.02`.

Shortlist требует:

- positive synthetic-minus-replay macro F1 минимум на `2/3` seeds;
- mean delta macro F1 не менее `+0.01` либо mean delta mel-F1 не менее `+0.015`;
- отсутствие guardrail failure по MCC;
- отсутствие gross artifact/leakage failure.

Это engineering screening, не confirmatory significance test. При трех seeds
минимальное двустороннее exact sign-flip p равно `0.25`.

## Следующий уровень подтверждения

Только прошедший stratum:

- расширяется до seeds `45-46`;
- затем проверяется nested lesion-group CV;
- calibration и operating point выбираются только на inner folds;
- сравнивается с real oversampling и своим replay;
- после замораживания протокола допускается единственная оценка locked test.

## Артефакты

Builder сохраняет:

- SHA256 входных train/scores CSV;
- class-specific thresholds;
- полный assignment всех кандидатов;
- выбранные synthetic CSV;
- synthetic train CSV;
- source replay rows/train CSV;
- counts, source reuse и overlap audit;
- `stage10_strata_manifest.json`.

Каждый training run дополнительно сохраняет стандартные config, environment,
sampling, model initialization, predictions, metrics и locked-test flag.

## Команды

Подготовка и запуск:

```bash
bash scripts/start_stage10_container.sh
```

Ручная подготовка split:

```bash
python tools/make_stage10_geometry_strata.py \
  --data-root /srv/research/projects/default/ham10000 \
  --train-csv splits/stage8/train_real.csv \
  --scores-csv /srv/research/projects/default/research-ml/outputs/reports/stage8_dino_geometry/synthetic_dino_scores.csv \
  --out-dir splits/stage10 \
  --target-classes mel,akiec,bkl \
  --dose-per-class 30 \
  --sample-weight 0.5 \
  --max-source-reuse-per-stratum 2 \
  --seed 20260727
```

## Известное инфраструктурное исправление

После обновления Ubuntu до kernel `7.0.0-28` Secure Boot отклонял DKMS-модуль
NVIDIA, подписанный незарегистрированным локальным ключом. Установлен
Canonical-signed пакет:

`linux-modules-nvidia-595-open-7.0.0-28-generic`.

После исправления:

- host `nvidia-smi`: RTX 5080, driver `595.84`, VRAM `16303 MiB`;
- Docker CUDA smoke test: passed.

## Журнал реализации и запуска

### 2026-07-27: implementation sprint

Коммит: `ca88f1c`.

Проверено:

- восемь server-side unit tests: passed;
- shell syntax launch scripts: passed;
- real-data dry-run: 4 strata по 90 строк, overlap `0`;
- synthetic/replay train size: `7318/7318` для каждой пары;
- source reuse max: `2`;
- ConvNeXt Base 384 BF16 forward/backward, batch 32: passed;
- peak GPU memory allocated/reserved: `11.97 / 12.51 GiB`;
- synthetic и replay DataLoader: по `7318` samples/epoch и одинаковые class
  counts.

### 2026-07-27: запуск screening matrix

Контейнер: `research-stage10-geometry`.

Первый run:

`outputs/stage10_synthetic_strict_id_384/20260727-135942_42`.

Начальный structured audit после двух эпох:

- контейнер: running;
- GPU utilization: `100%`;
- GPU memory: около `13.7 GiB`;
- sampling: weighted with replacement;
- dataset/samples per epoch: `7318/7318`;
- trainable parameters: `87,573,639`;
- `best.pt`, `val_metrics_best.json`, `val_predictions_best.csv`,
  `sampling_plan.json`, `model_initialization.json` созданы;
- locked test не запускался.

Промежуточные warm-up метрики не используются для научного вывода. Оценка
выполняется только после завершения matched synthetic/replay runs всех трех
seed.

### 2026-07-28: Stage 10 завершён

- 24/24 runs завершены с exit code `0`;
- все mandatory artifacts присутствуют;
- `test_evaluated=false` во всех runs;
- validation predictions совпадают по 1280 изображениям и 599 lesion groups;
- метрики независимо пересчитаны с максимальной ошибкой `1.08e-7`;
- выполнен hierarchical lesion-group bootstrap, 5000 повторов;
- решение и полные результаты:
  `docs/stage10_results_and_stage11_decision_2026-07-28.md`.
