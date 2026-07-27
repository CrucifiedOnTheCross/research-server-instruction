# Переход от самописного results index к MLflow и FiftyOne

Дата решения: 2026-07-27.

## Решение

Самописный dashboard на порту 8011 выводится из эксплуатации. Новая схема:

- **MLflow 3.14.0**: runs, resolved hyperparameters, epoch curves, итоговые
  метрики, system/GPU metrics и structured artifacts;
- **FiftyOne 1.19.0**: интерактивный просмотр synthetic/source/nearest-real
  изображений с фильтрами по классу, этапу отбора и geometry-признакам;
- **JupyterHub**: редактирование и прямой доступ к файлам проекта;
- JSON/CSV/YAML в `outputs/` остаются каноническими научными артефактами.

MLflow и FiftyOne являются представлениями над этими данными. Они не заменяют
локальные artifacts и не являются единственным местом хранения результатов.

## Почему MLflow

Официальная документация:
<https://mlflow.org/docs/latest/tracking>.

MLflow предоставляет готовые:

- список и сравнение runs;
- поиск по параметрам и метрикам;
- исторические metric curves по step/epoch;
- хранение и скачивание artifacts;
- experiment grouping;
- system metrics, включая GPU utilization, VRAM и power при наличии
  `nvidia-ml-py`.

Tracking server:
<https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/>.

Для текущего однопользовательского сервера выбран SQLite backend и локальный
artifact store на SSD. Это проще PostgreSQL/MinIO и достаточно при одном
последовательном training pipeline. Переход на PostgreSQL потребуется только при
нескольких параллельных writers или командной эксплуатации.

## Почему FiftyOne

Официальная документация:
<https://docs.voxel51.com/user_guide/using_datasets.html>.

MLflow хорошо показывает metrics и отдельные artifacts, но не заменяет
полноценный image dataset browser. FiftyOne позволяет:

- листать изображения без генерации HTML thumbnails;
- фильтровать по `label`, `selected_by_stage6`, `stage10_selected`,
  `stage10_stratum` и distance/geometry fields;
- переключать group slices:
  - `synthetic`;
  - `source`;
  - `nearest_real`;
- сохранять dataset metadata без копирования исходных изображений.

FiftyOne хранит только metadata и абсолютные пути. Изображения остаются в
`/srv/research/projects/default/ham10000`.

## Размещение

| Сервис | URL | Контейнер | Данные |
|---|---|---|---|
| MLflow | `http://10.200.1.180:5000` | `research-mlflow` | `tracking-data/mlflow` |
| FiftyOne | `http://10.200.1.180:5151` | `research-fiftyone` | `tracking-data/fiftyone` |
| JupyterHub | `http://10.200.1.180:8000` | existing | project files |

MLflow сохраняет включённым встроенный network security middleware. Адрес
лабораторного сервера явно разрешён одновременно в `--allowed-hosts` и
`--cors-allowed-origins`; небезопасный `*` намеренно не используется.
Настройка сверена с официальным руководством:
<https://mlflow.org/docs/latest/self-hosting/security/network/>.

FiftyOne использует отдельный внутренний MongoDB 7 container без опубликованного
сетевого порта. Это соответствует рекомендованному `FIFTYONE_DATABASE_URI`;
анонимная telemetry отключена.

Доступ ограничивается внутренней/VPN сетью и UFW. Публичный доступ без VPN не
документируется.

## Data contract MLflow

Каждый будущий training run получает:

- run name: `<experiment>/<timestamp>_<seed>`;
- tags:
  - experiment name;
  - seed;
  - source run directory;
  - code commit;
  - факт test evaluation;
- flattened resolved config как params;
- epoch metrics с исходным `epoch` как MLflow step;
- summary/best-validation/test metrics;
- JSON, CSV, YAML и plots как artifacts;
- ссылки на checkpoints в `checkpoint_locations.txt`;
- CPU, RAM, GPU, VRAM, power, network и disk system metrics.

Checkpoints не копируются в MLflow artifact store. Они велики, уже находятся на
SSD и имеют путь в run directory. Это предотвращает удвоение десятков гигабайт.

Training не зависит от доступности UI: при `tracking.fail_on_error=false`
ошибка MLflow записывается в обычный run log, а локальное обучение продолжается.

## Backfill существующих результатов

`tools/sync_mlflow_runs.py` импортирует только завершённые runs с:

- `summary.json`;
- `config.resolved.yaml`;
- `metrics.csv`.

Метрики не извлекаются из логов. Повторный запуск idempotent по
`tags.source_run_dir`. Отдельный experiment `HAM10000 Reports` получает
существующие CSV/JSON/PNG аналитических отчётов; HTML не импортируется.

Текущий объём во время Stage 10: 56 завершённых runs и 10 каталогов
аналитических отчётов. Число будет расти по мере завершения Stage 10.

## FiftyOne dataset

`tools/sync_fiftyone_synthetic.py` читает
`splits/stage10/stage10_candidate_assignments.csv` и создаёт persistent grouped
dataset `ham10000-synthetic-audit`.

Для каждой synthetic строки формируется группа до трёх изображений:

1. generated synthetic;
2. source real;
3. nearest real в feature space.

Metadata включает generation parameters, prompts, distances, feature margin,
manifold flag, Stage 6 selection и Stage 10 stratum/rank.

## Изменения training code

Добавлен `src/tracking.py`:

- lazy import MLflow;
- flattening config/metrics;
- epoch logging;
- system metrics;
- graceful failure;
- корректное завершение run как `FINISHED` или `KILLED`.

Новые параметры:

```yaml
tracking:
  enabled: true
  uri: http://10.200.1.180:5000
  experiment: HAM10000
  log_system_metrics: true
  fail_on_error: false
```

Stage 10 не изменяется во время выполнения. Новая интеграция применяется после
его завершения; Stage 10 импортируется backfill-скриптом.

## Вывод из эксплуатации HTML dashboard

Удалены:

- `tools/build_results_index.py`;
- scripts запуска results index/Jupyter на 8011.

`summarize_multiseed.py` сохраняет только JSON и CSV. Исторические HTML reports
не удаляются, поскольку являются частью аудита прошлых стадий, но новые стадии
не используют их как интерфейс.

После проверки MLflow/FiftyOne контейнер `research-ml-results-jupyter`
останавливается и удаляется, порт 8011 закрывается в UFW.

## Проверки перед эксплуатацией

- `python -m unittest tests.test_tracking -v`: 2/2;
- `python -m compileall -q src tools tracking`: успешно;
- `docker compose config -q`: успешно;
- MLflow `/health`: HTTP 200;
- повторный backfill: 0 imported, 56 skipped; reports: 0 imported,
  10 skipped;
- через UI проверены таблица runs, 71 метрика выбранного Stage 10 run,
  metric history и вкладки system metrics/artifacts;
- первый UI-аудит обнаружил блокировку CORS при загрузке runs; добавлен точный
  `--cors-allowed-origins http://10.200.1.180:5000`, после чего таблица
  загрузилась;
- FiftyOne: 1440 groups, 4320 samples, slices `synthetic`, `source`,
  `nearest_real`, missing media = 0;
- через UI проверена фактическая загрузка изображений и полей фильтрации;
- контейнер results dashboard удалён, правило UFW для 8011 удалено;
- MLflow/FiftyOne/Mongo имеют healthy status, наружу опубликованы только
  5000 и 5151.
