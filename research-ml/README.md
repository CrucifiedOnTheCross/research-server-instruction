# Research ML Experiments

Этот пакет нужен для контролируемых экспериментов по классификации изображений при дисбалансе классов и добавлении синтетики.

Главная идея: каждый запуск должен оставлять полный след для статьи:

- исходный и разрешенный конфиг;
- seed и настройки воспроизводимости;
- версии Python, PyTorch, CUDA, GPU;
- распределение классов по split;
- гиперпараметры модели, optimizer, scheduler, loss и sampler;
- метрики по эпохам;
- predictions на `val` и `test`;
- confusion matrix;
- `best.pt` и `last.pt`;
- диагностику synthetic-to-real gap в feature space.

## Почему такой дизайн

Каркас следует выводам из работ по long-tailed recognition и generative augmentation:

- синтетика должна оцениваться на real-only validation/test, а не на смешанном тесте;
- нужно сравнивать с простыми baseline: oversampling, class weights, focal loss, Balanced Softmax / logit adjustment;
- visual quality/FID недостаточно, потому полезно смотреть embedding diagnostics, real-vs-synthetic detectability, nearest-neighbor distance и межклассовую путаницу;
- synthetic ratio должен быть параметром эксперимента, а не автоматически доводиться до полного баланса.

## Формат CSV

Минимальный формат:

```csv
image_path,label,is_synthetic
images/img001.jpg,nv,0
synthetic/mel_0001.jpg,mel,1
```

`image_path` может быть абсолютным путем или путем относительно `data.root`.
`val.csv` и `test.csv` по умолчанию должны содержать только реальные изображения.

## Подготовка HAM10000

Официальный источник для стартового датасета - ISIC Archive, коллекция `66` / ISIC 2018 Task 3 Training. Скрипт использует официальный `isic-cli`, кэширует скачивание и строит `manifest.csv`, `splits/train.csv`, `splits/val.csv`, `splits/test.csv`, а также папки `images_by_class/`. Разбиение делается group-aware: все изображения одного `lesion_id` попадают только в один split, иначе получится утечка между train/val/test.

```bash
python tools/prepare_ham10000.py \
  --root /srv/research/projects/default/ham10000 \
  --source isic-cli \
  --collection-id 66 \
  --seed 42 \
  --link-mode symlink
```

Если датасет уже скачан вручную, положите изображения и metadata CSV внутрь `/srv/research/projects/default/ham10000` и запустите:

```bash
python tools/prepare_ham10000.py \
  --root /srv/research/projects/default/ham10000 \
  --source existing \
  --seed 42
```

## Установка на сервере

```bash
cd /srv/research/projects/default/research-ml
chmod +x scripts/*.sh
docker run --rm --gpus all \
  --user 1000:1006 --group-add 1006 \
  -v /srv/research/projects/default:/srv/research/projects/default \
  -w /srv/research/projects/default/research-ml \
  local/research-cuda-notebook:latest \
  bash scripts/setup_server_env.sh
```

Для RTX 50xx/Blackwell важно использовать современный PyTorch wheel. Если нужно явно указать CUDA build:

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 ./scripts/setup_server_env.sh
```

Если окружение уже собрано в Jupyter/Docker-образе, достаточно активировать его и установить недостающие зависимости.

## Рекомендуемый серверный запуск матрицы

```bash
cd /srv/research/projects/default/research-ml
chmod +x scripts/*.sh
./scripts/start_stage1_container.sh
```

Скрипт:

1. запустит detached Docker-контейнер с GPU;
2. задаст `--shm-size=16g` для PyTorch DataLoader;
3. подключит `/srv/research/projects/default`;
4. активирует `.venv`;
5. запустит `scripts/run_stage1_matrix.sh`.

Логи матрицы будут в `server-logs/`, live-лог можно смотреть через `docker logs -f research-ml-stage1`.

## Просмотр результатов по ссылке

```bash
cd /srv/research/projects/default/research-ml
./scripts/start_results_jupyter.sh
```

Ссылки:

- Dashboard: `http://10.200.1.180:8011/files/index.html?token=results`
- File browser: `http://10.200.1.180:8011/lab/tree/?token=results`

## Второй этап: targeted synthetic augmentation

Протокол второго этапа описан в `docs/stage2_synthetic_protocol.md`.

Ключевая идея: генерировать не все классы до полного баланса, а проблемные границы `mel/nv/bkl/akiec`, затем отбирать synthetic pool по feature-space критериям.

## Запуск обучения

```bash
python -m src.train --config configs/ham10000_stage1.yaml
```

Переопределение параметров без редактирования файла:

```bash
python -m src.train --config configs/ham10000_stage1.yaml \
  training.batch_size=128 \
  imbalance.loss=focal \
  imbalance.sampler=weighted \
  runtime.amp=bf16
```

## Контроль synthetic ratio

Если `train.csv` содержит реальные и синтетические изображения, можно создать контролируемые варианты:

```bash
python tools/apply_synthetic_ratio.py \
  --train-csv /srv/research/projects/default/ham10000/splits/train_full_synthetic_pool.csv \
  --out-csv /srv/research/projects/default/ham10000/splits/train_synth_0.50.csv \
  --ratio 0.50 \
  --seed 42
```

Балансировка до размера head-класса:

```bash
python tools/apply_synthetic_ratio.py \
  --train-csv /srv/research/projects/default/ham10000/splits/train_full_synthetic_pool.csv \
  --out-csv /srv/research/projects/default/ham10000/splits/train_synth_balance.csv \
  --balance-to-head \
  --seed 42
```

Потом в запуске:

```bash
python -m src.train --config configs/ham10000_stage1.yaml \
  data.train_csv=splits/train_synth_0.50.csv \
  experiment.name=ham10000_synth_050_balanced_softmax
```

## Диагностика синтетики

После обучения:

```bash
python -m src.analyze_embeddings \
  --run-dir outputs/ham10000_stage1_feature_aware_synthetic/20260708-120000_42 \
  --split train
```

Скрипт сохранит `embedding_diagnostics_train.json`, где будут:

- AUC классификатора `real vs synthetic` в эмбеддингах;
- расстояние synthetic-to-real внутри класса;
- расстояние synthetic-to-nearest-other-class;
- margin между своим и чужим классом;
- coverage proxy по ближайшим соседям.

## Практические режимы

Максимальная скорость:

```bash
runtime.deterministic=false runtime.amp=bf16 runtime.compile=true
```

Более строгая воспроизводимость:

```bash
runtime.deterministic=true runtime.amp=fp32 runtime.compile=false
```

Полная битовая воспроизводимость на GPU может снижать скорость и не всегда гарантируется всеми CUDA-операциями, поэтому режим фиксируется в артефактах запуска.

## Stage 8: исправленный протокол

Stage 8 не использует старый synthetic pool с чёрными полями. Полный pipeline:

```bash
bash scripts/start_stage8_container.sh
```

Он последовательно:

- создаёт новый group-aware train/validation/locked-test split;
- генерирует кандидатов только из Stage 8 train;
- прерывается, если pixel audit обнаруживает чёрные поля;
- отбирает синтетику независимым DINOv2 encoder;
- запускает четыре ConvNeXt Base 384 screening-ветки по seed 42-46;
- сохраняет multi-seed сводку после каждого завершённого запуска;
- не вычисляет метрики locked test до финального выбора метода.

Полное научное обоснование и критерии принятия решения:
`docs/stage8_methodology_repair_and_screening.md`.

Сводный анализ Stage 1-8 и оценка готовности статьи для «Компьютерной оптики»:
`docs/experiment_results_and_computer_optics_readiness_2026-07-24.md`.

Полный статистический аудит Stage 8, проверка пороговой эквивалентности,
сравнение с oversampling и план Stage 9:
`docs/stage8_full_statistical_analysis_and_article_plan_2026-07-24.md`.

Конспект современной литературы, использованной при интерпретации и
проектировании следующего этапа:
`docs/literature_update_synthetic_utility_and_long_tail_2026-07-24.md`.

Зафиксированный Stage 9 protocol для undersampling, Balanced Softmax, cRT,
source-matched replay и calibration:
`docs/stage9_controlled_utility_protocol.md`.

Итоговый анализ Stage 9, lesion-group bootstrap, calibration diagnostics и
план причинного geometry-dose эксперимента Stage 10:
`docs/stage9_results_and_stage10_plan_2026-07-27.md`.

Литературное обоснование, зафиксированные geometry strata, matched-replay
матрица и decision rules Stage 10:
`docs/stage10_literature_and_geometry_protocol.md`.

Условный протокол квалификации baseline Stage 11 с ConvNeXt-S/B, DINOv2
linear/full controls и формальным launch gate, закрытым до анализа Stage 10:
`docs/stage11_baseline_qualification_protocol.md`.

Обязательный порядок фиксации гипотез, изменений кода, проверок, результатов и
использованной литературы:
`docs/research_change_protocol.md`.

Пайплайн для подготовки evidence pack и LaTeX-черновика статьи под журнал
«Компьютерная оптика»:
`docs/computer_optics_article_pipeline.md`.
