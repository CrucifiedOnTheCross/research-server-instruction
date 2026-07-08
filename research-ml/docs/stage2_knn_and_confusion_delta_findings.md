# Stage 2 diagnostics: kNN audit and confusion delta

Дата: 2026-07-08  
Проект: HAM10000, несбалансированная классификация кожных новообразований, проверка полезности синтетических изображений.

## Зачем был нужен этот спринт

После Stage 2 стало ясно, что добавление синтетики не улучшило сильный real-only baseline. Поэтому следующий вопрос был не "какая метрика выше", а "почему синтетика не дала устойчивого выигрыша".

Этот спринт добавил две диагностики:

1. kNN-аудит синтетических изображений в feature-space лучшей real-only модели.
2. Confusion-delta отчеты: какие test-изображения Stage 2 исправляет, а какие ломает относительно Stage 1.

## Артефакты

Серверные HTML-отчеты доступны через results dashboard:

- Dashboard: `http://10.200.1.180:8011/files/index.html?token=results`
- kNN raw sample: `http://10.200.1.180:8011/files/stage2_diagnostics/knn_gallery/raw_sample40/index.html?token=results`
- kNN strict: `http://10.200.1.180:8011/files/stage2_diagnostics/knn_gallery/strict/index.html?token=results`
- kNN topk80: `http://10.200.1.180:8011/files/stage2_diagnostics/knn_gallery/topk80/index.html?token=results`
- Confusion delta folder: `http://10.200.1.180:8011/files/stage2_diagnostics/confusion_delta/?token=results`

Файлы на сервере:

- `/srv/research/projects/default/ham10000/reports/stage2_diagnostics/knn_gallery/`
- `/srv/research/projects/default/ham10000/reports/stage2_diagnostics/confusion_delta/`
- `/srv/research/projects/default/research-ml/tools/build_knn_gallery.py`
- `/srv/research/projects/default/research-ml/tools/compare_runs.py`

## kNN-аудит синтетики

Метод: для каждого synthetic image извлекаются признаки baseline ConvNeXt-модели. Затем ищется ближайший реальный train-пример того же класса и ближайший реальный train-пример из типичного confusing-класса. Панель показывает:

`source real -> synthetic -> nearest same class -> nearest confusing class`

Интерпретация margin:

- `same_distance` ниже - синтетика ближе к реальным примерам своего класса.
- `confusing_distance` ниже - синтетика ближе к опасному соседнему классу.
- `margin = confusing_distance - same_distance`.
- Положительный margin означает, что synthetic ближе к своему классу, чем к confusing-классу.
- Отрицательный margin означает потенциально вредный или неоднозначный synthetic sample.

### Сводка

| Pool | Total | Classes | Mean same distance | Mean confusing distance | Mean margin | Negative margin |
|---|---:|---|---:|---:|---:|---:|
| raw_sample40 | 120 | akiec=40, bkl=40, mel=40 | 0.3117 | 0.3220 | 0.0103 | 50 |
| strict | 43 | bkl=28, mel=15 | 0.0980 | 0.1797 | 0.0817 | 0 |
| topk80 | 240 | akiec=80, bkl=80, mel=80 | 0.2024 | 0.3589 | 0.1565 | 6 |

### Вывод по kNN

Raw-синтетика содержит много спорных изображений: 50 из 120 sampled examples имеют отрицательный margin. Это подтверждает, что простая генерация "похожих" изображений недостаточна.

Strict-фильтр дает наиболее чистые изображения в feature-space: отрицательных margin нет. Но он полностью выбросил `akiec`, поэтому как метод борьбы с дисбалансом он непригоден без отдельного механизма покрытия редких классов.

Topk80 является компромиссом: он сохраняет баланс по `akiec/bkl/mel`, имеет высокий mean margin и только 6 отрицательных margin. Но даже такой отбор не улучшил test-качество относительно Stage 1. Значит, одного feature-space-фильтра недостаточно: нужно учитывать не только "похоже на класс", но и полезность для decision boundary.

## Confusion-delta анализ

Метод: сравниваются предсказания двух run на одном и том же test split.

Статусы:

- `both_correct`: обе модели ответили верно.
- `candidate_only`: Stage 2 исправил ошибку Stage 1.
- `base_only`: Stage 2 сломал пример, который Stage 1 классифицировал верно.
- `both_wrong`: обе модели ошиблись.

### Stage1 CE weighted vs Stage2 topk80 CE weighted

Общий test count: 1502.

| Status | Count |
|---|---:|
| both_correct | 1256 |
| candidate_only | 57 |
| base_only | 55 |
| both_wrong | 134 |

Per-class:

| Target | base_only | candidate_only | both_wrong | Комментарий |
|---|---:|---:|---:|---|
| akiec | 3 | 2 | 11 | Нет выигрыша для ключевого редкого класса. |
| bcc | 1 | 3 | 7 | Небольшой плюс. |
| bkl | 18 | 12 | 27 | Синтетика чаще ломает, чем исправляет. |
| df | 2 | 1 | 2 | Малый класс, выводы осторожно. |
| mel | 21 | 13 | 32 | Важный клинический класс просел. |
| nv | 9 | 26 | 54 | Есть выигрыш на majority-классе, но это не главная цель. |
| vasc | 1 | 0 | 1 | Просадка на малом классе. |

Ключевой вывод: topk80 CE weighted почти балансирует исправления и поломки по всему test set, но ухудшает именно целевые minority/boundary классы `bkl` и `mel`.

### Stage1 CE weighted vs Stage2 strict CE weighted

Общий test count: 1502.

| Status | Count |
|---|---:|
| both_correct | 1260 |
| candidate_only | 69 |
| base_only | 51 |
| both_wrong | 122 |

Per-class:

| Target | base_only | candidate_only | both_wrong | Комментарий |
|---|---:|---:|---:|---|
| akiec | 2 | 6 | 7 | Улучшает `akiec`, хотя synthetic `akiec` в strict не осталось. Вероятно, эффект от общей регуляризации/границы. |
| bcc | 4 | 3 | 7 | Почти нейтрально. |
| bkl | 21 | 10 | 29 | Сильная просадка. |
| df | 4 | 0 | 3 | Просадка на малом классе. |
| mel | 16 | 14 | 31 | Почти нейтрально, но без уверенного выигрыша. |
| nv | 2 | 36 | 44 | Улучшение в основном на majority-классе. |
| vasc | 2 | 0 | 1 | Просадка. |

Ключевой вывод: strict CE weighted формально исправляет больше примеров, чем ломает, но его улучшение концентрируется на `nv`, а не на целевых проблемных классах.

### Stage1 balanced softmax vs Stage2 topk80 balanced softmax

Общий test count: 1502.

| Status | Count |
|---|---:|
| both_correct | 1257 |
| candidate_only | 65 |
| base_only | 71 |
| both_wrong | 109 |

Per-class:

| Target | base_only | candidate_only | both_wrong | Комментарий |
|---|---:|---:|---:|---|
| akiec | 5 | 6 | 8 | Почти нейтрально. |
| bcc | 1 | 7 | 7 | Улучшение. |
| bkl | 24 | 11 | 28 | Заметная просадка. |
| df | 2 | 0 | 4 | Просадка. |
| mel | 20 | 21 | 40 | Почти нейтрально, но много устойчивых ошибок. |
| nv | 19 | 18 | 20 | Нейтрально. |
| vasc | 0 | 2 | 2 | Небольшой плюс. |

Ключевой вывод: добавление topk80 к balanced softmax не дает стабильного улучшения, а `bkl` снова деградирует.

## Научная интерпретация

Текущие результаты поддерживают гипотезу:

> Полезность синтетики определяется не только визуальным качеством и не только близостью к manifold своего класса. Для повышения качества обучения нужны synthetic samples, которые одновременно реалистичны, покрывают редкие классы и добавляют информацию около decision boundary без внесения class-confusing артефактов.

Stage 2 показал важный отрицательный результат:

- Сгенерированная синтетика может быть визуально правдоподобной.
- Feature-space selection может удалить явный мусор.
- Но даже отобранная синтетика может не улучшать целевую обобщающую способность.

Это сильная основа для диплома/статьи, потому что она переводит задачу от "сгенерировать лучше" к "оценить и отобрать синтетику по полезности для обучения".

## Что делать в следующем этапе

Следующий этап стоит делать не как очередное увеличение количества synthetic images, а как controlled utility-aware selection.

Рекомендуемые направления:

1. Hard-case targeted generation.
   Генерировать синтетику не вокруг случайных minority samples, а вокруг реальных train-примеров, которые baseline считает трудными: низкая confidence, высокая entropy, близость к ошибкам `mel/bkl/akiec`.

2. Boundary-aware filtering.
   Оставлять synthetic samples, которые:
   - ближе к своему классу, чем к confusing-классу;
   - не являются слишком легкими near-duplicates;
   - имеют умеренную сложность для baseline, а не выглядят как out-of-distribution.

3. Ablation по количеству и весу синтетики.
   Проверить не только strict/topk80, но и:
   - topk20/topk40/topk80;
   - synthetic loss weight 0.25/0.5/1.0;
   - class-specific quotas, особенно для `akiec` и `mel`.

4. Separate real-vs-synthetic batch policy.
   Не смешивать synthetic как обычные реальные изображения без контроля. Лучше использовать batch sampler, где доля синтетики фиксирована и логируется.

5. Метрики отбора synthetic utility.
   Логировать для каждого synthetic:
   - distance to same class;
   - distance to confusing class;
   - margin;
   - baseline entropy/confidence;
   - source image id;
   - generation seed/prompt/strength;
   - был ли sample выбран;
   - как изменились test ошибки после добавления пула.

## Практический план следующего спринта

Цель: Stage 3 boundary-aware synthetic selection.

Минимальный набор:

1. Построить таблицу hard real train images:
   - baseline prediction на train/val;
   - confidence, entropy, margin между top-1/top-2;
   - target class;
   - correctness.

2. Сформировать generation candidates:
   - только `mel`, `akiec`, `bkl`;
   - приоритет: hard-but-correct и hard-wrong реальные изображения;
   - исключить lesion leakage между train/val/test через `lesion_id`.

3. Сделать selection score:
   - positive: class margin в kNN feature-space;
   - positive: baseline uncertainty;
   - negative: too close to source duplicate;
   - negative: closer to confusing class than own class.

4. Обучить controlled matrix:
   - baseline unchanged;
   - Stage 3 synthetic with small quota;
   - Stage 3 synthetic with medium quota;
   - Stage 3 synthetic loss weight 0.5.

5. Сравнить не только macro F1, но и:
   - recall/F1 по `mel`, `akiec`, `bkl`;
   - worst-class recall;
   - MCC;
   - ECE;
   - confusion-delta vs Stage 1;
   - real-vs-synthetic distinguishability.

## Вопросы к исследованию

1. Нужно ли оптимизировать синтетику под визуальное качество, если feature-space gap остается почти идеальным для real-vs-synthetic классификатора?
2. Является ли полезной синтетика, которая очень близка к своему классу, или такие изображения становятся near-duplicate augmentation без новой информации?
3. Можно ли предсказать полезность synthetic sample до обучения по признакам baseline-модели?
4. Какой баланс лучше для медицинской задачи: улучшать worst-class recall или macro F1, если эти цели расходятся?
5. Нужно ли делать отдельный selection policy для каждого класса, потому что `bkl`, `mel` и `akiec` реагируют на синтетику по-разному?

