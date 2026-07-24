# Stage 8: полный статистический анализ и план статьи

Дата анализа: 2026-07-24.

## Техническое резюме

Stage 8 не подтвердил добавочную пользу текущей synthetic+DINO ветки относительно
простого повторного предъявления реальных minority-примеров:

- `Real CE weighted`: macro F1 `0.7653 +/- 0.0230`, MCC `0.6404 +/- 0.0233`;
- `Synthetic+DINO weighted`: macro F1 `0.7472 +/- 0.0142`,
  MCC `0.6028 +/- 0.0327`;
- paired delta synthetic minus weighted: macro F1 `-0.0181`,
  MCC `-0.0376`;
- синтетика повысила mean melanoma recall на `0.0420`, но снизила melanoma
  precision на `0.0694` и melanoma F1 на `0.0420`.

Интервалы неопределенности перекрывают ноль для большинства сравнений. Это не
доказательство эквивалентности: заранее заданная equivalence margin отсутствует,
а пять seed недостаточны для сильного confirmatory inference.

Главный научный сигнал состоит в том, что синтетика сдвинула operating point в
сторону большей чувствительности к melanoma, но не улучшила общую discriminative
utility сверх weighted oversampling. Следующий этап обязан отделить threshold
shift, class-prior effect, representation gain и generator mismatch.

## Аудит данных и воспроизводимости

Анализ выполнен по структурированным артефактам, без извлечения результатов из
шумных логов:

- `summary.json`;
- `val_metrics_best.json`;
- `val_predictions_best.csv`;
- `config.resolved.yaml`;
- `class_counts.json`;
- `environment.json`;
- Stage 8 split summary, artifact audit, DINO geometry и multi-seed report.

Проверки:

| Проверка | Результат |
|---|---|
| Методов | 4 |
| Seed на метод | 5 (`42-46`) |
| Validation predictions на запуск | 1280 |
| Validation lesion groups | 599 |
| Synthetic rows в validation | 0 |
| Набор `image_id/target` между методами | Идентичен |
| Locked test | Не вычислялся |
| Source leakage из img2img в validation/test | Не обнаружен протоколом Stage 8 |

Validation support мал для `df=16`, `vasc=18`, `akiec=42`; seed-повторы не
оценивают неопределенность выбора split. Поэтому Stage 8 является screening, а
не финальным подтверждением.

## Методы Stage 8

| Метод | Sampling | Loss | Синтетика |
|---|---|---|---|
| Real CE natural | Natural | CE | Нет |
| Real CE weighted | Weighted random sampler | CE | Нет |
| Real Logit Adjustment natural | Natural | LA | Нет |
| Synthetic+DINO weighted | Weighted random sampler | CE | 240, weight `0.5` |

`Real CE weighted` является прямым oversampling control: synthetic-ветка
использует тот же sampler, но добавляет синтетическое содержание. Явного
random-undersampling control в Stage 8 нет.

## Средние результаты по пяти seed

| Метод | Macro F1 | Bal. acc | MCC | ECE | Worst recall | Mel recall |
|---|---:|---:|---:|---:|---:|---:|
| Real CE natural | 0.7454 | 0.7346 | 0.6518 | 0.1495 | 0.4873 | 0.5175 |
| Real CE weighted | **0.7653** | **0.7596** | **0.6404** | **0.1592** | **0.5955** | 0.5986 |
| Real LA natural | 0.7328 | 0.7457 | 0.6246 | 0.1756 | 0.5481 | 0.5986 |
| Synthetic+DINO weighted | 0.7472 | 0.7493 | 0.6028 | 0.1682 | 0.5864 | **0.6406** |

Жирное выделение обозначает лучший показатель только внутри этой screening
таблицы. ECE не recalibrated и не должен интерпретироваться отдельно от
temperature scaling.

## Paired seed analysis

Все методы сопоставлены по одинаковым seed.

| Метрика | Delta synth - weighted | Seed-t 95% CI | Победы synth | Exact sign-flip p |
|---|---:|---:|---:|---:|
| Macro F1 | -0.0181 | [-0.0513; 0.0151] | 1/5 | 0.1875 |
| Balanced accuracy | -0.0103 | [-0.0375; 0.0169] | 1/5 | 0.3125 |
| MCC | -0.0376 | [-0.0964; 0.0213] | 2/5 | 0.2500 |
| Mel precision | -0.0694 | [-0.1920; 0.0532] | 1/5 | 0.1875 |
| Mel recall | +0.0420 | [-0.1030; 0.1869] | 4/5 | 0.1875 |
| Mel F1 | -0.0420 | [-0.1281; 0.0442] | 2/5 | 0.3750 |

При `n=5` минимальное возможное двустороннее p exact sign-flip test равно
`0.0625`. Значимость не достигнута; отсутствие значимости не доказывает
отсутствие эффекта.

## Hierarchical bootstrap по seed и lesion group

Bootstrap повторно выбирает seed и целые lesion groups, сохраняя зависимость
между изображениями одного поражения. Выполнено 5000 итераций.

### Synthetic против weighted oversampling

| Метрика | Mean delta | 95% bootstrap CI | P(delta > 0) |
|---|---:|---:|---:|
| Macro F1 | -0.0182 | [-0.0530; 0.0154] | 0.140 |
| Balanced accuracy | -0.0107 | [-0.0447; 0.0200] | 0.249 |
| MCC | -0.0376 | [-0.0811; 0.0069] | 0.050 |
| Mel precision | -0.0691 | [-0.1497; 0.0230] | 0.067 |
| Mel recall | +0.0417 | [-0.0733; 0.1392] | 0.781 |
| Mel F1 | -0.0418 | [-0.1075; 0.0280] | 0.112 |

### Synthetic против natural sampling

| Метрика | Mean delta | 95% bootstrap CI | P(delta > 0) |
|---|---:|---:|---:|
| Macro F1 | +0.0017 | [-0.0272; 0.0296] | 0.546 |
| Balanced accuracy | +0.0143 | [-0.0194; 0.0491] | 0.806 |
| MCC | -0.0497 | [-0.0918; -0.0073] | 0.011 |
| Mel precision | -0.1610 | [-0.2676; -0.0668] | 0.0004 |
| Mel recall | +0.1230 | [0.0426; 0.2051] | 0.999 |
| Mel F1 | -0.0460 | [-0.1069; 0.0089] | 0.048 |

Против natural sampling рост melanoma recall выражен сильнее, но оплачивается
precision и MCC. Против matched weighted oversampling evidence of incremental
recall слабее, а общей utility нет.

## Проверка гипотезы «это только порог»

Для каждого seed к logits real-only моделей добавлялся offset класса `mel`.
Выбиралась точка с melanoma recall не ниже synthetic-модели, затем
максимизировались macro F1, MCC или mel-F1.

`Real CE weighted` после такого descriptive offset:

- доминировал synthetic по mel recall, mel precision, macro F1 и MCC в `3/5`
  seed;
- при выборе по macro F1 имел среднее преимущество над synthetic:
  macro F1 `+0.0020`, MCC `+0.0133`, mel-F1 `+0.0165`;
- при выборе по MCC имел среднее преимущество:
  macro F1 `+0.0019`, MCC `+0.0168`, mel-F1 `+0.0189`.

Это сильный диагностический сигнал пороговой эквивалентности, но не
confirmatory результат: offset оптимизирован и оценен на той же validation
выборке, которая использовалась для early stopping. В следующем этапе threshold
должен подбираться только на inner validation и проверяться на outer fold.

## Геометрия и качество синтетики

Сгенерировано 1440 кандидатов, выбрано 240: по 80 для `mel`, `akiec`, `bkl`.

Pixel audit не обнаружил прежний черный border failure:
`black_border_share` AUROC real-vs-synthetic около `0.505`. При этом
`border_std` AUROC около `0.714`, а `center_std` около `0.655`, то есть остаточный
low-level domain gap сохраняется.

DINOv2 PRDC:

| Класс | Precision | Coverage | Прошло strict filter | Взято |
|---|---:|---:|---:|---:|
| mel | 0.125 | 0.076 | 35 | 80 |
| akiec | 0.169 | 0.272 | 36 | 80 |
| bkl | 0.260 | 0.120 | 63 | 80 |

Для достижения квоты использовался top-k fallback. Следовательно, ветка не
является чистым тестом strict фильтра. Низкая precision/coverage согласуется с
отсутствием общего downstream выигрыша, но причинная связь пока не доказана.

## Ответ на вопрос об oversampling и undersampling

### Что уже известно

Synthetic+DINO не превзошел matched random oversampling. Это означает, что
текущая синтетика не добавила измеримую общую информацию сверх повторного
предъявления реальных minority samples.

### Что еще неизвестно

Stage 8 не содержит random undersampling. Поэтому утверждение «результат такой
же, как undersampling» пока невозможно. Undersampling может:

- ускорить обучение;
- уменьшить доминирование `nv`;
- потерять полезное разнообразие head-класса;
- изменить calibration и operating point.

Его нужно сравнить при двух бюджетах:

1. одинаковое число optimizer steps;
2. одинаковое число уникальных real images/epochs.

Без этих двух режимов сравнение будет смешивать метод балансировки и compute.

## Какие выводы допустимы для статьи

Поддерживаются:

1. Исправление border artifact и leakage-safe protocol не сделало текущую
   синтетику автоматически полезной.
2. Независимый DINOv2 top-k filter при квоте 80/class не превзошел weighted
   oversampling.
3. Наблюдаемый рост melanoma recall сопровождается снижением precision, mel-F1
   и MCC.
4. Threshold-equivalence является правдоподобным альтернативным объяснением.
5. Визуальная правдоподобность и отсутствие грубого артефакта недостаточны для
   downstream utility.

Пока не поддерживаются:

1. «Синтетика в целом бесполезна».
2. «Синтетика статистически эквивалентна oversampling».
3. «DINO distance причинно предсказывает utility».
4. «Метод улучшает обобщение на HAM10000 test или внешнем датасете».
5. «Рост recall клинически полезен» без фиксированной specificity/precision и
   анализа operating point.

## Научная гипотеза статьи

Рабочая гипотеза:

> Полезность синтетической аугментации при long-tail дерматоскопической
> классификации определяется не визуальным качеством само по себе, а
> совместным положением synthetic samples в независимом feature space,
> эффективной дозой, способом использования и добавочной информацией сверх
> class-prior/threshold controls.

Возможная новизна для «Компьютерной оптики»:

- lesion-aware leakage-safe protocol;
- разложение prior effect, threshold effect и representation effect;
- геометрическое дозирование `strict-ID / AID / OOD`;
- связь sample-level geometry с measured downstream utility;
- отрицательный, но воспроизводимый результат для off-the-shelf img2img и
  условия, при которых генерация начинает или не начинает работать.

## План Stage 9

### Sprint 9A. Сильные real-only и resampling controls

Фиксируются один split, ConvNeXt Base 384, augmentation, optimizer, scheduler и
compute budget.

Методы:

1. Natural CE.
2. Random oversampling CE.
3. Random undersampling CE.
4. Class-weighted CE или Balanced Softmax.
5. Logit Adjustment с natural sampler.
6. cRT: natural representation, balanced classifier head.
7. Source-matched replay: повторять реальные источники ровно в количестве и с
   весом, соответствующими synthetic manifest.

Primary: macro F1 и MCC. Guardrails: balanced accuracy, worst-class recall, ECE.
Secondary: per-class AUPRC/AUROC, melanoma sensitivity at fixed specificity.

### Sprint 9B. Nested threshold and calibration control

- inner validation: temperature scaling и mel logit offset;
- outer fold: единственная оценка выбранного порога;
- curves: melanoma sensitivity-specificity, precision-recall, decision curve;
- сравнение ranking и fixed-threshold utility.

Гипотеза H9B: synthetic recall gain исчезнет после честной threshold
calibration weighted real-only модели.

### Sprint 9C. Геометрическая доза

Для каждого класса создать непересекающиеся равные по размеру strata:

- `strict-ID`: ближайшая надежная область real manifold;
- `AID`: промежуточная distance/margin band;
- `OOD`: дальняя область;
- random eligible synthetic control.

Сравнивать при одинаковом количестве, synthetic weight и optimizer steps.
Запрещено добирать strict stratum top-k примерами из другой зоны.

Гипотеза H9C: utility немонотонна по расстоянию; AID может быть полезнее
strict-ID и OOD.

### Sprint 9D. Dose-response и utilization

Дозы: `0`, `0.1`, `0.25`, `0.5`, `1.0` относительно выбранного synthetic pool.
Для каждой дозы:

- natural sampler;
- weighted sampler;
- concatenation;
- replacement/source-matched replay.

Использовать successive halving: один seed для отсечения явно слабых веток,
три seed для shortlist, outer CV только для финалистов.

### Sprint 9E. Подтверждение

- group-aware 5-fold CV по `lesion_id`;
- одинаковые folds и seeds для paired comparison;
- минимум 2 seeds/fold для финалистов;
- hierarchical bootstrap whole lesion groups;
- exact paired permutation или mixed-effects model;
- Holm correction для secondary endpoints;
- заранее задать non-inferiority/equivalence margin;
- locked test открыть один раз после freeze;
- внешний test очистить от HAM10000/ISIC overlap по ID и perceptual hash.

## Критерий перехода к новому генератору

Переход к train-only LoRA/inpainting оправдан, если Stage 9A-D покажет хотя бы
одно из двух:

1. текущая synthetic utility связана с feature-space stratum или дозой, но
   ограничена distribution mismatch;
2. source-matched replay и calibrated threshold не воспроизводят выигрыш
   выбранной synthetic ветки.

Если эффект полностью воспроизводится resampling/threshold controls, следующий
научный шаг должен быть representation learning, а не более тяжелая генерация.

## План рукописи

1. Введение: почему визуального качества недостаточно.
2. Related work: imbalance controls, synthetic utility, medical diffusion,
   feature geometry.
3. Данные и leakage-safe split по lesion.
4. Генерация, pixel audit и независимый feature filter.
5. Controlled utilization matrix.
6. Статистика и calibration protocol.
7. Результаты: utility, operating point, geometry-dose relation.
8. Ablation: oversampling, undersampling, replay, threshold, strict/AID/OOD.
9. Ограничения: один домен, small rare classes, generator dependence.
10. Reproducibility: configs, manifests, seeds, environment, immutable reports.

## Артефакты анализа

Код:

- `tools/analyze_stage8_article.py`.

Локальный structured snapshot:

- `local_artifacts/stage8_article_analysis/`.

Выходы:

- `analysis/analysis_summary.json`;
- `analysis/stage8_seed_metrics.csv`;
- `analysis/prediction_alignment_checks.csv`;
- `analysis/paired_seed_comparisons.csv`;
- `analysis/hierarchical_bootstrap_comparisons.csv`;
- `analysis/mel_offset_frontier.csv`;
- `analysis/stage8_seed_comparison.png`.

Литературный конспект:

- `docs/literature_update_synthetic_utility_and_long_tail_2026-07-24.md`.

## Решение

Текущий материал дает содержательный отрицательный результат и сильную
методологическую основу, но еще не завершенную статью. Приоритет следующего
этапа: доказать, существует ли добавочная synthetic information после
resampling, replay и nested threshold controls. Только после этого имеет смысл
тратить GPU-бюджет на новый генератор.
