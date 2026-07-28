# Stage 11: квалификация baseline и решение для Stage 11B

Дата анализа: 2026-07-28.

## Executive summary

Stage 11A завершён: 12/12 runs, четыре архитектурных варианта, seeds 42-44.
Locked test не открывался. Анализ включает 15 matched runs вместе с Stage 8
ConvNeXt-B anchor, 1280 validation изображений, 599 lesion groups и 5000
hierarchical bootstrap повторов.

Основной baseline для следующей проверки — **ConvNeXt-Small**:

- 49.5M параметров против 87.6M у ConvNeXt-Base;
- macro F1 `0.7699 ± 0.0060`;
- MCC `0.6610 ± 0.0114`;
- balanced accuracy `0.7785`;
- macro AUPRC `0.8068`;
- mean runtime 25.9 минуты против 42.5 минуты у Stage 11 ConvNeXt-B;
- превосходит Stage 11 ConvNeXt-B по macro F1, MCC, balanced accuracy,
  macro AUPRC и melanoma F1 на 3/3 matched seeds.

DINOv2-B full получает самый высокий mean macro F1 (`0.7818`), но не является
лучшим основным baseline: MCC ниже (`0.6281`), melanoma F1 ниже (`0.4900`),
melanoma AUPRC ниже (`0.5022`), модель крупнее и медленнее. Его роль —
secondary architecture/representation sensitivity.

Единственная Stage 11B проверка:

> ConvNeXt-Small, `strict_id synthetic` против source-matched real replay,
> seeds 42-44, одинаковый recipe и закрытый test.

## Контроль целостности

- 12/12 Stage 11 runs имеют `summary.json`, resolved config, sampling plan,
  model initialization, class counts, validation metrics/predictions и
  `best.pt`;
- `test_evaluated=false` во всех runs;
- validation population совпадает для всех методов и seeds;
- 1280 уникальных изображений и одинаковые targets;
- split содержит 599 непересекающихся lesion groups;
- два диагностических pilot (`invalid_ema_no_warmup` и
  `invalid_cuda_lost_after_driver_reload`) не имеют `summary.json` и исключены;
- structured artifacts являются каноническим источником; training logs не
  использовались для метрик.

## Multi-seed результаты

| Метод | Macro F1 | MCC | Bal. acc | Macro AUPRC | ECE | Worst recall |
|---|---:|---:|---:|---:|---:|---:|
| Stage 8 ConvNeXt-B anchor | 0.7355 | **0.6598** | 0.7203 | 0.8000 | 0.1296 | 0.4580 |
| Stage 11 ConvNeXt-B | 0.7458 | 0.6384 | 0.7566 | 0.7733 | 0.1404 | 0.5625 |
| **Stage 11 ConvNeXt-S** | **0.7699** | **0.6610** | **0.7785** | **0.8068** | 0.1146 | 0.6667 |
| DINOv2-B linear | 0.5943 | 0.5049 | 0.5820 | 0.6279 | **0.0427** | 0.2500 |
| DINOv2-B full | **0.7818** | 0.6281 | **0.7882** | **0.8167** | **0.1055** | **0.6807** |

Низкий ECE linear probe не компенсирует резкое ухудшение discrimination и
classification utility. Это хороший пример того, почему calibration нельзя
рассматривать отдельно от качества ранжирования и решений.

## Paired seeds

### ConvNeXt-S против Stage 11 ConvNeXt-B

| Метрика | Mean delta | Победы |
|---|---:|---:|
| Macro F1 | +0.0241 | 3/3 |
| MCC | +0.0226 | 3/3 |
| Balanced accuracy | +0.0219 | 3/3 |
| Macro AUPRC | +0.0336 | 3/3 |
| ECE | -0.0259 | 3/3 лучше |
| Melanoma F1 | +0.0259 | 3/3 |

Lesion-group bootstrap CI для этих небольших различий пересекают ноль:
macro F1 `[-0.0120; 0.0616]`, MCC `[-0.0087; 0.0545]`. Поэтому утверждение
должно быть сформулировано как устойчивое practical dominance на трёх seeds,
а не как доказанное универсальное превосходство.

### ConvNeXt-S против Stage 8 anchor

- macro F1: `+0.0346`, bootstrap CI `[-0.0034; 0.0769]`;
- balanced accuracy: `+0.0582`, CI `[+0.0198; +0.0982]`;
- MCC: `+0.0010`, CI `[-0.0506; +0.0536]`;
- macro AUPRC: `+0.0068`, только 1/3 seed;
- melanoma F1: `+0.0229`, 3/3 seeds, но CI пересекает ноль.

Stage 11 recipe в основном улучшает class balance и argmax utility, но не
даёт убедительного общего выигрыша ranking quality.

### DINOv2 full

Против Stage 8 anchor:

- macro F1: `+0.0489`, CI `[+0.0058; +0.0968]`;
- balanced accuracy: `+0.0680`, CI `[+0.0235; +0.1164]`;
- MCC: `-0.0320`, CI `[-0.0823; +0.0146]`;
- melanoma F1: `-0.0285`, CI пересекает ноль;
- melanoma AUPRC: `-0.0552`, хуже на 3/3 seeds.

Против ConvNeXt-S преимущество macro F1 всего `+0.0143` и нестабильно
(2/3 seeds, CI `[-0.0223; +0.0580]`). Одновременно MCC ниже на `0.0330`, а
melanoma F1 ниже на `0.0514`. DINOv2 full не заменяет ConvNeXt-S как основной
baseline.

## Melanoma endpoints

| Метод | Precision | Recall | F1 | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|
| Stage 8 anchor | **0.603** | 0.469 | 0.518 | 0.823 | **0.557** |
| Stage 11 ConvNeXt-B | 0.408 | 0.704 | 0.515 | 0.862 | 0.524 |
| **Stage 11 ConvNeXt-S** | 0.426 | **0.744** | **0.541** | **0.892** | 0.544 |
| DINOv2 linear | 0.341 | 0.464 | 0.389 | 0.762 | 0.378 |
| DINOv2 full | 0.384 | 0.681 | 0.490 | 0.853 | 0.502 |

ConvNeXt-S создаёт более recall-oriented operating point. Однако melanoma
AUPRC всё ещё не превосходит старый anchor. Для статьи необходимо раздельно
сообщать threshold-free ranking и argmax/operating-point utility.

## Calibration и fixed specificity

Exploratory temperature scaling выполнен на group-held-out половине validation
и оценён на другой половине. Полный validation ранее использовался для early
stopping, поэтому это не confirmatory calibration.

Mean ECE после temperature scaling:

- Stage 8 anchor: 0.051;
- Stage 11 ConvNeXt-B: 0.055;
- ConvNeXt-S: 0.053;
- DINOv2 linear: 0.043;
- DINOv2 full: 0.072.

При calibration specificity 0.95 ConvNeXt-S имеет mean melanoma sensitivity
0.505, specificity 0.961 и precision 0.629. DINOv2 full: sensitivity 0.481,
specificity 0.944, precision 0.530. Это поддерживает выбор ConvNeXt-S.

## Вычислительная стоимость

| Метод | Mean runtime/run | Total params | Trainable params |
|---|---:|---:|---:|
| Stage 11 ConvNeXt-B | 42.5 мин | 87.6M | 87.6M |
| **ConvNeXt-S** | **25.9 мин** | **49.5M** | **49.5M** |
| DINOv2 linear | 8.4 мин | 86.1M | 5.4K |
| DINOv2 full | 40.5 мин | 86.1M | 86.1M |

ConvNeXt-S даёт лучший компромисс utility, melanoma endpoints, размера и
времени. DINO linear дешёв по обучаемым параметрам, но недостаточно силён.

## Решение Stage 11B

Зафиксирован один confirmatory screening contrast:

1. Backbone: `convnext_small.fb_in22k_ft_in1k_384`.
2. Recipe: тот же Stage 11 regularized real-only recipe.
3. Arm A: source-matched replay для `strict_id`.
4. Arm B: synthetic `strict_id`.
5. Seeds: 42, 43, 44.
6. Одинаковые sample counts, source mapping, optimizer и early stopping.
7. Primary endpoints: macro F1, MCC, balanced accuracy.
8. Co-primary minority diagnostics: melanoma F1 и AUPRC.
9. Inference: paired seeds и hierarchical lesion-group bootstrap.
10. Locked test остаётся закрытым.

Это проверяет переносимость главной гипотезы Stage 10 на более сильный и
эффективный baseline, не расширяя post-hoc пространство сравнений.

## Ограничения

- только три seeds;
- один внутренний HAM10000 validation split;
- validation участвует в early stopping;
- bootstrap учитывает seeds и lesion groups, но не заменяет внешний cohort;
- архитектуры имеют разные input sizes (384 и 392);
- DINOv2 и ConvNeXt могут по-разному ранжировать synthetic geometry;
- test остаётся закрытым до заранее определённого финального решения.

## Артефакты

`outputs/reports/stage11_analysis/`:

- `analysis_summary.json`;
- `artifact_integrity.csv`;
- `prediction_alignment.csv`;
- `seed_metrics.csv`;
- `method_summary.csv`;
- `paired_seed_comparisons.csv`;
- `hierarchical_lesion_bootstrap.csv`;
- `per_class_metrics.csv`;
- `calibration_summary.csv`;
- `fixed_specificity_diagnostics.csv`;
- `seed_metric_comparison.png`.

## Литературная трассируемость

Новые статьи для этой агрегации не использовались. Интерпретация следует уже
зафиксированным источникам о ConvNeXt, DINOv2, long-tail evaluation,
calibration и synthetic utility. Поэтому новые строки в Google Sheet на этом
шаге не добавлялись.
