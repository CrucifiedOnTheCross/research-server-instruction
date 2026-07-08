# Stage2 Synthetic Gap Findings

Дата: 2026-07-08.

Основание: первый диагностический запуск `tools/analyze_synthetic_gap.py` после внешнего review stage1-stage2.

## Что проверяли

Гипотеза диагностики: если synthetic features легко отделимы от real features, то модель может видеть синтетику как отдельный домен. В этом случае простое смешивание real+synthetic в train set может не улучшать real-only test, даже если картинки выглядят правдоподобно.

Baseline feature extractor:

- `stage1_cross_entropy_weighted`
- run: `outputs/stage1_cross_entropy_weighted/20260708-060820_42`
- backbone: `convnext_tiny.fb_in22k_ft_in1k`

Synthetic pools:

| Pool | Description |
|---|---|
| `raw` | все 960 generated images |
| `strict` | 43 изображения после строгого feature-space отбора |
| `topk80` | 240 изображений после `rule_or_topk80` отбора |

Real-vs-synthetic detector:

- feature vectors from baseline ConvNeXt;
- balanced real sample по тем же target classes;
- logistic regression;
- 5 stratified splits;
- metrics: AUROC, AUPRC.

## Результаты

| Pool | Real n | Synthetic n | Overall AUROC | Overall AUPRC |
|---|---:|---:|---:|---:|
| `raw` | 869 | 960 | **1.0000** | **1.0000** |
| `strict` | 43 | 43 | **0.9811** | **0.9784** |
| `topk80` | 240 | 240 | **1.0000** | **1.0000** |

Per-class detector:

| Pool | Class | Real n | Synthetic n | AUROC | AUPRC |
|---|---|---:|---:|---:|---:|
| `raw` | `akiec` | 229 | 320 | 1.0000 | 1.0000 |
| `raw` | `bkl` | 320 | 320 | 0.9999 | 0.9999 |
| `raw` | `mel` | 320 | 320 | 1.0000 | 1.0000 |
| `strict` | `bkl` | 28 | 28 | 0.9750 | 0.9673 |
| `strict` | `mel` | 15 | 15 | 1.0000 | 1.0000 |
| `topk80` | `akiec` | 80 | 80 | 1.0000 | 1.0000 |
| `topk80` | `bkl` | 80 | 80 | 0.9979 | 0.9980 |
| `topk80` | `mel` | 80 | 80 | 1.0000 | 1.0000 |

Артефакты на сервере:

- `project-default/ham10000/reports/stage2_diagnostics/index.html`
- `project-default/ham10000/reports/stage2_diagnostics/real_vs_synthetic_metrics.json`
- `project-default/ham10000/reports/stage2_diagnostics/pca_real_synthetic_raw.png`
- `project-default/ham10000/reports/stage2_diagnostics/pca_real_synthetic_strict.png`
- `project-default/ham10000/reports/stage2_diagnostics/pca_real_synthetic_topk80.png`
- `project-default/ham10000/reports/stage2_diagnostics/real_vs_synthetic_rows.csv`

## Интерпретация

Это сильное подтверждение synthetic-real gap.

Даже после strict selection синтетические features остаются почти идеально отделимыми от real features (`AUROC=0.9811`). Для raw и topk80 separation практически идеальный (`AUROC=1.0`). Это объясняет, почему stage2 не превзошел strong real-only baseline: классификатор получал не просто дополнительные варианты реальных классов, а отдельный synthetic domain.

Текущий результат усиливает вывод из `stage1_stage2_research_summary.md`: проблема не в том, что синтетики было мало или что model training не загрузил GPU. Проблема в том, что текущий generator/protocol создает данные, которые в embedding-space отличаются от real train.

## Что это меняет в плане

Нельзя считать `topk80` хорошим компромиссом только потому, что он дает больше coverage. Он дает coverage, но сохраняет почти идеальную real/synthetic detectability.

Нельзя запускать "больше такой же генерации". Вероятнее всего, это увеличит synthetic domain, а не real manifold coverage.

Следующий stage3 должен проверять не количество synthetic images, а снижение synthetic-real gap.

## Следующие действия

1. Построить kNN gallery:
   - synthetic image;
   - nearest real same-class;
   - nearest real confusing-class;
   - distances/margins.

2. Построить confusion-delta report:
   - stage1 CE weighted vs stage2 topk80 CE weighted;
   - stage1 Balanced Softmax vs stage2 Balanced Softmax variants.

3. Запустить prompt audit:
   - token length;
   - negative prompt truncation;
   - shorter negative prompt candidates.

4. Для stage3:
   - использовать boundary-conditioned source images;
   - снижать img2img strength;
   - проверять synthetic weight `< 1`;
   - добавить final real-only fine-tune;
   - рассмотреть synthetic-aware/domain-aware loss.

## Решение

Текущий synthetic path не закрывается, но его нужно менять. Наиболее вероятная формулировка научного вклада теперь:

> Синтетические изображения полезны при long-tailed medical image classification не тогда, когда они визуально похожи на целевой домен, а когда они не образуют отдельный synthetic domain в признаковом пространстве и улучшают coverage проблемных class-boundary regions.

