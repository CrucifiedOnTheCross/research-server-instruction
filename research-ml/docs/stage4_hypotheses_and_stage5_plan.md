# Stage 4 interpretation and Stage 5 plan

Дата: 2026-07-08  
Основание: Stage 1–4 results, external deep review, Stage 4 confusion-delta diagnostics.

## Что подтвердил Stage 4

Stage 4 проверял гипотезу, что вред Stage 3 идет не от всей синтетики, а главным образом от `bkl` synthetic samples.

Результаты:

| Run | Synthetic classes | Test macro F1 | Test bal acc | Worst recall | MCC | ECE |
|---|---|---:|---:|---:|---:|---:|
| Stage1 CE weighted | none | 0.7955 | 0.8279 | 0.7305 | 0.7673 | 0.0931 |
| Stage1 balanced softmax | none | 0.7989 | 0.7810 | 0.6347 | 0.7760 | 0.0954 |
| Stage3 utility weight 0.5 | mel, akiec, bkl | 0.7915 | 0.8063 | 0.6061 | 0.7653 | 0.0880 |
| Stage4 mel+akiec weight 0.5 | mel, akiec | 0.8044 | 0.7944 | 0.6407 | 0.7906 | 0.0784 |
| Stage4 mel-only weight 0.5 | mel | 0.7683 | 0.7860 | 0.6471 | 0.7520 | 0.0866 |

Ключевой вывод:

> `mel+akiec` без `bkl` synthetic впервые превзошел лучший Stage 1 по macro F1 и MCC. Это сильный сигнал в пользу class-specific synthetic policy.

Но Stage4 не закрыл все цели:

- balanced accuracy все еще ниже Stage1 CE weighted;
- worst-class recall все еще ниже Stage1 CE weighted;
- `mel-only` не является достаточным решением.

## Confusion-delta механизм

### Stage1 CE weighted vs Stage4 mel+akiec

| Status | Count |
|---|---:|
| both_correct | 1259 |
| candidate_only | 81 |
| base_only | 52 |
| both_wrong | 110 |

Per-class:

| Class | candidate_only | base_only | Вывод |
|---|---:|---:|---|
| akiec | 3 | 3 | Нейтрально. |
| bcc | 4 | 4 | Нейтрально. |
| bkl | 16 | 12 | `bkl` восстановился относительно Stage 3. |
| df | 0 | 2 | Просадка малого класса. |
| mel | 10 | 25 | Stage4 отдал часть mel-выигрыша Stage3. |
| nv | 48 | 4 | Большой плюс на majority class. |
| vasc | 0 | 2 | Просадка малого класса. |

### Stage3 utility weight 0.5 vs Stage4 mel+akiec

| Status | Count |
|---|---:|
| both_correct | 1259 |
| candidate_only | 81 |
| base_only | 55 |
| both_wrong | 107 |

Главное: относительно Stage3 модель Stage4 сильно восстанавливает `bkl`:

- `bkl candidate_only = 35`
- `bkl base_only = 5`

Это почти прямое подтверждение, что включение `bkl` synthetic в Stage3 было главным каналом вреда для `bkl`.

## Обновленная центральная гипотеза диссертации

Старая формулировка "синтетика помогает редким классам" слишком грубая.

Более точная гипотеза:

> Синтетические изображения улучшают редкие и трудноразличимые классы только тогда, когда они согласованы с реальной геометрией класса, покрывают полезную boundary-region, учитывают class-specific многомодальность и входят в обучение с контролируемой, класс-зависимой дозировкой.

## Проверяемые гипотезы Stage 5

### H1. Вред Stage3 был вызван bkl-синтетикой

Текущий статус: частично подтверждено Stage4.

Следующая проверка:

- подкластеризовать real `bkl`;
- измерить, к каким real `bkl` подкластером ближе synthetic `bkl`;
- посмотреть, какие подкластеры ассоциированы с ошибками `bkl -> mel`, `bkl -> nv`, `bkl -> akiec`.

### H2. Utility score смещен baseline-эмбеддингом

Следующая проверка:

- пересчитать synthetic utility score в feature-space:
  - Stage1 CE weighted;
  - Stage1 balanced softmax;
  - Stage4 mel+akiec;
- сравнить top-k overlap для `mel`, `akiec`, `bkl`;
- особенно проверить, нестабилен ли ranking для `bkl`.

Если overlap низкий, selection policy должна стать multi-encoder consensus, а не single-encoder score.

### H3. Synthetic-real gap class-specific

Следующая проверка:

- обучить/посчитать real-vs-synthetic detector по каждому классу отдельно;
- сравнить AUROC/AUPRC для raw, Stage2 topk80, Stage3 utility selected;
- проверить гипотезу:
  - `mel` synthetic менее отделим от real `mel`;
  - `bkl` synthetic сильнее отделим от real `bkl`.

### H4. Нужен per-class synthetic weight

Stage4 показал, что binary class gating работает лучше общего Stage3 pool.

Следующий компактный grid:

| mel weight | akiec weight | bkl weight |
|---:|---:|---:|
| 0.5 | 0.5 | 0.0 |
| 0.75 | 0.5 | 0.0 |
| 0.5 | 0.25 | 0.0 |
| 0.5 | 0.5 | 0.1 |

Запускать сначала single seed. Потом 3 seeds только для 1–2 лучших конфигураций.

### H5. Нужны сильные real-only baselines

Чтобы тезис "class-specific synthetic policy лучше простых методов" был убедительным, нужны дополнительные real-only baselines:

- LDAM-DRW;
- Balanced Contrastive Learning;
- возможно confusion-aware regularization как дополнительный baseline, если хватит времени.

## Источники, добавленные в Google Sheet

Google Sheet: `https://docs.google.com/spreadsheets/d/1AXcfUmUuuwwUTUefzN1XDyViwsK7tWMB2cMqw2mj4Mc/edit`

| Row | Источник | Зачем добавлен |
|---:|---|---|
| 49 | Cao et al., *Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss* | LDAM-DRW как обязательный strong real-only baseline. |
| 50 | Ren et al., *Balanced Meta-Softmax for Long-Tailed Visual Recognition* | Теоретическая основа Stage1 balanced softmax и label-distribution shift. |
| 51 | Zhu et al., *Balanced Contrastive Learning for Long-Tailed Visual Recognition* | Representation-learning baseline для проверки гипотезы о геометрии признаков. |

## Приоритетный план работ

1. Stage5 diagnostics без нового обучения:
   - per-class real-vs-synthetic detector;
   - BKL clustering;
   - multi-encoder utility overlap;
   - source audit hard-wrong vs hard-but-correct.

2. Stage5 compact training:
   - implement `synthetic_weight_per_class`;
   - run 3–4 single-seed per-class weight configs;
   - compare against Stage1 and Stage4.

3. Baseline strengthening:
   - implement LDAM-DRW;
   - plan BCL as separate representation-learning sprint.

4. Final dissertation package:
   - 3 seeds for Stage1 CE weighted;
   - 3 seeds for best real-only long-tail baseline;
   - 3 seeds for best class-specific synthetic policy;
   - optional short real-only fine-tune after synthetic pretraining.

