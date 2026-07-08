# Stage 3 implementation brief

Дата: 2026-07-08  
Основание: Stage 1 real-only baselines, Stage 2 synthetic augmentation, diagnostics sprint, external deep review 2.

## Главный вывод для следующего спринта

Нельзя продолжать линию "сгенерировать больше synthetic images" без смены критерия полезности. Stage 2 уже показал, что визуально правдоподобная и feature-space отобранная синтетика не гарантирует улучшения test-качества, особенно на `mel` и `bkl`.

Stage 3 должен проверять другую гипотезу:

> Synthetic images полезны только тогда, когда они class-aware, boundary-aware и discrepancy-aware: покрывают реальные hard cases, не превращаются в near-duplicates, не уходят к confusing-классам и вводятся в обучение с контролируемым весом.

## Что фиксируем перед новыми запусками

Минимальный аналитический долг:

1. Собрать единый run registry по всем Stage 1/2 runs.
2. Сохранить per-class precision/recall/F1 для всех запусков.
3. Сохранить confusion matrices и prediction CSV для лучших baseline и Stage 2.
4. Явно указать seed policy: Stage 3 сначала один seed для smoke/ablation, затем 3 seeds только для финальных 2-3 конфигураций.
5. Зафиксировать primary metric до запуска:
   - основная: macro F1;
   - safety/co-primary: worst-class recall;
   - обязательные secondary: balanced accuracy, MCC, ECE, per-class recall/F1 для `mel`, `akiec`, `bkl`.

## Stage 3: минимальная экспериментальная матрица

Baseline остается неизменным:

- `stage1_cross_entropy_weighted`
- `stage1_balanced_softmax_none`

Новые candidate-конфигурации:

| ID | Цель | Synthetic source | Selection | Training policy |
|---|---|---|---|---|
| S3-A | Проверить hard-case generation | только hard train cases `mel/akiec/bkl` | utility score top-k per class | CE weighted, synthetic weight 0.5 |
| S3-B | Проверить smaller synthetic dose | hard train cases `mel/akiec/bkl` | utility score top-k per class | CE weighted, synthetic weight 0.25 |
| S3-C | Проверить final real-only fine-tune | лучший из S3-A/S3-B | тот же selected pool | synthetic pretrain + short real-only fine-tune |

Пока не запускать широкую матрицу `balanced_softmax + synthetic`, если только S3-A/S3-B не дадут ясный сигнал.

## Hard-case mining

Для каждого real train/val изображения baseline должен сохранить:

- `image_id`
- `lesion_id`
- `split`
- `target`
- `pred`
- `is_correct`
- `confidence`
- `entropy`
- `top1_prob`
- `top2_prob`
- `top1_top2_margin`
- `loss`

Кандидаты для генерации:

- target in `mel, akiec, bkl`;
- приоритет 1: hard-but-correct, то есть correct samples with low confidence / high entropy;
- приоритет 2: hard-wrong, то есть ошибочные samples, но только если они визуально и метаданно валидны;
- не использовать val/test как source;
- не нарушать `lesion_id` split isolation.

## Utility score для synthetic sample

Для каждого synthetic sample логировать:

- `source_image_id`
- `source_lesion_id`
- `target`
- `prompt`
- `negative_prompt`
- `prompt_token_count`
- `seed`
- `strength`
- `guidance_scale`
- `num_inference_steps`
- `same_class_distance`
- `confusing_class_distance`
- `source_distance`
- `feature_margin = confusing_class_distance - same_class_distance`
- `baseline_entropy`
- `baseline_confidence`
- `selected`
- `selection_score`

Начальный selection score:

```text
score =
  + feature_margin
  + 0.25 * normalized_entropy
  - 0.25 * duplicate_penalty
  - 0.50 * negative_margin_penalty
```

Где:

- `feature_margin > 0` желательно;
- `duplicate_penalty` растет, если synthetic слишком близок к source image;
- `negative_margin_penalty` включается, если synthetic ближе к confusing-классу, чем к своему.

## Training policy

Synthetic images нельзя смешивать как полностью равные real examples без учета происхождения.

Минимальная реализация:

- в dataset добавить поле `is_synthetic`;
- в loss добавить `synthetic_weight`;
- логировать долю synthetic в каждом epoch;
- сохранять effective class counts с учетом веса;
- для S3-C выполнить короткий final fine-tune только на real train.

## Success criteria

Stage 3 считается успешным только если candidate:

1. Улучшает macro F1 относительно выбранного Stage 1 baseline.
2. Не ухудшает worst-class recall.
3. Не ухудшает `mel` recall/F1.
4. Имеет приемлемую calibration: ECE не хуже Stage 1 больше чем на малый допуск.
5. Confusion-delta показывает больше `candidate_only`, чем `base_only` на `mel/akiec/bkl` в сумме.

## Что подготовить в коде

1. `tools/export_hard_cases.py`
   Экспортирует hard real train/val cases по baseline predictions.

2. `tools/score_synthetic_utility.py`
   Считает feature distances, entropy, duplicate penalty и selection score.

3. Dataset changes
   Поддержка `is_synthetic` и `sample_weight`/`synthetic_weight`.

4. Configs
   - `configs/stage3_hardcase_synthetic_weight025.yaml`
   - `configs/stage3_hardcase_synthetic_weight05.yaml`
   - `configs/stage3_real_finetune.yaml`

5. Reports
   - kNN gallery для Stage 3 selected pool;
   - confusion-delta vs Stage 1;
   - per-class result table;
   - synthetic utility distribution.

