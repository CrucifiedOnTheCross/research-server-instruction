# Stage 10: geometry-conditioned synthetic utility and Stage 11 decision

Дата: 2026-07-28.

## Executive conclusion

Stage 10 завершён: 24/24 matched screening runs, четыре geometry strata,
synthetic и source-matched replay, seeds 42-44. Locked test не открывался.

Главный результат:

> Downstream utility синтетических дерматоскопических изображений зависит от
> их положения относительно real-class feature manifold. Близкий `strict_id`
> stratum улучшает MCC и ranking metrics относительно показа тех же source
> lesions, тогда как `random_remaining` статистически ухудшает MCC и mel F1.

Это не подтверждает универсальное превосходство synthetic augmentation.
`strict_id` сохраняется как единственный кандидат Stage 11B с обязательным
контролем mel operating point.

## Protocol integrity

- 4 strata: `strict_id`, `aid_radial`, `ood_far`, `random_remaining`;
- 2 arms: synthetic и source-matched replay;
- 3 matched seeds: 42, 43, 44;
- одинаковая synthetic/replay dose: 90 добавленных строк на stratum;
- validation: 1280 изображений, 599 lesion groups;
- validation population и target labels совпадают во всех 24 runs;
- `test_evaluated=false` во всех `summary.json`;
- обязательные config, sampling, initialization, prediction и checkpoint
  artifacts присутствуют;
- независимый пересчёт метрик из predictions: max absolute error `1.08e-7`;
- hierarchical bootstrap повторно выбирает seeds и целые lesion groups,
  5000 replicates.

Канонические результаты:

`outputs/reports/stage10_analysis/`

- `seed_metrics.csv`;
- `method_summary.csv`;
- `paired_seed_comparisons.csv`;
- `hierarchical_lesion_bootstrap.csv`;
- `per_class_discrimination.csv`;
- `calibration_summary.csv`;
- `fixed_specificity_diagnostics.csv`;
- `artifact_integrity.csv`;
- `prediction_alignment.csv`;
- `analysis_summary.json`.

## Paired synthetic-minus-replay results

Значения ниже являются absolute validation deltas. Bootstrap CI учитывает
неопределённость по трём seeds и целым lesion groups.

| Stratum | Metric | Mean delta | 95% lesion bootstrap CI | P(delta > 0) |
|---|---|---:|---:|---:|
| strict_id | Macro F1 | +0.0263 | [-0.0236; +0.0697] | 0.877 |
| strict_id | MCC | **+0.0661** | **[+0.0047; +0.1255]** | 0.984 |
| strict_id | Balanced accuracy | -0.0005 | [-0.0535; +0.0493] | 0.496 |
| strict_id | mel precision | +0.1193 | [-0.0055; +0.2318] | 0.970 |
| strict_id | mel recall | -0.0765 | [-0.1939; +0.0229] | 0.067 |
| strict_id | mel F1 | +0.0553 | [-0.0269; +0.1356] | 0.917 |
| aid_radial | Macro F1 | +0.0025 | [-0.0407; +0.0474] | 0.551 |
| aid_radial | MCC | +0.0167 | [-0.0229; +0.0567] | 0.806 |
| aid_radial | mel F1 | +0.0439 | [-0.0143; +0.1098] | 0.928 |
| ood_far | Macro F1 | -0.0241 | [-0.0572; +0.0069] | 0.063 |
| ood_far | MCC | -0.0282 | [-0.0665; +0.0077] | 0.059 |
| random_remaining | Macro F1 | -0.0194 | [-0.0620; +0.0167] | 0.145 |
| random_remaining | MCC | **-0.0488** | **[-0.0856; -0.0131]** | 0.005 |
| random_remaining | mel precision | **-0.0864** | **[-0.1350; -0.0402]** | 0.0002 |
| random_remaining | mel F1 | **-0.0688** | **[-0.1119; -0.0234]** | 0.0016 |

## Ranking versus threshold effects

`strict_id`:

- mean macro AUPRC delta `+0.0174`;
- знак положителен на 3/3 seeds;
- seed-level 95% t interval `[+0.0038; +0.0309]`;
- per-class AUPRC положителен для `akiec`, `bcc`, `bkl`, `df`, `mel` и `nv`;
- raw mel precision растёт, но raw mel recall падает на 3/3 seeds.

Следовательно, `strict_id` улучшает ranking/separation и MCC, но одновременно
сдвигает стандартный argmax operating point в сторону precision. Рост Macro F1
нельзя интерпретировать как доказанное улучшение чувствительности к melanoma.

`aid_radial`:

- global Macro F1 практически не меняется;
- mel F1 delta положителен на 3/3 seeds, но CI пересекает ноль;
- при exploratory fixed specificity 0.90 mean mel sensitivity составляет
  примерно `0.61` против `0.52` у replay;
- при specificity 0.95: `0.47` против `0.43`.

Это поддерживает operating-point гипотезу, но недостаточно для отдельного
Stage 11B arm.

`ood_far`:

- Macro F1 проигрывает на 3/3 seeds;
- seed-level 95% interval `[-0.0401; -0.0065]`;
- lesion bootstrap CI почти полностью отрицателен, но включает ноль.

`random_remaining`:

- проигрывает по MCC и mel F1 с bootstrap CI полностью ниже нуля;
- показывает, что equal dose и source matching не делают произвольную
  синтетику безопасной.

## Calibration diagnostics

Для каждого run выполнен exploratory group-held-out calibration split:
temperature и thresholds выбирались на одной половине lesion groups и
оценивались на другой.

Temperature scaling снижает ECE примерно до `0.05-0.07` во всех arms.
Калибровка не устраняет различия ranking и не превращает `random_remaining` в
полезный stratum.

Ограничение: весь validation split ранее использовался для early stopping,
поэтому calibration diagnostic не является независимым confirmatory test.

## Scientific interpretation

Stage 10 отвергает две упрощённые формулировки:

1. визуально качественная синтетика автоматически полезна;
2. synthetic gain можно объяснить дополнительными показами source lesions.

Matched replay control показывает дополнительный эффект generated pixels.
Однако знак эффекта определяется geometry stratum:

- близкий и identity-preserving stratum может улучшить learned separation;
- radial/intermediate samples преимущественно меняют mel operating point;
- far/random samples добавляют вредный distribution shift.

Наиболее защищаемый тезис статьи:

> Feature-space geometry является модератором downstream utility synthetic
> dermoscopic augmentation, а fidelity или количество synthetic samples сами
> по себе недостаточны.

## Stage 11 decision

Выбрана **ветка B с guardrail ветки D**:

- Stage 11A полностью квалифицирует real-only baseline;
- Stage 11B остаётся заблокированным до выбора baseline;
- единственная разрешённая utility-пара Stage 11B:
  `strict_id synthetic` versus `source-matched replay`;
- dose, sources, seeds и split сохраняются;
- `aid_radial`, `ood_far` и `random_remaining` не переносятся;
- primary Macro F1 дополняется MCC, macro/mel AUPRC и mel fixed-specificity
  diagnostics;
- падение mel recall запрещено скрывать ростом precision.

Stage 11A сохраняет четыре заранее подготовленных arms:

1. regularized ConvNeXt-Base;
2. regularized ConvNeXt-Small;
3. DINOv2-B/14 frozen linear probe;
4. DINOv2-B/14 full fine-tuning.

## Limitations

- один HAM10000 lesion-group split;
- только три seeds;
- validation использован и для early stopping, и для screening inference;
- bootstrap не создаёт независимую внешнюю когорту;
- нет внешней ISIC cohort;
- Stage 10 сравнивает geometry strata внутри одной ConvNeXt-B representation;
- locked test остаётся закрытым до окончательной фиксации метода.

До статьи winning comparison необходимо подтвердить repeated group-aware
splits или cross-validation и затем один раз оценить на locked test.

## Literature traceability

Новые статьи при анализе результатов не вводились. Интерпретация использует
источники, заранее зафиксированные в Stage 10/11 protocols и Google Sheet:

- Sajjadi et al., generative precision/recall;
- Kynkäänniemi et al., improved precision/recall;
- Naeem et al., density and coverage;
- MONICA long-tailed medical benchmark;
- ConvNeXt;
- DINOv2;
- Kumar et al. on fine-tuning feature distortion.

Поэтому новые строки Google Sheet для этого analysis sprint не требуются.

## Code and reproducibility

Добавлены:

- `tools/analyze_stage10_results.py`;
- `scripts/run_stage10_analysis.sh`.

Анализ не извлекает метрики из training logs. Он использует structured
JSON/CSV artifacts, пересчитывает метрики, проверяет population alignment и
сохраняет все промежуточные таблицы.
