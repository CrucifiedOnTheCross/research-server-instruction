# Stage 5: per-class synthetic utility experiments

Date: 2026-07-08

## Goal

Stage 5 checks whether the useful part of the synthetic pool can be controlled by class-specific dose instead of a binary "use / do not use" decision.

The motivation comes from Stage 3-4:

- Stage 3 utility-selected synthetic data improved some `mel` behavior but damaged `bkl`.
- Stage 4 `mel+akiec` synthetic data without `bkl` reached the best result so far: test macro F1 `0.8044`, MCC `0.7906`, ECE `0.0784`.
- Therefore the next hypothesis is not "more synthetic images are better", but "synthetic usefulness depends on class, feature location, and dose".

## Literature already used

The Stage 5 plan is based on the papers already added to the project Google Sheet and previous reports:

- He et al., "Deep Residual Learning for Image Recognition" / long-tail medical baseline context.
- Azizi et al., "Synthetic Data from Diffusion Models Improves ImageNet Classification".
- Derm-T2IM, text-to-image diffusion for skin lesion synthesis.
- Trabucco et al., dataset distillation/synthetic usefulness caveats.
- Cao et al., "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss" (LDAM-DRW).
- Ren et al., "Balanced Meta-Softmax for Long-Tailed Visual Recognition".
- Zhu et al., "Balanced Contrastive Learning for Long-Tailed Visual Recognition".

No new papers were introduced in this code sprint, so no new Google Sheet rows were required beyond the Stage 5 planning rows.

## Implemented code

New training capability:

- `training.synthetic_weight_per_class` in config.
- Real samples keep weight `1.0`.
- Synthetic samples can now use a class-specific scalar, for example `mel=0.75`, `akiec=0.5`, `bkl=0.0`.
- Each run writes `synthetic_weight_by_class.json` for reproducibility.

New diagnostics:

- `tools/stage5_hypothesis_diagnostics.py`
- Checks real-vs-synthetic separability by class using features from multiple trained encoders.
- Checks whether top-k utility selection is stable across encoders.
- Clusters real `bkl` features and assigns synthetic `bkl` images to those clusters.

## Diagnostic result

Diagnostics were written to:

`/srv/research/projects/default/ham10000/reports/stage5_diagnostics`

Сводные метрики отчета импортированы в MLflow:
`http://10.200.1.180:5000`. Исходные файлы остаются доступны через JupyterHub.

Key result:

- Real-vs-synthetic detector AUROC is approximately `1.0` for `mel`, `akiec`, and `bkl` across the tested encoders.
- This means the current generated images are highly distinguishable from real images in learned feature space.
- The result does not automatically mean synthetic images are useless: Stage 4 already showed benefit. It means quality/utility must be measured by downstream behavior, class dose, and feature placement rather than visual realism alone.

`bkl` cluster summary with the Stage 1 CE encoder:

| Cluster | Real bkl | Synthetic bkl | Mean synthetic distance |
| --- | ---: | ---: | ---: |
| 0 | 199 | 28 | 0.2893 |
| 1 | 136 | 25 | 0.3697 |
| 2 | 204 | 17 | 0.3445 |
| 3 | 230 | 10 | 0.3044 |

Interpretation: synthetic `bkl` covers several real-feature clusters, but it is still highly separable from real `bkl`. The likely issue is not only missing coverage, but domain/texture artifacts or feature shortcuts.

## Launched experiments

Container:

`research-stage5-per-class`

Script:

`scripts/run_stage5_per_class_weights.sh`

Experiments:

| Experiment | Train pool | Synthetic weights | Hypothesis |
| --- | --- | --- | --- |
| `stage5_mel075_akiec05_bkl0_ce_weighted` | Stage 4 `mel+akiec` | `mel=0.75`, `akiec=0.5`, `bkl=0.0` | More `mel` synthetic signal may recover Stage 3 `mel` gain without reintroducing harmful `bkl`. |
| `stage5_mel05_akiec025_bkl0_ce_weighted` | Stage 4 `mel+akiec` | `mel=0.5`, `akiec=0.25`, `bkl=0.0` | `akiec` may need lower synthetic dose if synthetic artifacts cause overfitting. |
| `stage5_mel05_akiec05_bkl01_ce_weighted` | Stage 3 full utility pool | `mel=0.5`, `akiec=0.5`, `bkl=0.1` | Very weak `bkl` synthetic weight may preserve possible boundary regularization without repeating Stage 3 harm. |

## First technical check

The first run started successfully:

- Run directory: `outputs/stage5_mel075_akiec05_bkl0_ce_weighted/20260708-144754_42`
- GPU utilization observed: `100%`
- VRAM observed: about `7 GB / 16 GB`
- Artifacts already present: `metrics.csv`, `metrics.jsonl`, `val_metrics_best.json`, `val_predictions_best.csv`, `best.pt`, `last.pt`, `config.resolved.yaml`, `synthetic_weight_by_class.json`.

First best validation snapshot observed during training:

| Metric | Value |
| --- | ---: |
| Val macro F1 | 0.7758 |
| Val balanced accuracy | 0.7722 |
| Val MCC | 0.7433 |
| Val ECE | 0.0754 |
| Worst class recall | 0.4485 |
| `mel` recall | 0.7186 |
| `akiec` recall | 0.6939 |
| `bkl` recall | 0.4485 |

This is not a final result. It only confirms that the metrics and artifact writing work correctly.

## What to inspect when runs finish

Primary comparison:

- Compare final test macro F1, MCC, balanced accuracy, ECE against Stage 4 `mel+akiec` and Stage 1 CE.
- Check per-class recall/F1 for `mel`, `akiec`, and especially `bkl`.
- Inspect confusion deltas:
  - Stage 4 vs Stage 5 best run.
  - Stage 1 CE vs Stage 5 best run.

Decision rules:

- If `mel075_akiec05_bkl0` improves `mel` without damaging `bkl`, keep class-specific synthetic weighting.
- If `akiec025` improves calibration or `akiec` precision/recall balance, use lower synthetic dose for artifact-sensitive classes.
- If `bkl01` still harms `bkl`, treat `bkl` synthetic generation as a failed current method and move to better generation/selection, not more training tweaks.

## Next hypotheses if Stage 5 confirms the pattern

- Add artifact-aware rejection: train a real-vs-synthetic detector and reject images with high synthetic probability before utility selection.
- Use source-lesion/session-aware constraints more strongly, because HAM10000 leakage and lesion identity are central to valid medical evaluation.
- Generate fewer but more controlled counterfactuals around hard real examples instead of class balancing by count.
- Test a long-tail loss baseline without synthetic data, such as LDAM-DRW or Balanced Softmax variants, to show whether synthetic augmentation is better than a simpler imbalance method.
