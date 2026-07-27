# Stage 6: feature geometry, selected synthetic visualizations, and regeneration criteria

Date: 2026-07-08

## Stage 5 outcome

Stage 5 finished successfully. Results were collected from structured artifacts (`summary.json`, `test_metrics.json`, `val_metrics_best.json`, `metrics.csv`), not from noisy logs.

| Experiment | Test macro F1 | Test bal acc | Test MCC | Test ECE | akiec recall | bkl recall | mel recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage 1 CE weighted | 0.7955 | 0.8279 | 0.7673 | 0.0931 | 0.7347 | 0.7636 | 0.7305 |
| Stage 3 utility, weight 0.5 | 0.7915 | 0.8063 | 0.7653 | 0.0880 | 0.6939 | 0.6061 | 0.7904 |
| Stage 4 mel+akiec, weight 0.5 | 0.8044 | 0.7944 | 0.7906 | 0.0784 | 0.7347 | 0.7879 | 0.6407 |
| Stage 5 mel0.75 akiec0.5 bkl0 | 0.7875 | 0.7971 | 0.7744 | 0.0705 | 0.6735 | 0.7394 | 0.6587 |
| Stage 5 mel0.5 akiec0.25 bkl0 | 0.7771 | 0.7987 | 0.7407 | 0.0824 | 0.7755 | 0.7697 | 0.5509 |
| Stage 5 mel0.5 akiec0.5 bkl0.1 | 0.7850 | 0.7737 | 0.7786 | 0.0790 | 0.5714 | 0.7697 | 0.6707 |

Interpretation:

- Stage 5 did not beat Stage 4 by final test macro F1.
- Stronger `mel` dose improved calibration but did not recover Stage 3 `mel` recall.
- Weak `bkl=0.1` did not solve the problem, so the next step should not be more scalar weighting. It should inspect feature-space geometry and generation quality.

## Literature used for Stage 6

The following sources were added to the Google Sheet:

- Sajjadi et al., "Assessing Generative Models via Precision and Recall", NeurIPS 2018.
- Kynkäänniemi et al., "Improved Precision and Recall Metric for Assessing Generative Models", NeurIPS 2019.
- Naeem et al., "Reliable Fidelity and Diversity Metrics for Generative Models", ICML 2020.

They motivate separating:

- fidelity / precision: whether synthetic samples lie inside the real class manifold;
- diversity / recall / coverage: whether synthetic samples cover the real class manifold;
- density: whether synthetic samples are concentrated in useful regions or sparse/off-domain areas.

## Stage 6 geometry diagnostics

Report:

`/srv/research/projects/default/ham10000/reports/stage6_feature_geometry/index.html`

Сводные метрики отчета импортированы в MLflow:
`http://10.200.1.180:5000`.

Key numbers with Stage 1 CE feature encoder:

| Class | Synthetic pool | PRDC precision | Density | Coverage | NN p80 | Duplicate rate | p10 margin | Passed geometry filter |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mel | 320 | 0.1187 | 0.0338 | 0.0488 | 0.4159 | 0.0000 | -0.1914 | 38 |
| akiec | 320 | 0.1219 | 0.0306 | 0.0655 | 0.5316 | 0.0000 | -0.2791 | 22 |
| bkl | 320 | 0.1750 | 0.0813 | 0.1287 | 0.3888 | 0.0000 | -0.0793 | 56 |

Interpretation:

- Precision and coverage are low for all classes. The generated pool is mostly outside the real-class feature manifold and covers little of the real manifold.
- `akiec` has the worst nearest-real distance and strongly negative low-tail margin, suggesting many generated examples are closer to confusing classes or off-manifold.
- `bkl` has somewhat better geometry than `mel/akiec`, but previous downstream results show it still harms training. This suggests that geometry alone is not sufficient; visual/clinical artifacts or confusing subtype semantics may matter.

## Stage 6 training

Created split:

`/srv/research/projects/default/ham10000/splits/stage6/train_stage6_geometry_mel_akiec.csv`

Launched:

`stage6_geometry_mel_akiec_ce_weighted`

The run is still in progress at the time of this note. Metrics must be read from the structured run directory after completion.

## Embedding visualizations

Report:

`/srv/research/projects/default/ham10000/reports/stage6_embedding_visuals/index.html`

Изображения и их геометрические признаки доступны в FiftyOne:
`http://10.200.1.180:5151`. Исходные PNG/CSV остаются доступны через
JupyterHub.

Files:

- `pca_all_classes.png`
- `pca_mel.png`
- `pca_akiec.png`
- `pca_bkl.png`
- `tsne_all_classes.png`
- `tsne_mel.png`
- `tsne_akiec.png`
- `tsne_bkl.png`
- `embedding_coordinates.csv`
- `embedding_visual_report.json`

Compared groups:

- real training samples;
- Stage 3 utility-selected synthetic images;
- Stage 4 `mel+akiec` selected synthetic images;
- Stage 6 geometry-filtered synthetic images.

PCA is the more stable global projection; t-SNE is useful for local neighborhoods but should not be used alone as evidence. The dissertation should cite the numeric geometry metrics as primary evidence and use these plots as interpretable visual support.

## Next hypothesis

The current synthetic generation should probably be regenerated, not only reweighted.

More promising generation criteria:

- generate fewer samples but condition them closer to real hard examples;
- explicitly avoid confusing-class directions in feature space;
- use post-generation rejection based on real-manifold precision and class margin;
- add clinical/visual artifact review for `bkl`, because feature geometry did not fully explain the downstream harm;
- compare against a non-synthetic long-tail baseline such as LDAM-DRW or Balanced Softmax before claiming synthetic benefit.
