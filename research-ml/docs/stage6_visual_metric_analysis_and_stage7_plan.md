# Stage 6 visual/metric analysis and Stage 7 problem checks

Date: 2026-07-08

## Input artifacts

Analysis used structured artifacts, not noisy logs:

- `test_metrics.json` for Stage 1, Stage 3, Stage 4, Stage 6.
- `stage6_feature_geometry_report.json`.
- `embedding_coordinates.csv`.
- PCA/t-SNE plots from `reports/stage6_embedding_visuals`.

Local downloaded copies for inspection:

`research-ml/local_artifacts/stage6_embedding_visuals`

## Metric summary

| Run | Macro F1 | Bal acc | MCC | ECE | akiec recall | bkl recall | mel recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage 1 CE weighted | 0.7955 | 0.8279 | 0.7673 | 0.0931 | 0.7347 | 0.7636 | 0.7305 |
| Stage 3 utility synthetic | 0.7915 | 0.8063 | 0.7653 | 0.0880 | 0.6939 | 0.6061 | 0.7904 |
| Stage 4 mel+akiec | 0.8044 | 0.7944 | 0.7906 | 0.0784 | 0.7347 | 0.7879 | 0.6407 |
| Stage 6 geometry mel+akiec | 0.7889 | 0.7958 | 0.7523 | 0.0969 | 0.6531 | 0.7939 | 0.7365 |

## Visual analysis

### mel

PCA and t-SNE show that Stage 3/4 `mel` synthetic samples lie as a broad peripheral band around the real `mel` cloud. Stage 6 geometry filtering moves part of the selected samples closer to the real cloud, but many selected points remain outside the dense real core.

This explains the metric pattern:

- Stage 3 improved `mel` recall to `0.7904`, but harmed `bkl`.
- Stage 4 excluded `bkl` synthetic and achieved best macro F1, but `mel` recall dropped to `0.6407`.
- Stage 6 recovered `mel` recall to `0.7365`, but macro F1/MCC/ECE worsened.

Interpretation: `mel` synthetic data can act as useful boundary regularization, but the current generated distribution is too broad/off-manifold for stable overall improvement.

### akiec

`akiec` is the clearest failure case. PCA/t-SNE show that many Stage 6 synthetic `akiec` points remain separated from the real `akiec` feature cloud, with several outlying islands.

Geometry confirms this:

- PRDC precision `0.1219`.
- Coverage `0.0655`.
- p80 nearest-real distance `0.5316`.
- p10 feature margin `-0.2791`.
- Only `22/320` samples passed the strict geometry filter.

Metric outcome:

- Stage 6 `akiec` recall fell to `0.6531`.

Interpretation: current `akiec` synthetic images should not be used for training without regeneration or much stricter rejection.

### bkl

`bkl` synthetic points visually overlap the real `bkl` region better than `akiec`, and geometry is also less bad:

- PRDC precision `0.1750`.
- Coverage `0.1287`.
- p80 nearest-real distance `0.3888`.
- p10 feature margin `-0.0793`.

But Stage 3 showed that adding `bkl` synthetic severely hurt `bkl` recall (`0.6061`). Therefore the failure is not fully explained by global feature distance. Likely causes:

- clinical/visual artifacts not captured by this encoder;
- synthetic `bkl` samples near confusing `mel/nv` boundary;
- shortcut texture patterns that distort the classifier boundary.

## Literature used

Already tracked in Google Sheet:

- Sajjadi et al., "Assessing Generative Models via Precision and Recall", NeurIPS 2018.
- Kynkäänniemi et al., "Improved Precision and Recall Metric for Assessing Generative Models", NeurIPS 2019.
- Naeem et al., "Reliable Fidelity and Diversity Metrics for Generative Models", ICML 2020.
- Menon et al., "Long-tail learning via logit adjustment", ICLR 2021.
- Kang et al., "Decoupling Representation and Classifier for Long-Tailed Recognition", ICLR 2020.
- Chen et al., "Augmented Conditioning Is Enough For Effective Training Image Generation", 2025.
- Li et al., "SAU: A Dual-Branch Network to Enhance Long-Tailed Recognition via Generative Models", 2024.

## Current conclusion

The project should not continue by simply adding more synthetic images or tuning a global synthetic weight.

The data suggest:

- `akiec` synthetic generation is currently harmful/off-manifold.
- `bkl` synthetic generation is ambiguous: feature overlap is not enough, and downstream harm remains.
- `mel` synthetic generation has a useful signal, but must be constrained.
- A strong real-only classifier/boundary baseline is necessary before claiming synthetic augmentation is beneficial.

## Stage 7 checks launched

Container:

`research-stage7-problem-checks`

Experiments:

1. `stage7_geometry_mel_only_ce_weighted`
   - Uses real training data plus only 80 Stage 6 geometry-filtered `mel` synthetic samples.
   - Excludes synthetic `akiec` and `bkl`.
   - Tests whether the useful `mel` signal can be retained without the `akiec/bkl` damage.

2. `stage7_real_logit_adjust_tau1_weighted`
   - Real-only.
   - Uses logit-adjusted cross-entropy.
   - Tests whether a simple long-tail boundary correction can match or beat synthetic augmentation.

First status:

- GPU active around `99%`.
- `stage7_geometry_mel_only_ce_weighted` started at `outputs/stage7_geometry_mel_only_ce_weighted/20260708-152407_42`.
- Stage 7 split: 7011 real + 80 synthetic `mel`.

## Next decision rules

- If `mel-only` improves `mel` recall without reducing macro F1/MCC below Stage 4, keep synthetic generation only for `mel`.
- If logit adjustment beats Stage 4, prioritize classifier-boundary methods over regeneration.
- If logit adjustment improves calibration but not recall, combine it with carefully filtered `mel` synthetic.
- If neither beats Stage 4, keep Stage 4 as current best and move to better generation: augmented conditioning / counterfactual generation / synthetic-aware training.
