# Stage 2 Synthetic Protocol

Цель этапа: проверить, даёт ли targeted image-conditioned synthetic augmentation прирост сверх сильных real-only baseline.

## Исследовательская логика

Baseline stage 1 показал, что главные проблемные границы:

- `mel` vs `nv`
- `mel` vs `bkl`
- `akiec` vs `bkl`
- `akiec` vs `bcc`

Поэтому этап 2 не балансирует все классы вслепую. Он строит синтетический пул для проблемных классов, затем отбирает изображения по feature-space критериям.

## Шаг 1. Генерация synthetic pool

```bash
cd /srv/research/projects/default/research-ml
docker run --rm --shm-size=16g --gpus all \
  --user 1000:1006 --group-add 1006 \
  -v /srv/research/projects/default:/srv/research/projects/default \
  -w /srv/research/projects/default/research-ml \
  local/research-cuda-notebook:latest \
  bash -lc 'source .venv/bin/activate && PYTHONPATH=. python tools/generate_synthetic_img2img.py --config configs/stage2_generation.yaml'
```

Артефакты:

- `ham10000/synthetic/<generation_name>/`
- `synthetic_manifest.csv`
- `generation_config.resolved.yaml`

## Шаг 2. Feature-aware selection

```bash
docker run --rm --shm-size=16g --gpus all \
  --user 1000:1006 --group-add 1006 \
  -v /srv/research/projects/default:/srv/research/projects/default \
  -w /srv/research/projects/default/research-ml \
  local/research-cuda-notebook:latest \
  bash -lc 'source .venv/bin/activate && PYTHONPATH=. python tools/select_synthetic_by_features.py \
    --baseline-run-dir outputs/stage1_cross_entropy_weighted/20260708-060820_42 \
    --synthetic-csv synthetic/ham10000_mel_boundary_img2img_v1/synthetic_manifest.csv \
    --out-dir /srv/research/projects/default/ham10000/splits/stage2 \
    --target-classes mel,akiec,bkl \
    --same-quantile 0.95 \
    --min-margin 0.05 \
    --max-per-class 500'
```

Артефакты:

- `selected_synthetic.csv`
- `synthetic_selection_diagnostics.csv`
- `synthetic_selection_report.json`
- `train_stage2_selected.csv`

## Шаг 3. Training matrix

```bash
docker run -d --name research-ml-stage2 --shm-size=16g --gpus all \
  --user 1000:1006 --group-add 1006 \
  -v /srv/research/projects/default:/srv/research/projects/default \
  -w /srv/research/projects/default/research-ml \
  local/research-cuda-notebook:latest \
  bash -lc 'source .venv/bin/activate && PYTHONPATH=. bash scripts/run_stage2_matrix.sh configs/ham10000_stage2_synthetic.yaml'
```

## Критерии успеха

Синтетика считается полезной, если она улучшает не только один показатель, а профиль:

- `macro F1` выше `0.7989` или хотя бы выше `0.7955`;
- `balanced accuracy` выше `0.8279`;
- `worst-class recall` не ниже `0.7305`;
- `ECE/Brier/NLL` не ухудшаются катастрофически;
- confusion по `mel -> nv`, `mel -> bkl`, `akiec -> bkl` снижается.

