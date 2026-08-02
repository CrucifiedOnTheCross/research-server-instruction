# Long-tail medical image classification

Independent research package for the ISIC-2019-LT study proposed in
`deep-research-report (1).md`. The package is intentionally isolated from
`research-ml`: it shares only the server and the official ISIC 2019 files.

## Stage 1 objective

1. Reproduce the published MONICA ISIC-2019-LT image lists at imbalance 100.
2. Audit their class mapping and lesion-level overlap.
3. Train the MONICA ResNet-50 cross-entropy baseline with structured artifacts.
4. Keep the MONICA test list closed until the validation workflow is verified.

The MONICA benchmark is retained for direct comparison, even though its public
lists are image-disjoint rather than lesion-disjoint. A separate lesion-disjoint
protocol will be built in Stage 2 and must not be mixed with MONICA scores.

## Prepare data

The server already contains the official ISIC 2019 archive under
`/srv/research/projects/default/isic2019`. Generate the pinned MONICA manifests:

```bash
python tools/prepare_isic2019_monica.py \
  --root /srv/research/projects/default/isic2019 \
  --ratios 100 200 500
```

Use `--download-images` only on a clean machine. The script downloads official
CSV files, fetches MONICA protocol arrays from pinned commit
`3dd808d6d578b9e0f9bf4ee1402727ff46d1c243`, verifies SHA-256, and writes CSV,
JSON, and checksum artifacts under `data_splits/monica/`.

## Run Stage 1

```bash
bash scripts/run_stage1_monica_ir100.sh
```

The default configuration uses the published MONICA baseline settings:
ResNet-50 with ImageNet weights, 224 px input, Adam, learning rate `3e-4`,
50 epochs, strong augmentation, and effective batch 256. On the RTX 5080 the
physical batch is 128 with two-step gradient accumulation. This hardware
adaptation is recorded and is not claimed to be bitwise identical to MONICA.

## Test

```bash
python -m unittest discover -s tests -v
```

## Evidence policy

- JSON/CSV/YAML artifacts are canonical; logs are diagnostic only.
- Every run stores resolved configuration, environment, split hashes, class
  counts, epoch metrics, best validation predictions, and checkpoint metadata.
- `evaluation.run_test` is `false` in Stage 1.
- Images and model checkpoints are not committed to git.

