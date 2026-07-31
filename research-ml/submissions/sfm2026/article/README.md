# SFM 2026 article draft

This directory contains a self-contained seven-page conference paper draft.
The manuscript follows the visual grammar observed in the official SFM 2024
proceedings while preserving the evidence-traceability rules from
`docs/computer_optics_article_pipeline.md`.

## Files

- `draft_sfm2026.tex`: editable Russian manuscript.
- `draft_sfm2026.pdf`: compiled and visually inspected draft.
- `figures/`: reproducible academic figures copied from `reports/sfm2026`.
- `article_metadata.json`: metadata used by the Computer Optics evidence-pack
  builder.
- `evidence_pipeline/`: generated evidence pack, agent brief, checklist, and
  candidate figure bundle.

## Rebuild

From this directory:

```powershell
xelatex -interaction=nonstopmode -halt-on-error draft_sfm2026.tex
xelatex -interaction=nonstopmode -halt-on-error draft_sfm2026.tex
```

To recreate the structured writing evidence from the repository root:

```powershell
python tools/build_computer_optics_article.py `
  --artifact-root reports/sfm2026 `
  --artifact-root reports/article_readiness_2026-07-31 `
  --out-dir submissions/sfm2026/article/evidence_pipeline `
  --metadata submissions/sfm2026/article/article_metadata.json `
  --language ru --make-zip
```

## Before submission

Confirm the official English transliteration of the speaker's name, the full
author list and order, coauthor consent, and whether the presentation is oral or
online oral. The manuscript reports validation results only; the locked test
remains closed and no clinical claim is made.
