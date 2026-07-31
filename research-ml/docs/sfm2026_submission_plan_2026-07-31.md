# Saratov Fall Meeting 2026: submission readiness and plan

## Decision

The project has enough completed material for a scientifically defensible SFM 2026 abstract and oral/poster presentation. It is not necessary or methodologically desirable to start another training experiment on the submission day.

The strongest conference-sized story is a controlled negative/mechanistic result:

> Feature-space quality and visual plausibility do not guarantee useful synthetic augmentation for long-tailed dermoscopic classification; source-matched replay and task-aware ranking diagnostics expose effects hidden by threshold metrics.

This is narrower and stronger than claiming that synthetic images generally improve classification.

## Official requirements checked on 31 July 2026

- SFM XXX is scheduled for 21-25 September 2026 in Saratov.
- The English homepage lists 31 July 2026 as both the abstract and registration deadline.
- The live registration form requests an English abstract of 200-250 words.
- The conference scope includes biomedical imaging and machine learning.
- The most suitable workshop is `Applications of Laser Molecular Imaging and Machine Learning VI`.
- The official pages conflict on some dates: the Russian information page lists an earlier abstract date, while full-text dates differ between the homepage and proceedings page. Registration should therefore be completed immediately and any uncertainty confirmed with the organizers.

### Format inferred from official SFM examples

Official SFM-2025 abstract pages use the following compact structure:

1. paper title;
2. author initials/names and numbered affiliations;
3. a continuous English abstract with problem, method, results, and conclusion;
4. keywords;
5. an optional funding acknowledgement;
6. speaker and affiliation metadata.

The proposed abstract follows this structure and emphasizes measured results rather than a literature review. The proceedings page confirms that conference papers may be submitted in Russian or English to `Problems of Optical Physics and Biophotonics` and undergo normal review, but no unambiguous SFM-2026 full-paper template was found on the official site. The full-paper layout must therefore be requested from the organizers instead of assuming that generic SPIE formatting applies.

Official pages:

- https://sfmconference.org/
- https://sfmconference.org/sfm26/
- https://sfmconference.org/sfm26/registration
- https://sfmconference.org/sfm26/conferences_workshops/
- https://sfmconference.org/proceedings/
- https://sfmconference.org/contacts/
- https://sfmconference.org/sfm25/conferences_workshops/nonlinear-dynamics-xvi/preliminary/3526/

## Evidence included in the abstract

1. HAM10000 lesion-group split and a closed locked test.
2. ConvNeXt-S at 384 x 384 as the qualified downstream model.
3. Equal-budget synthetic versus source-matched real replay.
4. Three paired seeds and 5,000-repeat hierarchical lesion-group bootstrap.
5. Stage 10 geometry interaction: strict-ID MCC `+0.066`, 95% CI `[+0.005, +0.125]`; random-remaining MCC `-0.049`, CI `[-0.086, -0.013]`.
6. Stronger confirmatory baseline: macro F1 `+0.012`, but macro AUPRC `-0.014` and melanoma AUPRC `-0.045`.
7. Stage 12: precision, density, and coverage lower in 11/12 encoder-class comparisons despite higher Vendi diversity in 11/12.
8. Stage 15A: complete crop-VAE-one-step pipeline MCC `-0.029`, 95% CI `[-0.058, -0.0004]`; crop explained most macro-AUPRC loss.

All abstract results are validation results. The abstract does not imply clinical utility and does not use locked-test outcomes.

## Why no new experiment today

- A new training run cannot be fully designed, replicated across seeds, bootstrapped, interpreted, and frozen before the deadline without compromising quality.
- Existing results already answer a coherent question with matched controls and uncertainty estimates.
- Opening the locked test on submission day would encourage post-hoc editing and is unnecessary for a conference abstract.
- The ongoing stronger-generator branch belongs to the next paper unless it passes its preregistered generator gate and an independent downstream protocol.

Only deterministic figure generation and evidence packaging are performed for this submission.

## Prepared package

- `submissions/sfm2026/abstract_en.txt`: submit-ready 200-250 word abstract.
- `submissions/sfm2026/registration_fields.md`: title, session, keywords, and author TODOs.
- `submissions/sfm2026/full_paper_outline.md`: proceedings-paper plan.
- `reports/sfm2026/evidence_manifest.json`: word count, hashes, and test-status declaration.
- `reports/sfm2026/figure_data/`: exact CSV data behind every quantitative figure.
- `reports/sfm2026/figures/`: publication-quality plots and internal generator examples.

Rebuild with:

```bash
python tools/build_sfm2026_submission.py
```

## Draft package completed on 31 July 2026

- The 247-word English abstract and ready-to-paste registration block are in
  `submissions/sfm2026/`.
- A seven-page Russian full-text draft was compiled and visually checked at
  `submissions/sfm2026/article/draft_sfm2026.pdf`.
- Descriptive English terminology in the Russian manuscript and figures was
  replaced with Russian equivalents. Model names, dataset names, metric
  abbreviations, and reference titles retain their standard form.
- Three academic figures are generated from structured Stage 10, 15A, and 16G
  artifacts. The locked test remains closed.
- Official SFM 2024 proceedings examples and formatting observations are listed
  in `references/sfm_examples/README.md`; extracted PDFs remain local and are
  not redistributed through git.

## Submission blockers owned by the authors

Before pressing Submit, confirm:

1. official English speaker name;
2. author list, order, and consent;
3. English affiliation wording;
4. presentation/participation format;
5. whether the organizers want the abstract pasted as plain text or with keywords in the same field.

If the web form rejects the submission because of the date discrepancy, contact the SFM-26 general secretary and secretary using the addresses on the official contacts page on the same day.
