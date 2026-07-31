# Agent brief: Computer Optics article draft

Generated: 2026-07-31T17:05:01

Target format: LaTeX (`draft_computer_optics.tex`) because Computer Optics accepts TEX and publishes an official TEX template.

## Non-negotiable journal constraints

- Article package: TEX or DOCX article, PDF under 10 MB, figures ZIP under 20 MB, signed first page scan if more than one author, and open-publication expert conclusion for Russian Federation authors.
- Regular article target: up to 12 pages. Abstract: 150-250 words, informative, no formulas, no undefined abbreviations, no numbered references.
- Keywords: 5-10 specific terms.
- Structure for a Russian article: title, authors, affiliations, abstract, keywords, citation; Introduction; numbered main sections; Conclusion; Acknowledgements if needed; References; author bios; English title/authors/abstract/keywords/citation/about authors.
- Do not use footnotes. Do not use LaTeX cross-references (`\label`, `\ref`, `\autoref`) for formulas, figures, tables, or sources. Write textual references manually: `рис. 1`, `табл. 1`, `[1]`.
- References must be in English, ordered by first citation, AMA-like, and include DOI when available.
- For Russian text, use decimal comma in prose and tables unless code/file names require dots.
- Figures must be separate files. Submission-preferred formats are JPEG/WMF; convert generated PNG files before final submission.

## Scientific writing policy for the next agent

- Every numerical claim must be traceable to `evidence_pack.json`.
- Separate completed evidence from pending pipeline status. In particular, Stage 8 screening is not a final result until multi-seed summaries and the predefined decision rule are complete.
- State the corrected research question: after removing acquisition artifacts and evaluation leakage, can independently selected synthetic dermoscopic images improve long-tail classification beyond strong real-only baselines?
- Do not present the old synthetic pool as a usable method. Treat it as a negative pilot that motivated protocol repair.
- Use the article scope "analysis and understanding of images / pattern recognition / digital image processing"; avoid making medical diagnostic claims beyond image-classification evaluation.

## Computer Optics style exemplars to imitate

- Recent image-classification papers put the aim, dataset/splitting protocol, model idea, and numerical results directly in the abstract.
- A Computer Optics paper on microscopic bacterial classification emphasizes strain-wise splitting as a generalization safeguard and reports both task-level and subgroup results.
- A Computer Optics paper on rare traffic sign augmentation frames synthetic data as a way to address rare classes, then validates utility by mixing synthetic and real data.

## Inputs

- Evidence pack: `evidence_pack.json`
- LaTeX draft: `draft_computer_optics.tex`
- Figure copies: `figures/`
- Compliance checklist: `computer_optics_checklist.md`

## Recommended generation sequence

1. Read `evidence_pack.json` completely.
2. Use `metric_files` for classification performance and `diagnostic_files` for synthetic-pool geometry, artifact audits, and domain-gap evidence.
3. Identify final methods and exclude incomplete/running artifacts.
4. Build a small final comparison table with mean/std over seeds when available.
5. Write the Russian article first.
6. Add English metadata block after the Russian author information.
7. Run the checklist and mark every unresolved item as TODO rather than silently filling it.
