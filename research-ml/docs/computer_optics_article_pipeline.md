# Pipeline for a Computer Optics article draft

This pipeline prepares a reproducible article-writing package for the journal
Computer Optics. The target format is LaTeX, because the journal accepts TEX and
publishes an official TEX template:

- guidelines: https://www.computeroptics.ru/guidelines.htm
- TEX template: https://www.computeroptics.ru/Guidelines/TemplateRuTex.zip

The pipeline does not pretend to write a final paper by itself. Its job is to
collect evidence from training artifacts, copy candidate figures, create a
journal-aware LaTeX draft, and give the next agent a strict brief so that the
scientific text is traceable to files.

## Why LaTeX

Computer Optics accepts DOCX or TEX. For automatic article generation, TEX is the
least ambiguous option: tables, equations, figures, and metadata can be generated
as text, diffed in git, reviewed by another agent, and compiled to PDF later.
Markdown is useful for notes, but it would still need conversion into the
journal template. TXT would lose figure/table structure.

## Quick start

Run from the `research-ml` directory:

```bash
python tools/build_computer_optics_article.py
```

The default output directory is:

```text
outputs/reports/computer_optics_article
```

It contains:

- `evidence_pack.json`: structured inputs for the article-writing agent;
- `draft_computer_optics.tex`: LaTeX draft in the style of the official template;
- `agent_brief.md`: detailed instructions for the next agent;
- `computer_optics_checklist.md`: submission and style checklist;
- `figures/`: copied candidate figures with stable names.

To create a zip with all generated assets:

```bash
python tools/build_computer_optics_article.py --make-zip
```

## Metadata file

Create a small JSON file when author metadata is known:

```json
{
  "title_ru": "TODO: краткое название без сокращений",
  "authors_ru": "И.О. Фамилия 1, И.О. Фамилия 2",
  "affiliations_ru": "1 Название организации, полный адрес",
  "keywords_ru": [
    "дермоскопические изображения",
    "классификация изображений",
    "дисбаланс классов",
    "синтетические данные",
    "глубокое обучение"
  ]
}
```

Then run:

```bash
python tools/build_computer_optics_article.py --metadata article_metadata.json
```

## Server artifacts

On the research server, point the script at the real output directories:

```bash
python tools/build_computer_optics_article.py \
  --artifact-root outputs \
  --artifact-root /srv/research/projects/default/ham10000/reports \
  --out-dir outputs/reports/computer_optics_article
```

The script scans JSON metric files, CSV summaries, Markdown methodology notes,
and figure files. It intentionally keeps only a bounded evidence set so the next
agent can read it completely.

## Writing rules for the agent

Use `agent_brief.md` as the controlling instruction. The most important rules:

- every numeric claim must come from `evidence_pack.json`;
- pending screening runs must not be described as final results;
- the old synthetic pool is a negative pilot, not the proposed method;
- do not use LaTeX `\label`, `\ref`, or automatic cross-references;
- write manual references such as `рис. 1`, `табл. 1`, `[1]`;
- keep References in English, in first-citation order, AMA-like, with DOI where
  available;
- for Russian text, use decimal comma in prose and final tables.

## Finalization checklist

Before submission, manually or with a dedicated formatting pass:

1. Download the official TEX template and compare the generated draft against it.
2. Replace every TODO with verified content.
3. Shorten the automatically generated metric table to the final selected
   experiments.
4. Convert PNG figures to JPEG/WMF if needed.
5. Compile the TEX to PDF and check that the PDF is under 10 MB.
6. Prepare the separate figures ZIP under 20 MB.
7. Add the signed first-page scan and open-publication expert conclusion where
   required.

