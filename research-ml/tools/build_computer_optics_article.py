from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import struct
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any


METRIC_FIELDS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "mcc",
    "worst_class_recall",
    "ece",
    "auroc_ovr_macro",
    "auprc_ovr_macro",
)

FIGURE_PATTERNS = (
    "pca",
    "tsne",
    "confusion",
    "geometry",
    "audit",
    "gallery",
    "embedding",
)

DEFAULT_KEYWORDS_RU = [
    "дермоскопические изображения",
    "классификация изображений",
    "дисбаланс классов",
    "синтетические данные",
    "глубокое обучение",
    "пространство признаков",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an evidence pack, agent brief, and LaTeX draft for a Computer Optics manuscript."
        )
    )
    parser.add_argument("--project-root", default=".", help="Usually the research-ml directory.")
    parser.add_argument("--out-dir", default="outputs/reports/computer_optics_article")
    parser.add_argument(
        "--artifact-root",
        action="append",
        default=None,
        help="Artifact root relative to project-root. Can be passed more than once.",
    )
    parser.add_argument("--metadata", default=None, help="Optional JSON with title/authors/affiliations.")
    parser.add_argument("--title", default="TODO: краткое название статьи без сокращений и формул")
    parser.add_argument("--language", choices=("ru", "en"), default="ru")
    parser.add_argument("--max-doc-chars", type=int, default=5000)
    parser.add_argument("--max-figures", type=int, default=8)
    parser.add_argument("--max-metric-files", type=int, default=30)
    parser.add_argument("--make-zip", action="store_true", help="Create submission_assets.zip with draft and figures.")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {"value": data}
    except Exception as exc:
        return {"_read_error": str(exc)}


def read_text_excerpt(path: Path, max_chars: int) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="cp1251", errors="replace")
    return text[:max_chars]


def safe_rel(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def is_under(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def is_generated_article_path(path: Path) -> bool:
    return any(part.startswith("computer_optics_article") for part in path.parts)


def should_skip_artifact(path: Path, exclude_dir: Path) -> bool:
    return is_under(path, exclude_dir) or is_generated_article_path(path)


def latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def fmt_float(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.4f}".replace(".", ",")


def compact_metrics(data: dict[str, Any]) -> dict[str, Any]:
    result = {key: data[key] for key in METRIC_FIELDS if key in data}
    per_class = data.get("per_class")
    if isinstance(per_class, dict):
        result["per_class"] = {
            label: {
                key: metrics.get(key)
                for key in ("precision", "recall", "f1", "support")
                if isinstance(metrics, dict) and key in metrics
            }
            for label, metrics in per_class.items()
        }
    return result


def discover_metric_files(
    roots: list[Path], project_root: Path, exclude_dir: Path, limit: int
) -> list[dict[str, Any]]:
    files: list[Path] = []
    for root in roots:
        if root.exists():
            files.extend(path for path in root.rglob("*metrics*.json") if not should_skip_artifact(path, exclude_dir))
            files.extend(path for path in root.rglob("summary.json") if not should_skip_artifact(path, exclude_dir))
            files.extend(path for path in root.rglob("multiseed_summary.json") if not should_skip_artifact(path, exclude_dir))
    unique = sorted(set(files), key=lambda p: p.stat().st_mtime, reverse=True)
    items: list[dict[str, Any]] = []
    for path in unique[:limit]:
        data = read_json(path)
        item = {
            "path": safe_rel(path, project_root),
            "mtime": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
            "metrics": compact_metrics(data),
        }
        if "experiments" in data:
            item["experiments"] = data["experiments"]
        items.append(item)
    return items


def compact_diagnostics(data: dict[str, Any]) -> dict[str, Any]:
    keep_keys = (
        "target_classes",
        "select_classes",
        "top_k_per_class",
        "selected_by_class",
        "black_border_auroc",
        "median_black_border_share",
        "synthetic_count",
        "real_count",
        "decision",
        "status",
    )
    result = {key: data[key] for key in keep_keys if key in data}
    class_metrics = data.get("class_metrics")
    if isinstance(class_metrics, list):
        result["class_metrics"] = [
            {
                key: item.get(key)
                for key in (
                    "label",
                    "n_real",
                    "n_synthetic",
                    "prdc_precision",
                    "prdc_density",
                    "prdc_coverage",
                    "nearest_real_distance_mean",
                    "center_shift_cosine_distance",
                    "duplicate_rate",
                    "unique_source_count",
                    "pass_geometry_filter_count",
                )
                if isinstance(item, dict) and key in item
            }
            for item in class_metrics
        ]
    return result or {key: data[key] for key in list(data)[:10] if not key.startswith("_")}


def discover_diagnostics(roots: list[Path], project_root: Path, exclude_dir: Path) -> list[dict[str, Any]]:
    patterns = (
        "*geometry*.json",
        "*audit*.json",
        "*report*.json",
        "embedding_visual_report.json",
        "multiseed_summary.json",
    )
    files: list[Path] = []
    for root in roots:
        if root.exists():
            for pattern in patterns:
                files.extend(path for path in root.rglob(pattern) if not should_skip_artifact(path, exclude_dir))
    items = []
    for path in sorted(set(files), key=lambda p: p.stat().st_mtime, reverse=True)[:30]:
        items.append(
            {
                "path": safe_rel(path, project_root),
                "mtime": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
                "diagnostics": compact_diagnostics(read_json(path)),
            }
        )
    return items


def read_csv_header(path: Path) -> dict[str, Any]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            first_rows = []
            for index, row in enumerate(reader):
                if index < 3:
                    first_rows.append(row)
                else:
                    break
            return {"path": path, "fields": reader.fieldnames or [], "sample_rows": first_rows}
    except Exception as exc:
        return {"path": path, "error": str(exc)}


def discover_tables(
    roots: list[Path], project_root: Path, exclude_dir: Path, limit: int = 20
) -> list[dict[str, Any]]:
    tables: list[dict[str, Any]] = []
    for root in roots:
        if root.exists():
            for path in sorted(root.rglob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True):
                if should_skip_artifact(path, exclude_dir):
                    continue
                info = read_csv_header(path)
                info["path"] = safe_rel(path, project_root)
                tables.append(info)
                if len(tables) >= limit:
                    return tables
    return tables


def image_size(path: Path) -> tuple[int | None, int | None]:
    try:
        with path.open("rb") as handle:
            header = handle.read(32)
            if header.startswith(b"\x89PNG\r\n\x1a\n"):
                width, height = struct.unpack(">II", header[16:24])
                return int(width), int(height)
            if header.startswith(b"\xff\xd8"):
                handle.seek(2)
                while True:
                    marker = handle.read(2)
                    if len(marker) < 2:
                        return None, None
                    while marker[0] != 0xFF:
                        marker = marker[1:] + handle.read(1)
                    code = marker[1]
                    size_bytes = handle.read(2)
                    if len(size_bytes) < 2:
                        return None, None
                    size = struct.unpack(">H", size_bytes)[0]
                    if 0xC0 <= code <= 0xC3:
                        data = handle.read(5)
                        height, width = struct.unpack(">HH", data[1:5])
                        return int(width), int(height)
                    handle.seek(size - 2, 1)
    except Exception:
        return None, None
    return None, None


def figure_score(path: Path) -> tuple[int, float]:
    name = path.name.lower()
    pattern_score = sum(1 for pattern in FIGURE_PATTERNS if pattern in name)
    return pattern_score, path.stat().st_mtime


def discover_figures(
    roots: list[Path], project_root: Path, out_dir: Path, max_figures: int
) -> list[dict[str, Any]]:
    candidates: list[Path] = []
    for root in roots:
        if root.exists():
            for suffix in ("*.png", "*.jpg", "*.jpeg"):
                candidates.extend(path for path in root.rglob(suffix) if not should_skip_artifact(path, out_dir))
    candidates = sorted(set(candidates), key=figure_score, reverse=True)[:max_figures]
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    figures = []
    for index, source in enumerate(candidates, start=1):
        ext = ".jpg" if source.suffix.lower() in (".jpg", ".jpeg") else source.suffix.lower()
        target_name = f"{index:02d}_{re.sub(r'[^A-Za-z0-9_-]+', '_', source.stem)[:48]}{ext}"
        target = figures_dir / target_name
        shutil.copy2(source, target)
        width, height = image_size(target)
        figures.append(
            {
                "number": index,
                "source_path": safe_rel(source, project_root),
                "draft_path": safe_rel(target, out_dir),
                "width_px": width,
                "height_px": height,
                "journal_note": "Convert PNG to JPEG/WMF before submission." if ext == ".png" else "JPEG-compatible.",
                "caption_ru": f"TODO: информативная подпись к результату на рис. {index}",
            }
        )
    return figures


def discover_docs(project_root: Path, max_chars: int) -> list[dict[str, Any]]:
    docs_dir = project_root / "docs"
    if not docs_dir.exists():
        return []
    docs = []
    for path in sorted(docs_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True):
        if path.name.startswith("external_"):
            continue
        docs.append({"path": safe_rel(path, project_root), "excerpt": read_text_excerpt(path, max_chars)})
    return docs


def load_metadata(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    return read_json(Path(path))


def build_evidence_pack(
    project_root: Path,
    roots: list[Path],
    metadata: dict[str, Any],
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "target_journal": "Computer Optics",
        "target_format": "LaTeX",
        "guidelines_url": "https://www.computeroptics.ru/guidelines.htm",
        "official_tex_template_url": "https://www.computeroptics.ru/Guidelines/TemplateRuTex.zip",
        "metadata": metadata,
        "metric_files": discover_metric_files(roots, project_root, out_dir, args.max_metric_files),
        "diagnostic_files": discover_diagnostics(roots, project_root, out_dir),
        "tables": discover_tables(roots, project_root, out_dir),
        "figures": discover_figures(roots, project_root, out_dir, args.max_figures),
        "docs": discover_docs(project_root, args.max_doc_chars),
    }


def metric_rows_latex(metric_files: list[dict[str, Any]]) -> str:
    rows = []
    for item in metric_files:
        metrics = item.get("metrics", {})
        if not metrics:
            continue
        rows.append(
            "        "
            + " & ".join(
                latex_escape(value)
                for value in (
                    item["path"],
                    fmt_float(metrics.get("macro_f1", "")),
                    fmt_float(metrics.get("balanced_accuracy", "")),
                    fmt_float(metrics.get("mcc", "")),
                    fmt_float(metrics.get("worst_class_recall", "")),
                    fmt_float(metrics.get("ece", "")),
                )
            )
            + r" \\"
        )
    if not rows:
        rows.append(r"        TODO &  &  &  &  &  \\")
    return "\n".join(rows)


def figures_latex(figures: list[dict[str, Any]]) -> str:
    blocks = []
    for fig in figures[:4]:
        path = fig["draft_path"]
        caption = latex_escape(fig["caption_ru"])
        blocks.append(
            "\n".join(
                [
                    r"\begin{figure}[h]",
                    r"\centering",
                    rf"\includegraphics[width=0.80\linewidth]{{{path}}}",
                    rf"\caption{{{caption}}}",
                    r"\end{figure}",
                ]
            )
        )
    return "\n\n".join(blocks) if blocks else "% TODO: add figures after first mention in the text."


def build_latex(evidence: dict[str, Any], title: str) -> str:
    metadata = evidence.get("metadata", {})
    authors = metadata.get("authors_ru", "TODO: И.О. Фамилия 1, И.О. Фамилия 2")
    affiliations = metadata.get(
        "affiliations_ru",
        "1 TODO: организация в именительном падеже, полный почтовый адрес",
    )
    keywords = metadata.get("keywords_ru", DEFAULT_KEYWORDS_RU)
    keywords_text = ", ".join(keywords)
    return rf"""\documentclass[10pt,a4paper]{{article}}
\usepackage[a4paper,top=2.0cm,bottom=1.5cm,left=2.0cm,right=2.0cm]{{geometry}}
\usepackage[utf8]{{inputenc}}
\usepackage[english,russian]{{babel}}
\usepackage{{tempora}}
\usepackage{{graphicx}}
\usepackage[labelsep=period]{{caption}}
\usepackage{{array}}
\usepackage{{enumitem}}
\usepackage[colorlinks=true,urlcolor=blue]{{hyperref}}
\newcolumntype{{L}}[1]{{>{{\raggedright\arraybackslash}}p{{#1}}}}
\captionsetup[table]{{name=Табл.,labelsep=period}}
\sloppy

\begin{{document}}

\begin{{center}}
    \section*{{{latex_escape(title)}}}
\end{{center}}

\begin{{center}}
\it{{{latex_escape(authors)}}}

\it{{{latex_escape(affiliations)}}}\\
TODO: указать автора для связи
\end{{center}}

\begin{{center}}
    {{\bf Аннотация}}
\end{{center}}

TODO: 150--250 слов без формул, ссылочных номеров и неопределённых сокращений. Аннотация должна назвать цель исследования, источник данных, проверяемый протокол, главные численные результаты и практическое значение. Не использовать общие фразы без результата.

\underline{{\it{{Ключевые слова}}}}: {latex_escape(keywords_text)}.

\underline{{\it{{Цитирование}}}}: TODO: ссылка для цитирования на русском языке после согласования авторов и названия.

\underline{{\it{{Citation}}}}: TODO: citation in English; DOI присваивается редакцией после принятия.

\begin{{center}}
    {{\bf Введение}}
\end{{center}}

TODO: сформулировать задачу классификации дермоскопических изображений при дисбалансе классов как задачу анализа и понимания изображений/распознавания образов. Первая ссылка в основном тексте должна быть [1]. Не ссылаться на таблицы, рисунки или формулы через \textbackslash ref.

\begin{{center}}
    {{\bf 1. Материалы и методы}}
\end{{center}}

TODO: описать HAM10000/ISIC, group-aware разбиение по lesion_id/group_id, реальные validation/test split, backbone, resolution, loss/sampling, seed protocol и запрет использования locked test до выбора финального метода.

\begin{{center}}
    {{\bf 2. Формирование и отбор синтетических изображений}}
\end{{center}}

TODO: отдельно описать отрицательный пилот со старым synthetic pool и исправленный Stage 8 protocol: train-only sources, center crop без padding, audit чёрных полей, независимый DINOv2 encoder, diversity/top-k selection. Не представлять pending screening как доказанный финальный результат.

\begin{{center}}
    {{\bf 3. Результаты}}
\end{{center}}

Табл. 1 суммирует численные артефакты, найденные автоматическим сборщиком. Перед подачей таблицу нужно вручную сократить до финальных сравнений методов и проверить, что каждая строка соответствует выбранному split.

\begin{{table}}[h]
    \begin{{center}}
    \caption{{Автоматически найденные метрики экспериментов}}
    \begin{{tabular}}{{|L{{56mm}}|L{{18mm}}|L{{18mm}}|L{{18mm}}|L{{18mm}}|L{{14mm}}|}}
        \hline
        Артефакт & Macro F1 & Bal. acc. & MCC & Worst recall & ECE \\
        \hline
{metric_rows_latex(evidence.get("metric_files", []))}
        \hline
    \end{{tabular}}
    \end{{center}}
\end{{table}}

TODO: дать текстовую интерпретацию только для завершённых запусков. Для screening использовать validation metrics; locked test упоминать только если он действительно был открыт по заранее заданному правилу.

{figures_latex(evidence.get("figures", []))}

\begin{{center}}
    {{\bf 4. Обсуждение}}
\end{{center}}

TODO: обсудить, почему визуальное качество синтетики недостаточно; связать utility с domain gap, coverage, class-conditional geometry и leakage safeguards. Ясно отделить инженерные ограничения от научного вывода.

\begin{{center}}
    {{\bf Заключение}}
\end{{center}}

TODO: 1 абзац с ответом на исследовательский вопрос, 1 абзац с ограничениями, 1 абзац с дальнейшим протоколом. Без новых чисел, которых не было в результатах.

\begin{{center}}
    {{\bf Благодарности}}
\end{{center}}

TODO: указать гранты/инфраструктуру или удалить раздел, если он не нужен.

\begin{{center}}
    {{\bf References}}
\end{{center}}

\begin{{enumerate}}[topsep=0px,itemsep=-1px]
    \item TODO: оформить источники в порядке цитирования по AMA; DOI обязателен, если есть.
\end{{enumerate}}

\newpage

\begin{{center}}
    {{\bf Сведения об авторах}}
\end{{center}}

TODO: для каждого автора 10--15 строк: ФИО, степень, звание/должность/место работы, интересы до 15 слов, e-mail, ORCID/web page.

\newpage

\begin{{center}}
    \section*{{TODO: English title}}
\end{{center}}

\begin{{center}}
TODO: Authors and affiliations in English
\end{{center}}

\begin{{center}}
    {{\bf Abstract}}
\end{{center}}

TODO: English abstract.

\underline{{\it{{Keywords}}}}: TODO.

\underline{{\it{{Citation}}}}: TODO.

\begin{{center}}
    {{\bf About authors}}
\end{{center}}

TODO: author information in English.

\end{{document}}
"""


def build_agent_brief(evidence: dict[str, Any]) -> str:
    return f"""# Agent brief: Computer Optics article draft

Generated: {evidence["generated_at"]}

Target format: LaTeX (`draft_computer_optics.tex`) because Computer Optics accepts TEX and publishes an official TEX template.

## Non-negotiable journal constraints

- Article package: TEX or DOCX article, PDF under 10 MB, figures ZIP under 20 MB, signed first page scan if more than one author, and open-publication expert conclusion for Russian Federation authors.
- Regular article target: up to 12 pages. Abstract: 150-250 words, informative, no formulas, no undefined abbreviations, no numbered references.
- Keywords: 5-10 specific terms.
- Structure for a Russian article: title, authors, affiliations, abstract, keywords, citation; Introduction; numbered main sections; Conclusion; Acknowledgements if needed; References; author bios; English title/authors/abstract/keywords/citation/about authors.
- Do not use footnotes. Do not use LaTeX cross-references (`\\label`, `\\ref`, `\\autoref`) for formulas, figures, tables, or sources. Write textual references manually: `рис. 1`, `табл. 1`, `[1]`.
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
"""


def build_checklist(evidence: dict[str, Any]) -> str:
    figure_lines = []
    for fig in evidence.get("figures", []):
        width = fig.get("width_px") or "?"
        height = fig.get("height_px") or "?"
        note = fig.get("journal_note", "")
        figure_lines.append(f"- [ ] Рис. {fig['number']}: `{fig['draft_path']}` ({width}x{height} px). {note}")
    if not figure_lines:
        figure_lines.append("- [ ] Добавить рисунки после первого упоминания в тексте.")
    return "\n".join(
        [
            "# Computer Optics submission checklist",
            "",
            "## Text",
            "",
            "- [ ] Название краткое, без формул, спецсимволов и сокращений.",
            "- [ ] Аннотация 150-250 слов, без формул и неопределённых аббревиатур.",
            "- [ ] 5-10 конкретных ключевых слов.",
            "- [ ] Введение не нумеруется; основной текст нумеруется вручную; заключение не нумеруется.",
            "- [ ] Нет footnotes, `\\label`, `\\ref`, `\\autoref` и автоматических перекрёстных ссылок.",
            "- [ ] Все сокращения раскрыты при первом употреблении.",
            "- [ ] Для русской статьи десятичные дроби в тексте и таблицах используют запятую.",
            "",
            "## Figures",
            "",
            *figure_lines,
            "- [ ] Все подписи начинаются с `Рис. N.` и не имеют точки в конце.",
            "- [ ] Итоговый ZIP с рисунками меньше 20 MB.",
            "",
            "## Tables And References",
            "",
            "- [ ] Все таблицы имеют подписи над таблицей: `Табл. N.` без точки в конце названия.",
            "- [ ] References на английском языке, по порядку цитирования.",
            "- [ ] DOI указан для каждого источника, где он существует.",
            "- [ ] Не используются неопубликованные материалы как источники.",
            "",
            "## Submission Package",
            "",
            "- [ ] TEX и PDF подготовлены; PDF меньше 10 MB.",
            "- [ ] Скан первой страницы с подписями всех авторов приложен, если авторов больше одного.",
            "- [ ] Экспертное заключение о возможности открытого опубликования подготовлено, если требуется.",
        ]
    )


def write_outputs(out_dir: Path, evidence: dict[str, Any], latex_text: str, make_zip: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "evidence_pack.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "draft_computer_optics.tex").write_text(latex_text, encoding="utf-8")
    (out_dir / "agent_brief.md").write_text(build_agent_brief(evidence), encoding="utf-8")
    (out_dir / "computer_optics_checklist.md").write_text(build_checklist(evidence), encoding="utf-8")
    if make_zip:
        with zipfile.ZipFile(out_dir / "submission_assets.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in out_dir.rglob("*"):
                if path.name == "submission_assets.zip" or path.is_dir():
                    continue
                archive.write(path, path.relative_to(out_dir))


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    out_dir = (project_root / args.out_dir).resolve()
    default_roots = ["outputs", "local_artifacts", "reports"]
    roots = [project_root / item for item in (args.artifact_root or default_roots)]
    metadata = load_metadata(args.metadata)
    evidence = build_evidence_pack(project_root, roots, metadata, args, out_dir)
    title = metadata.get("title_ru", args.title)
    latex_text = build_latex(evidence, title)
    write_outputs(out_dir, evidence, latex_text, args.make_zip)
    print(out_dir)


if __name__ == "__main__":
    main()
