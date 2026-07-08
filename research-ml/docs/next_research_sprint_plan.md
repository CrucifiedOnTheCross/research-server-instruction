# Next Research Sprint Plan

Дата: 2026-07-08.

Основание: внутреннее резюме `stage1_stage2_research_summary.md` и внешний разбор `external_deep_review_stage1_stage2.md`.

## Главная позиция

Нельзя сразу запускать новый большой раунд генерации. Stage2 уже показал, что простая связка `SD v1.5 img2img -> feature-space filtering -> normal mixed training` не превосходит сильные real-only baseline. Следующий спринт должен объяснить, **почему** синтетика не дала прирост, и закрыть воспроизводимость.

## Sprint A. Воспроизводимость

Цель: каждый запуск должен быть полностью отслеживаемым.

Задачи:

1. Превратить серверную папку `/srv/research/projects/default/research-ml` в git-tracked working copy или передавать commit hash из локального репозитория при запуске.
2. Добавить `run_registry.jsonl` в `outputs/`, куда каждый run пишет:
   - experiment name;
   - run dir;
   - command;
   - config path;
   - overrides;
   - git commit;
   - docker image;
   - hostname;
   - start/end time;
   - status;
   - selected synthetic manifest checksum, если есть.
3. Добавить checksum для:
   - train/val/test CSV;
   - synthetic manifest;
   - selected synthetic CSV;
   - resolved config.
4. Исправить `git_commit=None` в `environment.json`.

Ожидаемый результат:

- новый run нельзя интерпретировать без версии кода и manifest checksum;
- все будущие эксперименты пригодны для статьи.

## Sprint B. Диагностика stage2 без новой генерации

Цель: понять synthetic-real gap и влияние синтетики на границы классов.

Задачи:

1. Real-vs-synthetic detector:
   - общий AUROC/AUPRC;
   - per-class AUROC/AUPRC;
   - отдельно raw synthetic, strict selected, topk80 selected.
2. UMAP/t-SNE:
   - real train;
   - real test;
   - raw synthetic;
   - strict selected;
   - topk80 selected.
3. kNN audit gallery:
   - synthetic image;
   - nearest real same-class;
   - nearest real confusing-class;
   - feature distances/margins.
4. Confusion-delta report:
   - `stage1_cross_entropy_weighted` vs `stage2_topk80_ce_weighted`;
   - `stage1_balanced_softmax_none` vs `stage2_strict/topk80_balanced_softmax_none`.
5. Prompt audit:
   - token length for prompt and negative prompt;
   - detect truncation;
   - produce shorter negative prompt candidates.

Ожидаемый результат:

- HTML report в `ham10000/reports/stage2_diagnostics/`;
- JSON/CSV diagnostics для последующего анализа;
- решение: продолжать synthetic path или переключиться на representation-first baselines.

## Sprint C. Подтверждение устойчивости

Цель: отделить реальный эффект от шума одного seed.

Минимальная матрица:

| Config | Seeds |
|---|---:|
| `stage1_cross_entropy_weighted` | 3 |
| `stage1_balanced_softmax_none` | 3 |
| `stage2_topk80_ce_weighted` | 3 |
| `stage2_strict_ce_weighted` | 3, если хватит времени |

Метрики:

- mean/std macro F1;
- mean/std balanced accuracy;
- mean/std MCC;
- mean/std ECE;
- mean/std worst-class recall;
- paired bootstrap по `test_predictions.csv`, если реализуем.

Ожидаемый результат:

- доверие к выводу, что stage2 действительно не выигрывает;
- понимание дисперсии на HAM10000 split.

## Sprint D. Low-risk improvements before stage3

Цель: усилить baseline без новой генерации.

Кандидаты:

1. Logit adjustment.
2. LDAM/DRW.
3. Class-balanced loss.
4. Balanced contrastive / supervised contrastive head.
5. Temperature scaling and threshold analysis.
6. Final real-only fine-tune для synthetic-trained models.
7. Synthetic sample weight `0.25`, `0.5`, `1.0`.

Правило:

- не добавлять все сразу;
- каждая новая идея должна сравниваться с `stage1_cross_entropy_weighted` и `stage1_balanced_softmax_none`.

## Sprint E. Stage3 только после диагностики

Stage3 запускается, если diagnostics покажет, что synthetic path можно спасти.

Кандидат stage3 protocol:

1. Boundary-conditioned source selection:
   - выбирать real images с плохой margin;
   - target pairs: `mel/nv`, `mel/bkl`, `akiec/bkl`, `akiec/bcc`.
2. Short negative prompts:
   - без превышения CLIP token limit.
3. Img2img strength grid:
   - `0.15`, `0.25`, `0.35`.
4. Guidance grid:
   - `2.0`, `4.0`, `6.0`.
5. Training usage:
   - synthetic weight;
   - synthetic-aware branch;
   - final real-only fine-tune.

Критерий запуска большого multi-seed:

- stage3 single-seed должен превзойти `stage1_cross_entropy_weighted` хотя бы по двум из трех:
  - balanced accuracy;
  - worst-class recall;
  - macro F1;
- при этом ECE не должен ухудшиться более чем на 0.02.

## Decision rules

Если real-vs-synthetic AUROC высокий:

- синтетика легко отделима от real;
- приоритет: synthetic-aware training, lower synthetic weight, final real-only fine-tune.

Если UMAP показывает отдельные synthetic clusters:

- приоритет: boundary-conditioned generation и stricter selection;
- не увеличивать synthetic ratio.

Если strict selection снова отбрасывает `akiec`:

- не генерировать `akiec` через текущий SD v1.5 protocol;
- попробовать другой generator или source selection.

Если multi-seed подтверждает, что stage2 ниже baseline:

- сделать synthetic-path negative result частью научного вклада;
- усилить representation-first часть.

Если synthetic помогает только `akiec`, но ухудшает `mel`:

- не использовать общий synthetic training pool;
- перейти к pair-specific или class-specific sampling/weights.

## Deliverables

1. `outputs/run_registry.jsonl`
2. `ham10000/reports/stage2_diagnostics/index.html`
3. `ham10000/reports/stage2_diagnostics/real_vs_synthetic_metrics.json`
4. `ham10000/reports/stage2_diagnostics/umap_real_synthetic.png`
5. `ham10000/reports/stage2_diagnostics/knn_gallery/index.html`
6. `ham10000/reports/stage2_diagnostics/confusion_delta.md`
7. `ham10000/reports/stage2_diagnostics/prompt_audit.md`
8. `docs/stage3_protocol.md`, только после Sprint B-C.

## First implementation tasks

1. Add run registry and checksum utilities.
2. Add `tools/analyze_synthetic_gap.py`.
3. Add `tools/build_knn_gallery.py`.
4. Add `tools/audit_generation_prompts.py`.
5. Add `tools/compare_runs.py`.

Эти задачи не требуют новой генерации и почти не требуют GPU, кроме feature extraction для synthetic-gap анализа.
