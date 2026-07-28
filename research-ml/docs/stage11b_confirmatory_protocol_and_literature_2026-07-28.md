# Stage 11B: confirmatory transfer of strict-ID synthetic utility

Дата фиксации протокола: 2026-07-28, до запуска Stage 11B.

Статус запуска: `running`, container `research-stage11b-confirmatory`.

Проверки перед запуском:

- server unit tests: 25/25 passed;
- Stage 11B gate: open;
- locked test: не открывался;
- GPU smoke: batch 32, 384 px, BF16, peak VRAM 9.00 GiB, finite loss;
- первый training run: GPU utilization 99%, VRAM 10.5 GiB;
- commit зафиксированного протокола и кода: `3905b4d`.

## Научный вопрос

Stage 10 показал, что 90 синтетических изображений из `strict_id` stratum
превосходили source-matched real replay по MCC (`+0.0661`, lesion-bootstrap
95% CI `[+0.0047; +0.1255]`) и macro AUPRC (`+0.0174`, 3/3 seeds), но macro
F1 имел неопределённый CI, а melanoma recall снизился на `0.0765`.

Stage 11A затем выбрал более сильный и экономичный real-only baseline:
ConvNeXt-Small с 49.5M параметров, macro F1 `0.7699`, MCC `0.6610` и macro
AUPRC `0.8068`.

Stage 11B отвечает на один заранее заданный вопрос:

> Сохраняется ли преимущество содержимого отобранной синтетики над простым
> повторным предъявлением тех же реальных источников при переносе на
> квалифицированный ConvNeXt-Small recipe?

Это не тест того, полезна ли любая синтетика, и не сравнение генераторов.

## Литературное основание

### Yamaguchi, ACML/PMLR 2025

`Analyzing Diffusion Models on Synthesizing Training Datasets` показывает, что
diffusion samples могут концентрироваться около мод реального распределения и
хуже покрывать его внешние области. При этом они могут быть полезны как
интерполяционная добавка к real data, хотя не заменяют real dataset при
одинаковом размере.

Следствие для проекта:

- синтетика используется только как малая добавка к 7228 real rows;
- сравнение проводится с count- и source-matched real replay;
- utility доказывается downstream-метриками, а не реалистичностью.

Источник: https://proceedings.mlr.press/v260/yamaguchi25a.html

### Adamkiewicz et al., CVPR 2026

`When Pretty Isn't Useful` показывает, что рост визуального качества и
text-image alignment не гарантирует Synth-to-Real utility. Авторы связывают
провал с узким high-density/low-coverage распределением, ухудшением текстур и
высокочастотных деталей. В открытом коде отдельно считаются PRDC
precision/recall/density/coverage и Vendi diversity.

Следствие:

- Stage 11B не использует visual quality как endpoint;
- геометрический `strict_id` рассматривается как гипотеза о полезной области
  пространства признаков, которую нужно подтвердить downstream;
- после Stage 11B полезно отдельно проверить texture/frequency gap и coverage
  на дерматоскопическом encoder, но не добавлять эту проверку post-hoc в
  текущий confirmatory contrast.

Статья: https://arxiv.org/abs/2602.19946

Код: https://github.com/Bill2462/When-Pretty-Isn-t-Useful-codebase

### Yuan et al., ICML 2024

`Not Just Pretty Pictures: Toward Interventional Data Augmentation Using
Text-to-Image Generators` сравнивает prompting, conditioning и post-hoc
filtering. Наиболее важен тип conditioning; filtering не был стабильно полезен
во всех задачах. В коде pre-generated варианты привязаны к исходному
изображению, а во время обучения вариант выбирается внутри этой связи.

Следствие:

- сохраняется source provenance;
- source-matched replay является обязательным контролем;
- положительный результат `strict_id` нельзя обобщать на любой фильтр.

Статья: https://proceedings.mlr.press/v235/yuan24e.html

Код: https://github.com/YuanJianhao508/NotJustPrettyPictures

### Wang et al., CVPR 2024, Diff-Mix

Diff-Mix разделяет faithful foreground и contextual diversity, использует
управляемую image translation и явно задаёт долю синтетики в downstream
training (`syndata_p`). Открытая реализация подтверждает необходимость
контролировать generation strength, synthetic multiplier/ratio и downstream
recipe.

Следствие:

- в Stage 11B доза и вес синтетики фиксированы;
- изменение генератора, strength или synthetic ratio относится к следующей
  абляции, а не к текущей проверке;
- arm отличается от replay только содержанием добавленных изображений.

Статья: https://openaccess.thecvf.com/content/CVPR2024/html/Wang_Enhance_Image_Classification_via_Inter-Class_Image_Mixup_with_Diffusion_Model_CVPR_2024_paper.html

Код: https://github.com/Zhicaiwww/Diff-Mix

### SkinGenBench, arXiv 2025

SkinGenBench сравнивает StyleGAN2-ADA и DDPM на дерматоскопических данных и
оценивает одновременно FID/KID/IS, downstream classifiers и melanoma
endpoints. Работа полезна как близкий доменный benchmark, но её крупные
заявленные приросты нельзя напрямую сравнивать с нашим protocol без проверки
lesion-group split, состава HAM10000+MILK10K и equal-dose real controls.

Следствие:

- melanoma F1/AUPRC/recall должны сообщаться отдельно;
- generator architecture является будущим фактором, если текущая
  source-matched гипотеза подтвердится;
- Stage 11B сохраняет наш более строгий lesion-group protocol.

Статья: https://arxiv.org/abs/2512.17585

Код: https://github.com/adarsh-crafts/SkinGenBench

## Аудит открытых кодовых баз

Проверены:

1. `Bill2462/When-Pretty-Isn-t-Useful-codebase`: отдельные модули classifier,
   generation и distribution metrics; PRDC считается на balanced real/fake
   class slices, Vendi предусмотрен как дополнительная diversity metric.
2. `Zhicaiwww/Diff-Mix`: LoRA/DreamBooth generation, явные generation
   strength и multiplier, downstream `syndata_p`, отдельный long-tail runner.
3. `YuanJianhao508/NotJustPrettyPictures`: source-to-generated mapping,
   pre-generation и выбор связанной интервенции во время обучения.
4. `adarsh-crafts/SkinGenBench`: единый dermoscopy benchmark с generative,
   downstream и explainability уровнями.

Код из внешних репозиториев не копировался. В проект перенесены только
экспериментальные принципы: source provenance, controlled dose, downstream
utility, distribution coverage и minority endpoints.

## Зафиксированный дизайн

| Компонент | Значение |
|---|---|
| Dataset | HAM10000, lesion-group split |
| Base real train | 7228 строк |
| Validation | 1280 real-only изображений, 599 lesion groups |
| Locked test | закрыт, `evaluation.run_test=false` |
| Backbone | `convnext_small.fb_in22k_ft_in1k_384` |
| Resolution | 384 px, eval resize 438 + center crop |
| Seeds | 42, 43, 44 |
| Arm A | 90 source-matched real replay rows |
| Arm B | 90 `strict_id` synthetic rows |
| Dose | 30 `mel` + 30 `akiec` + 30 `bkl` |
| Effective added-row weight | 0.5 в обоих arms |
| Sampler | natural, без class-balanced resampling |
| Optimizer recipe | Stage 11 ConvNeXt-S: LR 1e-4, layer decay 0.925 |
| Schedule | 80 epochs, warmup 5, patience 15 |
| Regularization | drop path 0.2, label smoothing 0.05, EMA 0.9999 |

Natural sampler выбран потому, что Stage 11B должен переносить synthetic
contrast на квалифицированный Stage 11 baseline. Weighted sampler Stage 10 не
переносится: он менял бы одновременно baseline recipe и effective class
distribution.

## Preflight gate

`tools/check_stage11b_gate.py` закрывает запуск, если:

- не завершены три валидных ConvNeXt-S qualification runs;
- любой qualification run открыл locked test;
- configs отличаются от квалифицированного recipe;
- число или class dose добавлений различаются;
- 7228 базовых real rows не совпадают;
- synthetic rows не совпадают с зафиксированным `strict_id`;
- replay-to-synthetic source mapping нарушен;
- sample weight не равен 0.5 в обоих arms;
- train/source lesion groups пересекаются с validation/test;
- отсутствует structured artifact или provenance manifest.

Gate сохраняет SHA-256 всех входных CSV, manifest и eval splits в
`outputs/reports/stage11b_gate.json`.

## Endpoints и inference

Primary:

- macro F1;
- MCC;
- balanced accuracy.

Minority guardrails:

- melanoma precision, recall, F1;
- melanoma AUROC и AUPRC;
- sensitivity при fixed specificity 0.90 и 0.95.

Secondary:

- macro/per-class AUROC и AUPRC;
- worst-class recall;
- ECE, NLL и temperature-scaled diagnostics;
- runtime и best epoch.

Inference:

- paired differences synthetic minus replay для каждого seed;
- устойчивость знака на 3/3 seeds;
- exact sign-flip diagnostic;
- hierarchical bootstrap по seed и `lesion_id/group_id`, 5000 повторов;
- метрики пересчитываются из predictions, а не извлекаются из noisy logs.

## Заранее заданная интерпретация

`strong_positive`:

- macro F1, MCC и balanced accuracy имеют положительный mean delta;
- macro F1 и MCC положительны на 3/3 seeds;
- lesion-bootstrap lower CI выше нуля хотя бы для macro F1 или MCC;
- mean melanoma recall delta не ниже `-0.05`;
- melanoma AUPRC не ухудшается на всех трёх seeds.

`mixed_positive`:

- глобальные метрики положительны и устойчивы, но CI пересекает ноль или
  нарушен melanoma guardrail.

`weak_or_unstable_positive`:

- среднее улучшение есть, но знак нестабилен.

`null_or_negative`:

- нет согласованного глобального улучшения относительно replay.

Даже `strong_positive` остаётся внутренним validation confirmation и сам по
себе не разрешает открывать locked test.

## Код и запуск

- configs:
  - `configs/ham10000_stage11b_replay_strict_id_convnext_small_384.yaml`;
  - `configs/ham10000_stage11b_synthetic_strict_id_convnext_small_384.yaml`;
- gate: `tools/check_stage11b_gate.py`;
- training: `scripts/run_stage11b_confirmatory.sh`;
- container: `scripts/start_stage11b_container.sh`;
- analysis: `tools/analyze_stage11b_results.py`;
- analysis runner: `scripts/run_stage11b_analysis.sh`;
- regression tests: `tests/test_stage11b_protocol.py`.

Ожидаемое время на RTX 5080: около 2.5-3.5 часа для шести runs, затем
20-40 минут на calibration, bootstrap, проверку и документирование.

## Решение после Stage 11B

1. `strong_positive`: один заранее зафиксированный финальный locked-test
   contrast и подготовка внешней проверки ISIC/MILK10K.
2. `mixed_positive`: не открывать test; исследовать texture/frequency coverage
   и изменить generation/selection, сохранив source-matched controls.
3. `weak/null/negative`: сформулировать статью вокруг отрицательного результата:
   feature-space близость недостаточна, а простое replay не хуже синтетики.

## Фактические результаты

Дата анализа: 2026-07-28.

Статус:

- 6/6 runs завершены;
- seeds 42, 43, 44 для обоих arms;
- `test_evaluated=false` во всех runs;
- validation: 1280 изображений, 599 lesion groups;
- metric recomputation max absolute error: `8.32e-8`;
- hierarchical lesion bootstrap: 5000 повторов;
- prediction alignment и artifact integrity: passed.

### Multi-seed summary

| Arm | Macro F1 | MCC | Balanced acc. | Macro AUPRC | ECE | Mel F1 | Mel AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Synthetic strict-ID | 0.7693 ± 0.0095 | 0.6611 ± 0.0123 | 0.7779 ± 0.0092 | 0.7898 ± 0.0184 | 0.1261 | 0.5372 | 0.5024 |
| Source-matched replay | 0.7577 ± 0.0084 | 0.6515 ± 0.0046 | 0.7678 ± 0.0062 | **0.8034 ± 0.0191** | **0.1026** | 0.5295 | **0.5471** |

### Paired synthetic-minus-replay

| Endpoint | Mean delta | Synthetic wins | 95% seed-t CI |
|---|---:|---:|---:|
| Macro F1 | +0.0115 | 3/3 | [-0.0089; +0.0320] |
| Balanced accuracy | +0.0101 | 3/3 | [-0.0142; +0.0343] |
| MCC | +0.0096 | 2/3 | [-0.0243; +0.0434] |
| Macro AUPRC | **-0.0136** | **0/3** | [-0.0336; +0.0063] |
| Macro AUROC | -0.0094 | 1/3 | [-0.0368; +0.0180] |
| ECE | **+0.0235 хуже** | **0/3 лучше** | [-0.0215; +0.0685] |
| Melanoma F1 | +0.0077 | 2/3 | [-0.0735; +0.0889] |
| Melanoma recall | 0.0000 | 2/3 | [-0.0301; +0.0301] |
| Melanoma AUPRC | **-0.0447** | **0/3** | **[-0.0885; -0.0009]** |
| Melanoma AUROC | -0.0051 | 1/3 | [-0.0438; +0.0336] |

Lesion-group bootstrap:

| Endpoint | Mean delta | 95% CI | P(delta > 0) |
|---|---:|---:|---:|
| Macro F1 | +0.0120 | [-0.0098; +0.0350] | 0.884 |
| Balanced accuracy | +0.0101 | [-0.0143; +0.0353] | 0.811 |
| MCC | +0.0098 | [-0.0155; +0.0352] | 0.780 |
| Melanoma F1 | +0.0078 | [-0.0407; +0.0527] | 0.650 |
| Melanoma recall | +0.0004 | [-0.0523; +0.0535] | 0.487 |

Ни один primary endpoint не имеет bootstrap lower CI выше нуля.

### Operating points и calibration

При exploratory specificity 0.90 synthetic имеет mean melanoma sensitivity
`0.694`, replay `0.685`: разница мала и нестабильна. При specificity 0.95
synthetic sensitivity ниже: `0.491` против `0.528`; precision также ниже.

Temperature scaling уменьшает ECE у обоих arms, но это group-held-out
диагностика после validation-based early stopping. Она не исправляет ухудшение
melanoma ranking и не является confirmatory endpoint.

### Заранее заданное решение

Автоматическая классификация:

`weak_or_unstable_positive`.

Причины:

- глобальное направление macro F1/MCC/balanced accuracy положительное;
- macro F1 и balanced accuracy положительны на 3/3 seeds;
- MCC положителен только на 2/3 seeds;
- bootstrap CI primary endpoints пересекают ноль;
- macro AUPRC ухудшился на 3/3 seeds;
- melanoma AUPRC ухудшился на 3/3 seeds, mean delta `-0.0447`;
- melanoma guardrail не пройден.

Следовательно, сильная гипотеза Stage 11B **не подтверждена**. Strict-ID
синтетика немного меняет argmax decision geometry, но не демонстрирует
добавочной ranking information относительно предъявления реальных источников.
Наблюдаемая комбинация `macro F1 вверх / AUPRC вниз / ECE хуже` совместима с
изменением порогов и decision boundary, а не с устойчивым улучшением
представления редких классов.

Locked test не открывается.

## Следующая проверяемая гипотеза

Наиболее обоснованный следующий этап не должен повторять ещё один classifier
screen с тем же synthetic pool. Сначала требуется диагностировать, почему
`strict_id` близость не переносится в ranking utility:

1. Посчитать class-wise PRDC density/coverage и Vendi diversity в
   дерматоскопическом feature space, а не только nearest-real distance.
2. Проверить texture/high-frequency gap real-vs-synthetic, поскольку
   высокоуровневая DINOv2 близость может скрывать генеративные артефакты.
3. Оценить source-conditional novelty: расстояние synthetic до собственного
   source относительно ближайших real non-source neighbours.
4. Построить error-overlap и probability-shift анализ synthetic против replay,
   особенно для `mel`, `akiec` и `bkl`.
5. Перегенерировать только после определения failure mode: менять diversity,
   conditioning или strength, сохраняя count/source-matched replay.
6. Любой новый pool сначала проверять geometry/texture/coverage gate, затем
   выполнять один малый paired screening без открытия locked test.

Это переводит отрицательный Stage 11B результат в более сильный научный тезис:
близость к real manifold является недостаточным критерием полезности
синтетических медицинских изображений.

## Ограничения

- только три training seeds;
- один внутренний validation split участвует в early stopping;
- источник synthetic geometry определён DINOv2-признаками, classifier другой;
- 90 изображений дают малую statistical power для небольших эффектов;
- melanoma guardrail важнее красивого среднего macro F1;
- внешний cohort нужен до сильных заявлений о клиническом переносе.
