# Аудит готовности статьи по результатам Stage 1-16G

Дата среза: 2026-07-31.

## Решение

**Статью уже можно и нужно писать, но отправлять её в текущем состоянии
преждевременно.**

Материал поддерживает полноценную исследовательскую статью с отрицательным и
диагностическим результатом. Он не поддерживает статью с тезисом
«синтетические изображения повышают качество классификации».

Наиболее защищаемая формулировка:

> В контролируемой классификации дерматоскопических изображений визуальная
> правдоподобность, близость к реальному manifold в одном пространстве
> признаков и улучшение генеративных precision/coverage-метрик не гарантируют
> добавочную ranking utility относительно source-matched real replay.
> Полезность определяется одновременно геометрией в task-relevant
> представлениях, preprocessing, частотной структурой, сохранением диагноза и
> рабочей точкой классификатора.

Оценка готовности:

| Компонент | Статус |
|---|---|
| Научный вопрос | готов |
| Воспроизводимый код и structured artifacts | готов |
| Сильные real-only и простые imbalance baselines | готов |
| Source/count-matched replay control | готов |
| Multi-seed и lesion-group bootstrap | готов |
| Механистический анализ failure modes | готов |
| Причинное разложение generator pipeline | готов |
| Перенос data protocol на ISIC 2019 | готов |
| Одна финальная confirmatory оценка | не выполнена |
| Locked test | не открывался |
| Независимая внешняя utility cohort | не выполнена |
| Клиническая экспертная оценка синтетики | не выполнена |

Итого: **готовность к написанию около 80%, готовность к отправке около 60-65%**.

## Какую статью писать

### Рекомендуемое название

**Контролируемая оценка полезности синтетических дерматоскопических
изображений при дисбалансе классов**

Более исследовательский вариант:

**Почему близость в пространстве признаков не гарантирует полезность
синтетических дерматоскопических изображений**

### Тип статьи

Обычная исследовательская статья до 12 страниц, раздел журнала
«Анализ и понимание изображений, распознавание образов» или «Цифровая обработка
сигналов и изображений».

Тематика соответствует журналу: в «Компьютерной оптике» уже публиковались
работы по сегментации и классификации дерматоскопических изображений.
Актуальные правила допускают статьи на русском или английском языке,
рекомендуют объём до 12 страниц и требуют информативную аннотацию
150-250 слов:
<https://computeroptics.ru/guidelines.htm>.

### Не делать основной статьёй

1. Не пытаться описать Stage 1-16 как последовательный журнал экспериментов.
2. Не заявлять клиническое улучшение по одному росту melanoma recall.
3. Не представлять PRDC, DINOv2 distance, FID или визуальное качество как
   surrogate downstream utility.
4. Не объединять HAM10000 и ISIC 2019 метрики в одну таблицу как результаты
   одного датасета.
5. Не утверждать бесполезность всей синтетики: проверены конкретные SD1.5,
   selection и LoRA pipelines.

## Что считать основным доказательством

### 1. Исправленный HAM10000 protocol

Stage 8 устранил повторное использование test, грубую чёрную рамку,
source-lesion leakage и circular single-encoder selection. Все дальнейшие
сравнения использовали real-only validation, group-aware split и закрытый
locked test.

Stage 8 показал, что Synthetic+DINO не превзошёл weighted real-only baseline:

| Контраст synthetic minus weighted CE | Delta |
|---|---:|
| Macro F1 | -0.0181 |
| MCC | -0.0376 |
| Melanoma recall | +0.0420 |
| Melanoma precision | -0.0694 |
| Melanoma F1 | -0.0420 |

Это первый устойчивый результат статьи: рост recall без precision, F1 и MCC
является сдвигом operating point, а не общим улучшением модели.

### 2. Простые методы и source replay оказались сильными контролями

Stage 9 сравнил synthetic с oversampling, undersampling, Balanced Softmax,
cRT и source-matched replay.

- synthetic против source replay: macro F1 `-0.0006`, MCC `-0.0054`;
- synthetic против oversampling: macro F1 `-0.0306`, MCC `-0.0453`;
- synthetic превзошёл canonical undersampling, но undersampling использовал
  только 567 уникальных изображений за эпоху и терял разнообразие head-класса.

Поэтому статья отвечает на важный reviewer question: эффект не сравнивается
только с «ничего не делать»; он проверен против дешёвых методов и matched
реального контроля.

### 3. Geometry действительно модерирует эффект, но не гарантирует его

Stage 10 дал наиболее интересный exploratory механизм:

| Stratum, synthetic minus replay | Macro F1 | MCC | Macro AUPRC |
|---|---:|---:|---:|
| `strict_id` | +0.0263 | +0.0661 | +0.0174 |
| `aid_radial` | +0.0025 | +0.0167 | -0.002 |
| `ood_far` | -0.0241 | -0.0282 | -0.026 |
| `random_remaining` | -0.0194 | -0.0488 | отрицательный |

Для `strict_id` MCC bootstrap CI `[+0.0047; +0.1255]`, а для random MCC CI
`[-0.0856; -0.0131]`. Это поддерживает тезис о geometry-dependent utility.

Но Stage 11B на более сильном ConvNeXt-S показал, что положительный argmax
сигнал не переносится в ranking:

| Stage 11B, strict synthetic minus replay | Delta | Победы |
|---|---:|---:|
| Macro F1 | +0.0115 | 3/3 |
| MCC | +0.0096 | 2/3 |
| Macro AUPRC | -0.0136 | 0/3 |
| Melanoma AUPRC | -0.0447 | 0/3 |
| ECE | +0.0235, хуже | 0/3 лучше |

Ни один primary bootstrap interval не имел положительной нижней границы.
Это центральный результат: **threshold metrics и ranking metrics отвечают на
разные вопросы, а близость в одном encoder space недостаточна**.

### 4. Stage 12 объяснил механизм отрицательного переноса

По сравнению с matched replay:

- synthetic precision/density/coverage были ниже в 11 из 12
  `encoder x class` срезов;
- Vendi diversity была выше в 11 из 12, поэтому простого mode collapse нет;
- DINOv2 считал samples source-like, но task-specific ConvNeXt-S видел
  смещение за пределы source neighbourhood;
- частотный gap подтверждён в 14 из 15 paired class-metric intervals;
- для melanoma снизился positive `q10`, а negative `q90` вырос на `+0.1127`;
- главный confuser shift наблюдался между `akiec` и `mel`.

Следовательно, синтетика могла быть разнообразной, но разнообразие лежало не
в полезной class support и ухудшало ranking tail.

### 5. Улучшенный multi-encoder selector не решил проблему

Stage 13A увеличил coverage относительно `strict_id` во всех 12 encoder-class
срезах, precision в среднем на `+0.3083`, density на `+0.4317`. Однако Stage
13B снова проиграл replay:

| Metric | Synthetic minus replay |
|---|---:|
| Macro F1 | -0.0033 |
| MCC | -0.0022 |
| Macro AUPRC | -0.0136, 0/3 wins |
| Melanoma AUPRC | -0.0198, 0/3 wins |
| ECE | +0.0178, хуже |

Это усиливает новизну: даже multi-encoder PRDC improvement не является
достаточной objective function для отбора.

### 6. Stage 15A причинно разложил generator pipeline

Аудит установил, что 87 из 90 Stage 13B изображений были созданы при
`strength=0.05` и 30 шагах, то есть фактически примерно с одним denoising
step. При этом смешивались offline square crop, fixed view, SD1.5 VAE и UNet.

Stage 15A заранее зафиксировал четыре arms:

1. original source replay;
2. offline crop;
3. crop + VAE round-trip;
4. crop + VAE + one-step img2img.

| Причинный контраст | Macro AUPRC | Macro F1 | MCC |
|---|---:|---:|---:|
| Offline crop minus replay | -0.0064 | -0.0084 | -0.0154 |
| VAE minus crop | -0.0003 | +0.0028 | -0.0030 |
| One-step UNet minus VAE | -0.0016 | -0.0193 | -0.0104 |
| Full pipeline minus replay | -0.0083 | -0.0249 | -0.0288 |

Для полного pipeline MCC bootstrap CI `[-0.0578; -0.0004]`. Основной ranking
loss объясняется offline crop, а generic VAE сам по себе почти нейтрален.
Один UNet step ухудшает threshold behaviour без новой ranking information.

Это самый сильный законченный эксперимент текущей рукописи.

## Что даёт перенос на ISIC 2019

Stage 16 не следует смешивать с HAM10000 как ещё один fold. Это отдельная
scale-out ветка, которая усиливает раздел о переносимости протокола.

### Данные и baseline

- 25 331 official rows;
- 25 327 usable rows;
- 4 exact-duplicate/label-conflict rows quarantined;
- 13 926 connected groups;
- train/validation/locked test: 17 729 / 3 799 / 3 799;
- все восемь классов присутствуют во всех split;
- ConvNeXt-Small 384, batch 48, monitor macro AUPRC.

Balanced Softmax относительно Natural CE:

- macro AUPRC `+0.0011`, CI пересекает ноль;
- balanced accuracy `+0.0340`, CI `[+0.0153; +0.0548]`;
- macro F1 `-0.0161`;
- ECE `+0.0871`, хуже;
- melanoma AUPRC `-0.0119`, 0/3 seeds.

На большем мультицентровом наборе снова воспроизводится общий паттерн:
балансировка меняет operating point, но не гарантирует ranking или
calibration utility.

### Generator qualification

Сегментатор прошёл независимый train-only gate: Dice `0.9229`, IoU `0.8636`;
893 lesion-unique pseudo-masks допущены к генераторному smoke.

Но все генераторы были правильно заблокированы до downstream:

| Generator arm | Label agreement | Mask IoU | Решение |
|---|---:|---:|---|
| Historical SD1.5 img2img | 0.625 | 0.827 | fail, near-copy |
| Generic SD1.5 inpainting | 0.333 | 0.637 | fail |
| Domain-LoRA img2img | 0.312 | 0.610 | fail |
| Domain-LoRA inpainting | 0.333 | 0.680 | fail |

LoRA улучшила сохранение mask/background, но не diagnosis conditioning.
Это хороший независимый отрицательный результат для discussion, но не
downstream evidence и не доказательство бесполезности domain generators.

## Что включить в одну статью

### Основной текст

1. Постановка long-tail synthetic utility и опасность surrogate quality.
2. Leakage-safe lesion-group protocol и matched controls.
3. Квалификация ConvNeXt-S baseline.
4. Geometry-stratified synthetic versus replay.
5. Transfer check Stage 11B.
6. Multi-encoder, frequency и ranking-tail diagnostics.
7. Causal decomposition Stage 15A.
8. Ограниченная ISIC 2019 scale-out демонстрация протокола.

### Оставить в приложении или репозитории

- Stage 1-7;
- полную матрицу Stage 8/9;
- все calibration tables;
- per-class metrics для каждого seed;
- Stage 13 candidate-capacity iterations;
- infrastructure incidents;
- полные P0/P1 contact sheets.

### Не включать как отдельные claims

- post-hoc AUPRC-optimal checkpoints Stage 11B/13B;
- secondary source interaction Stage 16P по одному seed;
- temperature scaling после validation-based early stopping;
- визуальную оценку без врача как clinical realism.

## Рекомендуемые рисунки и таблицы

### Рисунки

1. Схема leakage-safe protocol: real train, source-conditioned generation,
   train-only selection, source-matched replay, real-only validation.
2. Forest plot Stage 10 geometry strata: synthetic minus replay по MCC и
   macro AUPRC.
3. Stage 12: PRDC coverage synthetic/replay в DINOv2 и ConvNeXt-S.
4. Stage 12: frequency shift и melanoma ranking-tail mechanism.
5. Stage 15A causal waterfall: replay -> crop -> VAE -> one-step UNet.
6. Небольшая generator qualification panel P0/P1 как отрицательный контроль.

### Основные таблицы

1. Dataset/split и leakage controls.
2. Strong real-only baseline qualification.
3. Stage 10 и Stage 11B paired effects.
4. Stage 12 mechanism summary.
5. Stage 15A causal contrasts.

При лимите 12 страниц ISIC 2019 baseline и P0/P1 лучше объединить в одну
короткую таблицу robustness/scale-out либо вынести в supplementary repository.

## Что блокирует отправку

### Обязательный минимум

1. **Заморозить статью до нового просмотра результатов.**
   Зафиксировать центральный тезис, primary contrast и statistical plan.
2. **Один раз открыть HAM10000 locked test только для Stage 15A.**
   Оценить уже сохранённые 12 checkpoints четырёх arms без переобучения и
   без выбора новой эпохи.
3. **Заранее определить multiplicity.**
   Primary contrast: full img2img pipeline minus original replay по macro
   AUPRC. Secondary causal contrasts: crop-replay, VAE-crop, UNet-VAE.
4. **После locked test не менять метод.**
   Любой дальнейший generator относится к другой статье.
5. **Собрать финальные checksum, config и environment tables.**

Такой пакет достаточен для честной внутренней controlled study в
«Компьютерной оптике», если выводы ограничены проверенными pipelines.

### Усиленный вариант

Дополнительно выполнить один из вариантов:

- repeated group-aware folds для Stage 15A A/D;
- независимую external evaluation на dataset без HAM/ISIC source overlap;
- независимую экспертную оценку небольшого стратифицированного generator
  sample.

Repeated folds лучше оценивают split uncertainty, но требуют нового обучения.
External cohort сильнее для generalization, однако нуждается в совместимом
label mapping. Экспертная оценка полезна для generator failure taxonomy, но
не заменяет downstream utility.

## Риски рецензирования

| Возможный вопрос | Текущий ответ |
|---|---|
| Почему отрицательный результат интересен? | Есть matched controls, geometry interaction, representation mismatch и causal decomposition |
| Может быть baseline слабый? | ConvNeXt-S квалифицирован против ConvNeXt-B и DINOv2 |
| Может быть достаточно oversampling? | Да; это часть результата, synthetic не превосходит его устойчиво |
| Может быть эффект только threshold? | Проверены AUPRC, fixed specificity, calibration и ranking tails |
| Может быть selection плохой? | Проверены single- и multi-encoder geometry, PRDC, Vendi и facility location |
| Может быть генератор слишком слабый? | Да, поэтому claims ограничены SD1.5/LoRA pipelines; P0/P1 отдельно провалили qualification |
| Есть ли leakage? | Stage 8+ использует lesion/source groups; audit не нашёл validation/test overlap |
| Почему нет test? | До отправки требуется единственное frozen opening |
| Есть ли клинические claims? | Нет; речь о классификации изображений и ranking endpoints |

## Предлагаемая структура рукописи

1. Введение.
2. Данные, lesion-group split и threat model.
3. Генерация, feature-space selection и matched controls.
4. Baseline qualification и метрики.
5. Geometry-dependent utility.
6. Почему положительный argmax signal не переносится в ranking.
7. Причинное разложение preprocessing, VAE и denoising.
8. Обсуждение переноса на ISIC 2019 и generator qualification.
9. Ограничения.
10. Заключение.

## Итог

Проект уже содержит публикационно интересный и достаточно редкий вклад:
не просто отрицательный benchmark, а последовательное объяснение, почему
синтетика может выглядеть качественной, быть близкой к real samples и
улучшать отдельные threshold metrics, но всё равно не добавлять полезной
ranking information.

**Рекомендация: начать писать рукопись сейчас, заморозить Stage 15A как
финальный HAM10000 contrast и подготовить единственное locked-test opening.
Stage 16G-P2 и более сильный lesion-aware generator вести как отдельную
следующую работу, чтобы не размывать текущую статью и не продолжать
оптимизацию по одной validation выборке.**

Машиночитаемая карта доказательств:
`reports/article_readiness_2026-07-31/evidence_matrix.csv`.
