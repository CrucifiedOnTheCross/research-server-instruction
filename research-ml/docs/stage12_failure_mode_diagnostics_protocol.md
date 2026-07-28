# Stage 12: failure-mode diagnostics after Stage 11B

Дата протокола: 2026-07-29.

## Исходный результат

Stage 11B не подтвердил сильную utility-гипотезу для 90 `strict_id`
изображений. По сравнению с count- и source-matched real replay синтетика
дала небольшое улучшение argmax-метрик, но ухудшила ranking и calibration:

- macro F1: `+0.0115`, 3/3 seeds, CI пересекает ноль;
- macro AUPRC: `-0.0136`, 0/3 seeds;
- melanoma AUPRC: `-0.0447`, 0/3 seeds;
- ECE: `+0.0235` (хуже).

Locked test остаётся закрытым.

## Проверяемая гипотеза

Близость к реальному manifold в пространстве DINOv2 является необходимым,
но недостаточным критерием полезности синтетических дерматоскопических
изображений. Провал ranking utility может быть связан с одним или несколькими
факторами:

1. `strict_id` имеет высокую локальную fidelity, но недостаточное coverage и
   diversity;
2. близость в DINOv2 не переносится в task-specific пространстве ConvNeXt-S;
3. синтетика почти копирует собственный source и не добавляет
   source-conditional novelty;
4. генерация меняет частотную/текстурную структуру, которую высокоуровневая
   embedding-метрика не обнаруживает;
5. синтетика сдвигает argmax boundary, но не улучшает ранжирование melanoma.

Stage 12 не обучает новые классификаторы и не выбирает новый synthetic pool.
Это снижает риск post-hoc оптимизации по validation.

## Литературное основание

### Kynkäänniemi et al., NeurIPS 2019

`Improved Precision and Recall Metric for Assessing Generative Models`
предлагает непараметрические manifold-оценки fidelity и coverage вместо одной
агрегированной оценки качества.

Источник: https://papers.nips.cc/paper_files/paper/2019/hash/0234c510bc6d908b28c70ff313743079-Abstract.html

### Naeem et al., ICML 2020

`Reliable Fidelity and Diversity Metrics for Generative Models` показывает
ограничения ранних precision/recall-метрик и предлагает density и coverage.
Stage 12 считает PRDC class-wise с `k=5`.

Источник: https://proceedings.mlr.press/v119/naeem20a.html

Код: https://github.com/clovaai/generative-evaluation-prdc

### Friedman and Dieng, TMLR 2023

`The Vendi Score` определяет diversity как экспоненту энтропии собственных
значений similarity matrix. В Stage 12 используется cosine similarity
нормированных признаков; дополнительно сохраняется effective rank.

Источник: https://openreview.net/forum?id=g97OHbQyk1

### Adamkiewicz et al., CVPR 2026

`When Pretty Isn't Useful` связывает низкую downstream utility современной
синтетики с mode concentration, недостаточным coverage и изменением
текстурно-частотных характеристик. Их работа обосновывает совместный анализ
PRDC, diversity и frequency gap, а не вывод по visual quality.

Источник: https://arxiv.org/abs/2602.19946

Код: https://github.com/Bill2462/When-Pretty-Isn-t-Useful-codebase

## Дизайн

### Данные

- 90 `strict_id` synthetic: 30 `mel`, 30 `akiec`, 30 `bkl`;
- 90 source-matched real replay presentations, связанных с synthetic по
  `source_image_id`; один real source может соответствовать нескольким
  synthetic variants;
- real train reference классов `mel`, `akiec`, `bkl`;
- validation predictions Stage 11B для seeds 42, 43, 44.

Lesion-level split не изменяется. Test CSV и test predictions не читаются.

### Пространства признаков

1. Независимый pretrained `vit_base_patch14_dinov2.lvd142m`, использованный
   для Stage 8 selection.
2. Три real-only ConvNeXt-S encoders из Stage 11, seeds 42, 43, 44.

Stage 11B checkpoints не используются для feature diagnostics, чтобы
избежать circularity. Из real reference удаляются все 90 source images:
иначе replay получил бы искусственно идеальную precision за счёт совпадения
с самим собой.

### Метрики

Class-wise для synthetic и source replay:

- PRDC precision, recall, density, coverage (`k=5`);
- Vendi score на cosine kernel;
- effective rank;
- mean nearest-reference cosine distance;
- source-to-synthetic cosine distance;
- nearest non-source real distance;
- novelty ratio: own-source distance / nearest non-source distance.

Частотно-текстурные признаки:

- low/mid/high radial Fourier energy fractions;
- spectral slope;
- RMS gradient energy.

Prediction diagnostics:

- synthetic-minus-replay probability shift отдельно для positives и
  negatives;
- изменение separation `mean(p|positive)-mean(p|negative)`;
- изменение AUPRC, precision/recall в top-K (`K = число positives`),
  квантилей positive scores и верхнего хвоста negative scores;
- class-wise сдвиг вероятности для каждого confusing negative class;
- argmax gains, losses и transition counts;
- отдельный фокус на `mel`, `akiec`, `bkl`.

### Неопределённость

- feature metrics повторяются в трёх real-only ConvNeXt-S spaces;
- source-paired frequency/novelty differences получают 5000-repeat bootstrap
  по `source_group_id`;
- prediction shifts агрегируются по paired seeds и lesion groups;
- результаты являются диагностическими, а не новой confirmatory проверкой.

## Критерии интерпретации

`coverage_failure`:

- synthetic coverage ниже replay в DINOv2 и в большинстве ConvNeXt-S seeds;
- Vendi/effective rank не компенсируют этот провал.

`representation_mismatch`:

- DINOv2 показывает близость/fidelity, но ConvNeXt-S precision/density или
  class margin заметно хуже.

`low_novelty`:

- own-source distance существенно ниже nearest non-source distance;
- synthetic coverage не выше replay при той же дозе.

`frequency_gap`:

- paired CI для high-frequency fraction, spectral slope или gradient energy
  не включает ноль и направление устойчиво по классам.

`boundary_without_ranking_gain`:

- число argmax gains положительно, но positive-negative probability
  separation для melanoma не растёт или уменьшается.

Регенерация разрешается только после определения доминирующего failure mode.
Следующий pool должен менять один фактор: conditioning/strength, diversity
или frequency preservation, сохраняя source/count-matched replay.

## Артефакты

Канонический каталог:
`outputs/reports/stage12_failure_mode_diagnostics`.

Ожидаются:

- `analysis_summary.json`;
- `feature_distribution_metrics.csv`;
- `source_conditional_novelty.csv`;
- `frequency_features.csv`;
- `frequency_paired_bootstrap.csv`;
- `prediction_probability_shifts.csv`;
- `ranking_tail_diagnostics.csv`;
- `confuser_probability_shifts.csv`;
- `argmax_transition_counts.csv`;
- caches `embeddings_*.npz` и соответствующие index CSV;
- PNG-графики без HTML dashboard.

## Фактическое выполнение

Дата: 2026-07-29.

- контейнер `research-stage12-diagnostics` завершён с exit code 0;
- 90 synthetic presentations, 90 replay presentations;
- 84 уникальных real sources;
- DINOv2 и три real-only ConvNeXt-S encoders;
- PRDC `k=5`;
- 5000 source-group bootstrap повторов;
- validation predictions seeds 42, 43, 44;
- `locked_test_evaluated=false`;
- 32/32 server regression tests passed;
- code version: `fb02eaf`.

### Distribution geometry

Синтетика имеет меньшие precision, density и coverage в 11 из 12
`encoder × class` сравнений. Одновременно Vendi score выше replay в 11 из 12
сравнений. Следовательно, наблюдается не простой diversity collapse:
синтетика разнообразнее относительно самой себя, но её разнообразие хуже
совпадает с полезной реальной поддержкой класса.

DINOv2:

| Class | Arm | Precision | Recall | Density | Coverage | Vendi |
|---|---|---:|---:|---:|---:|---:|
| mel | synthetic | 0.400 | 0.933 | 0.173 | 0.030 | 6.359 |
| mel | replay | 0.900 | 0.982 | 0.587 | 0.089 | 5.739 |
| akiec | synthetic | 0.400 | 0.951 | 0.180 | 0.112 | 6.808 |
| akiec | replay | 0.767 | 0.990 | 0.593 | 0.317 | 6.393 |
| bkl | synthetic | 0.567 | 0.937 | 0.227 | 0.031 | 6.616 |
| bkl | replay | 0.833 | 0.978 | 0.893 | 0.133 | 6.475 |

В ConvNeXt-S spaces эффект сильнее для `mel` и `bkl`. Усреднённый coverage:

- `mel`: synthetic 0.023, replay 0.165;
- `akiec`: synthetic 0.163, replay 0.236;
- `bkl`: synthetic 0.013, replay 0.136.

Высокий ConvNeXt PRDC recall у synthetic `mel/bkl` не противоречит низкому
coverage: удалённые и широко разбросанные synthetic points создают большие
generated radii. Именно поэтому Naeem et al. рекомендуют density/coverage как
более надёжное дополнение к precision/recall.

### Representation mismatch

Среднее отношение
`distance(synthetic, own source) / distance(synthetic, nearest non-source)`:

| Class | DINOv2 | ConvNeXt-S, mean over seeds |
|---|---:|---:|
| mel | 0.761 | 1.076 |
| akiec | 0.647 | 1.246 |
| bkl | 0.714 | 1.857 |

DINOv2 считает synthetic source-like, тогда как task-specific real-only
ConvNeXt-S, особенно для `bkl`, видит сдвиг за пределы собственного source
neighbourhood. Это подтверждает `representation_mismatch`: single-encoder
selection была недостаточной.

### Frequency and texture

14 из 15 paired class-metric CI не включают ноль.

| Class | Mid-frequency fraction | High-frequency fraction | Spectral slope | Gradient RMS |
|---|---:|---:|---:|---:|
| mel | +0.00581 | -0.00060, CI пересекает 0 | -0.2548 | +0.00263 |
| akiec | +0.00376 | -0.00122 | -0.1926 | +0.00117 |
| bkl | +0.00486 | -0.00168 | -0.2758 | +0.00142 |

Это не следует упрощать до тезиса «синтетика размыта»: gradient energy
увеличилась, а крайний high-frequency energy уменьшился для `akiec/bkl`.
Наблюдается систематическая перестройка спектра, совместимая с искусственными
границами/текстурами генератора.

### Почему melanoma AUPRC ухудшилась

Средняя melanoma probability separation выросла на `+0.0249`, но это скрывает
неоднородный сдвиг хвостов:

- melanoma AUPRC delta: `-0.0447`, 3/3 seeds;
- lower positive tail `q10`: `-0.0236`;
- upper negative tail `q90`: `+0.1127`;
- top-K precision, где `K=143`: `-0.0093` в среднем;
- наибольший mean `mel`-score сдвиг среди confusers наблюдается для `akiec`;
- `akiec` имеет наибольший `q95` confuser shift: `+0.1559`.

То есть часть melanoma positives становится хуже ранжирована, а достаточно
много negatives получают повышенный melanoma score. Argmax gains в seeds
43/44 не компенсируют этот ranking-tail regression.

### Решение по гипотезам

| Failure mode | Решение |
|---|---|
| Coverage failure | поддержан, 11/12 сравнений |
| Simple diversity/mode collapse | не поддержан: Vendi выше в 11/12 |
| Representation mismatch | поддержан |
| Frequency gap | поддержан, 14/15 CI |
| Melanoma ranking-tail failure | поддержан, 3/3 seeds |
| Простое «синтетика слишком похожа на source» | не поддержано task space |

## Научный вывод

Отбор по близости к real manifold в одном foundation-model space не
гарантирует downstream utility. `strict_id` синтетика добавляет разнообразие,
но это разнообразие частично лежит вне task-relevant class support и меняет
частотную структуру. В результате decision boundary может улучшить отдельные
argmax-метрики, одновременно ухудшая clinically relevant ranking.

Для статьи это более содержательный результат, чем «синтетика не помогла»:
показано, какие свойства single-encoder selection не контролирует и почему
source-matched replay остаётся сильнее по melanoma AUPRC.

## Stage 13A

Сначала анализируется существующий pool из 1440 кандидатов:

- 480 изображений на класс;
- strengths 0.15, 0.30, 0.45;
- 160 real sources на class.

Для всех кандидатов считаются DINOv2, ensemble real-only ConvNeXt-S и
frequency features. Новый selection gate должен:

1. требовать согласованную class fidelity в DINOv2 и ConvNeXt-S;
2. использовать двухсторонний novelty interval, а не минимальное расстояние;
3. ограничивать frequency deviation относительно собственного source;
4. разрешать не более одного synthetic variant на source;
5. выбирать facility-location subset, покрывающий недопредставленные real
   neighbourhoods;
6. отдельно штрафовать `mel ↔ akiec` confuser margin.

Если из текущего pool нельзя собрать 30 изображений на класс, которые проходят
gate, генерация меняется. Если pool достаточен, выбирается один новый набор из
90 изображений и выполняется только один paired ConvNeXt-S validation screen:
новая синтетика против source-matched replay, seeds 42, 43, 44. Existing
`strict_id` повторно не обучается.
