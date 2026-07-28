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
- 90 точных real sources, связанных по `source_image_id`;
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
- `argmax_transition_counts.csv`;
- caches `embeddings_*.npz` и соответствующие index CSV;
- PNG-графики без HTML dashboard.

