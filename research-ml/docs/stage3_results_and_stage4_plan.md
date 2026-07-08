# Stage 3 results and Stage 4 class-specific ablation plan

Дата: 2026-07-08  
Проект: HAM10000 synthetic augmentation for imbalanced medical image classification.

## Краткий вывод

Stage 3 подтвердил, что utility-aware synthetic selection лучше простой Stage 2 эвристики, но эффект оказался class-specific:

- `mel` заметно выигрывает от Stage 3 synthetic pool.
- `bkl` резко деградирует.
- общий macro F1 почти догоняет Stage 1, но не превосходит лучший real-only baseline.

Главная гипотеза следующего этапа:

> Synthetic augmentation нельзя включать одинаково для всех minority/boundary classes. Нужно class-specific synthetic policy: один класс может получать пользу, другой — вред из-за сдвига decision boundary.

## Сравнение ключевых run

| Run | Test macro F1 | Test bal acc | Worst recall | MCC | ECE | Главное наблюдение |
|---|---:|---:|---:|---:|---:|---|
| Stage1 CE weighted | 0.7955 | 0.8279 | 0.7305 | 0.7673 | 0.0931 | Лучший balanced accuracy и worst recall. |
| Stage1 balanced softmax | 0.7989 | 0.7810 | 0.6347 | 0.7760 | 0.0954 | Лучший macro F1. |
| Stage2 topk80 CE weighted | 0.7874 | 0.8039 | 0.6826 | 0.7637 | 0.0922 | Простая feature selection почти догнала baseline, но не победила. |
| Stage3 utility weight 0.25 | 0.7633 | 0.7605 | 0.5758 | 0.7540 | 0.0817 | Слишком слабый synthetic signal; ухудшает `bkl`. |
| Stage3 utility weight 0.5 | 0.7915 | 0.8063 | 0.6061 | 0.7653 | 0.0880 | Лучший synthetic run; улучшает `mel`, ломает `bkl`. |

## Per-class эффект Stage 3 weight 0.5

Относительно Stage1 CE weighted:

| Class | Stage1 recall | Stage3 recall | Delta | Интерпретация |
|---|---:|---:|---:|---|
| akiec | 0.7347 | 0.6939 | -0.0408 | Легкая просадка. |
| bcc | 0.8701 | 0.8961 | +0.0260 | Небольшой плюс. |
| bkl | 0.7636 | 0.6061 | -0.1576 | Главная проблема Stage 3. |
| df | 0.8235 | 0.7647 | -0.0588 | Просадка малого класса. |
| mel | 0.7305 | 0.7904 | +0.0599 | Главный выигрыш Stage 3. |
| nv | 0.9205 | 0.9404 | +0.0199 | Небольшой плюс. |
| vasc | 0.9524 | 0.9524 | 0.0000 | Нейтрально. |

Confusion-delta Stage1 CE weighted vs Stage3 weight 0.5:

| Class | Stage3 fixed Stage1 error | Stage3 broke Stage1 correct | Вывод |
|---|---:|---:|---|
| akiec | 3 | 5 | Нет пользы. |
| bcc | 2 | 0 | Плюс. |
| bkl | 8 | 34 | Сильный вред. |
| df | 1 | 2 | Почти нейтрально/минус. |
| mel | 18 | 8 | Сильный плюс. |
| nv | 34 | 14 | Плюс, но это majority class. |
| vasc | 1 | 1 | Нейтрально. |

## Научная интерпретация

Stage 3 не опровергает utility-aware selection. Наоборот, он показал, что общий synthetic pool скрывает разнонаправленные class-specific эффекты.

Практически это значит:

1. `mel` synthetic samples, выбранные по utility score, вероятно, дают полезные boundary examples.
2. `bkl` synthetic samples, даже после utility filtering, смещают границу в сторону ошибок `bkl -> mel/nv/akiec`.
3. Глобальный `synthetic_weight=0.5` лучше, чем `0.25`, но одного глобального веса недостаточно.
4. Следующая проверка должна быть не "еще больше синтетики", а "какие классы синтетики вообще стоит включать".

## Связь с литературой

Правило проекта: если статья используется для выбора следующего эксперимента или интерпретации результата, она фиксируется и в Google Sheet, и в MD-отчете стадии.

Google Sheet: `https://docs.google.com/spreadsheets/d/1AXcfUmUuuwwUTUefzN1XDyViwsK7tWMB2cMqw2mj4Mc/edit`

Результаты согласуются с текущим направлением исследований:

- He et al., ICLR 2023, *Is synthetic data from generative models ready for image recognition?* — synthetic images могут помогать recognition, но требуют осторожных стратегий применения, потому что визуальное качество не равно downstream utility: https://arxiv.org/abs/2210.07574
- Azizi et al., 2023, *Synthetic Data from Diffusion Models Improves ImageNet Classification* — diffusion augmentation может улучшать сильные классификаторы, но при class-conditional/domain-adapted генерации и масштабной настройке: https://arxiv.org/abs/2304.08466
- *Harnessing Synthetic Skin Lesion Data via Stable Diffusion Models* — skin-lesion synthetic augmentation может быть полезной, но в нашей постановке сильный baseline и lesion-aware split делают проверку строже: https://arxiv.org/html/2401.05159v1
- *Synthetic Data Generation for Long-Tail Medical Image Classification* — близкая постановка: diffusion-driven pipeline for medical long-tail classification with inpainting and OOD filtering; это поддерживает наш переход от общего img2img к class/region-aware generation и post-selection: https://arxiv.org/html/2605.03221v1
- *Understanding Trade-offs When Conditioning Synthetic Data* — качество и разнообразие conditioning создают trade-off; это поддерживает идею class-specific/condition-specific synthetic policy: https://arxiv.org/html/2507.02217v1

### Источники, использованные именно в Stage 3/4

| Google Sheet row | Источник | Как использован в этом этапе |
|---:|---|---|
| 2 | Jiang et al., *Synthetic Data Generation for Long-Tail Medical Image Classification* | Поддерживает идею, что для medical long-tail нужна не только генерация, но и post-selection/OOD filtering. |
| 37 | Li et al., *SAU: A Dual-Branch Network...* | Обосновывает учет real/synthetic discrepancy; наш `synthetic_weight` — простая проверка этой линии без новой архитектуры. |
| 42 | Chen et al., *Augmented Conditioning Is Enough...* | Поддерживает следующий шаг: image-conditioned / augmented-conditioned generation вместо class prompt only. |
| 45 | He et al., *Is synthetic data from generative models ready for image recognition?* | Методологическая опора для negative-result narrative: synthetic visual quality не гарантирует downstream utility. |
| 46 | Azizi et al., *Synthetic Data from Diffusion Models Improves ImageNet Classification* | Positive reference: синтетика может работать, если генератор, conditioning и протокол достаточно сильные. |
| 47 | Farooq et al., *Derm-T2IM...* | Близкий skin-lesion пример; использовать как background, но проверять lesion-aware split и class-wise effects. |
| 48 | Trabucco et al., *Understanding Trade-offs When Conditioning Synthetic Data* | Обосновывает Stage 4/следующий этап: class-specific coverage и conditioning trade-off. |

## Stage 4: запущенный следующий этап

Цель: проверить, является ли вред Stage 3 следствием именно `bkl` synthetic samples.

Запущены два class-specific ablation run:

1. `stage4_utility_mel_akiec_weight05_ce_weighted`
   - real train: 7011
   - synthetic: `mel=80`, `akiec=80`
   - excluded synthetic: `bkl`
   - synthetic weight: 0.5

2. `stage4_utility_mel_only_weight05_ce_weighted`
   - real train: 7011
   - synthetic: `mel=80`
   - excluded synthetic: `akiec`, `bkl`
   - synthetic weight: 0.5

Критерии успеха Stage 4:

- сохранить или улучшить `mel` recall относительно Stage 1;
- восстановить `bkl` recall относительно Stage 3;
- не ухудшить macro F1 относительно Stage3 weight 0.5;
- проверить confusion-delta по `mel/bkl/akiec`.

## Что делать после Stage 4

Если `mel_only` или `mel+akiec` улучшит баланс `mel/bkl`, следующий шаг:

1. Добавить per-class synthetic weights вместо глобального `synthetic_weight`.
2. Запустить финальную компактную матрицу:
   - real-only Stage1 CE weighted baseline, 3 seeds;
   - лучший Stage4 policy, 3 seeds;
   - лучший Stage4 policy + short real-only fine-tune, 3 seeds.
3. Для статьи формулировать основной вклад как:
   - strong negative result for naive synthetic augmentation;
   - evidence that synthetic utility is class-specific;
   - controlled diagnostics and class-specific policy for medical long-tail classification.
