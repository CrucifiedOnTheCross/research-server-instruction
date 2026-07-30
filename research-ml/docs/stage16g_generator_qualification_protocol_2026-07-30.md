# Stage 16G: квалификация генератора перед synthetic utility на ISIC 2019

Дата протокола: 2026-07-30

Статус: протокол и readiness gate реализованы; locked test закрыт.

## Исследовательский вопрос

Можно ли получить синтетические дерматоскопические изображения, которые:

1. сохраняют диагноз и морфологию поражения;
2. заполняют недопредставленные области train feature space;
3. не являются почти точными копиями исходных изображений;
4. улучшают downstream-классификацию по сравнению с равнодозовым
   source-matched real replay;
5. не ухудшают melanoma/SCC ranking, калибровку и worst-class recall?

Главный принцип: визуальная правдоподобность, FID и близость к real manifold
не считаются доказательством utility. Генератор допускается в Stage 16C только
после train-only квалификации и равнодозового replay-control.

## Аудит нашего текущего способа

Исторический pipeline использовал generic Stable Diffusion 1.5 img2img:

- исходное изображение необратимо центрировалось и обрезалось до квадрата;
- применялись generic class prompts;
- `strength=0.05/0.10`, CFG 6 и 30 inference steps;
- затем candidates фильтровались по DINOv2/ConvNeXt geometry, PRDC,
  frequency gates и facility-location coverage.

Stage 15A показал, что основной ranking loss возникал уже при fixed offline
crop, а VAE round-trip не объяснял весь эффект. Дополнительный слабый UNet
step ухудшал threshold metrics. Stage 13B показал, что улучшение geometry
candidate pool не исправляет дефект генератора: coverage-targeted synthetic
проиграл source-matched replay по macro AUPRC и melanoma AUPRC на 3/3 seeds.

Вывод: повторная настройка только порогов отбора не является достаточным
следующим этапом. Нужна смена conditioning и отдельная проверка сохранения
морфологии/диагноза.

## Что делали другие работы

### Derm-T2IM

Farooq et al. применили DreamBooth/few-shot fine-tuning Stable Diffusion,
заморозив text encoder. Seed data предварительно очищали DullRazor, а
генерацию выполняли text-to-image при 512 px. Авторы вручную отбрасывали
артефакты и демонстрировали управление размером поражения, числом невусов и
цветом кожи через prompt.

Ограничение для нас: опубликованный checkpoint обучен на бинарной
benign/malignant постановке, тогда как ISIC 2019 содержит восемь классов.
Поэтому Derm-T2IM разрешён только как feasibility comparator и не может
автоматически считаться корректным multiclass label generator.

### DiDGen

Shentu et al. предложили clinically detailed prompts, region-aware attention
loss и layout guidance. Обучение использует изображение вместе с
segmentation mask, отдельные lesion/skin tokens и согласование attention с
областью поражения. Код основан на Stable Diffusion 2.1 и ISIC 2018 Task 1.

Это наиболее содержательный кандидат для нашей гипотезы: conditioning
позволяет изменять внешний вид, сохраняя пространственную структуру поражения,
а attention-derived masks дают дополнительный проверяемый artifact.

Ограничения: опубликованный repository не предоставляет готовый checkpoint,
содержит несовместимые версии некоторых зависимостей и имеет
PolyForm Noncommercial license. На RTX 5080 16 GB сначала обязателен короткий
GPU smoke с gradient checkpointing/mixed precision, а не полный 20k-step run.

### Критические исследования synthetic utility

Bissoto et al. показали, что положительный эффект skin-lesion GAN synthesis
может проявляться прежде всего вне исходного распределения, а не как
устойчивый in-distribution выигрыш. Sagers et al. обнаружили, что медицинская
latent-diffusion синтетика обычно слабее дополнительного real data, эффект
насыщается при большой synthetic-to-real дозе, а source-conditioned samples
могут оказаться чрезмерно близкими к источнику.

Поэтому в Stage 16G обязательны source-near-duplicate gate, source-group
uniqueness и downstream replay-control.

### Classical lesion mixing

Perez et al. сравнили 13 augmentation regimes. Лучшей обычной комбинацией
были crop, affine transforms, flips и color jitter. Их lesion-mix переносил
маску поражения между изображениями, размывал границу и согласовывал
гистограмму. В бинарной постановке label определялся как melanoma, если
melanoma присутствовала хотя бы в одном источнике.

Для восьми классов такое правило неоднозначно. Lesion-mix не запускается как
основной arm, пока не определена clinically defensible multiclass label
policy.

## Волосы, цвет кожи и acquisition artifacts

### Волосы

Добавлять случайные волосы в основной synthetic arm нельзя. Волосы являются
не патологическим признаком, а acquisition nuisance. Их корреляция с классом
может создать shortcut. DullRazor/black-hat inpainting также может удалить
тонкие структуры поражения.

Решение:

- hair addition/removal не входит в основной генератор;
- позже допускается отдельный paired robustness stress test;
- одинаковое преобразование применяется к real replay и synthetic;
- результаты анализируются по subgroup с видимыми волосами.

### Цвет кожи

Обычный hue/saturation shift нельзя называть изменением skin tone. Цвет
дерматоскопии зависит от устройства, освещения, контактной жидкости и
постобработки. В текущей ISIC 2019 metadata нет проверенной Fitzpatrick/Monk
аннотации.

Решение:

- skin-tone conditioning отключён;
- он может быть включён только после появления валидированной tone
  annotation/model и отдельного fairness protocol;
- color constancy остаётся отдельной real-only preprocessing ablation.

## Как заполнять feature space

Заполняются не «дыры» validation/test, а только разреженные области train:

1. извлечь deterministic embeddings train изображений DINOv2 и независимым
   pretrained/real-only ConvNeXt;
2. внутри каждого редкого класса построить кластеры или k-center anchors;
3. выбирать независимые `group_id`, отдавая приоритет слабопокрытым кластерам;
4. генерировать несколько candidates на source lesion;
5. проверять label agreement независимыми real-only classifiers;
6. проверять mask IoU/морфологию и исключать near-duplicates источника;
7. выбирать coverage set с квотами по source/group и acquisition subgroup.

Ни embeddings, ни пороги не подбираются по validation или locked test.
Stage 13 уже показал, что PRDC/coverage без label/mask consistency
недостаточны.

## Кандидаты Stage 16G

| Arm | Назначение | Статус |
|---|---|---|
| generic SD1.5 img2img | исторический отрицательный control | не продвигается |
| Derm-T2IM | domain-checkpoint feasibility, binary only | pilot |
| DiDGen/SD2.1 region-aware | основной mask-conditioned кандидат | после gate |
| lesion mix | classical comparator | заблокирован label ambiguity |

## Последовательность выполнения

### G0. Train-only mask inventory

Скачать официальный ISIC 2018 Task 1 training-mask archive, сохранить SHA256,
извлечь маски безопасно и соединить их только с неизменяемым Stage 16 train
split. Совпадения с validation/locked test фиксируются только как count и не
попадают в manifest.

Артефакты:

- `splits/stage16g/mask_qualified_train.csv`;
- `splits/stage16g/mask_inventory.json`;
- `outputs/reports/stage16g_readiness.json`.

### G1. Reproducibility/VRAM smoke

Для DiDGen:

- checkout pinned revision
  `1ca3b085bbb42462e5a3cca8d0b4dfefb494b9c8`;
- отдельное зафиксированное environment;
- 8-16 train-only lesions, не более 20 optimization steps;
- fp16/bf16, batch 1, gradient accumulation/checkpointing;
- измерить peak VRAM, step time и проверить сохранение/возобновление;
- не использовать validation и locked test.

Если smoke не воспроизводится на 16 GB или pipeline требует неподтверждённых
ручных изменений, кандидат получает статус blocked, а не silently modified.

### G2. Train-only equal-budget pilot

Для mask-qualified редких классов:

- максимум 160 уникальных source lesions/class;
- четыре candidates/source;
- одинаковый candidate budget для сравниваемых generators;
- model revision, prompt, source/mask hashes, seed, scheduler, steps, CFG и
  precision сохраняются в manifest.

### G3. Qualification gates

Предварительные пороги:

- mask IoU не ниже 0.70;
- независимое label agreement не ниже 0.80;
- source cosine similarity не выше 0.995;
- PRDC precision не ниже 0.80;
- отсутствие validation/test references;
- один selected sample на source group;
- ручной blinded audit в FiftyOne.

Пороги считаются техническими gate, а не доказательством научной utility.

### G4. Stage 16C downstream confirmation

Только один прошедший generator сравнивается с:

1. Natural CE;
2. source-matched real replay той же дозы;
3. synthetic той же дозы.

Seeds 42-44, один и тот же ConvNeXt-Small initialization, split, optimizer,
epoch budget и checkpoint endpoint. Primary endpoint: validation macro AUPRC.
Дополнительно: macro F1, MCC, balanced accuracy, ECE, worst recall,
per-class AUPRC/AUROC, melanoma/SCC, fixed specificity, source subgroup и
lesion-group bootstrap. Locked test остаётся закрытым.

## Критерий научного успеха

Generator продвигается только при:

- mean macro AUPRC delta против replay не ниже +0.005;
- положительном знаке минимум на 2/3 seeds;
- MCC degradation не хуже 0.005;
- ECE degradation не хуже 0.02;
- отсутствии существенного ухудшения melanoma/SCC AUPRC;
- отсутствии evidence near-copying или source-group leakage.

Если generator не проходит, отрицательный результат всё равно научно полезен:
он локализует, какой уровень контроля не превращает synthetic fidelity в
downstream utility.

## Литература и код

1. Farooq et al., Derm-T2IM: https://arxiv.org/abs/2401.05159
2. Derm-T2IM checkpoint: https://huggingface.co/MAli-Farooq/Derm-T2IM
3. Shentu et al., DiDGen: https://papers.miccai.org/miccai-2025/0230-Paper4243.html
4. DiDGen code: https://github.com/junjie-shentu/DiDGen
5. Bissoto et al., critical synthetic-data analysis:
   https://www.ic.unicamp.br/~sandra/publication/2021-bissoto-isic-cvprw/
6. Sagers et al., latent diffusion medical classifiers:
   https://arxiv.org/abs/2308.12453
7. Perez et al., skin lesion augmentation:
   https://www.ic.unicamp.br/~sandra/pdf/papers/perez_ISIC18.pdf
8. Perez et al. code:
   https://github.com/fabioperez/skin-data-augmentation
9. Red-GAN mask-conditioned synthesis: https://arxiv.org/abs/2004.10734
10. Akrout et al., diffusion and closed-loop filtering:
    https://arxiv.org/abs/2301.04802
11. Official ISIC challenge data:
    https://challenge.isic-archive.com/data/

## Зафиксированные ограничения

- ISIC 2018 masks покрывают только часть ISIC 2019 train.
- Mask availability сама может быть source/domain-biased.
- DiDGen исследовался на ISIC 2018, а не на нашей полной 8-class ontology.
- Derm-T2IM binary label scope не переносится автоматически на восемь классов.
- Автоматические label classifiers могут разделять общие shortcuts.
- Validation остаётся пригодной для model selection, но не для выбора
  feature-space anchors или генеративных порогов.
