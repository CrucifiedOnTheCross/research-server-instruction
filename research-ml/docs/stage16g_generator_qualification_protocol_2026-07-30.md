# Stage 16G: квалификация генератора перед synthetic utility на ISIC 2019

Дата протокола: 2026-07-30

Статус: протокол и readiness gate реализованы; locked test закрыт.

Подготовка запускается контейнером
`scripts/start_stage16g_preparation_container.sh`, поскольку project `.venv`
привязан к Python CUDA-образа и не предназначен для выполнения на host.

Фактический G0 gate: закрыт для прямого DiDGen pilot. Из 2594 официальных
ISIC 2018 masks в Stage 16 train попали 303 изображения: 57 `mel` и 246
`nv`. Для `scc/bcc/ak/df/vasc` overlap равен нулю. Поэтому эти masks нельзя
выдавать за восьмиклассовую разметку ISIC 2019.

## Фактический результат Stage 16G-S

Дата завершения: 2026-07-30.

- segmentation train: 1822 официальных ISIC 2018 image-mask pairs;
- train-only qualification: 303 изображения Stage 16 train;
- исключено по Stage 16 validation/locked-test ID: 469;
- исключено по exact hash дополнительно: 0;
- locked test не открывался;
- лучший epoch: 29 из 30;
- qualification Dice: 0.92293;
- qualification IoU: 0.86355;
- время обучения: 1256 секунд.

Предварительно заданные gates `Dice >= 0.85` и `IoU >= 0.75` пройдены.
Это разрешает следующий технический шаг: inference pseudo-masks только на
редких train-классах. До генерации выполняются per-image и area-quartile
audit, component/confidence gates и визуальная проверка.

Техническая запись: первый запуск mask-audit завершился до inference из-за
отсутствия project root в `sys.path` при прямом запуске `tools/*.py`.
Импорт исправлен явно; научные данные и checkpoint не изменялись.

### Morphology audit и pseudo-mask gate

Per-image пересчёт checkpoint подтвердил aggregate результат:

- mean Dice: 0.92310;
- mean IoU: 0.86381;
- worst area-quartile Dice: 0.91574 для самых малых поражений;
- pseudo-mask candidates: 1089 независимых train groups;
- confidence/component gates прошли 972 изображения.

| Класс | Candidates | Passed |
|---|---:|---:|
| ak | 177 | 151 |
| bcc | 320 | 271 |
| df | 63 | 62 |
| mel | 320 | 304 |
| scc | 136 | 117 |
| vasc | 73 | 67 |

Этот gate подтверждает техническую пригодность сегментатора, но не
подтверждает корректность диагноза синтетики. Следующий шаг ограничен
визуальным аудитом pseudo-masks и generator smoke; downstream classifier
остаётся заблокирован.

Визуальный аудит публикуется как persistent FiftyOne dataset
`isic2019-stage16g-mask-audit`. Сохранённые views разделяют прошедшие,
отклонённые, фрагментированные, border-touching и low-confidence маски.

После visual audit до generator smoke зафиксирован дополнительный morphology
gate, не использующий downstream outcomes:

- площадь pseudo-mask должна находиться между 1-м и 99-м процентилями
  площади официальных qualification masks;
- `border_foreground_fraction <= 0.10`;
- предыдущие confidence/component/dominant-component gates сохраняются.

Фактический результат дополнительного gate:

- 893 pseudo-masks из 1089 допущены к generator smoke;
- все 893 относятся к уникальным `group_id`;
- `ak=144`, `bcc=255`, `df=56`, `mel=270`, `scc=106`,
  `vasc=62`;
- validation и locked test не использовались.

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

Перед генеративным smoke выполняется Stage 16G-S:

- обучить DeepLabV3-ResNet50 на официальных ISIC 2018 image-mask pairs,
  исключив все image IDs и точные hashes Stage 16 validation/locked test;
- 303 совпадающих изображения Stage 16 train использовать только как
  segmentation qualification;
- сохранить Dice/IoU по эпохам и checkpoint по validation Dice;
- только после Dice >= 0.85 и IoU >= 0.75 разрешить pseudo-masks редких
  train-классов;
- редкие pseudo-masks дополнительно проверить blinded audit в FiftyOne.

Этот дизайн воспроизводит недостающий segmentation step из Jiang et al.,
не обучаясь на downstream validation/test.

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
12. Jiang et al., ISIC 2019 class-conditioned inpainting and OOD selection:
    https://arxiv.org/abs/2605.03221
13. MedSAM official code: https://github.com/bowang-lab/MedSAM
14. BiomedParse official code: https://github.com/microsoft/BiomedParse

## Зафиксированные ограничения

- ISIC 2018 masks покрывают только часть ISIC 2019 train.
- Mask availability сама может быть source/domain-biased.
- DiDGen исследовался на ISIC 2018, а не на нашей полной 8-class ontology.
- Derm-T2IM binary label scope не переносится автоматически на восемь классов.
- Автоматические label classifiers могут разделять общие shortcuts.
- Validation остаётся пригодной для model selection, но не для выбора
  feature-space anchors или генеративных порогов.
# Stage 16G-P0: generator feasibility smoke

## Preregistered question

Does mask-conditioned inpainting produce a materially different, reproducible
train-only candidate pool while preserving the non-lesion background better
than the historical low-strength SD1.5 img2img control?

This is a generator-qualification stage. It does not train a downstream
classifier and it does not open validation or locked-test images.

## Design

- 24 anchors: 4 unique lesion groups from each of `mel`, `scc`, `bcc`, `ak`,
  `df`, and `vasc`.
- Anchors are selected deterministically across lesion-area quantiles from the
  893 masks that passed the Stage 16G segmentation and morphology gates.
- Two candidates per anchor and arm, giving 48 images per arm.
- `historical_sd15_img2img`: SD1.5 img2img at strength 0.10. This is a
  preregistered near-reconstruction negative control and cannot be promoted.
- `generic_sd15_mask_inpaint`: the dedicated SD1.5 inpainting checkpoint at
  strength 0.80, using the train-only predicted lesion mask.
- Every candidate records source, mask and output hashes, generator revision,
  prompt, seed and full resolved configuration.

The generic inpainting arm is only a feasibility control. It is not treated as
the final medical generator because it has not been adapted to dermoscopy.
Class-conditioned/domain-adapted LoRA or the reproducible part of DiDGen is
Stage 16G-P1 and is allowed only after P0 passes technical gates.

## Promotion gates

P0 must have no missing images, no evaluation overlap, deterministic seeds and
an immutable model revision. Subsequent qualification must measure non-lesion
background error, regenerated-mask IoU, independent class agreement,
source-near-duplicate similarity and artifact rejection. No downstream
classification run starts before those structured audits are complete.

## Implementation note

The dedicated inpainting checkpoint is intentional: Diffusers documents that
ordinary text-to-image checkpoints are compatible with inpainting but are less
effective than checkpoints trained for the task. LoRA is deferred because the
official Diffusers examples are starting points requiring task-specific data
adaptation; introducing it in P0 would confound conditioning with fine-tuning.

## Stage 16G-P0 factual result

Date: 2026-07-30. Both arms completed 48/48 images with zero hash failures.
All 24 anchors were unique train lesion groups; evaluation overlap by ID and
exact image hash was zero. The locked test remained closed.

| Arm | Label agreement | Source cosine | Mask IoU | Outside-mask MAE | Gate |
|---|---:|---:|---:|---:|---|
| historical SD1.5 img2img, strength 0.10 | 0.625 | 0.668 | 0.827 | 0.0208 | fail |
| generic SD1.5 mask inpainting, strength 0.80 | 0.333 | 0.328 | 0.637 | 0.0128 | fail |

The generic inpainting control failed the two central preregistered criteria:
independent real-only ConvNeXt-Small class agreement was below 0.80 and
segmentation IoU was below 0.70. Melanoma agreement was 0/8; five of eight
generated melanoma candidates were predicted as `nv`. Visual review confirmed
the metric failure: the generic checkpoint frequently generated eyes, noses,
lips and other non-dermoscopic structures inside the lesion mask.

The historical img2img control preserved lesion shape better, but also failed
class agreement. Its contact sheet shows near-reconstruction with only small
texture and colour changes. The classifier-feature cosine did not reliably
identify those visually obvious near-copies, so it must not be the only
duplicate diagnostic in the next qualification. Pixel/perceptual source
similarity and paired visual audit remain required.

### Decision

`p1_allowed=false`; no downstream classifier will be trained on either P0
pool. This rules out generic SD1.5 prompting as the main Stage 16 generator and
confirms that selection or OOD filtering cannot repair a generator that changes
diagnosis or produces out-of-domain anatomy.

The next permitted work is a train-only generator-development stage, not a
utility claim:

1. adapt a dermoscopy-domain generator using only Stage 16 train images;
2. keep mask conditioning and class prompts as separate, attributable factors;
3. add image-only and image-mask discriminator diagnostics before downstream
   training;
4. compare against the same 24 anchors and equal candidate budget;
5. require the full P0 quality gates before any Stage 16C classifier run.

Canonical artifacts:

- `outputs/stage16g_generator_smoke/qualification_per_image.csv`;
- `outputs/stage16g_generator_smoke/qualification_by_arm.csv`;
- `outputs/stage16g_generator_smoke/qualification_summary.json`;
- `outputs/stage16g_generator_smoke/contact_sheet_*.jpg`.

The paired visual audit is published as the persistent FiftyOne dataset
`isic2019-stage16g-generator-smoke` at `http://10.200.1.180:5151`.

## Stage 16G-P1 preregistration: train-only domain LoRA

P0 showed that mask conditioning alone preserves the background but does not
provide a dermoscopy prior. P1 therefore tests domain adaptation as a separate
factor. The implementation follows the current Diffusers text-to-image LoRA
training structure: frozen VAE/text encoder/base UNet, trainable attention LoRA,
class prompts, BF16, gradient checkpointing and Min-SNR weighting with
`gamma=5`.

The official DiDGen repository was reviewed at revision
`1ca3b085bbb42462e5a3cca8d0b4dfefb494b9c8`. Its released script performs
20,000-step SD2.1 fine-tuning with region-aware cross-attention loss, but ships
no checkpoint, uses an old pinned software stack and contains author-local
dataset paths. P1 does not claim to reproduce DiDGen. It tests the attributable
domain-adaptation component first on the supported project stack.

### Leakage and sampling controls

- only real Stage 16 train images are allowed;
- all 24 P0 anchor IDs and their complete lesion groups are excluded;
- one image per lesion group is retained;
- classes are sampled uniformly during optimization;
- validation and locked test are not used for loss, prompts or checkpoint
  selection;
- the LoRA base-model commit, data manifest and all hyperparameters are stored.

### P1 comparisons

The same 24 held-out train anchors and two candidates per anchor are reused:

1. `domain_lora_img2img`, strength 0.80: domain adaptation without a mask;
2. `domain_lora_mask_inpaint`, strength 0.80: domain adaptation plus mask
   conditioning.

They are interpreted together with P0's generic inpainting and low-strength
img2img controls. Downstream classification remains blocked until the adapted
inpainting arm passes the same class-agreement, mask-IoU, background,
near-duplicate and visual-artifact gates.

The full P1 budget is 5,000 optimization steps with effective batch size 4,
approximately 20,000 class-balanced image presentations. Checkpoints include
LoRA, optimizer, scheduler and exact global step every 500 steps. The container
uses an on-failure restart policy and resumes from the latest complete
checkpoint. A separate two-step output directory is required for the initial
VRAM/compatibility smoke and is never included in scientific summaries.

Technical smoke note: the first P1 VRAM smoke stopped before optimizer step 1
because PyTorch 2.12 does not accept a `dtype` keyword in `Tensor.cuda()`.
Tensor transfer was changed to `Tensor.to(device="cuda", dtype=...)`. No
training result, hyperparameter, data row or generator output was produced by
the failed attempt. PEFT is pinned to `0.20.0` for the project environment.

### Literature used for implementation

- Hugging Face Diffusers, official text-to-image LoRA training script and
  Min-SNR recommendation (`snr_gamma=5`);
- Shentu et al., DiDGen, MICCAI 2025 and Medical Image Analysis 2026;
- Jiang et al., *Synthetic Data Generation for Long-Tail Medical Image
  Classification*, arXiv:2605.03221;
- Shabu et al., *A Generative AI Approach for Reducing Skin Tone Bias in Skin
  Cancer Classification*, arXiv:2602.14356. This supports dermoscopy-domain
  LoRA feasibility but does not justify skin-tone conditioning in our main arm.
