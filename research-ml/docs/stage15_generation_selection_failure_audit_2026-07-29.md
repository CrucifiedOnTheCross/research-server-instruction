# Stage 15: аудит генерации, отбора и причин отрицательной utility синтетики

Дата аудита: 2026-07-29

Статус: retrospective audit завершён; locked test не открывался.

## Краткий научный вывод

В выполненных экспериментах не обнаружено признаков утечки validation/test в
train. Lesion-group split в downstream-обучении сохранён, поэтому основные
отрицательные выводы Stage 11B и Stage 13B нельзя объяснить простой утечкой.

Однако обнаружен критический методологический confound: 87 из 90 изображений
финального Stage 13B были получены при `strength=0.05`, `30` inference steps и
generic Stable Diffusion 1.5. Для diffusers это соответствует приблизительно
одному фактическому denoising step. Изображения при этом сначала необратимо
центрально обрезались из `600x450` до квадрата `512x512`, затем проходили через
generic 4-channel SD VAE. Следовательно, этот arm в основном проверял:

1. fixed center crop и resampling;
2. реконструкционные ошибки generic VAE;
3. один слабый шаг generic text-conditioned denoising;

а не пользу новых независимых дерматоскопических наблюдений.

Поэтому корректная формулировка текущего результата:

> Отобранные near-source преобразования generic SD1.5 не превзошли
> source-matched replay при lesion-aware validation. Это ещё не доказывает
> бесполезность domain-trained или lesion-conditioned генерации.

## Проверенные риски

### 1. Split leakage

Статус: критической утечки не найдено.

- Downstream split использует `source_group_id` исходного поражения.
- Синтетические изображения допускаются только от train lesions.
- Validation и locked test не использовались при генерации.
- Locked test в Stage 11B, 13B и 14P не открывался.

Оставшийся риск: synthetic manifest присваивает каждому изображению уникальный
`group_id`. Без специального использования `source_group_id` повторное
автоматическое разбиение такого manifest могло бы разнести потомков одного
поражения по разным split. Текущий Stage 8+ pipeline использует
`source_group_id`, поэтому исторические результаты не затронуты. Для будущих
этапов это должно оставаться fail-closed invariant.

### 2. Независимость исходных поражений

Статус: найден дефект отбора, влияние на этапы неодинаково.

До исправления генератор и Stage 13 selector ограничивали повтор
`source_image_id`, но не повтор `source_group_id/lesion_id`.

| Набор | Строк | Уникальных source images | Уникальных source lesions | Вывод |
|---|---:|---:|---:|---|
| Stage 10 `strict_id` | 90 | 84 | 83 | эффективное разнообразие меньше заявленной дозы |
| Stage 13B final | 90 | 90 | 90 | финальный набор случайно удовлетворяет lesion uniqueness |
| Stage 13 low-strength pool | 480 | 480 | 438 | pool содержит несколько снимков одних lesions |
| Stage 13 BKL expansion | 160 | 160 | 153 | pool содержит несколько снимков одних lesions |

Между двумя Stage 13 candidate pools не было пересечения по
`source_image_id`, но было 15 общих `source_group_id`. Финальный выбранный
набор всё же получился lesion-unique, поэтому результат Stage 13B остаётся
валидным. Для Stage 10 сравнение synthetic/replay было pair-matched и потому
не смещено в пользу одного arm, но его нельзя описывать как сравнение на 90
независимых lesions.

Исправление: Stage 13 facility selection, capacity gate и final gate теперь
проверяют `source_group_id`, с fallback на `source_image_id` только для старых
manifest без group metadata.

### 3. Фактическая сила генерации

Статус: критический design flaw.

Stage 13 использовал:

- `stable-diffusion-v1-5/stable-diffusion-v1-5`;
- generic, не дообученный на дерматоскопии pipeline;
- `strength=[0.05, 0.10]`;
- `num_inference_steps=30`;
- `guidance_scale=6`;
- 87/90 финальных изображений при `strength=0.05`;
- 3/90 при `strength=0.10`.

В official diffusers img2img documentation `strength` определяет долю
denoising trajectory. При `30 * 0.05` pipeline выполняет примерно один
denoising step. Это очень близко к VAE round-trip, а не к генерации нового
варианта поражения.

Pixel audit относительно точного offline center-cropped источника:

| Сравнение | Подмножество | MAE | PSNR |
|---|---|---:|---:|
| source crop vs generated | все 90 | 0.0190 | 32.14 dB |
| source crop vs SD1.5 VAE round-trip | strength 0.05, n=87 | 0.01375 | 34.90 dB |
| source crop vs generated | strength 0.05, n=87 | 0.01882 | 32.23 dB |
| generated vs VAE round-trip | strength 0.05, n=87 | 0.01357 | 34.93 dB |
| source crop vs generated | strength 0.10, n=3 | — | 29.54 dB |

Интерпретация: большинство выбранных samples почти совпадают с
реконструкцией источника через SD1.5 VAE с очень малой дополнительной
модификацией.

### 4. Неэквивалентный preprocessing synthetic и replay

Статус: критический confound, требующий отдельного factorial control.

Для генерации исходное изображение `600x450` обрабатывается `ImageOps.fit`:
центрально обрезается до квадрата и сохраняется как фиксированное `512x512`.
При этом сохраняется около 75% площади исходного кадра.

Source-matched replay читает исходный `600x450` файл, а stochastic
`RandomResizedCrop` применяется заново каждую эпоху. Следовательно, arms
различаются одновременно:

- наличием/отсутствием diffusion;
- fixed offline crop против online stochastic view;
- SD VAE bottleneck;
- интерполяцией и частотными артефактами;
- prompt/negative prompt.

Текущий design не позволяет приписать отрицательный эффект именно denoising
или качеству отбора.

### 5. Согласованность checkpoint и научной endpoint

Статус: найден существенный analysis mismatch.

Модель сохранялась по `val/macro_f1`, тогда как поздние этапы интерпретировали
macro AUPRC и melanoma AUPRC как основные ranking/clinical endpoints.
Macro F1 зависит от argmax thresholding, а AUPRC оценивает ранжирование.

Retrospective learning-curve audit по лучшей macro AUPRC эпохе:

| Stage | Arm | seed 42 | seed 43 | seed 44 |
|---|---|---:|---:|---:|
| 11B | replay | 0.80790 | 0.80779 | 0.82701 |
| 11B | synthetic strict-ID | 0.80909 | 0.80848 | 0.82263 |
| 13B | replay coverage | 0.81680 | 0.81255 | 0.82774 |
| 13B | synthetic coverage | 0.81467 | 0.80167 | 0.82015 |

При post-hoc AUPRC-optimal checkpoint Stage 11B почти равен replay
`(+0.0012, +0.0007, -0.0044)`, то есть macro-F1 checkpoint преувеличивал
strict-pool ranking loss. Stage 13B проигрывает replay на всех трёх seeds
даже при таком оптимистичном выборе, поэтому его отрицательный вывод
устойчив.

Эти числа нельзя использовать как confirmatory результат: эпоха выбрана
post-hoc на той же validation. В следующем протоколе checkpoint endpoint
должна быть заранее согласована с primary endpoint, либо должен применяться
фиксированный epoch/cross-fitting.

### 6. Label quality исходных изображений

Статус: не является главным объяснением.

Train metadata:

- serial imaging: 3704;
- histopathology: 2886;
- expert consensus: 599;
- confocal: 39.

Final selections:

- strict-ID: 85 histopathology, 5 expert consensus;
- Stage 13 coverage: 80 histopathology, 10 expert consensus.

Значит, отрицательный эффект нельзя объяснить преимущественно слабо
подтверждёнными labels.

## Что говорят наши метрики

Stage 12 показал:

- synthetic хуже replay по precision/density/coverage в 11 из 12
  encoder/class срезов;
- Vendi diversity чаще выше, но coverage ниже;
- frequency-domain gap присутствует в 14 из 15 проверок;
- melanoma ranking tails ухудшаются;
- DINOv2 и ConvNeXt не полностью согласуются.

Это не выглядит как простой mode collapse. Скорее, синтетика разнообразна в
неполезных направлениях, но не покрывает необходимые real feature regions.
Pixel/VAE audit уточняет механизм: generic VAE и фиксированный crop вносят
частотный/domain shift, а почти нулевой denoising не добавляет достаточной
семантической вариативности.

## Ошибки отбора и остающиеся ограничения

1. Multi-encoder agreement не является независимой клинической проверкой.
   Он может отбирать classifier-friendly texture shortcuts.
2. Положительный margin в нескольких encoders тяготеет к лёгким
   прототипическим samples, а не к полезным boundary/AID cases.
3. Facility location улучшает feature coverage относительно candidate pool,
   но не исправляет плохой candidate generator.
4. Нет lesion-mask consistency, dermoscopic-attribute preservation или
   независимого label-preservation classifier, обученного на другом
   источнике.
5. `negative_prompt` систематически удаляет волосы, линейки и маркеры. Это
   может создавать acquisition shortcut относительно реальных данных.
6. Model ID сохраняется, но revision/checkpoint hash генератора не закреплён.
7. Доза 30/class мала, а PRDC с `k=5` имеет заметную дисперсию; geometry
   metrics следует сопровождать bootstrap/seed sensitivity.
8. Ранний Stage 2 извлекал real reference embeddings через stochastic train
   loader. Это могло вносить повторы и случайность. Stage 13 использовал
   deterministic eval transforms, поэтому его итог не затронут.

## Сопоставление с исследованиями

### Generic generation недостаточно

Исследование Bissoto et al. показало, что визуально правдоподобная skin-lesion
синтетика редко даёт устойчивый in-distribution выигрыш без строгих controls.
Это согласуется с нашими negative/tied results.

Работа *When Pretty Isn't Useful* показывает, что современные T2I-модели
могут улучшать fidelity и prompt following без роста training utility:
high-density/low-coverage и texture/frequency gap сохраняются. Наш Stage 12
воспроизводит тот же тип расхождения в медицинском домене.

### Medical VAE и lesion focus важны

В работе *Diffusion-based skin disease data augmentation with fine-grained
detail preservation and interpolation for data diversity* generic 4-channel
VAE прямо связывается с потерей
мелких медицинских деталей. Авторы обучают 8-channel VAE на HAM10000,
используют lesion masks и class-conditional latent diffusion. Это наиболее
прямое объяснение нашего VAE/frequency gap.

DiDGen дообучает SD2.1 с клинически структурированными prompts и
region-aware attention loss, генерируя согласованные image-mask pairs.
LF-VAR использует lesion-focused VQ-VAE и quantitative lesion measurements.
Обе работы подтверждают, что lesion-aware conditioning и domain-trained
representation предпочтительнее generic SD1.5.

Derm-T2IM использует DreamBooth domain adaptation, но решает binary
benign/malignant задачу и не является прямым drop-in baseline для нашей
семиклассовой постановки.

### Генерировать нужно не только прототипы

DiffuLT мотивирует AID/decision-boundary samples для long-tail learning.
Iterative Online Image Synthesis адаптирует class dose по текущей accuracy и
использует classifier guidance. Это поддерживает следующий вектор, но такие
методы следует проверять только после устранения crop/VAE confound.

## Stage 15A: причинное разложение generator pipeline

Перед переходом к более тяжёлому генератору нужен минимальный equal-dose
factorial experiment. Иначе новая модель будет сравниваться с
методологически неоднозначным baseline.

Все arms:

- ConvNeXt-Small, тот же validated training recipe;
- seeds 42, 43, 44;
- 30 samples/class для `mel`, `akiec`, `bkl`;
- одинаковые 90 lesion-unique источников;
- source/count matched;
- locked test закрыт;
- один заранее выбранный checkpoint endpoint;
- structured artifacts, MLflow и FiftyOne.

| Arm | Преобразование | Изолируемый эффект |
|---|---|---|
| A | original source replay | reference |
| B | exact offline `512x512` center crop, без VAE | fixed crop/resampling |
| C | B + SD1.5 VAE encode/decode, без UNet denoising | VAE bottleneck |
| D | current SD1.5 img2img, strength 0.05 | один denoising step/prompt |

Primary comparisons:

- B − A: цена offline geometry/fixed view;
- C − B: цена generic 4-channel VAE;
- D − C: вклад UNet/prompt при почти нулевом denoising;
- D − A: воспроизведение текущего composite effect.

Primary endpoint: validation macro AUPRC или заранее фиксированный
multi-objective checkpoint rule. Macro F1, MCC, balanced accuracy, ECE,
worst recall, melanoma AUPRC и fixed-specificity — secondary endpoints.
Использовать paired seeds и hierarchical lesion-group bootstrap.

Decision rules:

- если B существенно хуже A, сначала исправлять preprocessing/view diversity;
- если C хуже B, generic VAE запрещается как основной generator backbone;
- если D не лучше C, prompt/UNet при strength 0.05 не создаёт полезной
  вариативности;
- только после этого запускать Stage 15B с domain generator.

## Stage 15B: условный generator pilot

При подтверждённом VAE/generator failure:

1. основной кандидат — воспроизводимый class-conditional HAM10000 diffusion
   с domain-trained 8-channel VAE;
2. lesion-aware кандидат — DiDGen-style region-aware fine-tuning или
   mask-conditioned inpainting;
3. LF-VAR допускается только после checkpoint, license, code и RTX 5080
   16 GB VRAM gate;
4. Derm-T2IM допускается как binary/domain adaptation reference, но не как
   главный семиклассовый comparator.

Новая генерация должна сохранять:

- checkpoint revision/hash;
- VAE architecture и checkpoint hash;
- фактическое число denoising steps;
- prompt/conditioning/mask;
- source lesion ID;
- source и generated preprocessing;
- reconstruction MSE, MS-SSIM/LPIPS и frequency diagnostics;
- изображения до отбора и после отбора в FiftyOne.

## Изменения, внесённые после аудита

- Stage 13 facility selector теперь обеспечивает уникальность
  `source_group_id`, а не только `source_image_id`.
- Capacity и final gates проверяют количество независимых source lesions.
- Добавлены regression tests на два изображения одного поражения.
- Исторические manifests и результаты не изменены.

## Литература и кодовые базы

1. Hugging Face Diffusers. Stable Diffusion Img2Img API:
   https://huggingface.co/docs/diffusers/main/api/pipelines/stable_diffusion/img2img
2. Bissoto A., Valle E., Avila S. GAN-Based Data Augmentation and
   Anonymization for Skin-Lesion Analysis: A Critical Review. CVPRW 2021:
   https://openaccess.thecvf.com/content/CVPR2021W/ISIC/html/Bissoto_GAN-Based_Data_Augmentation_and_Anonymization_for_Skin-Lesion_Analysis_A_Critical_CVPRW_2021_paper.html
3. Kim M. et al. Diffusion-based skin disease data augmentation with
   fine-grained detail preservation and interpolation for data diversity.
   PLOS ONE, 2025:
   https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0331404
   Code: https://github.com/raddshing/skin-disease-diffusion
4. Shentu et al. DiDGen: Diffusion-based Dermoscopic Image Generation with
   Structured Clinical Prompts. MICCAI 2025:
   https://papers.miccai.org/miccai-2025/0230-Paper4243.html
   Code: https://github.com/junjie-shentu/DiDGen
5. Sun et al. Controllable Skin Synthesis via Lesion-Focused Vector
   Autoregression Model. MICCAI 2025:
   https://papers.miccai.org/miccai-2025/0183-Paper0807.html
   Code: https://github.com/echosun1996/LF-VAR
6. Farooq et al. Derm-T2IM: Harnessing Synthetic Skin Lesion Data via Stable
   Diffusion Models for Enhanced Skin Disease Classification using ViT and
   CNN. 2024: https://arxiv.org/abs/2401.05159
   Model: https://huggingface.co/MAli-Farooq/Derm-T2IM
7. Zhao et al. DiffuLT: Diffusion for Long-Tail Recognition. NeurIPS 2024:
   https://papers.neurips.cc/paper_files/paper/2024/hash/de7858e3e7f9f0f7b2c7bfdc86f6d928-Abstract-Conference.html
8. Liang et al. Diffusion Curriculum. ICCV 2025:
   https://openaccess.thecvf.com/content/ICCV2025/html/Liang_Diffusion_Curriculum_Synthetic-to-Real_Data_Curriculum_via_Image-Guided_Diffusion_ICCV_2025_paper.html
9. Tschandl et al. The HAM10000 dataset. Scientific Data, 2018:
   https://www.nature.com/articles/sdata2018161
10. Adamkiewicz et al. When Pretty Isn't Useful. CVPR 2026:
    https://github.com/Bill2462/When-Pretty-Isn-t-Useful-codebase
11. Li S., Lin Y., Chen H., Cheng K.-T. Iterative Online Image Synthesis via
    Diffusion Model for Imbalanced Classification. MICCAI 2024:
    https://papers.miccai.org/miccai-2024/427-Paper0901.html
12. Miętkiewicz Ł., Ciechanowski L., Jemielniak D. The Skin Game:
    Revolutionizing Standards for AI Dermatology Model Comparison. 2025:
    https://arxiv.org/abs/2502.02500

## Итог для статьи

Найденная проблема не обесценивает проект. Напротив, она делает научную
историю точнее: feature-space quality filtering не может компенсировать
неподходящий generator backbone и preprocessing confound. Публикационно
интересный вклад может состоять в causal decomposition:

1. отделить crop, VAE и denoising;
2. показать, какие geometry/frequency metrics предсказывают downstream
   utility;
3. сравнить generic near-copy generation с lesion-aware/domain-trained
   generation при source- и lesion-matched controls;
4. доказать, что visual fidelity или encoder agreement сами по себе
   недостаточны.
