# Stage 11: квалификация baseline и условный план после Stage 10

Дата фиксации: 2026-07-27.

Статус: **код подготовлен, запуск заблокирован до завершения и разбора Stage 10**.

## Зачем нужен Stage 11

Stage 10 отвечает на вопрос о downstream utility различных geometry strata
синтетики при фиксированном ConvNeXt-B baseline. Этого достаточно для причинного
сравнения внутри Stage 10, но недостаточно для сильного утверждения статьи:

> эффект синтетики не должен быть следствием слабого, нестабильного или
> архитектурно специфичного baseline.

Stage 11 квалифицирует representation/training baseline отдельно от выбора
синтетических данных. Он не заменяет Stage 10 и не меняет его протокол.

## Главные гипотезы

### H11.1: текущий ConvNeXt-B может быть недооптимизирован

Наблюдения Stage 8-9:

- высокая дисперсия best epoch между seeds;
- weighted CE достигает очень малого train loss при заметном val loss;
- для отдельных synthetic/replay arms наблюдается падение best-to-last;
- текущий recipe не использует layer-wise LR decay, EMA или label smoothing.

Проверка: сравнить существующий `stage8_real_ce_natural_384` с тем же
ConvNeXt-B, но с более консервативным transfer-learning recipe.

### H11.2: ConvNeXt-B может быть избыточным для HAM10000

ConvNeXt-B имеет около 88 млн параметров при 7228 уникальных real train images.
ConvNeXt-S имеет около 49.5 млн параметров. Если Small сохраняет средний macro F1
и MCC, но уменьшает variance/overfit, он является более защищаемым baseline.

### H11.3: вывод о synthetic utility может зависеть от representation family

DINOv2-B/14 используется в двух режимах:

1. frozen backbone + linear head;
2. full fine-tuning.

Linear probe проверяет, достаточно ли уже существующей геометрии признаков.
Full fine-tuning проверяет достижимое качество ViT-backbone. Эти режимы нельзя
смешивать в один вывод: высокая linear-probe метрика говорит о переносимости
представления, а не о превосходстве end-to-end classifier.

### H11.4: более сложная модель не считается автоматически более сильной

Кандидат становится новым baseline только при согласованном выигрыше по seeds и
клинически важным метрикам. Один лучший seed или более высокая вычислительная
стоимость не являются основанием для promotion.

## Литературные основания

### MONICA

Ju et al., *MONICA: Benchmarking on Long-tailed Medical Image Classification*:
<https://arxiv.org/abs/2410.02010>.

MONICA показывает, что выводы long-tail medical classification часто становятся
несопоставимыми из-за разных backbone, split и training protocol. В Stage 11
архитектура меняется только на одном real-only split, при одинаковых seeds и
закрытом test.

### ConvNeXt

Liu et al., *A ConvNet for the 2020s*, CVPR 2022:
<https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html>.

Официальный код:
<https://github.com/facebookresearch/ConvNeXt>.

Использованные идеи:

- AdamW и cosine schedule;
- EMA как доступный элемент recipe;
- label smoothing;
- layer-wise LR decay при transfer learning;
- отдельные 384 px checkpoints;
- ConvNeXt-S как capacity control для ConvNeXt-B.

Важная реализационная поправка: официальный код группирует ConvNeXt примерно в
12 частей, тогда как `timm 1.0.27` создаёт 38 уровней для Small. Буквальный
`layer_decay=0.8` в `timm` даёт минимальный scale около `0.00026` и почти
замораживает ранние слои. Используется `0.925`, что даёт нижний scale около
`0.056` и приблизительно сохраняет глубину затухания официального
`0.8^13`.

### DINOv2

Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*:
<https://arxiv.org/abs/2304.07193>.

Официальный код:
<https://github.com/facebookresearch/dinov2>.

Использованные идеи:

- frozen linear evaluation как самостоятельная диагностика representation;
- ViT-B/14 как сильный, но выполнимый на RTX 5080 architecture control;
- full fine-tuning оценивается отдельно;
- размер 392 выбран как ближайший к 384 размер, кратный patch size 14.

### Fine-tuning может искажать pretrained features

Kumar et al., *Fine-Tuning can Distort Pretrained Features and Underperform
Out-of-Distribution*, ICLR 2022:
<https://arxiv.org/abs/2202.10054>.

Работа обосновывает парное сравнение linear probe и full fine-tuning. Она не
доказывает OOD-эффект в HAM10000: для такого утверждения потребуется отдельная
внешняя когорта.

Все три новые литературные записи добавлены в Google Sheet проекта, строки
67-69. MONICA уже была зафиксирована ранее.

## Матрица Stage 11A

Existing anchor:

| Experiment | Architecture | Training |
|---|---|---|
| `stage8_real_ce_natural_384` | ConvNeXt-B 384 | текущий natural CE |

Новые arms:

| Experiment | Params | Effective batch | Основное отличие |
|---|---:|---:|---|
| `stage11_real_convnext_base_regularized_384` | ~88M | 32 | EMA, smoothing 0.05, normalized layer decay |
| `stage11_real_convnext_small_regularized_384` | 49.5M | 32 | capacity control с тем же recipe |
| `stage11_real_dinov2_base_linear_392` | 86.6M total, trainable head only | 64 | frozen representation |
| `stage11_real_dinov2_base_finetune_392` | 86.6M | 32 | full fine-tuning, batch 16 × accumulation 2 |

Каждый новый arm: seeds `42, 43, 44`, всего 12 runs. Test остаётся закрытым:
`evaluation.run_test=false`.

Stage 11A использует natural sampling. Это намеренно изолирует качество
representation/training recipe от weighted sampler. Long-tail method выбирается
только после этой квалификации.

## Критерии выбора baseline

Primary:

- validation macro F1, mean и standard deviation по трём seeds.

Co-primary/guardrails:

- MCC;
- balanced accuracy;
- ECE до post-hoc calibration;
- mel precision, recall, F1 и AUPRC;
- per-class AUROC/AUPRC;
- worst-class recall;
- best epoch и best-to-last gap;
- trainable parameters, elapsed time и peak VRAM.

Promotion требует:

1. полного набора seeds `42-44`;
2. отсутствия test evaluation;
3. одинакового lesion-group split;
4. отсутствия деградации MCC и mel AUPRC, скрытой ростом macro F1;
5. оценки uncertainty по lesion-group bootstrap;
6. содержательной величины эффекта, а не только знака среднего.

Три seeds дают screening evidence, но не заменяют независимые group-aware folds.
Финальный baseline для статьи должен быть подтверждён group-aware repeated
split или k-fold на train+validation без открытия locked test.

## Условные ветви после Stage 10

### Ветка A: ни один synthetic stratum не превосходит matched replay

- Stage 11A выполняется полностью.
- Главный научный результат формулируется как отрицательный, но содержательный:
  feature-space proximity/quality недостаточны для utility.
- Новый synthetic запуск на всех backbone не выполняется.
- После выбора baseline проверяется только strongest synthetic stratum против
  source-matched replay и random oversampling.

### Ветка B: один stratum устойчиво превосходит replay

- Stage 11A выбирает сильный baseline.
- В Stage 11B winning stratum и matched replay повторяются на выбранном baseline.
- Для проверки architecture dependence допускается второй backbone, но только
  для одной заранее выбранной пары.
- Synthetic dose и source matching остаются теми же, что в Stage 10.

### Ветка C: знак эффекта нестабилен между seeds

- Не увеличивать число методов.
- Сначала провести lesion-group bootstrap и error-overlap analysis.
- Затем увеличить seeds до 5 для одной пары или выполнить group-aware folds.
- Geometry threshold/dose не менять одновременно.

### Ветка D: geometry stratum полезен только для mel operating point

- Отдельно анализировать ranking и threshold effects.
- Сравнить raw и calibrated probabilities.
- Не заявлять общий classifier improvement, если macro F1 растёт за счёт
  ухудшения MCC, AUPRC или других классов.

## Что сознательно не включено до результатов

- Mixup/CutMix: они смешивают визуальную семантику поражений и добавляют новый
  augmentation confound.
- новый генератор;
- увеличение synthetic dose;
- одновременная замена backbone и sampler в utility experiment;
- locked-test selection;
- MiSLAS/LDAM-DRW в Stage 11A.

MiSLAS или LDAM-DRW рассматриваются для Stage 11B как один сильный long-tail
control после выбора representation baseline. Добавлять несколько таких методов
сейчас означало бы расходовать GPU до ответа Stage 10.

## Реализованные изменения кода

### Training engine

Добавлены опциональные поля:

- `training.layer_decay`;
- `training.label_smoothing`;
- `training.model_ema`;
- `training.model_ema_decay`;
- `training.model_ema_warmup`.

При значениях по умолчанию поведение старых стадий не меняется.

Для моделей с фиксированным positional embedding добавлено опциональное
`model.img_size`. DINOv2 в `timm` по умолчанию ожидает 518 px; без явного
`img_size=392` первый batch завершился бы assertion error.

Label smoothing применяется только к train CE. Evaluation loss использует
обычный CE. Для focal/Balanced Softmax/logit adjustment ненулевое smoothing
завершается явной ошибкой вместо неявного смешивания методов.

EMA model используется для validation model selection и сохраняется в
checkpoint с `weights_source=ema`. Online model продолжает получать градиенты.

Layer decay реализован через `timm.create_optimizer_v2`, после чего `lr_scale`
явно применяется к каждой parameter group. Для scheduler используется общий
cosine multiplier, сохраняющий отношение LR между слоями.

Каждый новый run сохраняет:

- `optimizer_groups.json` с числом параметров, LR scale и weight decay;
- `lr_min` и `lr_max` в epoch metrics;
- `weights_source` в checkpoint и `summary.json`;
- полный resolved config и прежние artifacts.

### Gate

`tools/check_stage11_gate.py` требует:

- ровно 24 Stage 10 runs;
- все 8 experiment × 3 seed combinations;
- `test_evaluated=false`;
- целостные `summary`, config, model initialization, sampling plan, class counts,
  validation metrics/predictions и checkpoint;
- `status: approved`;
- `stage10_analysis_reviewed: true`;
- наличие всех включённых Stage 11 configs.

Gate пишет `outputs/reports/stage11_gate.json`. Пока решение имеет статус
`pending_stage10`, запуск обязан завершаться до создания training container.

## Проверки реализации

В отдельной server staging-копии, без изменения live Stage 10:

- `timm 1.0.27` подтверждён;
- обе ConvNeXt и DINOv2 model names доступны;
- `ModelEmaV3` доступна;
- 15/15 unit tests прошли;
- ConvNeXt-S: 49 460 071 параметр, 76 optimizer groups;
- DINOv2-B/14: 86 585 095 параметров, 28 optimizer groups;
- LR ratio сохраняется после scheduler step;
- pretrained DINOv2-B/14 успешно выполнил CPU forward `1×3×392×392`;
- gate подтверждён закрытым при `pending_stage10`;
- локальный host прошёл `compileall`; PyTorch на host не установлен, поэтому
  ML unit tests выполнялись в server image;
- GPU smoke не запускался, чтобы не конкурировать с активным Stage 10.

Во время проверки была обнаружена и исправлена критическая ошибка интеграции:
сырой `lr_scale` из `timm` не применяется стандартным PyTorch scheduler
автоматически. Без исправления layer decay существовал бы только в metadata.

Также smoke forward выявил, что DINOv2 без явного `model.img_size` ожидает
`518×518`; конфиги и model factory исправлены до запуска.

## Ресурсы lab-bio

- RTX 5080 16 GB;
- BF16;
- channels-last;
- 24 GB shared memory;
- 12 DataLoader workers, prefetch 4, persistent workers;
- 24 OMP/MKL threads;
- Hugging Face/Torch cache на server SSD.

Effective batch фиксируется в 32 для full fine-tuning сравнений. ConvNeXt-S не
получает больший batch, потому что это изменило бы optimization dynamics и
ослабило архитектурное сравнение. Frozen DINO linear probe использует batch 64.

Ожидаемая длительность 12 новых runs: ориентировочно 12-20 часов в зависимости
от early stopping и фактической скорости DINOv2. Оценка будет уточнена после
первого run каждой architecture family.

## Действия после завершения Stage 10

1. Собрать structured Stage 10 artifacts, не извлекать метрики из noisy logs.
2. Выполнить paired synthetic-versus-replay analysis и lesion-group bootstrap.
3. Выбрать ветку A-D и записать rationale в `configs/stage11_decision.yaml`.
4. При необходимости сократить/изменить `enabled_configs`.
5. Установить `stage10_analysis_reviewed: true` и `status: approved`.
6. Сделать отдельный commit решения.
7. Запустить `scripts/start_stage11_container.sh`.
8. После Stage 11A выбрать final baseline и только затем определить Stage 11B.

## Открытые научные вопросы

- Достаточно ли внутреннего HAM10000 validation для выбора representation, или
  до статьи необходим внешний ISIC cohort?
- Следует ли считать mel AUPRC co-primary вместе с macro F1?
- Если Small равен Base, считать ли меньшую variance и compute достаточным
  основанием для promotion?
- Если DINO linear probe силён, а fine-tuning слабее, нужен ли adapter/partial
  fine-tuning вместо полного обновления backbone?
- Должен ли winning synthetic stratum переноситься между ConvNeXt и DINOv2
  feature spaces, или architecture dependence является отдельным результатом?
