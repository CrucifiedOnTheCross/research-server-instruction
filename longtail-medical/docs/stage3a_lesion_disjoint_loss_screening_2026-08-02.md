# Stage 3A: lesion-disjoint long-tail loss screening

Date: 2026-08-02  
Status: completed; validation-only analysis finalized
Test state: Stage 2 test was opened once and is now permanently closed  
Selection data: lesion-disjoint train and validation only

## Scientific question

Which established long-tail objective improves generalization to previously
unseen lesions when backbone, pretrained initialization, optimizer,
augmentation, split, batch, epoch count, and model seeds are held fixed?

This is a screening stage, not the final claim. Its result selects CE and at
most two methods for Stage 3B confirmation across new lesion-disjoint split
seeds 101, 202, and 303. No Stage 3A decision may use the Stage 2 test.

## Frozen matrix

Six objectives are trained for seeds 42, 43, and 44 (18 runs):

| Arm | Frozen implementation |
|---|---|
| CE | ordinary multiclass cross-entropy |
| Weighted CE | inverse class frequency, normalized to mean weight 1 |
| Focal | multiclass focal CE, gamma 2 |
| CB-Focal | effective-number weights, beta 0.9999, gamma 2 |
| Balanced Softmax | CE on `logits + log(class_count)` |
| LDAM-DRW | margin proportional to `n_c^-1/4`, max margin 0.5, cosine classifier, scale 30; effective-number reweighting only after epoch 40 |

Post-hoc Logit Adjustment with tau 1 is derived from CE validation
predictions and does not retrain a duplicate model. This is deliberate:
training-time logit adjustment with tau 1 has the same objective as Balanced
Softmax. The post-hoc variant uses `log(p_c) - log(pi_c)` and is recorded as a
derived validation-only method.

All training runs use ResNet-50 with ImageNet-1K v2 initialization, Adam with
learning rate 0.0003 and zero weight decay, physical batch 256, bfloat16,
224-pixel crops, the Stage 2 augmentations, exactly 50 epochs, and no early
stopping. `last.pt` is primary; `best.pt` is secondary and selected only by
lesion-level validation MCC.

LDAM's normalized classifier is an explicit component of its canonical
implementation. It is saved as `classifier_type=cosine_normed_linear`; other
arms use the Stage 2 linear classifier. Therefore any LDAM effect is attributed
to the canonical LDAM-DRW package, not to the margin in isolation.

## Endpoints and decision rule

Primary screening endpoint: lesion-level validation MCC from `last.pt`.

Key secondary endpoints: lesion-level balanced accuracy, macro AUPRC, MEL/SCC/AK
recall and AUPRC, worst-class recall, ECE, NLL, Brier score, image-level metrics,
seed sign stability, epoch trajectory, elapsed time, and peak VRAM.

Ranking is based on multi-seed mean, paired-seed effects versus CE, and lesion
bootstrap uncertainty. A method is not promoted solely because one seed is
best. CE and at most two methods advance. Test remains unavailable. Calibration
is reported but does not replace the primary ranking metric.

## Reproducibility artifacts

Every run must contain resolved YAML and JSON configuration, environment,
split hashes, run signature, class counts, model and loss initialization,
epoch CSV, last/best validation metrics and predictions, `last.pt`, `best.pt`,
and `summary.json` with `test_evaluated=false`. MLflow uses project, stage,
split, method, seed, protocol, checkpoint, and test-policy tags.

Readiness requires exactly 18 valid training runs from one commit and three
derived post-hoc runs. Configurations contain no `test_csv`; split artifacts
must state `test_loaded=false` and `test_evaluated=false`.

## Literature and implementation audit

- Lin et al., Focal Loss for Dense Object Detection, ICCV 2017:
  https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf
- Cui et al., Class-Balanced Loss Based on Effective Number of Samples, CVPR 2019:
  https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html
- Cao et al., Label-Distribution-Aware Margin Loss, NeurIPS 2019:
  https://proceedings.neurips.cc/paper/2019/hash/621461af90cadfdaf0e8d4cc25129f91-Abstract.html
- Ren et al., Balanced Meta-Softmax, NeurIPS 2020:
  https://proceedings.neurips.cc/paper/2020/hash/2ba61cc3a8f44143e1f2f13b2b729ab3-Abstract.html
- Menon et al., Long-tail learning via logit adjustment, ICLR 2021:
  https://research.google/pubs/long-tail-learning-via-logit-adjustment/
- Ju et al., MONICA medical long-tail benchmark:
  https://arxiv.org/abs/2410.02010

Official code snapshots inspected before implementation:

- MONICA commit `3dd808d6d578b9e0f9bf4ee1402727ff46d1c243`;
- LDAM-DRW commit `2536330f2afdaa65618323cb5a5850efccce762a`;
- Balanced Meta-Softmax commit `34a61e432881816c2da14d577d6ed63501288f5f`.

The audit confirmed Balanced Softmax adds log class counts to logits; the
effective-number implementations normalize weights to the number of classes;
canonical LDAM uses `n_c^-1/4`, maximum margin 0.5, scale 30, and deferred
reweighting. MONICA defers reweighting until 80% of its configured epochs.

## Stage 3B gate

After Stage 3A, freeze CE and no more than two promoted methods before creating
or inspecting any new split test. Stage 3B must separate split variability from
model-seed variability. Synthetic augmentation belongs to Stage 3C and must be
compared with exposure-matched oversampling and simpler augmentation controls.

## Итог выполнения

Stage 3A завершён без технических или протокольных отклонений. Readiness
зафиксировал `ready=true`, 18 обученных запусков и 3 производных запуска Logit
Adjustment. Все 18 запусков выполнены на commit `f3ca1e4`, содержат 50 эпох,
`last.pt`, `best.pt`, разрешённые конфигурации, хеши разбиения, инициализацию
модели и функции потерь, предсказания и метрики validation. Во всех артефактах
`test_evaluated=false`; `test_csv` отсутствует. Закрытый после Stage 2 test не
загружался и не использовался.

В validation содержится 190 независимых групп поражений: `nv` 33, `mel` 15,
`bcc` 19, `bkl` 24, `ak` 28, `scc` 18, `vasc` 28 и `df` 25. Поэтому
неопределённость для `mel`, `scc` и других редких классов остаётся существенной
даже при детерминированном обучении.

## Статистический анализ

Первичный анализ использует `last.pt`, как было зафиксировано до обучения.
`best.pt`, выбранный по lesion-level MCC на той же validation, является только
анализом чувствительности и не меняет первичное решение.

Для сравнений с CE выполнен 10 000-кратный иерархический парный бутстрэп:

1. три сида выбирались с возвращением;
2. внутри каждого выбранного сида группы `lesion_id` выбирались с возвращением
   отдельно в каждом классе;
3. для каждой реплики вычислялась разность метода и CE;
4. итоговая реплика усреднялась по выбранным сидам.

Такой расчёт учитывает зависимость нескольких изображений одного поражения и
случайность инициализации модели, но не заменяет повторение на новых вариантах
разбиения. Независимые сравнения выполняются шестью процессами с фиксированными
последовательностями случайных чисел. Параллельная версия сократила время
анализа с 19.6 до 6.9 секунды; SHA-256 выходного JSON совпал с последовательной
версией.

## Первичный результат: `last.pt`

Все значения представлены как среднее ± стандартное отклонение по сидам
42–44. `ΔMCC` вычислен относительно парного CE с тем же сидом. Доверительный
интервал получен иерархическим бутстрэпом.

| Метод | Lesion MCC | ΔMCC | 95% ДИ ΔMCC | Знак по сидам | Balanced accuracy | Macro AUPRC | ECE | NLL | Worst recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CE | 0.4715 ± 0.0328 | 0 | — | — | 0.5155 | 0.5770 | 0.3404 | 3.135 | 0.1274 |
| Weighted CE | 0.4654 ± 0.0164 | −0.0061 | [−0.0569; 0.0477] | 1/3 | 0.5099 | 0.6038 | 0.3052 | 2.521 | 0.1119 |
| Focal | 0.4454 ± 0.0222 | −0.0261 | [−0.0736; 0.0213] | 1/3 | 0.4889 | 0.5846 | 0.2657 | 2.208 | 0.0370 |
| CB-Focal | 0.4776 ± 0.0267 | +0.0061 | [−0.0486; 0.0563] | 2/3 | 0.5196 | **0.6073** | **0.2146** | **1.836** | 0.0926 |
| Balanced Softmax | 0.4703 ± 0.0314 | −0.0012 | [−0.0684; 0.0746] | 1/3 | 0.5150 | 0.5967 | 0.3203 | 2.360 | 0.1274 |
| **LDAM-DRW** | **0.5295 ± 0.0212** | **+0.0580** | **[0.0190; 0.0987]** | **3/3** | **0.5683** | 0.5928 | 0.3045 | 4.096 | **0.1618** |
| Logit Adjustment | 0.5019 ± 0.0270 | +0.0304 | [−0.0040; 0.0627] | 2/3 | 0.5499 | 0.5991 | 0.3022 | 2.356 | 0.1600 |

Для LDAM-DRW вероятность положительной разности MCC в бутстрэп-распределении
равна 0.9985, а balanced accuracy — 0.9944. Для Logit Adjustment интервал MCC
слегка пересекает ноль, но прирост balanced accuracy равен +0.0344 с 95% ДИ
[0.0032; 0.0638]. У остальных методов интервалы MCC и balanced accuracy
включают ноль.

### Разделение ранжирования и порогового решения

LDAM-DRW дал крупнейший прирост MCC и balanced accuracy, но только +0.0158
macro AUPRC относительно CE. Следовательно, его основной эффект связан не
только с улучшением общего ранжирования: изменились границы решения между
классами. Высокий NLL 4.096 показывает, что вероятности LDAM-DRW остаются
чрезмерно уверенными и перед практическим применением требуют отдельной
калибровки.

CB-Focal не дал убедительного прироста MCC, однако показал лучший macro AUPRC,
ECE и NLL. Это полезный отрицательно-положительный результат: метод улучшает
вероятностное ранжирование и калибровку, но в текущем фиксированном режиме не
превращает это улучшение в устойчивый выигрыш первичной пороговой метрики.

Weighted CE и Balanced Softmax также увеличили macro AUPRC относительно CE,
но их MCC практически не изменился. Обычный Focal оказался хуже CE по MCC и
worst-class recall; его нельзя продвигать только на основании общей литературы
о дисбалансе.

## Метрики на уровне изображений

Lesion-level оценка является основной, но направление эффекта дополнительно
проверено на отдельных изображениях.

| Метод | Image MCC | Balanced accuracy | Macro F1 | Macro AUPRC | ECE | NLL |
|---|---:|---:|---:|---:|---:|---:|
| CE | 0.3916 | 0.4525 | 0.4025 | 0.5223 | 0.4420 | 4.136 |
| Weighted CE | 0.4010 | 0.4642 | 0.4270 | 0.5490 | 0.3891 | 3.305 |
| Focal | 0.3973 | 0.4567 | 0.4016 | 0.5233 | 0.3351 | 2.790 |
| CB-Focal | 0.4186 | 0.4792 | 0.4338 | **0.5520** | **0.2920** | **2.395** |
| Balanced Softmax | 0.4129 | 0.4758 | 0.4369 | 0.5265 | 0.4097 | 3.256 |
| **LDAM-DRW** | **0.4311** | **0.4942** | **0.4587** | 0.5305 | 0.4625 | 5.594 |
| Logit Adjustment | 0.4096 | 0.4733 | 0.4370 | 0.5361 | 0.4050 | 3.216 |

LDAM-DRW сохраняет лучший MCC, balanced accuracy и macro F1 и на уровне
изображений. Одновременно ухудшение ECE/NLL становится ещё заметнее, что
подтверждает необходимость не смешивать качество классификации и качество
вероятностной калибровки.

## Редкие и клинически важные классы

Ниже приведены lesion-level F1 / AUPRC для первичного `last.pt`.

| Метод | MEL | AK | SCC | DF |
|---|---:|---:|---:|---:|
| CE | 0.390 / 0.444 | 0.335 / 0.457 | 0.249 / 0.384 | 0.255 / 0.432 |
| Weighted CE | 0.375 / 0.468 | 0.425 / 0.501 | 0.188 / 0.265 | 0.275 / **0.569** |
| Focal | 0.393 / 0.426 | 0.426 / 0.498 | 0.068 / 0.364 | 0.255 / 0.487 |
| CB-Focal | 0.400 / 0.436 | 0.469 / **0.535** | 0.139 / 0.308 | 0.267 / 0.547 |
| Balanced Softmax | 0.434 / **0.531** | 0.384 / 0.449 | 0.206 / 0.336 | 0.303 / 0.461 |
| **LDAM-DRW** | **0.473** / 0.441 | **0.483** / 0.516 | 0.295 / 0.323 | **0.334** / 0.436 |
| Logit Adjustment | 0.441 / 0.481 | 0.367 / 0.468 | **0.365 / 0.397** | 0.273 / 0.475 |

LDAM-DRW повысил MEL F1 без снижения recall: recall равен 0.711 как у CE, а
рост F1 обусловлен улучшением точности. Для AK он повысил recall с 0.238 до
0.417 и F1 с 0.335 до 0.483. Для SCC наилучший пороговый результат дал Logit
Adjustment: recall вырос с 0.185 до 0.315, F1 — с 0.249 до 0.365, а AUPRC — с
0.384 до 0.397.

Balanced Softmax дал лучший MEL AUPRC 0.531 и чувствительность 0.511 при
специфичности 0.95, но его средний MEL recall при argmax снизился с 0.711 до
0.644. Это прямой пример того, почему AUPRC и фиксированная специфичность
нельзя заменять одной итоговой метрикой.

Для SCC Weighted CE, Focal, CB-Focal и Balanced Softmax ухудшили AUPRC и/или
F1 относительно CE. Усиление редких классов не является равномерным: один
способ компенсации дисбаланса может помочь AK или DF и одновременно навредить
SCC.

## Чувствительность к выбору контрольной точки

| Метод | Best epoch, среднее | Last MCC | Best MCC | Δ best−last | Last AUPRC | Best AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| CE | 33.7 | 0.4715 | 0.5121 | +0.0406 | 0.5770 | 0.6012 |
| Weighted CE | 18.7 | 0.4654 | 0.5552 | +0.0898 | 0.6038 | 0.6399 |
| Focal | 37.7 | 0.4454 | 0.5150 | +0.0697 | 0.5846 | 0.6305 |
| CB-Focal | 23.0 | 0.4776 | 0.5479 | +0.0702 | 0.6073 | 0.6267 |
| Balanced Softmax | 19.3 | 0.4703 | **0.5747** | **+0.1043** | 0.5967 | **0.6648** |
| LDAM-DRW | 47.0 | **0.5295** | 0.5462 | +0.0167 | 0.5928 | 0.6043 |
| Logit Adjustment | — | 0.5019 | 0.5273 | +0.0253 | 0.5991 | 0.6155 |

Balanced Softmax выглядит лучшим при validation-selected `best.pt`, но его
лучшие эпохи нестабильны: 30, 6 и 22. Поскольку `best.pt` выбирался на той же
validation, по которой сравниваются методы, этот результат оптимистичен и не
может заменить заранее заданный `last.pt`. Большой разрыв best−last у Weighted
CE, CB-Focal и Balanced Softmax указывает на переобучение или на нестабильную
траекторию после раннего максимума.

LDAM-DRW принципиально отличается: лучшие эпохи 50, 44 и 47, а средний разрыв
best−last составляет только +0.0167 MCC. Его выигрыш не зависит от удачного
раннего снимка модели и потому является наиболее надёжным результатом Stage
3A.

## Вычислительная стоимость

Все обучаемые методы использовали в среднем 11.04 GiB VRAM. Один 50-эпоховый
запуск занимал 11.4–11.8 минуты; суммарное GPU-время 18 запусков составило 3.48
часа. Различия функций потерь практически не изменили стоимость обучения.
Logit Adjustment не требует повторного обучения и вычисляется из сохранённых
CE-предсказаний.

## Научный вывод

Главная гипотеза Stage 3A подтверждена частично: корректная функция потерь
может улучшить обобщение на новые поражения при неизменных данных и
архитектуре, но большинство популярных long-tail методов не дают устойчивого
прироста первичной метрики.

1. **LDAM-DRW — единственный убедительно положительный обучаемый метод.**
   Прирост lesion MCC +0.058, одинаковый знак 3/3 и 95% ДИ полностью выше нуля.
2. **Logit Adjustment — перспективный почти бесплатный второй метод.** Он
   улучшил balanced accuracy и особенно SCC, хотя ДИ для MCC слегка включает
   ноль.
3. **CB-Focal — лучший по калибровке и macro AUPRC, но не по первичному MCC.**
   Его следует сохранить как объясняющий результат, а не продвигать по
   первичному критерию.
4. **Balanced Softmax чувствителен к выбору эпохи.** Сильный `best.pt` при
   слабом `last.pt` требует независимого подтверждения, а не post-hoc смены
   checkpoint policy.
5. **Обычный Focal не подходит для текущего протокола.** Он ухудшил MCC и
   worst-class recall, особенно SCC.

## Решение для Stage 3B

До создания новых разбиений фиксируется следующий минимальный набор:

- обучать **CE** как неизменный контроль;
- обучать **LDAM-DRW** как единственный продвинутый обучаемый arm;
- вычислять **Logit Adjustment tau=1** из CE-предсказаний без отдельного
  обучения.

Stage 3B должен использовать новые lesion-disjoint split seeds 101, 202 и 303
и отделить вариативность разбиения от вариативности инициализации. Основной
анализ остаётся `last.pt`; `best.pt` приводится отдельно. Для LDAM-DRW заранее
добавляется калибровочная ветка на validation, но калибровка не используется
для выбора основной модели. Синтетические данные не допускаются до Stage 3C.

## Ограничения

- Stage 3A использует одно фиксированное разбиение; три model seeds не измеряют
  вариативность между разбиениями.
- Validation одновременно используется для скрининга методов и анализа
  `best.pt`, поэтому secondary-оценки не являются независимым подтверждением.
- Для MEL имеется только 15 независимых поражений, для SCC — 18; классовые
  выводы требуют широких интервалов и повторения на новых split seeds.
- Иерархический бутстрэп учитывает lesion clustering и model seeds, но не
  создаёт новую популяцию источников или устройств съёмки.
- LDAM-DRW является каноническим пакетом margin + cosine classifier + DRW;
  текущий этап не разделяет причинный вклад этих компонентов.
- Закрытый test уже был использован Stage 2 и навсегда исключён из дальнейшего
  выбора. Stage 3A не даёт новой test-оценки и не должен представляться как
  окончательная внешняя валидация.
