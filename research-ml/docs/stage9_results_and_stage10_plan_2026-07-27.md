# Stage 9: результаты контролируемой проверки utility синтетики

Дата анализа: 2026-07-27.

## Краткий научный вывод

Текущая ветка `Synthetic + DINO` не показала добавочной discriminative utility
по сравнению с повторным предъявлением исходных реальных изображений и с
обычным random oversampling.

Главное matched comparison:

- synthetic против source-matched replay: разница macro F1 `-0.0006`,
  lesion-group bootstrap 95% CI `[-0.0536; 0.0518]`;
- разница MCC `-0.0054`, 95% CI `[-0.0762; 0.0586]`;
- разница melanoma F1 `-0.0053`, 95% CI `[-0.0586; 0.0452]`.

Практически совпавшие средние не доказывают статистическую эквивалентность:
использовано только три seed, а equivalence margins заранее не задавались.
Однако результат не дает evidence, что изменения, внесенные генератором,
полезнее повторного предъявления source images.

Synthetic уверенно лучше только canonical undersampling. Этот baseline удаляет
большую часть реального разнообразия и не является достаточным основанием для
заявления о пользе генерации. Лучшим screening-методом по macro F1 остается
`Real CE oversampling`.

## Что было проанализировано

Stage 8 anchors, seeds `42-44`:

- `stage8_real_ce_natural_384`;
- `stage8_real_ce_weighted_384`;
- `stage8_real_logit_adjust_natural_384`;
- `stage8_synthetic_dino_384`.

Stage 9 controls, seeds `42-44`:

- `stage9_real_ce_undersample_384`;
- `stage9_real_balanced_softmax_natural_384`;
- `stage9_source_replay_weighted_384`;
- `stage9_crt_weighted_384`.

Во всех сравнениях использованы ConvNeXt Base, размер `384`, одинаковый
group-aware validation split и реальные изображения в evaluation.

## Аудит целостности

Результаты извлечены из JSON/CSV, без анализа `run.log`.

| Проверка | Результат |
|---|---|
| Stage 9 runs | 12/12 |
| Seed на метод | 3 (`42-44`) |
| Validation rows на запуск | 1280 |
| Validation lesion groups | 599 |
| Synthetic rows в validation | 0 |
| `image_id/target` между всеми runs | Идентичны |
| `summary.json: test_evaluated` | `false` для 12/12 |
| Sampling/model/trainable artifacts | Полны для 12/12 |
| Calibration diagnostic | Присутствует для 12/12 |
| Пересчет метрик из predictions | Совпал с JSON, max abs error `< 5e-7` |
| Контейнер Stage 9 | Завершился с exit code 0 |

После перезагрузки сервера `nvidia-smi` перестал связываться с драйвером.
Это произошло после завершения обучения и не влияет на сохраненные артефакты,
но GPU stack необходимо восстановить до следующего запуска.

## Средние результаты по matched seeds

Значения приведены как mean по трем seed; `+/-` означает sample SD.

| Метод | Macro F1 | Bal. acc | MCC | ECE | Mel P | Mel R | Mel F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Real CE natural | 0.7355 +/- 0.0113 | 0.7203 | **0.6598** | 0.1296 | **0.6027** | 0.4685 | **0.5183** |
| Real CE oversampling | **0.7681 +/- 0.0276** | **0.7655** | 0.6397 | 0.1570 | 0.4379 | 0.6317 | 0.5154 |
| Real Logit Adjustment | 0.7377 +/- 0.0173 | 0.7437 | 0.6294 | 0.1716 | 0.4333 | 0.5874 | 0.4920 |
| Synthetic + DINO | 0.7384 +/- 0.0103 | 0.7483 | 0.5944 | 0.1560 | 0.3752 | 0.6131 | 0.4571 |
| Real CE undersampling | 0.6086 +/- 0.0516 | 0.6799 | 0.4489 | **0.1258** | 0.2578 | 0.6550 | 0.3635 |
| Real Balanced Softmax | 0.7513 +/- 0.0097 | 0.7571 | 0.6268 | 0.1844 | 0.4099 | 0.6224 | 0.4926 |
| Source-matched replay | 0.7386 +/- 0.0397 | 0.7449 | 0.6000 | 0.1676 | 0.3916 | 0.5711 | 0.4627 |
| cRT | 0.7320 +/- 0.0095 | 0.7439 | 0.6170 | 0.1643 | 0.3817 | **0.6783** | 0.4883 |

Низкий raw ECE undersampling не делает его лучшим: модель одновременно имеет
худшие macro F1, MCC и ranking metrics. Calibration оценивается отдельно.

## Ключевые paired comparisons

### Synthetic против source-matched replay

| Метрика | Mean delta | 95% hierarchical bootstrap CI | P(delta > 0) |
|---|---:|---:|---:|
| Macro F1 | -0.0006 | [-0.0536; 0.0518] | 0.492 |
| Balanced accuracy | +0.0028 | [-0.0308; 0.0406] | 0.549 |
| MCC | -0.0054 | [-0.0762; 0.0586] | 0.442 |
| Mel precision | -0.0164 | [-0.1004; 0.0590] | 0.343 |
| Mel recall | +0.0425 | [-0.0969; 0.1926] | 0.714 |
| Mel F1 | -0.0053 | [-0.0586; 0.0452] | 0.417 |

Source replay имеет особенно высокую seed-variance: macro F1
`0.7818 / 0.7037 / 0.7302`. Поэтому совпадение средних нельзя превращать в
утверждение об эквивалентности.

### Synthetic против real oversampling

| Метрика | Mean delta | 95% hierarchical bootstrap CI | P(delta > 0) |
|---|---:|---:|---:|
| Macro F1 | -0.0306 | [-0.0711; 0.0097] | 0.061 |
| Balanced accuracy | -0.0181 | [-0.0527; 0.0153] | 0.133 |
| MCC | -0.0453 | [-0.0955; 0.0089] | 0.047 |
| Mel precision | -0.0626 | [-0.1735; 0.0680] | 0.166 |
| Mel recall | -0.0181 | [-0.1549; 0.0988] | 0.405 |
| Mel F1 | -0.0580 | [-0.1292; 0.0190] | 0.058 |

На matched seeds synthetic не сохранил даже прежний описательный выигрыш по
melanoma recall. Это поддерживает объяснение через variance/operating point, а
не через устойчивое улучшение представления.

### Synthetic против undersampling

| Метрика | Mean delta | 95% hierarchical bootstrap CI |
|---|---:|---:|
| Macro F1 | +0.1308 | [0.0715; 0.1905] |
| MCC | +0.1452 | [0.0984; 0.1917] |
| Mel F1 | +0.0930 | [0.0281; 0.1512] |

Undersampling использует только `567` уникальных изображений за эпоху и теряет
разнообразие head-класса. Этот результат показывает вред такого режима, а не
уникальную пользу генерации.

## Проверка Stage 9 shortlist

Ни одна новая ветка не проходит заранее зафиксированный shortlist полностью:

- source replay против oversampling: macro F1 `-0.0295`, MCC `-0.0397`;
- cRT против oversampling: macro F1 `-0.0361`, MCC `-0.0227`;
- Balanced Softmax: macro F1 `-0.0169`, MCC `-0.0129`;
- undersampling значительно хуже;
- synthetic против oversampling: macro F1 `-0.0297`, MCC `-0.0453`.

Следовательно, перенос текущей synthetic-конфигурации на locked test запрещен.

## Ranking quality

Mean melanoma ranking metrics:

| Метод | Mel AUROC | Mel AUPRC |
|---|---:|---:|
| Real CE natural | 0.8227 | 0.5575 |
| Real CE oversampling | 0.8505 | 0.5657 |
| Synthetic + DINO | 0.8316 | 0.5190 |
| Real Balanced Softmax | 0.8332 | 0.5236 |
| Source-matched replay | 0.8236 | 0.5107 |
| cRT | **0.8537** | **0.5877** |
| Real CE undersampling | 0.7820 | 0.4219 |

cRT дает интересный диагностический сигнал: высокая melanoma AUROC/AUPRC и
recall при более слабом global macro F1. Значит, качество representation и
выбор operating point нужно отделять от argmax multiclass utility. Это не
основание объявить cRT победителем по primary endpoint.

Полные per-class AUROC/AUPRC сохранены в
`reports/stage9_analysis_20260727/per_class_discrimination.csv`.

## Calibration и fixed specificity

Calibration diagnostic является exploratory: общий validation split ранее
участвовал в early stopping. Внутри diagnostic целые lesion groups разделены на
`299` calibration и `300` evaluation groups.

Temperature scaling снизил средний ECE evaluation subset:

- real oversampling: примерно `0.17 -> 0.06`;
- synthetic: примерно `0.17 -> 0.07`;
- source replay: примерно `0.17 -> 0.05`;
- cRT: примерно `0.18 -> 0.06`.

При целевой specificity `0.90` фактическая mean evaluation specificity была
около `0.88-0.90`; melanoma sensitivity:

- oversampling `0.62`;
- Balanced Softmax `0.60`;
- cRT `0.58`;
- synthetic `0.55`;
- source replay `0.55`;
- undersampling `0.45`.

При specificity `0.95` mean sensitivity:

- oversampling `0.48`;
- Balanced Softmax `0.49`;
- cRT `0.48`;
- synthetic `0.44`;
- source replay `0.44`;
- undersampling `0.38`.

Таким образом, текущая synthetic-ветка не дает преимущество в clinically
interpretable operating points. Temperature scaling улучшает confidence
calibration, но не меняет argmax classification и не создает ranking utility.

## Статистические ограничения

1. При трех seed минимальное двустороннее p exact sign-flip test равно `0.25`.
   Поэтому ни один screening comparison не может достичь `p < 0.05`.
2. Hierarchical bootstrap повторно выбирает seed и целые lesion groups, но три
   seed недостаточны для надежной оценки training variance.
3. Использован один fixed validation split. Неопределенность выбора split не
   измерена.
4. Validation использовался для early stopping; calibration diagnostic не
   является независимой confirmatory оценкой.
5. Locked test остается закрытым.
6. Результат относится к текущему генератору, pool, dose, DINO-фильтру и
   ConvNeXt protocol; он не доказывает бесполезность синтетики вообще.

## Научная интерпретация

Stage 9 усиливает центральную гипотезу будущей статьи:

> Визуальное качество и близость к реальному manifold недостаточны. Полезность
> synthetic augmentation определяется добавочной информацией относительно
> source replay, положением относительно class boundary, покрытием редких
> внутриклассовых мод и согласованием с downstream representation.

Текущий top-k DINO selection не удовлетворяет этой формулировке: strict filter
прошли только `35 mel`, `36 akiec`, `63 bkl`, а квота была заполнена fallback.
Поэтому Stage 8/9 проверили смесь in-distribution и менее подходящих кандидатов.

## План Stage 10

### 10A. Causal geometry-dose screening

Использовать равный dose без fallback, например `30` изображений на класс:

1. strict in-distribution;
2. boundary band: близко к собственному и соседнему классу;
3. OOD/far;
4. random eligible;
5. source-matched real replay для каждого stratum;
6. real oversampling control.

Фиксируются class composition, sample weight, sampler, optimizer steps,
backbone, augmentation и seeds. Главная проверка: есть ли interaction
`stratum x synthetic/replay`, а не просто отличие одного synthetic arm.

### 10B. Regeneration gate

Если существующего pool недостаточно для неперекрывающихся strata:

- генерировать больший pool по grid `strength x guidance x prompt seed`;
- сохранять provenance до отбора;
- запрещать quota-filling top-k fallback;
- проверять low-level domain classifier и DINO coverage до обучения;
- не обучать downstream модель, пока stratum не достигнет заранее заданной
  мощности и не пройдет leakage/pixel audit.

### 10C. Confirmatory protocol

Только geometry-stratum, который превосходит свой source replay и не уступает
oversampling по screening margins, переносится в:

- nested lesion-group cross-validation;
- минимум пять outer folds;
- calibration/threshold selection только внутри inner folds;
- paired outer-fold analysis;
- заранее заданные primary endpoints и multiplicity control;
- один финальный locked-test запуск после замораживания метода.

## Решение после Stage 9

1. Не продолжать увеличивать dose текущего mixed top-k synthetic pool.
2. Не открывать locked test.
3. Считать real oversampling текущим reference baseline.
4. Перейти к Stage 10A как к проверке механизма полезности синтетики.
5. Использовать cRT как ranking diagnostic, но не как primary classifier.
6. До GPU-экспериментов восстановить NVIDIA driver/container runtime.

## Воспроизводимые материалы

Анализ:

```bash
python tools/analyze_stage9_results.py \
  --artifact-root local_artifacts/stage9_structured_20260727 \
  --out-dir reports/stage9_analysis_20260727 \
  --bootstrap-replicates 5000
```

Сохраненные таблицы:

- `seed_metrics.csv`;
- `method_summary.csv`;
- `paired_seed_comparisons.csv`;
- `hierarchical_lesion_bootstrap.csv`;
- `per_class_discrimination.csv`;
- `calibration_summary.csv`;
- `calibration_offset_outcomes.csv`;
- `fixed_specificity_diagnostics.csv`;
- `stage9_artifact_integrity.csv`;
- `prediction_alignment.csv`;
- `analysis_summary.json`.

## Использованная литература

Новые статьи для вычислительного анализа не добавлялись. Интерпретация и
постановка controls опираются на уже внесенные в Google Sheet проекта работы:

1. Kang et al., *Decoupling Representation and Classifier for Long-Tailed
   Recognition*, <https://arxiv.org/abs/1910.09217>.
2. Menon et al., *Long-tail learning via logit adjustment*,
   <https://arxiv.org/abs/2007.07314>.
3. MONICA, <https://arxiv.org/abs/2410.02010>.
4. Исследование random oversampling/undersampling и calibration,
   <https://doi.org/10.1186/s40537-023-00857-7>.
5. DiffuLT,
   <https://papers.nips.cc/paper_files/paper/2024/hash/de7858e3e7f9f0f7b2c7bfdc86f6d928-Abstract-Conference.html>.

Подробные конспекты находятся в
`docs/literature_update_synthetic_utility_and_long_tail_2026-07-24.md`.
