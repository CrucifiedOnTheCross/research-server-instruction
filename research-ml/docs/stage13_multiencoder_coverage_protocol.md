# Stage 13: multi-encoder coverage-targeted synthetic selection

Дата фиксации: 2026-07-29.

## Основание

Stage 12 показал:

- PRDC precision/density/coverage `strict_id` ниже replay в 11/12 сравнений;
- Vendi выше в 11/12, поэтому проблема не сводится к mode collapse;
- DINOv2 и real-only ConvNeXt-S расходятся в оценке source novelty;
- frequency gap поддержан в 14/15 paired comparisons;
- melanoma AUPRC ухудшена на 3/3 seeds из-за ranking-tail regression.

Следовательно, ещё более строгий nearest-DINO отбор не является обоснованным.

## Литература

### DiffuLT, NeurIPS 2024

DiffuLT выделяет approximately-in-distribution samples: полезные примеры могут
немного отклоняться от real distribution, но слишком далёкие samples вредны.
Это обосновывает двухсторонний novelty interval вместо минимизации расстояния.

Статья: https://papers.neurips.cc/paper_files/paper/2024/hash/de7858e3e7f9f0f7b2c7bfdc86f6d928-Abstract-Conference.html

### Wei, Iyer, Bilmes, ICML 2015

`Submodularity in Data Subset Selection and Active Learning` связывает
nearest-neighbour subset selection с монотонной субмодулярной facility-location
задачей. Жадный алгоритм используется для выбора представительного набора при
ограничении мощности.

Статья: https://proceedings.mlr.press/v37/wei15.html

### PRDC и Vendi

PRDC разделяет fidelity и coverage, Vendi измеряет внутреннее разнообразие
набора. Они используются совместно: высокий Vendi не компенсирует низкие
density/coverage.

- https://proceedings.mlr.press/v119/naeem20a.html
- https://openreview.net/forum?id=g97OHbQyk1

## Stage 13A: train-only selection

Вход:

- 1440 synthetic candidates;
- классы `mel`, `akiec`, `bkl`;
- strengths 0.15, 0.30, 0.45;
- 160 sources на класс;
- 7228 real train rows;
- DINOv2 и три real-only ConvNeXt-S checkpoints.

Validation predictions и validation metrics не участвуют в выборе.
Locked test не читается.

### Candidate features

Для каждого encoder:

- own-source cosine distance;
- source local radius: расстояние до 5-го real same-class neighbour;
- local novelty ratio: own-source distance / source radius;
- nearest same-class non-source distance;
- nearest other-class distance;
- class margin;
- inside-real-manifold indicator.

Frequency:

- low/mid/high radial Fourier energy;
- spectral slope;
- gradient RMS;
- robust z-score относительно real train class;
- paired source-frequency distance.

### Предварительно заданные tiers

Tier A:

- inside real manifold в 3/4 encoders;
- positive class margin в 3/4 encoders;
- local novelty ratio в `[0.25, 1.50]` в 3/4 encoders;
- frequency robust max-z `<= 4`;
- source-frequency distance не выше class median pool.

Tier B:

- inside real manifold в 2/4 encoders;
- positive class margin в 3/4 encoders;
- novelty interval в 2/4 encoders;
- frequency robust max-z `<= 5`;
- source-frequency distance не выше class p75.

Если Tier A содержит 30 уникальных sources, используется только Tier A.
Иначе Tier B может заполнить недостающие позиции. Не более одного synthetic
variant на source.

### Facility-location objective

Для каждого класса выбираются 30 кандидатов. Similarity к каждому real train
примеру усредняется по четырём encoder spaces после масштабирования на median
real local radius. Реальные точки, не покрытые прежним `strict_id`, получают
больший вес.

Жадный шаг выбирает кандидата с максимальным marginal facility gain при
ограничении source uniqueness. При равном gain используется predeclared
quality score, а затем `image_id`.

## Gate Stage 13A -> Stage 13B

Новый subset сравнивается с прежним `strict_id` на общей real reference,
из которой исключён union source images обоих наборов.

Gate открывается только если:

1. выбрано 30 изображений каждого класса и 90 уникальных sources;
2. PRDC coverage выше минимум в 8/12 `encoder × class` сравнений;
3. mean coverage delta положительна;
4. mean precision delta не ниже `-0.05`;
5. mean density delta не ниже `-0.05`;
6. frequency-source distance ниже `strict_id` минимум для 2/3 классов;
7. Tier B составляет не более 50% subset;
8. locked test не использован.

Если gate закрыт, новые classifier runs не запускаются.

## Stage 13B

При открытом gate выполняется единственное paired сравнение:

- 90 coverage-targeted synthetic;
- 90 source-matched real replay;
- ConvNeXt-S qualified recipe;
- seeds 42, 43, 44;
- synthetic/replay weight 0.5;
- natural sampler;
- validation only, `evaluation.run_test=false`.

Existing `strict_id` не переобучается: Stage 11B остаётся историческим
контролем. Primary и minority endpoints совпадают со Stage 11B.

## Фактический Stage 13A на исходном pool

Дата: 2026-07-29.

Все четыре encoder caches и frequency features успешно рассчитаны. Первый
запуск остановился на попытке выбрать 30 строк; pipeline исправлен так, чтобы
candidate shortage создавал structured fail-closed manifest, а не traceback.

Gate закрыт:

| Class | Tier A rows | Tier B rows | A+B rows | Unique sources | Required |
|---|---:|---:|---:|---:|---:|
| mel | 8 | 21 | 29 | 26 | 30 |
| akiec | 60 | 67 | Tier A достаточно | 59 Tier A | 30 |
| bkl | 2 | 9 | 11 | 11 | 30 |

Training CSV не создан, Stage 13B не запущен.

Основные bottlenecks:

- `bkl`: Tier A novelty проходит 1.7%, positive margin 20.2%;
- `mel`: Tier A novelty проходит 4.2%, positive margin 34.6%;
- frequency max-z проходит 83.5–90.8%, поэтому frequency gate не является
  главным источником shortage;
- почти все допустимые строки имеют strength 0.15;
- strength 0.45 не даёт ни одного A/B кандидата для `mel` и `bkl`;
- task-space local novelty ratio монотонно растёт со strength.

Это подтверждает, что исходный минимум strength 0.15 уже слишком сильно
сдвигает `mel/bkl` в real-only ConvNeXt-S spaces.

## Stage 13A2: low-strength regeneration

Проверяется только один новый фактор:

- strengths: 0.05 и 0.10 вместо 0.15, 0.30, 0.45;
- те же Stable Diffusion 1.5, prompts и negative prompts;
- guidance scale 6.0;
- 30 inference steps;
- center crop 512;
- те же target classes;
- 160 deterministic real sources на класс;
- один вариант на source и strength;
- всего 960 изображений.

После генерации применяется тот же multi-encoder/frequency gate без изменения
порогов. Новые изображения не смешиваются с исходным pool при facility
selection; старый `strict_id` добавляется только как evaluation control.

Реализация генерации поддерживает идемпотентное продолжение после остановки:
готовые файлы с валидными строками manifest не генерируются повторно. Resume
разрешён только при полном совпадении resolved config. Порядок классов и
соответствие seed исходному изображению явно детерминированы и не зависят от
`PYTHONHASHSEED`.

## Фактический Stage 13A2

Дата: 2026-07-29.

Полный pool из 960 изображений и все четыре encoder caches рассчитаны без
ошибок. Gate снова закрыт до обучения:

| Class | Tier A | Tier B | A+B unique sources | Required |
|---|---:|---:|---:|---:|
| mel | 28 | 29 | 47 | 30 |
| akiec | 115 | 83 | 100 Tier A | 30 |
| bkl | 4 | 23 | 22 | 30 |

По strength:

- `mel 0.05`: 43 допустимых источника; `0.10`: 14;
- `akiec 0.05`: 130; `0.10`: 68;
- `bkl 0.05`: 20; `0.10`: 7.

Frequency max-z проходит 142–152 из 160 строк на class/strength. Для `bkl
0.05` novelty Tier B проходит 40/160, positive margin 71/160, но полное
пересечение критериев даёт только 20 источников. Следовательно, снижение
strength улучшило соответствие пространству, однако исходного бюджета
источников недостаточно для `bkl`.

## Stage 13A3: независимое расширение bkl source budget

Меняется один фактор: число независимых исходных `bkl` lesions при уже
лучшем strength 0.05.

- генерируются 160 дополнительных `bkl` изображений;
- source IDs из Stage 13A2 явно исключаются;
- один вариант на source;
- Stable Diffusion, prompt, negative prompt, guidance scale, steps, crop и
  seed фиксируются в resolved config;
- объединённый candidate pool содержит 1120 строк: исходные 960 плюс 160
  дополнительных `bkl`;
- пороги Tier A/B, one-source constraint, facility-location objective и
  gate не изменяются;
- при shortage либо geometry gate failure обучение снова не запускается.

Такой дизайн проверяет достаточность candidate/source coverage, а не
смягчает определение пригодного изображения после просмотра результата.
DiffuLT мотивирует фильтрацию approximately-in-distribution синтетики, а
Diffusion Curriculum показывает необходимость контролировать спектр
image-guidance и distribution gap. В этой стадии используется только
эмпирически пригодный край спектра 0.05; curriculum training пока не
проверяется.

## Фактический Stage 13A3 и решение о Stage 13B

Дата: 2026-07-29.

Дополнительные 160 `bkl` sources увеличили допустимый набор до 39 строк из
35 уникальных источников. Итоговый subset:

- `mel`: 18 Tier A + 12 Tier B;
- `akiec`: 30 Tier A;
- `bkl`: 8 Tier A + 22 Tier B;
- всего 90 строк, 90 уникальных source lesions;
- Tier B fraction 0.3778.

Geometry gate открыт:

- coverage wins: 12/12;
- mean coverage delta против `strict_id`: +0.1547;
- mean precision delta: +0.3083;
- mean density delta: +0.4317;
- frequency wins: 3/3;
- locked test не использован.

Stage 13B запускается как paired confirmatory comparison:

- `stage13b_synthetic_coverage_convnext_small_384`;
- `stage13b_replay_coverage_convnext_small_384`;
- seeds 42, 43, 44;
- 80 epochs с одинаковым qualified ConvNeXt-S recipe;
- early stopping только по validation;
- `evaluation.run_test=false`.

Новая ссылка:

- Liang Y., Bhardwaj S., Zhou T. *Diffusion Curriculum:
  Synthetic-to-Real Data Curriculum via Image-Guided Diffusion*. ICCV,
  2025.
  https://openaccess.thecvf.com/content/ICCV2025/html/Liang_Diffusion_Curriculum_Synthetic-to-Real_Data_Curriculum_via_Image-Guided_Diffusion_ICCV_2025_paper.html
