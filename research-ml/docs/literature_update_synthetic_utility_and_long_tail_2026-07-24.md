# Обновление литературы: полезность синтетики и long-tail классификация

Дата проверки: 2026-07-24.

## Назначение

Этот файл фиксирует статьи, реально использованные при интерпретации Stage 8 и
проектировании следующего этапа. Для каждой работы разделены результат авторов,
ограничения переноса и конкретное следствие для нашего протокола. Те же работы
внесены в Google Sheets «Мониторинг статей — дисбаланс, управляемая генерация и
устойчивые признаки».

Основной вопрос проекта:

> Когда синтетические дерматоскопические изображения добавляют полезную
> информацию сверх изменения априоров классов, повторного предъявления реальных
> minority-примеров и настройки порога решения?

## Новые ключевые источники 2026 года

### Li et al. Diffusion-Based Data Augmentation for Image Recognition

- Источник: <https://arxiv.org/abs/2603.08364>
- Тип: систематический анализ generator fine-tuning, generation и utilization.
- Релевантный результат: на fine-grained и медицинских задачах визуально
  правдоподобная синтетика может не улучшать классификацию; слабая real-guidance
  иногда полезнее дообученного генератора.
- Ограничение: агрегирует несколько задач и не заменяет проверку на нашем
  lesion-aware split.
- Следствие: качество генератора нельзя оценивать только по картинкам, FID или
  близости к real. Нужны matched downstream controls и отдельная ось utilization.

### Ma, Lyu, Zhang. When Does Synthetic Data Augmentation Improve
Score-Based Imbalanced Classification?

- Источник: <https://arxiv.org/abs/2606.26053>
- Тип: теория score-based классификации при дисбалансе.
- Релевантный результат: эффект синтетики раскладывается на изменение
  эффективного class prior и ошибку синтетического распределения. При хорошо
  специфицированной модели фундаментального выигрыша ранжирования может не быть;
  фиксированные threshold-метрики могут улучшиться из-за сдвига operating point.
- Ограничение: теория не специфична для изображений и не доказывает, что наша
  модель хорошо специфицирована.
- Следствие: сравнивать AUROC/AUPRC, calibrated threshold и class-specific logit
  offset; отдельно проверять изменение ranking и изменение decision threshold.

### Ma, Zhang. Synthetic Augmentation in Imbalanced Learning

- Источник: <https://arxiv.org/abs/2601.16120>
- Тип: теория и validation-tuned synthetic size.
- Релевантный результат: оптимальная доза синтетики может быть немонотонной;
  балансировка до head-класса не является универсально оптимальной.
- Следствие: Stage 9 должен подбирать synthetic dose только внутри training
  folds, сохраняя locked test закрытым.

### Dombrowski et al. The Learnability Gap in Medical Latent Diffusion

- Источник: <https://arxiv.org/abs/2605.17087>
- Тип: медицинская latent diffusion, включая ISIC2019.
- Релевантный результат: высокая реконструкционная или визуальная точность
  latent autoencoder не гарантирует сохранение дискриминативной информации;
  авторы наблюдают существенный learnability gap.
- Ограничение: исследует latent reconstruction/generation, а не наш точный
  img2img pipeline.
- Следствие: оценивать class separability, source-copying, precision/coverage и
  downstream utility; визуальный контроль необходим, но недостаточен.

### Jiang et al. Synthetic Data Generation for Long-Tail Medical Image
Classification: A Case Study in Skin Lesions

- Источник: <https://arxiv.org/abs/2605.03221>
- Тип: diffusion inpainting, маски, class conditioning и OOD selection.
- Релевантный результат: управляемое изменение области поражения и фильтрация
  могут улучшать long-tail skin lesion classification.
- Ограничение переноса: описанный proportional image-level cross-validation не
  дает нам достаточной гарантии lesion/patient isolation; численные результаты
  нельзя напрямую сравнивать с нашим group-aware split.
- Следствие: inpainting/mask-preserving генератор является кандидатом только
  после доказательства, что проблема Stage 8 находится в генераторе, а не в
  дозе, sampler или operating point.

## Сильные baseline и геометрия признаков

### MONICA: Benchmarking on Medical Image Classification with Long-Tailed Data

- Источник: <https://arxiv.org/abs/2410.02010>
- Релевантный результат: resampling остается сильным baseline; результат зависит
  от предобучения, sampler, checkpoint selection и разделения representation и
  classifier. Среди сильных методов рассматриваются GCL и MiSLAS.
- Следствие: синтетика обязана сравниваться на одном split/backbone/budget с
  random oversampling и сильным representation/classifier baseline.

### ECL: Class-Enhanced Contrastive Learning for Long-Tailed Skin Lesion
Classification

- Источник: <https://arxiv.org/abs/2307.04136>
- Релевантный результат: class-aware contrastive representation learning может
  повышать разделимость редких классов сильнее, чем только коррекция loss.
- Следствие: нужен real-only representation-level control. Иначе улучшение
  feature geometry нельзя приписать синтетическим изображениям.

### Scholz et al. Imbalance-aware loss functions improve medical image
classification

- Источник: <https://proceedings.mlr.press/v250/scholz24a.html>
- Релевантный результат: дифференцируемые MCC/F1 losses совместно с batch
  sampling могут улучшать minority-метрики медицинских классификаторов.
- Следствие: добавить один воспроизводимый loss-level control, если primary
  endpoints статьи включают MCC и macro F1.

### Kang et al. Decoupling Representation and Classifier for Long-Tailed
Recognition

- Источник: <https://arxiv.org/abs/1910.09217>
- Релевантный результат: representation, обученное на естественном
  распределении, и отдельно перебалансированный classifier могут быть сильнее
  end-to-end resampling.
- Следствие: добавить cRT-подобную ветку с frozen encoder и balanced classifier.

### Menon et al. Long-tail learning via logit adjustment

- Источник: <https://arxiv.org/abs/2007.07314>
- Релевантный результат: class-prior correction может менять decision boundary
  без добавления изображений.
- Следствие: Logit Adjustment запускать с natural sampler и отдельно проверять
  post-hoc offsets. Комбинация weighted sampler + LA смешивает два механизма.

## Контроли resampling и калибровки

### Yang et al. The impact of random oversampling and random undersampling

- Источник: <https://doi.org/10.1186/s40537-023-00857-7>
- Релевантный результат: в большом исследовании resampling не давал
  универсального выигрыша AUROC, а часть ухудшения calibration исправлялась
  recalibration.
- Ограничение: исследование не image-specific.
- Следствие: random oversampling и random undersampling должны быть явными
  контролями; ECE оценивать вместе с temperature scaling и threshold analysis.

### DiffuLT: Diffusion for Long-tail Recognition Without External Knowledge

- Источник:
  <https://papers.nips.cc/paper_files/paper/2024/hash/de7858e3e7f9f0f7b2c7bfdc86f6d928-Abstract-Conference.html>
- Релевантный результат: полезными могут быть approximately-in-distribution
  примеры, а не только ближайшие к real samples.
- Ограничение: feature distance не гарантирует медицинскую корректность.
- Следствие: сравнить равные по размеру зоны `strict-ID`, `AID band` и `OOD`;
  текущий nearest top-k нельзя считать теоретически оптимальным.

## Синтез литературы для нашей статьи

Литература не поддерживает универсальный тезис «чем реалистичнее синтетика, тем
выше качество классификатора». Более точная причинная схема:

1. Генератор определяет семантическую корректность и synthetic-to-real gap.
2. Фильтр определяет положение синтетики относительно class manifold.
3. Доза и sampler изменяют эффективные априоры и частоту предъявления классов.
4. Loss/head/threshold меняют decision boundary.
5. Downstream результат возникает из взаимодействия всех четырех механизмов.

Поэтому научная новизна должна заключаться не в еще одном наборе сгенерированных
изображений, а в контролируемом разложении utility:

`utility = prior/reweighting effect + representation effect - distribution mismatch`.

Это рабочая концептуальная модель, а не доказанное равенство. Stage 9 должен
операционализировать каждый член отдельным matched control.

## Вопросы, которые остаются открытыми

1. Дает ли синтетика прирост ranking (`AUROC`, `AUPRC`) или только меняет порог?
2. Повторяет ли source-matched real replay эффект тех же 240 synthetic samples?
3. Существует ли AID-зона с большей utility, чем strict nearest-neighbor filter?
4. Связан ли per-sample utility с DINO distance, class margin, source similarity
   и артефактными признаками?
5. Сохраняется ли эффект при group-aware cross-validation?
6. Можно ли воспроизвести результат на внешних изображениях без HAM10000 overlap?
7. Улучшает ли inpainting/LoRA генератор семантическую корректность настолько,
   чтобы превзойти сильный real-only long-tail baseline?

## Правило дальнейшего использования источников

В текст статьи включается только утверждение, подтвержденное первичным источником
и нашим соответствующим экспериментом. Численные результаты чужих работ не
переносятся в таблицу сравнения как напрямую сопоставимые, если различаются
split, backbone, resolution, preprocessing или leakage control.
