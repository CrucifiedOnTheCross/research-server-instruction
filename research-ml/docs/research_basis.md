# Research Basis For The Experiment Code

Код спроектирован под первый этап исследования: проверить, какие свойства синтетики делают ее полезной при дисбалансе классов.

## Опорные выводы

- Synthetic augmentation может помогать немонотонно: важны mismatch генератора, synthetic ratio и real-only validation/test.
- Diffusion augmentation нужно разбирать на этапы: настройка/условие генератора, генерация, использование синтетики в обучении классификатора.
- Для long-tail важно сравнивать генеративные подходы с простыми baseline: re-sampling, class weights, focal loss, Balanced Softmax, logit adjustment.
- Визуальное качество не является достаточной метрикой: полезны real-vs-synthetic detectability, embedding overlap, class confusion, per-class recall, calibration.
- Если синтетика образует отдельные tight clusters или ближе к confusing class, она может вредить даже при хорошей визуальной правдоподобности.

## Что обязательно сохраняет запуск

- `config.resolved.yaml`
- `environment.json`
- `class_to_idx.json`
- `class_counts.json`
- `metrics.csv`
- `metrics.jsonl`
- `run.log`
- `best.pt`
- `last.pt`
- `val_metrics_best.json`
- `test_metrics.json`
- `val_predictions_best.csv`
- `test_predictions.csv`
- `embedding_diagnostics_<split>.json` после запуска `src.analyze_embeddings`

## Минимальная матрица первого этапа

1. Real-only baseline.
2. Real-only + weighted sampler.
3. Real-only + focal loss.
4. Real-only + Balanced Softmax.
5. Synthetic ratio 0.25 / 0.5 / 1.0 / balanced.
6. Synthetic naive vs filtered by feature diagnostics.
7. Финальная оценка только на real-only validation/test.

## Synthetic Ratio Protocol

Для каждой доли синтетики фиксировать:

- исходный train pool;
- seed отбора;
- число real/synthetic объектов по классам;
- генератор и его параметры;
- способ фильтрации;
- downstream loss/sampler;
- real-only validation/test результат.

Не смешивать в одной таблице эффекты генерации, sampler и loss: сначала прогнать простые baseline, затем добавлять синтетику поверх лучшего и поверх нейтрального baseline.
