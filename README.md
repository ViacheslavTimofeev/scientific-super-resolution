# sr_pet_project

## Что это за проект

В этом проекте реализованы репродуцируемый бенчмарк и инференс пайплайн для задачи super-resolution на микро-КТ снимках пород с использованием CNN и трансформеров. LR (low-resolution) изображения создаются из HR (high-resolution) и затем масштабируются к желаемому размеру. Конечным результатом является SR (super-resolution) размеров x2, x4 или x8 от LR.

## Цель проекта

- Сравнить bicubic, CNN, SwinIR
- Получить воспроизводимую оценку качества
- Экспорт и хранение модели в ONNX
- Сравнить latency PyTorch vs ONNX runtime
- Завернуть inference в API

## Что входит в MVP

- 1 датасет
- 3 варианта inference: bicubic, CNN, SwinIR
- Метрики: PSNR, SSIM
- Сохранение изображений-примеров
- Экспортированная лучшая модель в ONNX
- latency benchmark
- FastAPI endpoint

## Не входит в MVP

- arbitrary-scale SR
- сложный трекинг экспериментов
- docker orchestration
- distributed training
- полноценный UI

## Промежуточные результаты

- [ ] train/eval запускаются из CLI
- [ ] результаты пишутся в outputs
- [ ] evaluation сохраняет таблицу метрик и изображения
- [ ] ONNX output численно близок к PyTorch
- [ ] benchmark выдает latency summary
- [ ] API принимает изображение и возвращает SR

## Главный pipeline

1. конфиг описывает запуск
2. датасет и transforms готовят LR/HR пары
3. модель получает LR и выдает SR
4. train-loop обучает
5. eval-loop считает метрики и сохраняет визуализации
6. лучшая модель экспортируется в ONNX
7. benchmark сравнивает PyTorch и ONNX
8. API использует один из inference backends
