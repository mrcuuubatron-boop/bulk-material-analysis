# Анализ сыпучих материалов по фотографиям

В проекте реализован воспроизводимый конвейер:

1. загрузка фотографии;
2. сегментация методом Отсу;
3. морфологическая очистка и выбор основной связной компоненты;
4. расчёт площади, bbox, центра и текстурных признаков;
5. медиана/MAD-фильтрация ряда наблюдений;
6. обучение nearest-centroid по признакам;
7. экспорт CSV, масок, наложений и графиков.

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Формат обучающих фотографий

```text
example_data/
  sand/
    sand_001.jpg
  cement/
    cement_001.jpg
  gravel/
    gravel_001.jpg
```

## Быстрый пример

```bash
python scripts/generate_samples.py

python sv_filter.py --pipeline train \
  --image-input-dir example_data \
  --experience-model component_experience.json \
  --output-dir output

python sv_filter.py --pipeline infer \
  --image-input-dir example_data/sand \
  --experience-model component_experience.json \
  --output-dir output/infer \
  --prediction-csv-output predictions.csv
```

Результаты:

- `output/infer/measurements_raw.csv` — исходные измерения;
- `output/infer/measurements_filtered.csv` — стабильные измерения;
- `output/infer/predictions.csv` — предсказанный материал и уверенность;
- `output/infer/masks/` — бинарные маски;
- `output/infer/overlays/` — исходное фото с выделенной областью;
- `output/infer/feature_trends.png` — графики до и после фильтрации.

## GUI и статистика

Установите зависимости и запустите локальное приложение:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Для удобного запуска в Linux используйте готовый скрипт:

```bash
./run_gui.sh
```

Скрипт самостоятельно создаёт `.venv`, устанавливает зависимости и запускает
GUI. Дополнительные параметры Streamlit можно передать после имени скрипта:

```bash
./run_gui.sh --server.port 8502
```

GUI содержит вкладки для анализа фотографий, обучения модели, истории запусков
и работы с датасетом. История сохраняется в `output/statistics.sqlite3` и включает
скорость обработки, количество кадров, валидные измерения, материалы и уверенность
классификации.

### Оценка объёма кучи

Во вкладке анализа на первом изображении проведите линию по известному эталону:

- метровой рейке или полоске длиной 1 м;
- короткой стороне листа A4 — 0.210 м;
- длинной стороне листа A4 — 0.297 м.

После калибровки программа переводит площадь маски в м² и оценивает объём
в м³. Для скрытой глубины кучи доступны два режима: приблизительно использовать
ширину видимого основания или ввести глубину вручную. Геометрическая модель
может быть эллипсоидальной, пирамидальной или прямоугольной.

Оценка по одной фотографии является приближённой: камера не видит скрытую часть
кучи. Для дипломного эксперимента рекомендуется фотографировать кучу сбоку,
размещать эталон в той же плоскости и при возможности вводить глубину вручную.
Масса рассчитывается как `масса = объём × плотность определённого материала`.

## Датасет

Для дипломной работы подходит открытый [Concrete Aggregate PSD Imaging Dataset](https://data.uni-hannover.de/dataset/concrete-aggregate-psd-imaging-dataset):
природный песок, речной гравий и переработанный бетонный заполнитель, лицензия
CC BY 4.0. Архивы занимают около 474 и 494 МБ.

Скачать часть датасета можно без ручного копирования URL:

```bash
python scripts/download_dataset.py --part coarse
python scripts/download_dataset.py --part fine
```

После скачивания разложите архивы в формат, который понимает обучение:

```bash
python scripts/prepare_dataset.py --part all
```

Будут созданы классы `river_gravel`, `recycled_concrete` и `river_sand`,
а также файлы `metadata_coarse.csv` и `metadata_fine.csv` с исходными PSD
значениями. После подготовки можно указать `datasets/training` во вкладке
«Обучение» GUI или выполнить:

```bash
python sv_filter.py --pipeline train \
  --image-input-dir datasets/training \
  --experience-model component_experience.json \
  --output-dir output/aggregate-training
```
