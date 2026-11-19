"""Local Streamlit interface for the bulk-material analysis pipeline."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import time

from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from material_analysis.database import StatisticsDatabase
from material_analysis.models import ExperienceModel
from material_analysis.pipeline import analyze_directory
from material_analysis.training import train_model
from material_analysis.volume import Calibration


ROOT = Path(__file__).resolve().parent
DEFAULT_DB = ROOT / "output" / "statistics.sqlite3"
DATASET_URL = (
    "https://data.uni-hannover.de/dataset/"
    "concrete-aggregate-psd-imaging-dataset"
)


def save_uploaded_files(files, target: Path) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    for uploaded in files:
        destination = target / Path(uploaded.name).name
        destination.write_bytes(uploaded.getvalue())
    return target


def read_predictions(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def render_volume_calibration(image_path: Path | None) -> Calibration | None:
    st.subheader("Калибровка масштаба и объёма")
    st.caption(
        "На первом фото проведите линию по известному отрезку. "
        "Например, по метровой рейке или по стороне листа A4."
    )
    reference_type = st.radio(
        "Эталон на фотографии",
        ["Метровая полоска", "Лист A4"],
        horizontal=True,
        help="Линия должна совпадать с отрезком известной длины на снимке.",
    )
    if reference_type == "Метровая полоска":
        reference_metres = 1.0
        st.info("Проведите линию ровно по метровой полоске.")
    else:
        a4_side = st.selectbox(
            "Какую сторону листа A4 измеряете?",
            ["Короткую — 0.210 м", "Длинную — 0.297 м"],
            help="Проведите линию по выбранной стороне листа.",
        )
        reference_metres = 0.210 if a4_side.startswith("Короткую") else 0.297

    reference_pixels = 0.0
    if image_path is not None and image_path.exists():
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            display_width = min(image.width, 900)
            display_height = round(image.height * display_width / image.width)
            background = image.resize((display_width, display_height))
        canvas = st_canvas(
            fill_color="rgba(255, 0, 0, 0.1)",
            stroke_width=3,
            stroke_color="#ff0000",
            background_image=background,
            update_streamlit=True,
            height=display_height,
            width=display_width,
            drawing_mode="line",
            key="calibration_line",
        )
        objects = (canvas.json_data or {}).get("objects", [])
        if objects:
            line = objects[-1]
            displayed_pixels = (
                (float(line.get("x2", 0)) - float(line.get("x1", 0))) ** 2
                + (float(line.get("y2", 0)) - float(line.get("y1", 0))) ** 2
            ) ** 0.5
            reference_pixels = displayed_pixels * image.width / display_width
            st.success(f"Эталонная линия: {reference_pixels:.1f} пикселей")
        else:
            st.warning("Проведите красную линию по эталонному отрезку.")
    else:
        reference_pixels = st.number_input(
            "Длина эталона в пикселях",
            min_value=1.0,
            value=100.0,
            help="Ручной режим, если изображение ещё не выбрано.",
        )

    depth_mode = st.radio(
        "Как оценивать глубину кучи?",
        ["По ширине видимого основания", "Задать глубину вручную"],
        help=(
            "Одна фотография не видит скрытую глубину. По умолчанию глубина "
            "приближается шириной кучи; для более точного результата задайте её вручную."
        ),
    )
    manual_depth = 0.0
    if depth_mode == "Задать глубину вручную":
        manual_depth = st.number_input(
            "Оценочная глубина кучи, м",
            min_value=0.01,
            value=1.0,
            help="Введите расстояние от переднего до заднего края кучи.",
        )
    shape = st.selectbox(
        "Геометрическая модель кучи",
        ["Эллипсоид — рекомендуемый вариант", "Пирамида", "Прямоугольный объём"],
        help=(
            "Модель учитывает, что фотография показывает только проекцию кучи. "
            "Эллипсоид обычно даёт наиболее осторожную оценку."
        ),
    )
    shape_name = {
        "Эллипсоид — рекомендуемый вариант": "ellipsoid",
        "Пирамида": "pyramid",
        "Прямоугольный объём": "box",
    }[shape]
    if reference_pixels <= 0:
        return None
    return Calibration.from_reference(
        reference_pixels,
        reference_metres,
        depth_mode="manual" if depth_mode == "Задать глубину вручную" else "width",
        manual_depth_m=manual_depth,
        shape=shape_name,
    )


def render_parameter_explanations() -> None:
    with st.sidebar.popover("❓ Что означают параметры?"):
        st.markdown("### Пояснения")
        st.markdown(
            """
            **Метод Отсу** автоматически ищет границу между фоном и материалом
            по яркости пикселей. Он превращает фотографию в чёрно-белую маску.

            **Смещение порога Отсу** вручную изменяет найденную границу:
            положительное значение выделяет больше светлых областей, отрицательное
            — делает выделение строже. Начинайте со значения `0`.

            **Морфологическая очистка** убирает мелкий шум маски и закрывает
            небольшие разрывы внутри объекта. Больший размер сильнее сглаживает
            маску, но может удалить мелкие детали. Обычно подходит `3`.

            **MAD (Median Absolute Deviation)** — медианное абсолютное отклонение.
            Это устойчивый показатель разброса чисел. Он помогает найти случайный
            выброс, например один неудачный кадр, не искажая результат средним
            значением.

            **Коэффициент MAD** определяет строгость удаления выбросов:
            меньше — фильтр строже, больше — мягче. Рекомендуется `3.5`.
            """
        )


def render_sidebar() -> tuple[Path, Path, bool, int, int, float]:
    st.sidebar.header("Параметры конвейера")
    render_parameter_explanations()
    output_dir = Path(
        st.sidebar.text_input(
            "Папка для результатов",
            "output/gui",
            help="Сюда будут сохранены CSV-файлы, маски, изображения и графики.",
        )
    )
    model_path = Path(
        st.sidebar.text_input(
            "Файл обученной модели",
            "component_experience.json",
            help="JSON-файл с эталонными признаками материалов.",
        )
    )
    dark_object = st.sidebar.checkbox(
        "Материал темнее фона",
        value=True,
        help="Включите, если материал на фото темнее окружающего фона.",
    )
    morphology_size = st.sidebar.slider(
        "Сила очистки маски",
        1,
        7,
        3,
        step=2,
        help=(
            "Морфологическая очистка удаляет мелкий шум и закрывает разрывы "
            "в выделенной области. Значение 3 — стандартный вариант."
        ),
    )
    threshold_offset = st.sidebar.slider(
        "Ручная поправка порога Отсу",
        -40,
        40,
        0,
        help=(
            "Изменяет автоматически найденную границу яркости. 0 — полностью "
            "автоматический режим; положительные значения выделяют больше областей."
        ),
    )
    mad_k = st.sidebar.slider(
        "Строгость удаления выбросов (MAD)",
        1.0,
        8.0,
        3.5,
        step=0.5,
        help=(
            "MAD — медианное абсолютное отклонение. Меньшее значение удаляет "
            "больше подозрительных кадров, большее — меньше."
        ),
    )
    return output_dir, model_path, dark_object, morphology_size, threshold_offset, mad_k


def main() -> None:
    st.set_page_config(page_title="Bulk Material Lab", page_icon="🪨", layout="wide")
    st.title("🪨 Анализ сыпучих материалов")
    st.caption("Выделение материала → измерения → очистка данных → классификация → аналитика")
    output_dir, model_path, dark_object, morphology_size, threshold_offset, mad_k = render_sidebar()
    database = StatisticsDatabase(DEFAULT_DB)

    tab_analysis, tab_training, tab_stats, tab_dataset = st.tabs(
        ["Анализ фото", "Обучение", "Статистика", "База данных"]
    )

    with tab_analysis:
        st.subheader("Анализ серии фотографий")
        uploaded = st.file_uploader(
            "Загрузите фотографии одной серии",
            type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"],
            accept_multiple_files=True,
        )
        input_dir_text = st.text_input(
            "Или укажите существующую папку с фотографиями",
            "example_data/sand",
            help="В папке должны находиться изображения JPG, PNG или другого поддерживаемого формата.",
        )
        preview_path = None
        if uploaded:
            preview_dir = Path(tempfile.mkdtemp(prefix="bulk-preview-"))
            preview_path = preview_dir / Path(uploaded[0].name).name
            preview_path.write_bytes(uploaded[0].getvalue())
        elif Path(input_dir_text).is_dir():
            preview_path = next(
                (path for path in sorted(Path(input_dir_text).iterdir())
                 if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}),
                None,
            )
        calibration = render_volume_calibration(preview_path)
        if st.button("Запустить анализ", type="primary", disabled=not uploaded and not Path(input_dir_text).is_dir()):
            work_dir = Path(tempfile.mkdtemp(prefix="bulk-analysis-"))
            input_dir = save_uploaded_files(uploaded, work_dir) if uploaded else Path(input_dir_text)
            try:
                model = ExperienceModel.load(model_path) if model_path.exists() else None
                started = time.perf_counter()
                measurements, filtered, predictions_path = analyze_directory(
                    input_dir,
                    output_dir,
                    experience_model=model,
                    morphology_size=morphology_size,
                    threshold_offset=threshold_offset,
                    dark_object=dark_object,
                    mad_k=mad_k,
                    calibration=calibration,
                )
                duration = time.perf_counter() - started
                predictions = read_predictions(predictions_path)
                run_id = database.record_run(
                    "infer",
                    input_dir,
                    output_dir,
                    measurements,
                    duration,
                    predictions,
                )
                st.success(f"Анализ завершён: запуск #{run_id}, {duration:.2f} с")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Изображений", len(measurements))
                col2.metric("Валидных", sum(row.valid for row in measurements))
                col3.metric("После фильтра", len(filtered))
                col4.metric("мс/изображение", f"{duration * 1000 / max(len(measurements), 1):.1f}")
                if predictions:
                    st.dataframe(predictions, use_container_width=True)
                plot_path = output_dir / "feature_trends.png"
                if plot_path.exists():
                    st.image(str(plot_path), caption="Признаки до и после MAD-фильтрации")
            except (OSError, ValueError, json.JSONDecodeError) as error:
                st.error(f"Ошибка анализа: {error}")

    with tab_training:
        st.subheader("Обучение классификатора материалов")
        st.info(
            "Подготовьте папку с подпапками классов, например: "
            "sand/, cement/, gravel/. Для каждого класса желательно несколько фотографий."
        )
        training_dir = st.text_input(
            "Папка с обучающими фотографиями",
            "example_data",
            help="Каждая подпапка считается отдельным классом материала.",
        )
        density_text = st.text_input(
            "Плотности материалов, кг/м³",
            "sand=1600,cement=1400,gravel=1800",
            help="Нужно для оценки массы: масса = объём × плотность.",
        )
        if st.button("Обучить модель"):
            source = Path(training_dir)
            if not source.is_dir():
                st.error(f"Каталог не найден: {source}")
            else:
                try:
                    densities = {}
                    for item in density_text.split(","):
                        if item.strip():
                            name, value = item.split("=", 1)
                            densities[name.strip()] = float(value)
                    started = time.perf_counter()
                    model = train_model(source, model_path)
                    model.density_kg_m3 = densities
                    model.save(model_path)
                    duration = time.perf_counter() - started
                    st.success(f"Модель сохранена: {model_path} ({duration:.2f} с)")
                    st.json({"materials": list(model.materials), "density_kg_m3": densities})
                except (OSError, ValueError) as error:
                    st.error(f"Ошибка обучения: {error}")

    with tab_stats:
        st.subheader("История запусков и аналитика")
        summary = database.summary()
        columns = st.columns(5)
        columns[0].metric("Запусков", summary["runs"])
        columns[1].metric("Изображений", summary["images"])
        columns[2].metric("Валидных", summary["valid_images"])
        columns[3].metric("Среднее время запуска", f"{summary['avg_duration']:.2f} с")
        columns[4].metric("Средняя скорость", f"{summary['avg_image_ms']:.1f} мс/фото")
        st.subheader("По материалам")
        material_rows = database.material_statistics()
        if material_rows:
            st.dataframe(material_rows, use_container_width=True)
        else:
            st.info("Статистика появится после запуска инференса с моделью.")
        st.subheader("Последние запуски")
        st.dataframe(database.runs(), use_container_width=True)

    with tab_dataset:
        st.subheader("Рекомендуемый открытый датасет")
        st.markdown(
            "**Concrete Aggregate PSD Imaging Dataset** содержит изображения "
            "природного песка, речного гравия и переработанного бетонного заполнителя."
        )
        st.markdown(f"[Открыть страницу датасета]({DATASET_URL})")
        st.write("Лицензия: Creative Commons Attribution 4.0 International (CC BY 4.0).")
        st.write("Доступны архивы M_fine.zip и M_coarse.zip, каждый около 0.5 ГБ.")
        st.warning(
            "Это большой датасет. Скачайте архивы вручную, распакуйте их и "
            "сформируйте подпапки классов перед обучением."
        )
        st.code(
            "mkdir -p datasets/aggregate\n"
            "curl -L 'https://data.uni-hannover.de/dataset/"
            "6f844e22-12ed-48a7-9ccb-b502f8121650/resource/"
            "ae8f74b2-dcbd-46f8-81f6-ec6aab979768/download/m_coarse.zip' "
            "-o datasets/aggregate/m_coarse.zip",
            language="bash",
        )


if __name__ == "__main__":
    main()
