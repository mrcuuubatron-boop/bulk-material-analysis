from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .models import Measurement, ExperienceModel
from .features import extract_measurement, kmeans_color, feature_vector
from .filtering import robust_filter
from .volume import Calibration, estimate_volume


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def analyze_image(
    image_path: str | Path,
    output_dir: str | Path,
    morphology_size: int = 3,
    threshold_offset: int = 0,
    dark_object: bool = True,
    class_hint: str = "",
):
    return extract_measurement(
        image_path,
        morphology_size=morphology_size,
        threshold_offset=threshold_offset,
        dark_object=dark_object,
        class_hint=class_hint,
    )


def save_overlay(rgb: np.ndarray, mask: np.ndarray, path: Path) -> None:
    image = Image.fromarray(rgb)
    overlay = Image.new("RGBA", image.size, (255, 0, 0, 0))
    overlay_pixels = overlay.load()
    ys, xs = np.nonzero(mask)
    for x, y in zip(xs, ys):
        overlay_pixels[int(x), int(y)] = (255, 0, 0, 90)
    combined = Image.alpha_composite(image.convert("RGBA"), overlay)
    combined.save(path)


def save_mask(mask: np.ndarray, path: Path) -> None:
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def write_measurements_csv(path: Path, measurements: list[Measurement]) -> None:
    if not measurements:
        return
    fieldnames = list(measurements[0].to_dict().keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for measurement in measurements:
            writer.writerow(measurement.to_dict())


def save_feature_plot(raw: list[Measurement], filtered: list[Measurement], output_dir: Path) -> None:
    fields = [
        ("area_px", "Площадь (px)"),
        ("width_px", "Ширина (px)"),
        ("height_px", "Высота (px)"),
        ("cx", "Центр X (px)"),
        ("cy", "Центр Y (px)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()
    for axis, (field, title) in zip(axes, fields):
        raw_values = [getattr(row, field) for row in raw]
        filtered_values = [getattr(row, field) for row in filtered]
        positions = list(range(len(raw_values)))
        axis.plot(positions, raw_values, "o-", color="#d62728", label="До фильтрации", alpha=0.75)
        if filtered_values:
            axis.plot(positions[:len(filtered_values)], filtered_values, "o-", color="#1f77b4", label="После фильтрации")
        axis.set_title(title)
        axis.set_xlabel("Номер изображения")
        axis.set_ylabel("Значение")
        axis.grid(alpha=0.25)
    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_trends.png", dpi=150)
    plt.close(fig)


def analyze_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    experience_model: ExperienceModel | None = None,
    morphology_size: int = 3,
    threshold_offset: int = 0,
    dark_object: bool = True,
    mad_k: float = 3.5,
    calibration: Calibration | None = None,
) -> tuple[list[Measurement], list[Measurement], Path]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = output_dir / "masks"
    overlays_dir = output_dir / "overlays"
    masks_dir.mkdir(exist_ok=True)
    overlays_dir.mkdir(exist_ok=True)

    image_paths = sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    measurements: list[Measurement] = []

    for index, image_path in enumerate(image_paths):
        measurement, rgb, _gray, mask = extract_measurement(
            image_path,
            morphology_size=morphology_size,
            threshold_offset=threshold_offset,
            dark_object=dark_object,
        )
        measurements.append(measurement)
        stem = f"{index + 1:03d}_{image_path.stem}"
        save_mask(mask, masks_dir / f"{stem}_mask.png")
        save_overlay(rgb, mask, overlays_dir / f"{stem}_overlay.png")

    write_measurements_csv(output_dir / "measurements_raw.csv", measurements)
    filtered = robust_filter(measurements, mad_k=mad_k)
    write_measurements_csv(output_dir / "measurements_filtered.csv", filtered)
    save_feature_plot(measurements, filtered, output_dir)

    predictions_path = output_dir / "predictions.csv"
    if experience_model is not None:
        predictions = predict_measurements(
            measurements,
            experience_model,
            image_dir=input_dir,
            morphology_size=morphology_size,
            threshold_offset=threshold_offset,
            dark_object=dark_object,
            calibration=calibration,
        )
        write_predictions_csv(predictions_path, predictions)
    return measurements, filtered, predictions_path


def predict_measurements(
    measurements: list[Measurement],
    model: ExperienceModel,
    image_dir: str | Path,
    morphology_size: int = 3,
    threshold_offset: int = 0,
    dark_object: bool = True,
    calibration: Calibration | None = None,
) -> list[dict]:
    image_dir = Path(image_dir)
    predictions = []
    for measurement in measurements:
        image_path = image_dir / measurement.filename
        _measurement, rgb, _gray, mask = extract_measurement(
            image_path,
            morphology_size=morphology_size,
            threshold_offset=threshold_offset,
            dark_object=dark_object,
        )
        components = kmeans_color(rgb, mask, k=3)
        vector = np.asarray(feature_vector(measurement, components), dtype=np.float64)

        best_name = ""
        best_distance = float("inf")
        for name, record in model.materials.items():
            centroid = np.asarray(record["centroid"], dtype=np.float64)
            scale = np.asarray(record["scale"], dtype=np.float64)
            distance = float(np.linalg.norm((vector - centroid) / scale))
            if distance < best_distance:
                best_distance = distance
                best_name = name

        distance_limit = 1.0
        confidence = 1.0 / (1.0 + max(best_distance, 0.0) / distance_limit)
        if calibration is not None:
            volume = estimate_volume(measurement, calibration)
            area_m2 = volume["area_m2"]
            volume_m3 = volume["volume_m3"]
        else:
            area_m2 = measurement.area_px * model.pixel_scale_x * model.pixel_scale_y
            volume_m3 = area_m2 * model.height_estimate_m
        density = float(model.density_kg_m3.get(best_name, 0.0))
        mass_kg = volume_m3 * density
        predictions.append({
            "filename": measurement.filename,
            "predicted_material": best_name,
            "confidence": round(confidence, 6),
            "distance": round(best_distance, 6),
            "area_m2": round(area_m2, 6),
            "volume_m3": round(volume_m3, 6),
            "depth_m": round(volume["depth_m"], 6) if calibration is not None else "",
            "density_kg_m3": density,
            "mass_kg": round(mass_kg, 6),
        })
    return predictions


def write_predictions_csv(path: Path, predictions: list[dict]) -> None:
    if not predictions:
        return
    fieldnames = [
        "filename",
        "predicted_material",
        "confidence",
        "distance",
        "area_m2",
        "volume_m3",
        "depth_m",
        "density_kg_m3",
        "mass_kg",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)
