from pathlib import Path
import numpy as np

from .models import Measurement, Component
from .segmentation import segment_image


def extract_measurement(
    image_path: str | Path,
    morphology_size: int = 3,
    threshold_offset: int = 0,
    dark_object: bool = True,
    class_hint: str = "",
) -> tuple[Measurement, np.ndarray, np.ndarray, np.ndarray]:
    rgb, gray, mask = segment_image(
        image_path,
        morphology_size=morphology_size,
        threshold_offset=threshold_offset,
        dark_object=dark_object,
    )
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return (
            Measurement(
                filename=Path(image_path).name,
                area_px=0.0,
                width_px=0.0,
                height_px=0.0,
                cx=0.0,
                cy=0.0,
                mean_gray=0.0,
                std_gray=0.0,
                valid=False,
                class_hint=class_hint,
            ),
            rgb,
            gray,
            mask,
        )

    xmin, xmax = int(xs.min()), int(xs.max())
    ymin, ymax = int(ys.min()), int(ys.max())
    area = float(mask.sum())
    width = float(xmax - xmin + 1)
    height = float(ymax - ymin + 1)
    cx = float(np.mean(xs))
    cy = float(np.mean(ys))
    gray_values = gray[mask]
    mean_gray = float(np.mean(gray_values))
    std_gray = float(np.std(gray_values))

    h, w = gray.shape
    touches_border = xmin == 0 or ymin == 0 or xmax >= w - 1 or ymax >= h - 1
    valid = bool(area >= 32 and not touches_border)
    measurement = Measurement(
        filename=Path(image_path).name,
        area_px=area,
        width_px=width,
        height_px=height,
        cx=cx,
        cy=cy,
        mean_gray=mean_gray,
        std_gray=std_gray,
        valid=valid,
        class_hint=class_hint,
    )
    return measurement, rgb, gray, mask


def kmeans_color(rgb: np.ndarray, mask: np.ndarray, k: int = 3, max_iterations: int = 50) -> list[Component]:
    pixels = rgb[mask].astype(np.float64) / 255.0
    if len(pixels) == 0:
        return []
    k = min(k, len(pixels))
    rng = np.random.default_rng(42)
    centroids = pixels[rng.choice(len(pixels), size=k, replace=False)].copy()
    labels = np.zeros(len(pixels), dtype=np.int64)

    for _ in range(max_iterations):
        distances = np.linalg.norm(pixels[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1)
        updated = np.array([
            pixels[labels == cluster].mean(axis=0) if np.any(labels == cluster)
            else centroids[cluster]
            for cluster in range(k)
        ])
        if np.allclose(updated, centroids):
            break
        centroids = updated

    counts = np.bincount(labels, minlength=k)
    fractions = counts.astype(np.float64) / counts.sum()
    order = np.argsort(-fractions)
    return [
        Component(
            mean_rgb=tuple((centroids[index] * 255.0).tolist()),
            fraction=float(fractions[index]),
        )
        for index in order
    ]


def feature_vector(measurement: Measurement, components: list[Component]) -> list[float]:
    fractions = [component.fraction for component in components]
    reds = [component.mean_rgb[0] for component in components]
    greens = [component.mean_rgb[1] for component in components]
    blues = [component.mean_rgb[2] for component in components]
    while len(fractions) < 3:
        fractions.append(0.0)
        reds.append(0.0)
        greens.append(0.0)
        blues.append(0.0)
    return [
        measurement.area_px,
        measurement.width_px,
        measurement.height_px,
        measurement.mean_gray,
        measurement.std_gray,
        fractions[0],
        reds[0],
        greens[0],
        blues[0],
    ]
