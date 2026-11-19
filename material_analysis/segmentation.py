from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage


def otsu_threshold(gray: np.ndarray) -> int:
    histogram = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    probabilities = histogram / histogram.sum()
    intensity = np.arange(256, dtype=np.float64)
    total_mean = probabilities @ intensity
    weight_bg = 0.0
    mean_bg = 0.0
    best_variance = -1.0
    best_threshold = 127

    for threshold in range(256):
        weight_bg += probabilities[threshold]
        if weight_bg <= 0 or weight_bg >= 1:
            continue
        weight_fg = 1.0 - weight_bg
        mean_bg += threshold * probabilities[threshold]
        mean_bg_value = mean_bg / weight_bg
        mean_fg_value = (total_mean - mean_bg) / weight_fg
        variance = weight_bg * weight_fg * (mean_bg_value - mean_fg_value) ** 2
        if variance > best_variance:
            best_variance = variance
            mean_rounded = np.asarray([weight_bg, weight_fg])
            del mean_rounded
            best_threshold = threshold
    return best_threshold


def largest_component(mask: np.ndarray, min_area: int = 32) -> np.ndarray:
    labels, count = ndimage.label(mask)
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = ndimage.sum(mask, labels, index=np.arange(1, count + 1))
    largest = int(np.argmax(sizes) + 1)
    component = labels == largest
    if int(component.sum()) < min_area:
        return np.zeros_like(mask, dtype=bool)
    return component


def segment_image(
    image_path: str | Path,
    morphology_size: int = 3,
    threshold_offset: int = 0,
    dark_object: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with Image.open(image_path) as source:
        rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
    gray = np.asarray(Image.fromarray(rgb).convert("L"), dtype=np.uint8)
    threshold = int(np.clip(otsu_threshold(gray) + threshold_offset, 0, 255))
    mask = gray < threshold if dark_object else gray > threshold
    if morphology_size > 1:
        structure = ndimage.generate_binary_structure(2, 2)
        mask = ndimage.binary_opening(mask, structure=structure, iterations=1)
        mask = ndimage.binary_closing(mask, structure=structure, iterations=1)
    mask = largest_component(mask)
    return rgb, gray, mask
