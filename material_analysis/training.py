from pathlib import Path

import numpy as np

from .models import ExperienceModel
from .features import extract_measurement, kmeans_color, feature_vector


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def train_model(
    input_dir: str | Path,
    model_path: str | Path,
    k_clusters: int = 3,
    morphology_size: int = 3,
    threshold_offset: int = 0,
    dark_object: bool = True,
    pixel_scale_x: float = 1.0,
    pixel_scale_y: float = 1.0,
    height_estimate_m: float = 0.0,
) -> ExperienceModel:
    input_dir = Path(input_dir)
    vectors_by_material: dict[str, list[np.ndarray]] = {}

    class_dirs = sorted(path for path in input_dir.iterdir() if path.is_dir())
    if class_dirs:
        sources = [(path.name, path) for path in class_dirs]
    else:
        sources = [(input_dir.name, input_dir)]

    for material_name, source in sources:
        image_paths = sorted(
            path for path in source.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        for image_path in image_paths:
            measurement, rgb, _gray, mask = extract_measurement(
                image_path,
                morphology_size=morphology_size,
                threshold_offset=threshold_offset,
                dark_object=dark_object,
                class_hint=material_name,
            )
            if not measurement.valid:
                continue
            components = kmeans_color(rgb, mask, k=k_clusters)
            vectors_by_material.setdefault(material_name, []).append(
                np.asarray(feature_vector(measurement, components), dtype=np.float64)
            )

    if not vectors_by_material:
        raise ValueError("No valid training images found")

    model = ExperienceModel(
        pixel_scale_x=pixel_scale_x,
        pixel_scale_y=pixel_scale_y,
        height_estimate_m=height_estimate_m,
    )
    all_vectors = np.vstack([vector for values in vectors_by_material.values() for vector in values])
    global_scale = np.maximum(np.std(all_vectors, axis=0), 1.0)

    for material_name, vectors in vectors_by_material.items():
        matrix = np.vstack(vectors)
        centroid = np.mean(matrix, axis=0)
        scale = np.maximum(np.std(matrix, axis=0), global_scale * 0.25)
        model.materials[material_name] = {
            "centroid": centroid.tolist(),
            "scale": scale.tolist(),
            "examples": len(vectors),
        }

    model.save(model_path)
    return model
