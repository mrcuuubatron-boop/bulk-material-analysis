import numpy as np

from .models import Measurement


def mad_filter(values: list[float], k: float = 3.5) -> tuple[list[float], list[bool]]:
    if not values:
        return [], []
    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    if mad == 0:
        nonzero_deviations = np.abs(array - median)
        nonzero_deviations = nonzero_deviations[nonzero_deviations > 0]
        if len(nonzero_deviations):
            mad = float(np.min(nonzero_deviations))
        else:
            return array.tolist(), [True] * len(array)
    keep = np.abs(array - median) <= k * mad
    return array[keep].tolist(), keep.tolist()


def robust_filter(measurements: list[Measurement], mad_k: float = 3.5) -> list[Measurement]:
    if not measurements:
        return []

    fields = ("area_px", "width_px", "height_px", "cx", "cy")
    keep_mask = np.ones(len(measurements), dtype=bool)
    for field in fields:
        values = [getattr(row, field) for row in measurements]
        _filtered, keep = mad_filter(values, k=mad_k)
        keep_mask &= np.asarray(keep, dtype=bool)

    filtered = []
    for row, keep in zip(measurements, keep_mask):
        if keep and row.valid:
            filtered.append(row)
    return filtered
