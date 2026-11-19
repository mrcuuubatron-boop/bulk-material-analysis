"""Approximate metric volume estimation from a calibrated 2-D pile silhouette."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .models import Measurement


@dataclass(frozen=True)
class Calibration:
    """Pixel-to-metre calibration and a visible-pile shape assumption."""

    metres_per_pixel_x: float
    metres_per_pixel_y: float
    depth_mode: str = "width"
    manual_depth_m: float = 0.0
    shape: str = "ellipsoid"

    @classmethod
    def from_reference(
        cls,
        reference_pixels: float,
        reference_metres: float,
        depth_mode: str = "width",
        manual_depth_m: float = 0.0,
        shape: str = "ellipsoid",
    ) -> "Calibration":
        if reference_pixels <= 0 or reference_metres <= 0:
            raise ValueError("Reference length must be greater than zero")
        scale = reference_metres / reference_pixels
        return cls(
            metres_per_pixel_x=scale,
            metres_per_pixel_y=scale,
            depth_mode=depth_mode,
            manual_depth_m=manual_depth_m,
            shape=shape,
        )


def estimate_volume(measurement: Measurement, calibration: Calibration) -> dict[str, float]:
    """Estimate volume from silhouette area and an explicit depth assumption.

    A single RGB image cannot recover hidden depth. The result is therefore an
    engineering estimate: projected area × estimated depth × shape coefficient.
    """
    area_m2 = (
        measurement.area_px
        * calibration.metres_per_pixel_x
        * calibration.metres_per_pixel_y
    )
    width_m = measurement.width_px * calibration.metres_per_pixel_x
    height_m = measurement.height_px * calibration.metres_per_pixel_y
    if calibration.depth_mode == "manual":
        depth_m = calibration.manual_depth_m
    else:
        depth_m = width_m

    coefficients = {
        "ellipsoid": math.pi / 6.0,
        "pyramid": 1.0 / 3.0,
        "box": 1.0,
    }
    if calibration.shape not in coefficients:
        raise ValueError(f"Unknown pile shape: {calibration.shape}")
    volume_m3 = area_m2 * depth_m * coefficients[calibration.shape]
    return {
        "area_m2": area_m2,
        "width_m": width_m,
        "height_m": height_m,
        "depth_m": depth_m,
        "shape_coefficient": coefficients[calibration.shape],
        "volume_m3": volume_m3,
    }
