from dataclasses import asdict, dataclass
from pathlib import Path
import json
from typing import Iterable


@dataclass
class Measurement:
    filename: str
    area_px: float
    width_px: float
    height_px: float
    cx: float
    cy: float
    mean_gray: float
    std_gray: float
    valid: bool
    class_hint: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Component:
    mean_rgb: tuple[float, float, float]
    fraction: float


@dataclass
class ExperienceModel:
    pixel_scale_x: float = 1.0
    pixel_scale_y: float = 1.0
    height_estimate_m: float = 0.0
    density_kg_m3: dict[str, float] | None = None
    materials: dict[str, dict] | None = None

    def __post_init__(self) -> None:
        if self.density_kg_m3 is None:
            self.density_kg_m3 = {}
        if self.materials is None:
            self.materials = {}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ExperienceModel":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "materials" not in data:
            raise ValueError("Invalid experience model: 'materials' is required")
        return cls(**data)
