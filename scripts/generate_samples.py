from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "example_data"


def make_sample(path: Path, material: str, index: int) -> None:
    rng = np.random.default_rng(1000 + index)
    height, width = 480, 640
    background = np.empty((height, width, 3), dtype=np.uint8)
    background[:] = rng.normal(220, 8, (1, 1, 3)).clip(0, 255)
    image = Image.fromarray(background, mode="RGB")
    draw = ImageDraw.Draw(image)

    cx = 320 + index * 8 - 20
    cy = 240 + index * 3 - 8
    if material == "sand":
        box = (cx - 175, cy - 135, cx + 175, cy + 135)
        color = (190, 155, 95)
        roughness = 14
    elif material == "cement":
        box = (cx - 165, cy - 125, cx + 165, cy + 125)
        color = (150, 145, 138)
        roughness = 18
    else:
        box = (cx - 185, cy - 145, cx + 185, cy + 145)
        color = (125, 120, 112)
        roughness = 24

    draw.ellipse(box, fill=color)
    array = np.asarray(image).copy()
    yy, xx = np.ogrid[:height, :width]
    ellipse = ((xx - cx) / 180) ** 2 + ((yy - cy) / 140) ** 2 <= 1
    noise = rng.normal(0, roughness, ellipse.sum())
    pixels = array[ellipse].astype(np.float32)
    pixels[:, 0] = (pixels[:, 0] + noise).clip(0, 255)
    pixels[:, 1] = (pixels[:, 1] + noise * 0.5).clip(0, 255)
    pixels[:, 2] = (pixels[:, 2] + noise * 0.25).clip(0, 255)
    array[ellipse] = pixels
    Image.fromarray(array, mode="RGB").save(path)


def main() -> None:
    for material in ("sand", "cement", "gravel"):
        directory = OUTPUT / material
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(6):
            make_sample(directory / f"{material}_{index + 1:03d}.jpg", material, index)
    print(f"Generated samples in {OUTPUT}")


if __name__ == "__main__":
    main()
