"""Prepare downloaded aggregate archives for the current training pipeline.

The source archive stores all images in one directory and keeps PSD references
in a separate directory. This script creates class folders and metadata.csv.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil
import zipfile


ARCHIVE_PARTS = {
    "coarse": ("m_coarse.zip", "coarse"),
    "fine": ("m_fine.zip", "fine"),
}
LABELS = {
    "CN_2_16_Rhein": "river_gravel",
    "CN_2_16_Rezy": "recycled_concrete",
    "CN_02_Rhein": "river_sand",
}


def source_label(filename: str) -> str | None:
    stem = Path(filename).stem
    for prefix, label in LABELS.items():
        if stem.startswith(prefix + "_"):
            return label
    return None


def prepare_part(
    archive_path: Path,
    output_dir: Path,
    part: str,
    limit: int | None = None,
) -> int:
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")
    rows: list[dict[str, str]] = []
    with zipfile.ZipFile(archive_path) as archive:
        image_names = [
            name for name in archive.namelist()
            if name.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        image_names.sort()
        if limit is not None:
            image_names = image_names[:limit]
        for image_name in image_names:
            label = source_label(Path(image_name).name)
            if label is None:
                continue
            image_target_dir = output_dir / label
            image_target_dir.mkdir(parents=True, exist_ok=True)
            image_target = image_target_dir / Path(image_name).name
            with archive.open(image_name) as source, image_target.open("wb") as target:
                shutil.copyfileobj(source, target)

            reference_name = image_name.replace("/img/", "/ref/").rsplit(".", 1)[0] + ".txt"
            reference = ""
            if reference_name in archive.namelist():
                reference = archive.read(reference_name).decode("utf-8").strip()
            rows.append({
                "filename": image_target.name,
                "material": label,
                "part": part,
                "psd_reference": reference,
                "source_archive": archive_path.name,
            })

    metadata_path = output_dir / f"metadata_{part}.csv"
    write_header = not metadata_path.exists()
    with metadata_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [
            "filename", "material", "part", "psd_reference", "source_archive"
        ])
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare aggregate archives for training")
    parser.add_argument("--part", choices=(*ARCHIVE_PARTS, "all"), default="all")
    parser.add_argument("--archive-dir", type=Path, default=Path("datasets/aggregate"))
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/training"))
    parser.add_argument("--limit", type=int, help="Process only the first N images per selected part")
    args = parser.parse_args()

    parts = list(ARCHIVE_PARTS) if args.part == "all" else [args.part]
    total = 0
    for part in parts:
        archive_name, _ = ARCHIVE_PARTS[part]
        count = prepare_part(args.archive_dir / archive_name, args.output_dir, part, args.limit)
        total += count
        print(f"{part}: prepared {count} images")
    print(f"Total prepared: {total}")
    print(f"Training directory: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
