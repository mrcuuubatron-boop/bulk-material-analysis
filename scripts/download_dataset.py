"""Download the public aggregate dataset archives.

The archives are large (about 474 MB and 494 MB). Use --part coarse or fine.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve


DATASET_URLS = {
    "coarse": (
        "https://data.uni-hannover.de/dataset/6f844e22-12ed-48a7-9ccb-"
        "b502f8121650/resource/ae8f74b2-dcbd-46f8-81f6-ec6aab979768/"
        "download/m_coarse.zip"
    ),
    "fine": (
        "https://data.uni-hannover.de/dataset/6f844e22-12ed-48a7-9ccb-"
        "b502f8121650/resource/0544af00-d50b-4617-953c-76681e07bea0/"
        "download/m_fine.zip"
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Download aggregate image dataset")
    parser.add_argument("--part", choices=DATASET_URLS, default="coarse")
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/aggregate"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / f"m_{args.part}.zip"
    print(f"Downloading {args.part} dataset to {target}")
    urlretrieve(DATASET_URLS[args.part], target)
    print(f"Saved {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
