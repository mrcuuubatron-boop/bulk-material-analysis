import argparse
from pathlib import Path

from .models import ExperienceModel
from .training import train_model
from .pipeline import analyze_directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Классификация и анализ сыпучих материалов по фотографиям"
    )
    parser.add_argument("--mode", choices=("image",), default="image")
    parser.add_argument("--pipeline", choices=("train", "infer"), required=True)
    parser.add_argument("--image-input-dir", type=Path, required=True)
    parser.add_argument("--experience-model", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--prediction-csv-output", type=Path)
    parser.add_argument("--k-clusters", type=int, default=3)
    parser.add_argument("--morphology-size", type=int, default=3)
    parser.add_argument("--threshold-offset", type=int, default=0)
    parser.add_argument("--dark-object", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mad-k", type=float, default=3.5)
    parser.add_argument("--pixel-scale-x", type=float, default=1.0)
    parser.add_argument("--pixel-scale-y", type=float, default=1.0)
    parser.add_argument("--height-estimate-m", type=float, default=0.0)
    parser.add_argument(
        "--densities",
        default="",
        help="Плотности материалов в кг/м³, например: sand=1600,cement=1400,gravel=1800",
    )
    return parser


def parse_densities(value: str) -> dict[str, float]:
    densities = {}
    for item in value.split(","):
        if not item.strip():
            continue
        name, density = item.split("=", 1)
        densities[name.strip()] = float(density.strip())
    return densities


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.pipeline == "train":
        model_path = args.experience_model or args.output_dir / "component_experience.json"
        model = train_model(
            args.image_input_dir,
            model_path,
            k_clusters=args.k_clusters,
            morphology_size=args.morphology_size,
            threshold_offset=args.threshold_offset,
            dark_object=args.dark_object,
            pixel_scale_x=args.pixel_scale_x,
            pixel_scale_y=args.pixel_scale_y,
            height_estimate_m=args.height_estimate_m,
        )
        model.density_kg_m3 = parse_densities(args.densities)
        model.save(model_path)
        print(f"Model trained: {model_path}")
        print(f"Materials: {', '.join(model.materials)}")
        return 0

    if args.experience_model is None:
        parser.error("--experience-model is required for infer")
    model = ExperienceModel.load(args.experience_model)
    _raw, _filtered, predictions_path = analyze_directory(
        args.image_input_dir,
        args.output_dir,
        experience_model=model,
        morphology_size=args.morphology_size,
        threshold_offset=args.threshold_offset,
        dark_object=args.dark_object,
        mad_k=args.mad_k,
    )
    if args.prediction_csv_output:
        args.prediction_csv_output.parent.mkdir(parents=True, exist_ok=True)
        args.prediction_csv_output.write_text(
            predictions_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    print(f"Analysis complete: {args.output_dir}")
    return 0
