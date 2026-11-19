import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_samples import make_sample
from material_analysis.segmentation import otsu_threshold, largest_component
from material_analysis.features import extract_measurement
from material_analysis.filtering import mad_filter, robust_filter
from material_analysis.models import Measurement
from material_analysis.training import train_model
from material_analysis.pipeline import analyze_directory
from material_analysis.models import ExperienceModel
from material_analysis.volume import Calibration, estimate_volume


class CoreTests(unittest.TestCase):
    def test_otsu_and_component(self):
        import numpy as np
        gray = np.vstack([np.full((40, 80), 200), np.full((40, 80), 50)])
        threshold = otsu_threshold(gray)
        self.assertLess(threshold, 200)
        mask = np.zeros((20, 20), dtype=bool)
        mask[2:12, 3:13] = True
        mask[18, 18] = True
        component = largest_component(mask, min_area=32)
        self.assertEqual(int(component.sum()), 100)

    def test_mad_filter_removes_outlier(self):
        filtered, keep = mad_filter([10, 10, 11, 10, 100], k=3.5)
        self.assertNotIn(100.0, filtered)
        self.assertFalse(keep[-1])

    def test_calibrated_volume_and_mass_geometry(self):
        measurement = Measurement("pile.jpg", 10000, 100, 100, 50, 50, 80, 5, True)
        calibration = Calibration.from_reference(
            reference_pixels=100,
            reference_metres=1,
            shape="box",
        )
        result = estimate_volume(measurement, calibration)
        self.assertAlmostEqual(result["area_m2"], 1.0)
        self.assertAlmostEqual(result["depth_m"], 1.0)
        self.assertAlmostEqual(result["volume_m3"], 1.0)

    def test_full_synthetic_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            training = root / "training"
            for material in ("sand", "cement", "gravel"):
                directory = training / material
                directory.mkdir(parents=True)
                for index in range(5):
                    make_sample(directory / f"{material}_{index}.jpg", material, index)

            model_path = root / "experience.json"
            train_model(training, model_path)
            model = ExperienceModel.load(model_path)
            self.assertEqual(set(model.materials), {"sand", "cement", "gravel"})

            inference = root / "inference"
            inference.mkdir()
            make_sample(inference / "sample.jpg", "sand", 2)
            output = root / "output"
            measurements, filtered, predictions_path = analyze_directory(
                inference, output, experience_model=model
            )
            self.assertEqual(len(measurements), 1)
            self.assertEqual(len(filtered), 1)
            self.assertTrue(predictions_path.exists())
            self.assertTrue((output / "feature_trends.png").exists())
            self.assertTrue((output / "predictions.csv").exists())


if __name__ == "__main__":
    unittest.main()
