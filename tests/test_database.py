import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from material_analysis.database import StatisticsDatabase
from material_analysis.models import Measurement


class DatabaseTests(unittest.TestCase):
    def test_records_run_and_aggregates_statistics(self):
        with tempfile.TemporaryDirectory() as tmp:
            database = StatisticsDatabase(Path(tmp) / "stats.sqlite3")
            rows = [
                Measurement("a.jpg", 100, 10, 10, 5, 5, 100, 2, True),
                Measurement("b.jpg", 80, 9, 9, 5, 5, 110, 3, False),
            ]
            run_id = database.record_run(
                "infer",
                "input",
                "output",
                rows,
                0.2,
                [{"filename": "a.jpg", "predicted_material": "sand", "confidence": 0.8}],
            )
            self.assertEqual(run_id, 1)
            summary = database.summary()
            self.assertEqual(summary["runs"], 1)
            self.assertEqual(summary["images"], 2)
            self.assertEqual(summary["valid_images"], 1)
            self.assertEqual(database.material_statistics()[0]["material"], "sand")


if __name__ == "__main__":
    unittest.main()
