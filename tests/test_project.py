import csv
import tempfile
import unittest
from pathlib import Path

from leaf_disease.metrics import classification_metrics
from prepare_data import prepare


class DataPreparationTests(unittest.TestCase):
    def test_stratified_splits_and_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            for label in ("healthy", "diseased"):
                (source / label).mkdir(parents=True)
                for index in range(10):
                    (source / label / f"leaf_{index}.jpg").write_bytes(b"example")
            output = root / "prepared"
            counts = prepare(source, output, "binary", seed=42)
            self.assertEqual(sum(counts.values()), 20)
            self.assertTrue(all(counts[(split, label)] > 0
                                for split in ("train", "val", "test")
                                for label in ("healthy", "diseased")))
            with (output / "manifest.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 20)
            self.assertEqual(len({row["source"] for row in rows}), 20)
            self.assertTrue(all((output / row["destination"]).is_file() for row in rows))

    def test_plantvillage_folder_mapping(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            for name in ("Tomato___healthy", "Tomato___Early_blight"):
                (source / name).mkdir(parents=True)
                for index in range(3):
                    (source / name / f"{index}.png").write_bytes(b"example")
            counts = prepare(source, root / "prepared", "plantvillage", seed=1)
            self.assertEqual(sum(counts.values()), 6)


class MetricTests(unittest.TestCase):
    def test_diseased_is_positive_class(self):
        result = classification_metrics([0, 0, 1, 1], [0, 1, 0, 1],
                                        {"diseased": 0, "healthy": 1})
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["diseased_precision"], 0.5)
        self.assertEqual(result["diseased_recall"], 0.5)
        self.assertEqual(result["diseased_f1"], 0.5)


if __name__ == "__main__":
    unittest.main()
