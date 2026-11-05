"""Small synthetic checks; these do not validate model quality or train models."""

import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from experiment_driver import get_cv_splits, make_cv_bins
from full_pipeline_eval_core import apply_router_mask, bin_codes
import structural_strategy_common as structural
from utils import load_dataset, load_test_dataset, save_results


class GeometryFeaturesTests(unittest.TestCase):
    def test_black_and_textured_images_have_finite_deterministic_features(self):
        images = {
            "black": np.zeros((72, 90, 3), dtype=np.float32),
            "textured": np.random.default_rng(42).random((72, 90, 3), dtype=np.float32),
        }
        dimensions = {
            "spatial_layout": 24,
            "distance_features": 18,
            "angle_features": 20,
            "perspective_cues": 7,
        }
        features = {}
        for name, image in images.items():
            with self.subTest(image=name):
                original = image.copy()
                parts = structural.geometry_aware_parts(image)
                repeated = structural.geometry_aware_parts(image)
                self.assertEqual(set(parts), set(dimensions))
                for group, size in dimensions.items():
                    self.assertEqual(parts[group].shape, (size,))
                    self.assertEqual(parts[group].dtype, np.dtype("float32"))
                    self.assertTrue(np.isfinite(parts[group]).all(), group)
                    np.testing.assert_array_equal(parts[group], repeated[group])
                np.testing.assert_array_equal(image, original)
                features[name] = parts
        self.assertFalse(np.array_equal(
            features["black"]["angle_features"], features["textured"]["angle_features"]
        ))


class RoutingAndSplitsTests(unittest.TestCase):
    def test_bin_boundaries(self):
        values = np.array([0.25, 0.999, 1.0, 1.999, 2.0, 2.78])
        codes = bin_codes(values)
        np.testing.assert_array_equal(codes, [0, 0, 1, 1, 2, 2])
        self.assertEqual(codes.dtype, np.dtype("int8"))

    def test_router_selects_masked_values_clips_and_preserves_inputs(self):
        y = np.array([0.25, 1.0, 2.0, 2.78])
        baseline = np.array([0.5, 1.2, 2.2, 2.5])
        direct = np.array([2.0, -1.0, 1.0, 4.0])
        baseline_before, direct_before = baseline.copy(), direct.copy()
        result = apply_router_mask(y, baseline, direct, np.array([False, True, False, True]))
        np.testing.assert_allclose(result, [0.5, 0.25, 2.2, 2.78])
        self.assertEqual(result.dtype, np.dtype("float32"))
        np.testing.assert_array_equal(baseline, baseline_before)
        np.testing.assert_array_equal(direct, direct_before)

    def test_router_handles_all_and_no_selected_rows(self):
        y = np.array([0.25, 2.78])
        baseline = np.array([0.5, 2.5])
        direct = np.array([1.2, 1.8])
        for selected, expected in [(False, baseline), (True, direct)]:
            with self.subTest(selected=selected):
                actual = apply_router_mask(y, baseline, direct, np.full(2, selected))
                np.testing.assert_allclose(actual, expected)

    def test_cv_folds_are_deterministic_disjoint_stratified_and_cover_every_row(self):
        y = np.linspace(0.25, 2.78, 80, dtype=np.float32)
        bins = make_cv_bins(y)
        splits = get_cv_splits(y, n_splits=5)
        repeated = get_cv_splits(y, n_splits=5)
        self.assertEqual(len(splits), 5)
        validation_counts = np.zeros(len(y), dtype=int)
        for (train, validation), (again_train, again_validation) in zip(splits, repeated):
            np.testing.assert_array_equal(train, again_train)
            np.testing.assert_array_equal(validation, again_validation)
            self.assertEqual(len(np.intersect1d(train, validation)), 0)
            np.testing.assert_array_equal(np.sort(np.concatenate([train, validation])), np.arange(len(y)))
            np.testing.assert_array_equal(np.bincount(bins[validation], minlength=8), np.full(8, 2))
            validation_counts[validation] += 1
        np.testing.assert_array_equal(validation_counts, np.ones(len(y), dtype=int))


class DataAndSubmissionTests(unittest.TestCase):
    def test_submission_csv_has_header_sequential_padded_ids_and_prediction_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "nested" / "submission.csv"
            predictions = [1.5, 0.25, 2.75]
            save_results(predictions, output_path=path)
            with path.open(encoding="utf-8", newline="") as file:
                rows = list(csv.reader(file))
        self.assertEqual(rows[0], ["ID", "Distance"])
        self.assertEqual([row[0] for row in rows[1:]], ["000", "001", "002"])
        self.assertEqual([float(row[1]) for row in rows[1:]], predictions)

    def test_empty_submission_has_only_header(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "submission.csv"
            save_results([], output_path=path)
            self.assertEqual(path.read_text(encoding="utf-8"), "ID,Distance\n")

    def test_synthetic_data_preserves_label_order_and_sorts_test_filenames(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data = root / "data"
            (data / "train_images").mkdir(parents=True)
            (data / "test_images").mkdir()
            # Reversed labels and leading zeroes catch accidental sorting or ID coercion.
            (data / "train_labels.csv").write_text("ID,distance\n002,2.0\n001,0.5\n", encoding="utf-8")
            for image_id, value in [("001", 25), ("002", 200)]:
                Image.new("RGB", (12, 12), (value, value, value)).save(data / "train_images" / f"{image_id}.png")
            for filename, value in [("010.png", 210), ("002.png", 35), ("001.png", 10)]:
                Image.new("RGB", (12, 12), (value, value, value)).save(data / "test_images" / filename)
            (data / "test_images" / "ignored.txt").write_text("not an image", encoding="utf-8")

            for load_rgb in (False, True):
                with self.subTest(load_rgb=load_rgb):
                    config = {"data_dir": data, "downsample_factor": 30, "load_rgb": load_rgb}
                    images, distances = load_dataset(config)
                    self.assertEqual(images.shape, (2, 300 if load_rgb else 100))
                    np.testing.assert_array_equal(distances, [2.0, 0.5])
                    np.testing.assert_array_equal(images[0], np.full(images.shape[1], 200))
                    np.testing.assert_array_equal(images[1], np.full(images.shape[1], 25))
                    test_images = np.asarray(load_test_dataset(config))
                    self.assertEqual(test_images.shape, (3, images.shape[1]))
                    np.testing.assert_array_equal(test_images[:, 0], [10, 35, 210])

            # The active geometry feature loader must use the same image ordering.
            with patch.object(structural, "ROOT", root):
                self.assertEqual([p.name for p in structural._image_paths("train")], ["002.png", "001.png"])
                self.assertEqual([p.name for p in structural._image_paths("test")], ["001.png", "002.png", "010.png"])


if __name__ == "__main__":
    unittest.main()
