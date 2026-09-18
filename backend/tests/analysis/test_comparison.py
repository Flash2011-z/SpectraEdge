"""Shared filtering and detector reuse for comparison analysis."""

import unittest
from unittest.mock import patch

import numpy as np

from backend.analysis import compare_detectors
from backend.analysis import comparison
from backend.analysis.components import analyze_objects, filter_small_components
from backend.detection import laplacian, prewitt, sobel, threshold_edges, zero_crossing_edges
from backend.signal_ops import gaussian_blur


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        y, x = np.indices((15, 19))
        self.image = ((x >= 6) * 150 + (y >= 9) * 75).astype(np.float64)

    def test_filters_once_and_reuses_same_signal_for_every_detector(self):
        before = self.image.copy()
        filtered = gaussian_blur(self.image, 1.2, 9)
        with patch.object(comparison, "gaussian_blur", return_value=filtered) as blur:
            with patch.object(comparison, "sobel", wraps=sobel) as sobel_call:
                with patch.object(comparison, "prewitt", wraps=prewitt) as prewitt_call:
                    with patch.object(comparison, "laplacian", wraps=laplacian) as laplacian_call:
                        result = compare_detectors(self.image, 1.2, 9, 96, 96, 20)
        blur.assert_called_once()
        self.assertIs(sobel_call.call_args.args[0], filtered)
        self.assertIs(prewitt_call.call_args.args[0], filtered)
        self.assertIs(laplacian_call.call_args.args[0], filtered)
        np.testing.assert_array_equal(self.image, before)
        self.assertIs(result.filtered, filtered)

    def test_results_match_existing_decisions_and_object_analysis(self):
        result = compare_detectors(self.image, 0, 3, 96, 96, 20)
        expected = {
            "sobel": threshold_edges(sobel(result.filtered)[2], 96),
            "prewitt": threshold_edges(prewitt(result.filtered)[2], 96),
            "laplacian": zero_crossing_edges(laplacian(result.filtered), 20),
        }
        for name, edge_map in expected.items():
            with self.subTest(detector=name):
                entry = getattr(result, name)
                np.testing.assert_array_equal(entry.edge_map, edge_map)
                analysis_mask = filter_small_components(edge_map, 2) if name == "laplacian" else edge_map
                self.assertEqual(entry.object_list, analyze_objects(analysis_mask).objects)
                self.assertEqual(entry.object_count, len(entry.object_list))
                self.assertEqual(entry.edge_pixel_count, int(np.count_nonzero(edge_map)))
                expected_average = (sum(item["area"] for item in entry.object_list) /
                                    len(entry.object_list) if entry.object_list else 0.0)
                self.assertEqual(entry.average_object_area, expected_average)
                self.assertEqual(entry.edge_map.shape, self.image.shape)
                self.assertEqual(entry.edge_map.dtype, np.uint8)
                self.assertGreaterEqual(entry.processing_time, 0)
        self.assertEqual(result.sobel.threshold, 96)
        self.assertEqual(result.sobel.threshold_type, "gradient_magnitude")
        self.assertEqual(result.prewitt.threshold, 96)
        self.assertEqual(result.laplacian.threshold, 20)
        self.assertEqual(result.laplacian.threshold_type, "zero_crossing_contrast")
        self.assertEqual(result.sobel.threshold_label, "Gradient magnitude threshold")
        self.assertEqual(result.prewitt.threshold_label, "Gradient magnitude threshold")
        self.assertEqual(result.laplacian.threshold_label, "Zero-crossing contrast threshold")
        self.assertIsNone(result.sobel.minimum_component_area)
        self.assertEqual(result.laplacian.minimum_component_area, 2)

    def test_laplacian_cleanup_changes_objects_but_not_raw_edges(self):
        raw = np.zeros(self.image.shape, dtype=np.uint8)
        raw[0, 0] = 255
        raw[4, 5:8] = 255
        with patch.object(comparison, "zero_crossing_edges", return_value=raw):
            default = compare_detectors(self.image, 0, 3, 96, 96, 20)
            strict = compare_detectors(self.image, 0, 3, 96, 96, 20, 4)
        np.testing.assert_array_equal(default.laplacian.edge_map, raw)
        np.testing.assert_array_equal(strict.laplacian.edge_map, raw)
        self.assertEqual([item["area"] for item in default.laplacian.object_list], [3])
        self.assertEqual(strict.laplacian.object_list, [])
        self.assertEqual(default.laplacian.edge_pixel_count, 4)

    def test_detector_specific_thresholds_change_only_their_own_edge_counts(self):
        low = compare_detectors(self.image, 0, 3, 96, 96, 0)
        high = compare_detectors(self.image, 0, 3, 96, 96, 500)
        np.testing.assert_array_equal(low.sobel.edge_map, high.sobel.edge_map)
        np.testing.assert_array_equal(low.prewitt.edge_map, high.prewitt.edge_map)
        self.assertNotEqual(np.count_nonzero(low.laplacian.edge_map),
                            np.count_nonzero(high.laplacian.edge_map))

    def test_invalid_threshold_is_rejected(self):
        for threshold in (-1, np.nan, np.inf, True, "20"):
            with self.subTest(threshold=threshold), self.assertRaises((TypeError, ValueError)):
                compare_detectors(self.image, 1, 7, 96, 96, threshold)


if __name__ == "__main__":
    unittest.main()
