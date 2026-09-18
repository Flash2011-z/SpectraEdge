"""Independent scale passes and persistence fusion expectations."""

import unittest
from unittest.mock import patch

import numpy as np

from backend.analysis import multi_scale_edges
from backend.analysis import multiscale
from backend.detection import laplacian, prewitt, sobel, threshold_edges, zero_crossing_edges
from backend.signal_ops import gaussian_blur


class MultiScaleTests(unittest.TestCase):
    def setUp(self):
        y, x = np.indices((13, 17))
        self.image = ((x >= 5) * 120 + (y >= 8) * 80).astype(np.float64)

    def test_every_scale_starts_from_same_original_array(self):
        before = self.image.copy()
        with patch("backend.analysis.multiscale.gaussian_blur", wraps=gaussian_blur) as blur:
            result = multi_scale_edges(self.image, "Sobel", [0, 1.2, 2], 10, 2)
        self.assertEqual(blur.call_count, 3)
        for call in blur.call_args_list:
            self.assertIs(call.args[0], blur.call_args_list[0].args[0])
            np.testing.assert_array_equal(call.args[0], before)
        np.testing.assert_array_equal(self.image, before)
        self.assertEqual([scale.kernel_size for scale in result.scales], [3, 9, 13])

    def test_shapes_float_responses_and_binary_outputs_are_preserved(self):
        for detector in ("Sobel", "Prewitt", "Laplacian"):
            with self.subTest(detector=detector):
                result = multi_scale_edges(self.image, detector, [0, 1.2, 2], 10)
                self.assertEqual(result.sigma_values, (0.0, 1.2, 2.0))
                self.assertEqual(result.support_count, 2)
                for scale in result.scales:
                    self.assertEqual(scale.filtered.shape, self.image.shape)
                    self.assertEqual(scale.edge_map.shape, self.image.shape)
                    self.assertEqual(scale.filtered.dtype.kind, "f")
                    self.assertEqual(scale.edge_map.dtype, np.uint8)
                self.assertEqual(result.persistence_map.shape, self.image.shape)
                self.assertEqual(result.fused_edge_map.shape, self.image.shape)
                self.assertEqual(result.persistence_map.dtype, np.uint8)
                self.assertEqual(result.fused_edge_map.dtype, np.uint8)

    def test_persistence_is_count_and_fusion_uses_inclusive_support(self):
        edge_masks = [
            np.array([[0, 255, 255], [0, 0, 255]], dtype=np.uint8),
            np.array([[0, 0, 255], [255, 0, 255]], dtype=np.uint8),
            np.array([[0, 0, 0], [255, 0, 255]], dtype=np.uint8),
        ]
        with patch("backend.analysis.multiscale.gaussian_blur", side_effect=lambda image, sigma, size: image.copy()):
            with patch("backend.analysis.multiscale.sobel", return_value=(
                np.zeros((2, 3)), np.zeros((2, 3)), np.ones((2, 3)))):
                with patch("backend.analysis.multiscale.threshold_edges", side_effect=edge_masks):
                    result = multi_scale_edges(np.zeros((2, 3)), "Sobel", [0, 1, 2], 0, 2)
        np.testing.assert_array_equal(result.persistence_map, [[0, 1, 2], [2, 0, 3]])
        np.testing.assert_array_equal(result.fused_edge_map, [[0, 0, 255], [255, 0, 255]])
        self.assertGreaterEqual(int(result.persistence_map.min()), 0)
        self.assertLessEqual(int(result.persistence_map.max()), len(result.scales))

    def test_single_scale_exactly_matches_normal_detector_analysis(self):
        for detector, sigma, threshold in (("Sobel", 1.2, 20), ("Prewitt", 2, 20),
                                           ("Laplacian", 0, 20)):
            with self.subTest(detector=detector):
                result = multi_scale_edges(self.image, detector, [sigma], threshold)
                scale = result.scales[0]
                filtered = gaussian_blur(self.image, sigma, scale.kernel_size)
                if detector == "Laplacian":
                    expected = zero_crossing_edges(laplacian(filtered), threshold)
                else:
                    function = sobel if detector == "Sobel" else prewitt
                    expected = threshold_edges(function(filtered)[2], threshold)
                np.testing.assert_array_equal(scale.edge_map, expected)
                np.testing.assert_array_equal(result.fused_edge_map, expected)
                np.testing.assert_array_equal(result.persistence_map, expected != 0)
                self.assertEqual(result.support_count, 1)

    def test_repeated_calls_are_deterministic(self):
        first = multi_scale_edges(self.image, "Laplacian", [0.8, 1.6, 3.2], 12, 2)
        second = multi_scale_edges(self.image, "Laplacian", [0.8, 1.6, 3.2], 12, 2)
        np.testing.assert_array_equal(first.persistence_map, second.persistence_map)
        np.testing.assert_array_equal(first.fused_edge_map, second.fused_edge_map)
        for left, right in zip(first.scales, second.scales):
            np.testing.assert_array_equal(left.filtered, right.filtered)
            np.testing.assert_array_equal(left.edge_map, right.edge_map)

    def test_invalid_sigmas_detector_threshold_and_support(self):
        invalid_sigmas = ([], [1, 1], [-0.1], [5.1], [np.nan], [np.inf], [True], ["1"],
                          range(6), 1.2, "1,2")
        for sigmas in invalid_sigmas:
            with self.subTest(sigmas=sigmas), self.assertRaises((TypeError, ValueError)):
                multi_scale_edges(self.image, "Sobel", sigmas, 10)
        for detector in ("sobel", "Canny", None):
            with self.subTest(detector=detector), self.assertRaises(ValueError):
                multi_scale_edges(self.image, detector, [1], 10)
        for threshold in (-1, np.nan, np.inf, True, "10"):
            with self.subTest(threshold=threshold), self.assertRaises((TypeError, ValueError)):
                multi_scale_edges(self.image, "Sobel", [1], threshold)
        for support in (0, 4, 1.5, True):
            with self.subTest(support=support), self.assertRaises((TypeError, ValueError)):
                multi_scale_edges(self.image, "Sobel", [0, 1, 2], 10, support)


if __name__ == "__main__":
    unittest.main()
