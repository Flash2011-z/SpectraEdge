"""Analytical expectations, independent of a library Sobel implementation."""

import unittest
from unittest.mock import patch

import numpy as np

from backend.detection import sobel, threshold_edges
from backend.signal_ops import convolve2d, gaussian_blur


class SobelTests(unittest.TestCase):
    def test_constant_images_have_zero_gradients_and_no_edges(self):
        for shape in [(5, 7), (1, 1), (1, 7), (5, 1)]:
            with self.subTest(shape=shape):
                gx, gy, magnitude = sobel(np.full(shape, 123, dtype=np.uint8))
                for output in (gx, gy, magnitude, threshold_edges(magnitude, 0)):
                    np.testing.assert_array_equal(output, np.zeros(shape))

    def test_rightward_ramp_is_positive_x_with_reflected_boundaries(self):
        # A unit ramp gives (1+2+1) * (right-left) = 4*2 = 8.
        image = np.tile(np.arange(5), (3, 1))
        gx, gy, magnitude = sobel(image)
        expected = np.tile([0, 8, 8, 8, 0], (3, 1))
        np.testing.assert_array_equal(gx, expected)
        np.testing.assert_array_equal(gy, np.zeros((3, 5)))
        np.testing.assert_array_equal(magnitude, expected)

    def test_fractional_constant_offset_is_removed_without_erasing_weak_edges(self):
        image = np.full((3, 5), 123.00000000000001)
        for output in sobel(image):
            np.testing.assert_array_equal(output, np.zeros(image.shape))
        image[:, 2:] += 1e-10
        _, _, magnitude = sobel(image)
        self.assertTrue(np.all(magnitude[:, 1:3] > 0))
        self.assertEqual(np.count_nonzero(threshold_edges(magnitude, 0)), 6)

    def test_downward_ramp_is_positive_y(self):
        image = np.tile(np.arange(5)[:, None], (1, 3))
        gx, gy, magnitude = sobel(image)
        expected = np.tile(np.array([0, 8, 8, 8, 0])[:, None], (1, 3))
        np.testing.assert_array_equal(gx, np.zeros((5, 3)))
        np.testing.assert_array_equal(gy, expected)
        np.testing.assert_array_equal(magnitude, expected)

    def test_known_step_and_reverse_step_preserve_signed_large_responses(self):
        for sign in (1, -1):
            with self.subTest(sign=sign):
                image = np.tile([0, 0, 255, 255, 255], (3, 1))
                if sign < 0:
                    image = 255 - image
                gx, gy, magnitude = sobel(image.astype(np.uint8))
                expected = np.tile([0, 1020, 1020, 0, 0], (3, 1))
                np.testing.assert_array_equal(gx, sign * expected)
                np.testing.assert_array_equal(gy, np.zeros((3, 5)))
                np.testing.assert_array_equal(magnitude, expected)
                self.assertEqual(gx.dtype.kind, "f")

    def test_two_axis_magnitude_is_euclidean_length(self):
        y, x = np.indices((5, 7))
        gx, gy, magnitude = sobel(3 * x - 4 * y)
        np.testing.assert_array_equal(gx[1:-1, 1:-1], 24)
        np.testing.assert_array_equal(gy[1:-1, 1:-1], -32)
        np.testing.assert_array_equal(magnitude[1:-1, 1:-1], 40)
        np.testing.assert_allclose(magnitude, np.sqrt(gx * gx + gy * gy))

    def test_calls_existing_true_convolution_twice(self):
        with patch("backend.detection.sobel.convolve2d", wraps=convolve2d) as convolution:
            sobel([[1, 2], [3, 4]])
        self.assertEqual(convolution.call_count, 2)

    def test_readonly_noncontiguous_input_is_unchanged(self):
        image = np.arange(70.0).reshape(7, 10)[:, ::2]
        before = image.copy()
        image.setflags(write=False)
        for output in sobel(image):
            self.assertEqual(output.shape, image.shape)
            self.assertEqual(output.dtype.kind, "f")
            self.assertFalse(np.shares_memory(output, image))
        np.testing.assert_array_equal(image, before)

    def test_invalid_images_raise_clear_errors(self):
        for image in ([], [1, 2], [[[1]]], [[1], [2, 3]], [[np.nan]], [[np.inf]]):
            with self.subTest(image=image), self.assertRaisesRegex(ValueError, "image"):
                sobel(image)
        for image in ([[True]], [[1j]], [["1"]], np.array([[1]], dtype=object)):
            with self.subTest(image=image), self.assertRaisesRegex(TypeError, "image"):
                sobel(image)


class ThresholdTests(unittest.TestCase):
    def test_strict_equality_and_zero_threshold(self):
        magnitude = np.array([[0, 95.9, 96, 96.1, 1020]])
        np.testing.assert_array_equal(threshold_edges(magnitude, 96), [[0, 0, 0, 255, 255]])
        np.testing.assert_array_equal(threshold_edges(magnitude, 0), [[0, 255, 255, 255, 255]])

    def test_threshold_does_not_normalize_or_modify_magnitude(self):
        magnitude = np.array([[0.0, 100.0, 2000.0]])
        before = magnitude.copy()
        edges = threshold_edges(magnitude, 1443)
        np.testing.assert_array_equal(edges, [[0, 0, 255]])
        np.testing.assert_array_equal(magnitude, before)
        self.assertEqual(edges.dtype, np.uint8)
        self.assertFalse(np.shares_memory(edges, magnitude))

    def test_increasing_threshold_only_removes_edges(self):
        image = np.random.default_rng(220).integers(0, 256, (17, 23))
        filtered = gaussian_blur(image, 1.2, 7)
        _, _, magnitude = sobel(filtered)
        previous = np.ones(image.shape, dtype=bool)
        for threshold in (0, 32, 96, 200, 500, 1443):
            selected = threshold_edges(magnitude, threshold) != 0
            self.assertFalse(np.any(selected & ~previous))
            previous = selected
        self.assertFalse(previous.any())

    def test_invalid_thresholds(self):
        for value in (-1, np.nan, np.inf, -np.inf, 10**1000):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "threshold"):
                threshold_edges([[1]], value)
        for value in (True, np.bool_(False), "96", 1j, None, [96]):
            with self.subTest(value=value), self.assertRaisesRegex(TypeError, "threshold"):
                threshold_edges([[1]], value)

    def test_invalid_magnitude(self):
        for value in ([], [1], [[-1]], [[np.nan]], [[np.inf]], [[1], [2, 3]]):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "magnitude"):
                threshold_edges(value, 0)
        for value in ([[True]], [[1j]], [["1"]]):
            with self.subTest(value=value), self.assertRaisesRegex(TypeError, "magnitude"):
                threshold_edges(value, 0)


if __name__ == "__main__":
    unittest.main()
