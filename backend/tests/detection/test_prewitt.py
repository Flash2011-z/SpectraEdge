"""Analytical Prewitt expectations and numerical boundaries."""

import unittest
from unittest.mock import patch

import numpy as np

from backend.detection import prewitt
from backend.signal_ops import convolve2d


class PrewittTests(unittest.TestCase):
    def test_flat_images_are_exactly_zero(self):
        for shape in ((5, 7), (1, 1), (1, 7), (5, 1)):
            for output in prewitt(np.full(shape, 123.00000000000001)):
                np.testing.assert_array_equal(output, np.zeros(shape))

    def test_horizontal_and_vertical_ramps(self):
        image = np.tile(np.arange(5), (3, 1))
        expected = np.tile([0, 6, 6, 6, 0], (3, 1))
        for vertical in (False, True):
            gx, gy, magnitude = prewitt(image.T if vertical else image)
            np.testing.assert_array_equal(gy if vertical else gx, expected.T if vertical else expected)
            np.testing.assert_array_equal(gx if vertical else gy, 0)
            np.testing.assert_array_equal(magnitude, expected.T if vertical else expected)

    def test_signed_unclipped_step(self):
        for sign in (-1, 1):
            gx, gy, magnitude = prewitt(sign * np.tile([0, 0, 255, 255, 255], (3, 1)))
            expected = np.tile([0, 765, 765, 0, 0], (3, 1))
            np.testing.assert_array_equal(gx, sign * expected)
            np.testing.assert_array_equal(gy, 0)
            np.testing.assert_array_equal(magnitude, expected)

    def test_two_axis_magnitude(self):
        y, x = np.indices((5, 7))
        gx, gy, magnitude = prewitt(3 * x - 4 * y)
        np.testing.assert_array_equal(gx[1:-1, 1:-1], 18)
        np.testing.assert_array_equal(gy[1:-1, 1:-1], -24)
        np.testing.assert_array_equal(magnitude[1:-1, 1:-1], 30)

    def test_shape_float_storage_and_input_unchanged(self):
        image = np.arange(70.0).reshape(7, 10)[:, ::2]
        before = image.copy()
        image.setflags(write=False)
        for output in prewitt(image):
            self.assertEqual(output.shape, image.shape)
            self.assertEqual(output.dtype.kind, "f")
            self.assertFalse(np.shares_memory(output, image))
        np.testing.assert_array_equal(image, before)

    def test_reuses_manual_convolution(self):
        with patch("backend.detection.prewitt.convolve2d", wraps=convolve2d) as convolution:
            prewitt([[1, 2], [3, 4]])
        self.assertEqual(convolution.call_count, 2)

    def test_invalid_inputs(self):
        for image in ([], [1], [[np.nan]], [[np.inf]], [[1], [2, 3]]):
            with self.assertRaises(ValueError):
                prewitt(image)
        for image in ([[True]], [[1j]], [["1"]]):
            with self.assertRaises(TypeError):
                prewitt(image)
