"""Signed Laplacian and explicit zero-crossing semantics."""

import unittest
from unittest.mock import patch

import numpy as np

from backend.detection import laplacian, zero_crossing_edges
from backend.signal_ops import convolve2d


class LaplacianTests(unittest.TestCase):
    def test_flat_images_are_exactly_zero(self):
        for shape in ((5, 7), (1, 1), (1, 7), (5, 1)):
            response = laplacian(np.full(shape, 123.00000000000001))
            np.testing.assert_array_equal(response, np.zeros(shape))
            np.testing.assert_array_equal(zero_crossing_edges(response, 0), 0)

    def test_step_has_localized_signed_response(self):
        for sign in (-1, 1):
            image = sign * np.tile([0, 0, 255, 255, 255], (3, 1))
            for vertical in (False, True):
                expected = sign * np.tile([0, 255, -255, 0, 0], (3, 1))
                np.testing.assert_array_equal(laplacian(image.T if vertical else image),
                                              expected.T if vertical else expected)

    def test_kernel_centre_and_no_clipping(self):
        image = np.zeros((5, 5))
        image[2, 2] = 255
        expected = np.zeros((5, 5))
        expected[2, 2] = -1020
        expected[1, 2] = expected[3, 2] = expected[2, 1] = expected[2, 3] = 255
        np.testing.assert_array_equal(laplacian(image), expected)

    def test_shape_float_storage_and_input_unchanged(self):
        image = np.arange(70.0).reshape(7, 10)[:, ::2]
        before = image.copy()
        image.setflags(write=False)
        response = laplacian(image)
        self.assertEqual(response.shape, image.shape)
        self.assertEqual(response.dtype.kind, "f")
        self.assertFalse(np.shares_memory(response, image))
        np.testing.assert_array_equal(image, before)

    def test_reuses_manual_convolution(self):
        with patch("backend.detection.laplacian.convolve2d", wraps=convolve2d) as convolution:
            laplacian([[1, 2], [3, 4]])
        convolution.assert_called_once()

    def test_invalid_images_and_responses(self):
        for function in (laplacian, lambda image: zero_crossing_edges(image, 0)):
            for image in ([], [1], [[np.nan]], [[np.inf]], [[1], [2, 3]]):
                with self.assertRaises(ValueError):
                    function(image)
            for image in ([[True]], [[1j]], [["1"]]):
                with self.assertRaises(TypeError):
                    function(image)


class ZeroCrossingTests(unittest.TestCase):
    def test_strict_threshold_and_both_endpoints(self):
        response = np.array([[0., 2., -3., 0.]])
        for threshold in (0, 4.9, 5, 6):
            expected = [[0, 255, 255, 0]] if threshold < 5 else [[0, 0, 0, 0]]
            np.testing.assert_array_equal(zero_crossing_edges(response, threshold), expected)
            np.testing.assert_array_equal(zero_crossing_edges(response.T, threshold), np.array(expected).T)

    def test_exact_zero_ties_and_plateaus(self):
        for response, expected in (([2, 0, -3], [0, 255, 0]),
                                   ([2, 0, 3], [0, 0, 0]),
                                   ([2, 0, 0, -3], [0, 0, 0, 0])):
            np.testing.assert_array_equal(zero_crossing_edges([response], 4), [expected])
            np.testing.assert_array_equal(zero_crossing_edges(np.array(response)[:, None], 4),
                                          np.array(expected)[:, None])
            np.testing.assert_array_equal(zero_crossing_edges([response], 5), 0)

    def test_no_diagonal_or_border_wrap_crossings(self):
        np.testing.assert_array_equal(zero_crossing_edges([[1, 0], [0, -1]], 0), 0)
        np.testing.assert_array_equal(zero_crossing_edges([[1, 0, 0, -1]], 0), 0)

    def test_weak_signs_survive_and_same_sign_contrast_is_not_an_edge(self):
        np.testing.assert_array_equal(zero_crossing_edges([[1e-12, -1e-12]], 0), 255)
        np.testing.assert_array_equal(zero_crossing_edges([[1, 1000]], 0), 0)

    def test_mask_and_input_unchanged(self):
        response = np.array([[1., -2.], [0., 3.]])
        before = response.copy()
        response.setflags(write=False)
        edges = zero_crossing_edges(response, 0)
        self.assertEqual(edges.shape, response.shape)
        self.assertEqual(edges.dtype, np.uint8)
        self.assertFalse(np.shares_memory(edges, response))
        np.testing.assert_array_equal(response, before)

    def test_threshold_monotonicity(self):
        response = laplacian(np.random.default_rng(220).integers(0, 256, (13, 17)))
        previous = np.ones(response.shape, dtype=bool)
        for threshold in (0, 96, 200, 500, 2040):
            selected = zero_crossing_edges(response, threshold) != 0
            self.assertFalse(np.any(selected & ~previous))
            previous = selected
        self.assertFalse(previous.any())

    def test_invalid_thresholds(self):
        for threshold in (-1, np.nan, np.inf, 10**1000):
            with self.assertRaises(ValueError):
                zero_crossing_edges([[1]], threshold)
        for threshold in (True, np.bool_(False), "1", 1j, None, [1]):
            with self.assertRaises(TypeError):
                zero_crossing_edges([[1]], threshold)
