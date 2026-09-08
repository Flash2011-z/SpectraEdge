"""Checks for Gaussian weights, smoothing, and the sigma-zero bypass."""

import unittest

import numpy as np

from backend.signal_ops import gaussian_blur, gaussian_kernel


class GaussianKernelTests(unittest.TestCase):
    def test_normalization_and_symmetry(self):
        kernel = gaussian_kernel(5, 1.2)

        self.assertEqual(kernel.shape, (5, 5))
        self.assertTrue(np.issubdtype(kernel.dtype, np.floating))
        self.assertTrue(np.isfinite(kernel).all())
        self.assertTrue((kernel > 0).all())
        self.assertAlmostEqual(float(kernel.sum()), 1.0, places=14)
        np.testing.assert_allclose(kernel, kernel[::-1, :], rtol=0, atol=1e-15)
        np.testing.assert_allclose(kernel, kernel[:, ::-1], rtol=0, atol=1e-15)
        np.testing.assert_allclose(kernel, kernel.T, rtol=0, atol=1e-15)
        self.assertGreater(kernel[2, 2], kernel[2, 1])
        self.assertGreater(kernel[2, 1], kernel[1, 1])

    def test_known_three_by_three_weights(self):
        # This sigma makes exp(-1/(2*sigma^2)) = 1/2. The unnormalised
        # centre, side, and corner weights are 1, 1/2, and 1/4; their sum is 4.
        sigma = 1 / np.sqrt(2 * np.log(2))
        expected = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=float) / 16
        np.testing.assert_allclose(gaussian_kernel(3, sigma), expected, rtol=0, atol=1e-15)

    def test_one_by_one_kernel(self):
        np.testing.assert_array_equal(gaussian_kernel(1, 1.5), [[1.0]])

    def test_numpy_scalar_parameters(self):
        kernel = gaussian_kernel(np.int64(3), np.float32(1.2))
        self.assertEqual(kernel.shape, (3, 3))
        self.assertAlmostEqual(float(kernel.sum()), 1.0, places=14)

    def test_extreme_positive_sigma_stays_normalized(self):
        # Tiny sigma approaches a centred impulse; huge sigma approaches
        # uniform weights on this finite support. Neither should produce NaN.
        tiny_sigma = np.nextafter(0.0, 1.0)
        with np.errstate(all="raise"):
            narrow = gaussian_kernel(3, tiny_sigma)
            wide = gaussian_kernel(3, np.finfo(float).max)
        np.testing.assert_array_equal(narrow, [[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        np.testing.assert_allclose(wide, np.full((3, 3), 1 / 9), rtol=0, atol=1e-15)

    def test_invalid_kernel_sizes(self):
        for size in (0, -1, -3, 2, 4):
            with self.subTest(size=size):
                with self.assertRaisesRegex(ValueError, "kernel_size"):
                    gaussian_kernel(size, 1)
        for size in (3.0, True, np.bool_(True), "3", None, [3]):
            with self.subTest(size=repr(size)):
                with self.assertRaisesRegex(TypeError, "kernel_size"):
                    gaussian_kernel(size, 1)

    def test_invalid_sigma_values(self):
        for sigma in (0, -1, np.nan, np.inf, -np.inf, 10**400):
            with self.subTest(sigma=sigma):
                with self.assertRaisesRegex(ValueError, "sigma"):
                    gaussian_kernel(3, sigma)
        for sigma in (True, np.bool_(False), "1", None, 1 + 0j, [1]):
            with self.subTest(sigma=repr(sigma)):
                with self.assertRaisesRegex(TypeError, "sigma"):
                    gaussian_kernel(3, sigma)


class GaussianBlurTests(unittest.TestCase):
    def test_constant_image_remains_constant_including_borders(self):
        for value in (127, -25, 300):
            with self.subTest(value=value):
                image = np.full((4, 6), value, dtype=float)
                result = gaussian_blur(image, sigma=1.3, kernel_size=5)
                self.assertEqual(result.shape, image.shape)
                np.testing.assert_allclose(result, image, rtol=0, atol=1e-12)

    def test_impulse_spreads_into_known_gaussian_weights(self):
        image = np.zeros((5, 5))
        image[2, 2] = 1
        sigma = 1 / np.sqrt(2 * np.log(2))
        expected = np.zeros((5, 5))
        expected[1:4, 1:4] = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16

        result = gaussian_blur(image, sigma=sigma, kernel_size=3)
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-15)

    def test_sigma_zero_returns_unchanged_floating_point_copy(self):
        for dtype in (np.uint8, np.float32, np.float64):
            with self.subTest(dtype=dtype):
                image = np.array([[1, 2], [3, 4]], dtype=dtype)
                result = gaussian_blur(image, sigma=0, kernel_size=3)
                self.assertTrue(np.issubdtype(result.dtype, np.floating))
                np.testing.assert_array_equal(result, image)
                self.assertFalse(np.shares_memory(result, image))
                result[0, 0] = -100
                self.assertEqual(image[0, 0], 1)

    def test_sigma_zero_preserves_unclipped_values_and_accepts_lists(self):
        image = [[-5.5, 300.25]]
        np.testing.assert_array_equal(gaussian_blur(image, 0, 1), image)

    def test_smoothing_does_not_modify_the_input(self):
        image = np.arange(12, dtype=np.uint8).reshape(3, 4)
        original = image.copy()
        result = gaussian_blur(image, 1.2, 3)
        np.testing.assert_array_equal(image, original)
        self.assertFalse(np.shares_memory(result, image))

    def test_invalid_images_are_rejected_even_when_sigma_is_zero(self):
        for sigma in (0, 1):
            for image in ([], [[]], [1, 2], np.zeros((2, 2, 3)), [[np.nan]], [[np.inf]]):
                with self.subTest(sigma=sigma, image=repr(image)):
                    with self.assertRaisesRegex(ValueError, "image"):
                        gaussian_blur(image, sigma, 3)
            for image in ([["1"]], [[True]], [[1 + 0j]], [[None]]):
                with self.subTest(sigma=sigma, image=repr(image)):
                    with self.assertRaisesRegex(TypeError, "image"):
                        gaussian_blur(image, sigma, 3)

    def test_invalid_parameters_are_rejected_even_for_bypass(self):
        for sigma in (0, 1):
            for size in (0, 2, -1):
                with self.subTest(sigma=sigma, size=size):
                    with self.assertRaisesRegex(ValueError, "kernel_size"):
                        gaussian_blur([[1]], sigma, size)
            for size in (3.0, True, "3"):
                with self.subTest(sigma=sigma, size=size):
                    with self.assertRaisesRegex(TypeError, "kernel_size"):
                        gaussian_blur([[1]], sigma, size)
        for sigma in (-1, np.nan, np.inf, -np.inf):
            with self.subTest(sigma=sigma):
                with self.assertRaisesRegex(ValueError, "sigma"):
                    gaussian_blur([[1]], sigma, 3)
        for sigma in (True, "0", None, 1 + 0j):
            with self.subTest(sigma=repr(sigma)):
                with self.assertRaisesRegex(TypeError, "sigma"):
                    gaussian_blur([[1]], sigma, 3)


if __name__ == "__main__":
    unittest.main()
