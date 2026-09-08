"""Analytical Fourier cases, spectrum scaling, and shared input validation."""

import unittest
from unittest.mock import patch

import numpy as np

from backend.signal_ops import dft1d, fft2d, fft_spectrum, gaussian_blur, inverse_2d


class FFT2DTests(unittest.TestCase):
    def test_hand_calculated_transform_is_complex_and_unshifted(self):
        # For length two, the DFT takes the sum and difference. Applying
        # this along both axes gives DC=10, horizontal=-2, and vertical=-4.
        result = fft2d([[1, 2], [3, 4]])
        expected = np.array([[10, -2], [-4, 0]], dtype=complex)
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-12)
        self.assertTrue(np.issubdtype(result.dtype, np.complexfloating))

    def test_zero_image_has_zero_transform(self):
        result = fft2d(np.zeros((3, 5)))
        np.testing.assert_array_equal(result, np.zeros((3, 5), dtype=complex))

    def test_constant_image_has_only_unshifted_dc(self):
        for shape in ((4, 6), (5, 7)):
            with self.subTest(shape=shape):
                image = np.full(shape, 3.0)
                expected = np.zeros(shape, dtype=complex)
                expected[0, 0] = 3 * image.size
                np.testing.assert_allclose(fft2d(image), expected, rtol=0, atol=1e-12)

    def test_origin_unit_impulse_has_all_one_coefficients(self):
        image = np.zeros((5, 7))
        image[0, 0] = 1
        np.testing.assert_allclose(fft2d(image), np.ones(image.shape), rtol=0, atol=1e-12)

    def test_offset_unit_impulse_has_unit_magnitude_and_expected_phase(self):
        height, width = 5, 7
        image = np.zeros((height, width))
        image[1, 2] = 1
        u, v = np.indices(image.shape)

        # Only the sample at (1, 2) contributes to the defining DFT sum.
        expected = np.exp(-2j * np.pi * (u / height + 2 * v / width))
        result = fft2d(image)
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-12)
        np.testing.assert_allclose(np.abs(result), np.ones(image.shape), rtol=0, atol=1e-12)

    def test_periodic_sine_has_expected_signed_frequency_coefficients(self):
        height, width = 5, 7
        y, x = np.indices((height, width))
        image = np.sin(2 * np.pi * (y / height + 2 * x / width))

        # sin(theta) = (exp(j*theta) - exp(-j*theta)) / (2j).
        # Integer periods give only two bins, each of magnitude H*W/2.
        expected = np.zeros(image.shape, dtype=complex)
        expected[1, 2] = -0.5j * image.size
        expected[-1, -2] = 0.5j * image.size
        np.testing.assert_allclose(fft2d(image), expected, rtol=0, atol=1e-12)

    def test_rectangular_non_power_of_two_and_singleton_shapes_round_trip(self):
        for shape in ((3, 5), (6, 10), (7, 11), (1, 7), (7, 1), (1, 1)):
            with self.subTest(shape=shape):
                image = np.arange(np.prod(shape), dtype=float).reshape(shape) - 2.5
                transform = fft2d(image)
                self.assertEqual(transform.shape, shape)
                self.assertEqual(fft_spectrum(image).shape, shape)
                # Independent defining sums verify the forward result too;
                # a forward/inverse pair alone could hide matching mistakes.
                expected_rows = np.array([dft1d(row) for row in image])
                expected = np.array([dft1d(column) for column in expected_rows.T]).T
                np.testing.assert_allclose(transform, expected, rtol=1e-12, atol=1e-12)
                restored = inverse_2d(transform)
                np.testing.assert_allclose(restored, image, rtol=0, atol=1e-12)


class FFTSpectrumTests(unittest.TestCase):
    def test_zero_image_has_zero_floating_point_spectrum(self):
        image = np.zeros((3, 5), dtype=np.uint8)
        result = fft_spectrum(image)
        np.testing.assert_array_equal(result, np.zeros(image.shape))
        self.assertTrue(np.issubdtype(result.dtype, np.floating))

    def test_constant_image_dc_is_centred_for_even_and_odd_dimensions(self):
        for shape in ((4, 6), (5, 7), (4, 7), (1, 1)):
            for value in (-3.0, 3.0):
                with self.subTest(shape=shape, value=value):
                    image = np.full(shape, value)
                    expected = np.zeros(shape)
                    expected[shape[0] // 2, shape[1] // 2] = np.log1p(abs(value) * image.size)
                    np.testing.assert_allclose(fft_spectrum(image), expected, rtol=0, atol=1e-12)

    def test_unit_impulse_has_uniform_log_magnitude(self):
        image = np.zeros((5, 7))
        image[1, 2] = 1
        expected = np.full(image.shape, np.log(2))
        np.testing.assert_allclose(fft_spectrum(image), expected, rtol=0, atol=1e-12)

    def test_periodic_sine_peaks_are_at_expected_shifted_locations(self):
        height, width = 5, 7
        y, x = np.indices((height, width))
        image = np.sin(2 * np.pi * (y / height + 2 * x / width))
        expected = np.zeros(image.shape)

        # Centring moves unshifted bins (1, 2) and (4, 5) by (2, 3),
        # wrapping at the dimensions: the peaks land at (3, 5) and (1, 1).
        expected[3, 5] = np.log1p(image.size / 2)
        expected[1, 1] = np.log1p(image.size / 2)
        np.testing.assert_allclose(fft_spectrum(image), expected, rtol=0, atol=1e-12)

    def test_magnitudes_are_not_independently_normalized_or_quantized(self):
        small = fft_spectrum(np.full((3, 5), 2.0))
        large = fft_spectrum(np.full((3, 5), 300.0))
        self.assertAlmostEqual(float(small[1, 2]), np.log1p(30), places=12)
        self.assertAlmostEqual(float(large[1, 2]), np.log1p(4500), places=12)
        self.assertGreater(large[1, 2], small[1, 2])
        self.assertTrue(np.issubdtype(large.dtype, np.floating))

    def test_spectrum_calls_fft2d(self):
        image = np.array([[1.0]])
        with patch("backend.signal_ops.fourier.fft2d", return_value=np.array([[3 + 4j]])) as transform:
            result = fft_spectrum(image)
        transform.assert_called_once()
        self.assertIs(transform.call_args.args[0], image)
        np.testing.assert_allclose(result, [[np.log1p(5)]], rtol=0, atol=1e-12)

    def test_existing_gaussian_blur_outputs_are_accepted(self):
        image = np.full((5, 7), 12, dtype=np.uint8)
        smoothed = gaussian_blur(image, sigma=1.2, kernel_size=3)
        np.testing.assert_allclose(fft_spectrum(smoothed), fft_spectrum(image), rtol=0, atol=1e-12)

    def test_centring_moves_every_bin_correctly_for_odd_and_even_shapes(self):
        for shape in ((4, 6), (5, 7), (4, 7), (1, 7), (7, 1)):
            with self.subTest(shape=shape):
                height, width = shape
                coefficients = np.arange(1, height * width + 1).reshape(shape) * (1 + 1j)
                expected = np.empty(shape)
                for row in range(height):
                    for column in range(width):
                        expected[(row + height // 2) % height, (column + width // 2) % width] = (
                            np.log1p(abs(coefficients[row, column]))
                        )
                with patch("backend.signal_ops.fourier.fft2d", return_value=coefficients):
                    result = fft_spectrum(np.zeros(shape))
                np.testing.assert_allclose(result, expected, rtol=0, atol=1e-12)

    def test_gaussian_smoothed_impulse_reduces_outer_spectrum_without_rescaling(self):
        image = np.zeros((7, 9))
        image[3, 4] = 1
        # The impulse is far from every border, so its smoothed weights
        # still sum to one. DC is unchanged while high frequencies weaken.
        smoothed = gaussian_blur(image, sigma=1.2, kernel_size=3)
        original_spectrum = fft_spectrum(image)
        smoothed_spectrum = fft_spectrum(smoothed)
        self.assertAlmostEqual(float(smoothed_spectrum[3, 4]), np.log(2), places=12)
        self.assertLess(smoothed_spectrum[0, 0], original_spectrum[0, 0])
        self.assertLessEqual(float(smoothed_spectrum.max()), float(original_spectrum.max()) + 1e-12)


class FourierInputTests(unittest.TestCase):
    def test_inputs_are_unchanged_and_results_have_independent_storage(self):
        for operation in (fft2d, fft_spectrum):
            for dtype in (np.uint8, np.int16, np.float32, np.float64):
                with self.subTest(operation=operation.__name__, dtype=dtype):
                    image = np.arange(15, dtype=dtype).reshape(3, 5)
                    original = image.copy()
                    result = operation(image)
                    np.testing.assert_array_equal(image, original)
                    self.assertFalse(np.shares_memory(result, image))
                    result[0, 0] = -100
                    np.testing.assert_array_equal(image, original)

    def test_read_only_noncontiguous_input_is_supported(self):
        backing = np.arange(48, dtype=float).reshape(6, 8)
        original = backing.copy()
        image = backing[::2, ::2]
        image.setflags(write=False)
        for operation in (fft2d, fft_spectrum):
            with self.subTest(operation=operation.__name__):
                result = operation(image)
                self.assertEqual(result.shape, image.shape)
                np.testing.assert_array_equal(backing, original)

    def test_empty_non_2d_and_ragged_inputs_are_rejected(self):
        invalid_images = ([], [[]], np.empty((0, 3)), 5, [1, 2], np.zeros((2, 2, 3)), [[1], [2, 3]])
        for operation in (fft2d, fft_spectrum):
            for image in invalid_images:
                with self.subTest(operation=operation.__name__, image=repr(image)):
                    with self.assertRaisesRegex(ValueError, "image.*2D"):
                        operation(image)

    def test_nonreal_and_nonnumeric_inputs_are_rejected(self):
        invalid_images = ([["1"]], [[True]], [[1 + 0j]], [[None]], np.array([[1]], dtype=object))
        for operation in (fft2d, fft_spectrum):
            for image in invalid_images:
                with self.subTest(operation=operation.__name__, image=repr(image)):
                    with self.assertRaisesRegex(TypeError, "image.*real numeric"):
                        operation(image)

    def test_nonfinite_values_are_rejected(self):
        for operation in (fft2d, fft_spectrum):
            for value in (np.nan, np.inf, -np.inf):
                with self.subTest(operation=operation.__name__, value=value):
                    with self.assertRaisesRegex(ValueError, "image.*finite"):
                        operation([[1, value]])


if __name__ == "__main__":
    unittest.main()
