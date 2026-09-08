"""Numerical checks for convolution, padding, and input validation."""

import unittest

import numpy as np

from backend.signal_ops import convolve2d


class Convolve2DTests(unittest.TestCase):
    def test_asymmetric_kernel_uses_convolution_not_correlation(self):
        image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        kernel = np.array([[1, 2, 0], [0, 0, 0], [0, 0, 0]])

        # The flipped kernel has bottom row [0, 2, 1]. At the centre,
        # convolution is 2*8 + 1*9 = 25; unflipped correlation would give 5.
        # Reflect padding supplies the neighbours for the other positions.
        expected = np.array([[13, 16, 17], [22, 25, 26], [13, 16, 17]], dtype=float)

        result = convolve2d(image, kernel)
        np.testing.assert_array_equal(result, expected)
        self.assertNotEqual(result[1, 1], 5)

    def test_reflection_padding_and_same_output_shape(self):
        image = np.array([[1, 2], [3, 4]])

        # NumPy's reflect mode does not repeat the boundary sample:
        # padded image = [[4,3,4,3], [2,1,2,1], [4,3,4,3], [2,1,2,1]].
        # Summing each 3x3 neighbourhood gives these hand-calculated totals.
        expected = np.array([[27, 24], [21, 18]], dtype=float)

        result = convolve2d(image, np.ones((3, 3)))
        self.assertEqual(result.shape, image.shape)
        np.testing.assert_array_equal(result, expected)

    def test_negative_response_is_preserved_for_unsigned_image(self):
        image = np.array([[30, 20, 10]], dtype=np.uint8)
        result = convolve2d(image, [[1, 0, -1]])
        np.testing.assert_array_equal(result, [[0.0, -20.0, 0.0]])
        self.assertTrue(np.issubdtype(result.dtype, np.floating))

    def test_values_above_255_are_not_clipped_or_wrapped(self):
        image = np.array([[200, 255]], dtype=np.uint8)
        np.testing.assert_array_equal(convolve2d(image, [[2]]), [[400.0, 510.0]])

    def test_fractional_coefficients_and_input_lists(self):
        result = convolve2d([[1, 2], [3, 4]], [[0.25]])
        np.testing.assert_array_equal(result, [[0.25, 0.5], [0.75, 1.0]])
        self.assertTrue(np.issubdtype(result.dtype, np.floating))

    def test_rectangular_kernel_and_single_column_image(self):
        result = convolve2d([[1], [2], [4]], [[1], [0], [-1]])
        np.testing.assert_array_equal(result, [[0.0], [3.0], [0.0]])

    def test_kernel_larger_than_single_pixel_image(self):
        result = convolve2d([[5]], np.ones((3, 5)))
        self.assertEqual(result.shape, (1, 1))
        np.testing.assert_array_equal(result, [[75.0]])

    def test_inputs_are_unchanged_and_output_is_independent(self):
        image = np.array([[1.0, 2.0], [3.0, 4.0]])
        kernel = np.array([[1.0]])
        original_image = image.copy()
        original_kernel = kernel.copy()

        result = convolve2d(image, kernel)
        np.testing.assert_array_equal(image, original_image)
        np.testing.assert_array_equal(kernel, original_kernel)
        self.assertFalse(np.shares_memory(result, image))
        self.assertFalse(np.shares_memory(result, kernel))

    def test_invalid_shapes_and_empty_arrays(self):
        for invalid in ([], [[]], 5, [1, 2], np.zeros((2, 2, 1)), [[1], [2, 3]]):
            with self.subTest(argument="image", value=repr(invalid)):
                with self.assertRaisesRegex(ValueError, "image"):
                    convolve2d(invalid, [[1]])
            with self.subTest(argument="kernel", value=repr(invalid)):
                with self.assertRaisesRegex(ValueError, "kernel"):
                    convolve2d([[1]], invalid)

    def test_even_kernel_dimensions_are_rejected(self):
        for shape in ((2, 3), (3, 2), (2, 2), (1, 2), (2, 1)):
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(ValueError, "odd"):
                    convolve2d([[1]], np.ones(shape))

    def test_nonreal_and_nonnumeric_arrays_are_rejected(self):
        for invalid in ([["1"]], [[True]], [[1 + 0j]], [[None]], np.array([[1]], dtype=object)):
            with self.subTest(argument="image", value=repr(invalid)):
                with self.assertRaisesRegex(TypeError, "image"):
                    convolve2d(invalid, [[1]])
            with self.subTest(argument="kernel", value=repr(invalid)):
                with self.assertRaisesRegex(TypeError, "kernel"):
                    convolve2d([[1]], invalid)

    def test_nonfinite_values_are_rejected(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(argument="image", value=value):
                with self.assertRaisesRegex(ValueError, "image.*finite"):
                    convolve2d([[value]], [[1]])
            with self.subTest(argument="kernel", value=value):
                with self.assertRaisesRegex(ValueError, "kernel.*finite"):
                    convolve2d([[1]], [[value]])


if __name__ == "__main__":
    unittest.main()
