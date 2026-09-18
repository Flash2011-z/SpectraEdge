"""Tests for reproducible grayscale noise operations."""

import unittest

import numpy as np

from backend.signal_ops import add_gaussian_noise, add_salt_pepper_noise


class NoiseTests(unittest.TestCase):
    def test_same_seed_is_deterministic_and_different_seed_changes_output(self):
        image = np.full((30, 40), 128.0)
        for operation, strength in ((add_gaussian_noise, 12), (add_salt_pepper_noise, 0.2)):
            with self.subTest(operation=operation.__name__):
                first = operation(image, strength, 220)
                np.testing.assert_array_equal(first, operation(image, strength, 220))
                self.assertFalse(np.array_equal(first, operation(image, strength, 221)))

    def test_shape_float_output_and_input_are_preserved(self):
        for operation, strength in ((add_gaussian_noise, 12), (add_salt_pepper_noise, 0.3)):
            with self.subTest(operation=operation.__name__):
                image = np.arange(20, dtype=np.uint8).reshape(4, 5)
                original = image.copy()
                result = operation(image, strength, 10)
                self.assertEqual(result.shape, image.shape)
                self.assertTrue(np.issubdtype(result.dtype, np.floating))
                self.assertFalse(np.shares_memory(result, image))
                np.testing.assert_array_equal(image, original)

    def test_gaussian_noise_has_expected_large_sample_statistics_and_is_unclipped(self):
        image = np.full((600, 700), 128.0)
        difference = add_gaussian_noise(image, 15, 42) - image
        self.assertAlmostEqual(float(difference.mean()), 0, delta=0.1)
        self.assertAlmostEqual(float(difference.std()), 15, delta=0.1)
        high = add_gaussian_noise(np.full((100, 100), 255.0), 20, 42)
        self.assertGreater(float(high.max()), 255)

    def test_salt_pepper_only_replaces_pixels_with_intensity_extremes(self):
        image = np.full((200, 250), 127.0)
        result = add_salt_pepper_noise(image, 0.25, 99)
        self.assertTrue(set(np.unique(result)) <= {0.0, 127.0, 255.0})
        corrupted = np.count_nonzero(result != image) / image.size
        self.assertAlmostEqual(corrupted, 0.25, delta=0.01)
        self.assertGreater(np.count_nonzero(result == 0), 0)
        self.assertGreater(np.count_nonzero(result == 255), 0)
        np.testing.assert_array_equal(
            add_salt_pepper_noise(image, 1, 1) != 127,
            np.ones(image.shape, dtype=bool),
        )

    def test_zero_strength_returns_an_independent_equal_array(self):
        image = np.arange(12, dtype=float).reshape(3, 4)
        for operation in (add_gaussian_noise, add_salt_pepper_noise):
            result = operation(image, 0, 4)
            np.testing.assert_array_equal(result, image)
            self.assertFalse(np.shares_memory(result, image))

    def test_invalid_strength_seed_and_images_are_rejected(self):
        for value in (-1, np.nan, np.inf, -np.inf):
            with self.subTest(gaussian_strength=value):
                with self.assertRaises(ValueError):
                    add_gaussian_noise([[1]], value, 0)
        for value in (-0.1, 1.1, np.nan, np.inf):
            with self.subTest(salt_pepper_strength=value):
                with self.assertRaises(ValueError):
                    add_salt_pepper_noise([[1]], value, 0)
        for value in (True, "1", None, 1 + 0j):
            with self.subTest(strength=repr(value)):
                with self.assertRaises(TypeError):
                    add_gaussian_noise([[1]], value, 0)
        for seed in (-1, 2**32):
            with self.subTest(seed=seed):
                with self.assertRaises(ValueError):
                    add_gaussian_noise([[1]], 1, seed)
        for seed in (True, 1.5, "1", None):
            with self.subTest(seed=repr(seed)):
                with self.assertRaises(TypeError):
                    add_gaussian_noise([[1]], 1, seed)
        for image in ([], [1, 2], [[[1]]], [[np.nan]]):
            with self.subTest(image=repr(image)):
                with self.assertRaises((TypeError, ValueError)):
                    add_salt_pepper_noise(image, 0.1, 0)


if __name__ == "__main__":
    unittest.main()
