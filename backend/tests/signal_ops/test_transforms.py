"""Independent sums and analytical checks for the manual transform core."""

import cmath
import unittest
from unittest.mock import patch

import numpy as np

from backend.signal_ops import (
    dft1d,
    fft1d,
    fft_radix2,
    idft1d,
    ifft1d,
    ifft_radix2,
    inverse_2d,
    transform_2d,
)
from backend.signal_ops import transforms


def defining_sum_1d(values, inverse=False):
    """Small test oracle: literal sums, no plans or periodic root table."""
    length = len(values)
    sign = 1 if inverse else -1
    scale = length if inverse else 1
    return np.array([
        sum(
            values[n] * cmath.exp(sign * 2j * cmath.pi * k * n / length)
            for n in range(length)
        ) / scale
        for k in range(length)
    ], dtype=np.complex128)


def defining_sum_2d(values, inverse=False):
    """Full 2D defining sum, independent of the row/column implementation."""
    height, width = values.shape
    sign = 1 if inverse else -1
    scale = values.size if inverse else 1
    expected = np.empty(values.shape, dtype=np.complex128)
    for u in range(height):
        for v in range(width):
            expected[u, v] = sum(
                values[y, x] * cmath.exp(
                    sign * 2j * cmath.pi * (u * y / height + v * x / width)
                )
                for y in range(height)
                for x in range(width)
            ) / scale
    return expected


class DirectDFTTests(unittest.TestCase):
    def test_hand_calculated_four_point_transform_and_inverse(self):
        values = [1, 2, 3, 4]
        expected = [10, -2 + 2j, -2, -2 - 2j]
        np.testing.assert_allclose(dft1d(values), expected, rtol=0, atol=1e-12)
        # Inverse is checked against a known spectrum, not just our forward output.
        np.testing.assert_allclose(idft1d(expected), values, rtol=0, atol=1e-12)

    def test_hand_calculated_complex_signal(self):
        values = [1 + 1j, 2 - 1j]
        expected = [3, -1 + 2j]
        np.testing.assert_allclose(dft1d(values), expected, rtol=0, atol=1e-12)
        np.testing.assert_allclose(idft1d(expected), values, rtol=0, atol=1e-12)

    def test_forward_and_inverse_match_literal_sums(self):
        for length in (1, 2, 3, 4, 5, 6, 7, 9):
            values = np.arange(length) / 3 + 1j * np.arange(length)[::-1] / 2
            for operation, inverse in ((dft1d, False), (idft1d, True)):
                with self.subTest(length=length, operation=operation.__name__):
                    expected = defining_sum_1d(values, inverse)
                    np.testing.assert_allclose(operation(values), expected, rtol=0, atol=1e-12)

    def test_direct_dft_does_not_use_fft_plans(self):
        with patch.object(transforms, "_make_plan", side_effect=AssertionError("FFT used")):
            np.testing.assert_allclose(dft1d([1, 2]), [3, -1], rtol=0, atol=1e-12)
            np.testing.assert_allclose(idft1d([3, -1]), [1, 2], rtol=0, atol=1e-12)


class FFTVectorTests(unittest.TestCase):
    def test_radix2_matches_direct_dft_for_real_and_complex_vectors(self):
        generator = np.random.default_rng(220)
        for length in (1, 2, 4, 8, 16, 32):
            real = generator.normal(size=length)
            for values in (real, real + 1j * generator.normal(size=length)):
                with self.subTest(length=length, complex=np.iscomplexobj(values)):
                    np.testing.assert_allclose(fft_radix2(values), dft1d(values), atol=1e-12)
                    np.testing.assert_allclose(ifft_radix2(values), idft1d(values), atol=1e-12)

    def test_arbitrary_fft_matches_direct_dft_without_changing_length(self):
        generator = np.random.default_rng(221)
        # Prime, odd composite, even composite, powers of two, and singleton.
        for length in (1, 2, 3, 5, 6, 7, 8, 9, 10, 15, 17, 25, 31):
            real = generator.normal(size=length)
            for values in (real, real + 1j * generator.normal(size=length)):
                with self.subTest(length=length, complex=np.iscomplexobj(values)):
                    result = fft1d(values)
                    self.assertEqual(result.shape, values.shape)
                    self.assertEqual(result.dtype, np.dtype(np.complex128))
                    np.testing.assert_allclose(result, dft1d(values), rtol=1e-12, atol=1e-12)
                    np.testing.assert_allclose(ifft1d(values), idft1d(values), atol=1e-12)

    def test_zero_constant_and_impulse_analytical_cases(self):
        for length in (1, 4, 7, 9):
            for operation in (dft1d, fft1d):
                with self.subTest(length=length, operation=operation.__name__):
                    np.testing.assert_allclose(operation(np.zeros(length)), 0, atol=1e-12)
                    expected = np.zeros(length, dtype=complex)
                    expected[0] = length * (2 - 3j)
                    np.testing.assert_allclose(
                        operation(np.full(length, 2 - 3j)), expected, atol=1e-12
                    )
                    impulse = np.zeros(length)
                    impulse[0] = 1
                    np.testing.assert_allclose(operation(impulse), np.ones(length), atol=1e-12)

    def test_shifted_impulse_preserves_phase(self):
        for length in (4, 7, 9):
            with self.subTest(length=length):
                impulse = np.zeros(length)
                impulse[2] = 1
                expected = np.exp(-2j * np.pi * np.arange(length) * 2 / length)
                result = fft1d(impulse)
                np.testing.assert_allclose(result, expected, rtol=0, atol=1e-12)
                np.testing.assert_allclose(np.abs(result), 1, rtol=0, atol=1e-12)

    def test_periodic_complex_tone_has_one_peak_with_correct_phase(self):
        for length in (8, 9, 11):
            with self.subTest(length=length):
                amplitude = 2 - 3j
                values = amplitude * np.exp(2j * np.pi * 2 * np.arange(length) / length)
                expected = np.zeros(length, dtype=complex)
                expected[2] = amplitude * length
                np.testing.assert_allclose(fft1d(values), expected, rtol=0, atol=1e-12)

    def test_inverse_known_coefficients_have_positive_phase_and_correct_scale(self):
        for operation in (idft1d, ifft_radix2, ifft1d):
            with self.subTest(operation=operation.__name__):
                np.testing.assert_allclose(operation([1, 1, 1, 1]), [1, 0, 0, 0], atol=1e-12)
                np.testing.assert_allclose(operation([8, 0, 0, 0]), [2, 2, 2, 2], atol=1e-12)
                np.testing.assert_allclose(operation([0, 4, 0, 0]), [1, 1j, -1, -1j], atol=1e-12)

    def test_forward_and_inverse_reconstruct_both_directions(self):
        generator = np.random.default_rng(222)
        for length in (1, 2, 4, 7, 9, 16, 23):
            values = generator.normal(size=length) + 1j * generator.normal(size=length)
            pairs = [(dft1d, idft1d), (fft1d, ifft1d)]
            if length & (length - 1) == 0:
                pairs.append((fft_radix2, ifft_radix2))
            for forward, inverse in pairs:
                with self.subTest(length=length, forward=forward.__name__):
                    np.testing.assert_allclose(inverse(forward(values)), values, atol=1e-12)
                    np.testing.assert_allclose(forward(inverse(values)), values, atol=1e-12)

    def test_radix2_rejects_non_power_of_two_lengths(self):
        for operation in (fft_radix2, ifft_radix2):
            for length in (3, 5, 6, 9, 12):
                with self.subTest(operation=operation.__name__, length=length):
                    with self.assertRaisesRegex(ValueError, "power of two"):
                        operation(np.ones(length))


class TransformPlanTests(unittest.TestCase):
    def test_bit_reversal_and_stage_twiddles_are_reused(self):
        with patch.object(transforms.np, "exp", wraps=np.exp) as exponentials:
            plan = transforms._Radix2Plan(8)
            np.testing.assert_array_equal(plan.reversed_indices, [0, 4, 2, 6, 1, 5, 3, 7])
            self.assertEqual([width for width, _ in plan.stages], [2, 4, 8])
            self.assertEqual(exponentials.call_count, 3)
            plan.transform(np.ones(8, dtype=complex))
            transforms._apply_plan(plan, np.ones(8, dtype=complex), inverse=True)
            self.assertEqual(exponentials.call_count, 3)

    def test_bluestein_internal_padding_and_chirp_spectrum_are_reused(self):
        for length in (3, 6, 9, 17):
            with self.subTest(length=length):
                plan = transforms._BluesteinPlan(length)
                padded_length = plan.convolution_length
                self.assertGreaterEqual(padded_length, 2 * length - 1)
                self.assertEqual(padded_length & (padded_length - 1), 0)
                self.assertLess(padded_length // 2, 2 * length - 1)
                kernel_before = plan.kernel_transform.copy()
                # Only the input chirp transform and inverse convolution are
                # evaluated each time: the kernel transform is already stored.
                with patch.object(plan.radix_plan, "transform", wraps=plan.radix_plan.transform) as run:
                    result = plan.transform(np.ones(length, dtype=complex))
                    self.assertEqual(run.call_count, 2)
                self.assertEqual(result.shape, (length,))
                np.testing.assert_array_equal(plan.kernel_transform, kernel_before)

    def test_power_of_two_lengths_bypass_bluestein(self):
        with patch.object(transforms, "_BluesteinPlan", side_effect=AssertionError("chirp used")):
            for length in (1, 2, 4, 8, 16):
                with self.subTest(length=length):
                    np.testing.assert_allclose(ifft1d(fft1d(np.ones(length))), 1, atol=1e-12)

    def test_fast_transform_paths_never_fall_back_to_direct_dft(self):
        with patch.object(transforms, "_direct_dft", side_effect=AssertionError("slow DFT used")):
            for length in (1, 4, 7, 9):
                values = np.arange(length)
                np.testing.assert_allclose(ifft1d(fft1d(values)), values, atol=1e-12)
            plane = np.ones((3, 5))
            np.testing.assert_allclose(inverse_2d(transform_2d(plane)), plane, atol=1e-12)

    def test_2d_calls_reuse_one_plan_per_unique_axis_length(self):
        for shape in ((4, 4), (3, 5), (1, 7), (1, 1)):
            for operation in (transform_2d, inverse_2d):
                with self.subTest(shape=shape, operation=operation.__name__):
                    with patch.object(transforms, "_make_plan", wraps=transforms._make_plan) as make:
                        operation(np.ones(shape))
                        self.assertEqual(
                            sorted(call.args[0] for call in make.call_args_list),
                            sorted(set(shape)),
                        )
                        make.reset_mock()
                        operation(np.ones(shape))
                        # A second calculation builds its own plans, not a global cache.
                        self.assertEqual(make.call_count, len(set(shape)))


class Transform2DTests(unittest.TestCase):
    def test_forward_and_inverse_match_independent_full_2d_sums(self):
        generator = np.random.default_rng(223)
        for shape in ((2, 4), (3, 5), (4, 3), (3, 6), (1, 5), (7, 1), (1, 1)):
            values = generator.normal(size=shape) + 1j * generator.normal(size=shape)
            for operation, inverse in ((transform_2d, False), (inverse_2d, True)):
                with self.subTest(shape=shape, operation=operation.__name__):
                    result = operation(values)
                    self.assertEqual(result.shape, shape)
                    self.assertEqual(result.dtype, np.dtype(np.complex128))
                    expected = defining_sum_2d(values, inverse)
                    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-12)

    def test_inverse_single_bin_gives_expected_complex_plane_wave(self):
        height, width = 3, 5
        spectrum = np.zeros((height, width), dtype=complex)
        spectrum[1, 2] = height * width
        y, x = np.indices(spectrum.shape)
        expected = np.exp(2j * np.pi * (y / height + 2 * x / width))
        np.testing.assert_allclose(inverse_2d(spectrum), expected, rtol=0, atol=1e-12)

    def test_complex_forward_inverse_round_trip(self):
        generator = np.random.default_rng(224)
        for shape in ((4, 8), (5, 7), (6, 9), (1, 7), (7, 1), (1, 1)):
            with self.subTest(shape=shape):
                values = generator.normal(size=shape) + 1j * generator.normal(size=shape)
                np.testing.assert_allclose(inverse_2d(transform_2d(values)), values, atol=1e-12)
                np.testing.assert_allclose(transform_2d(inverse_2d(values)), values, atol=1e-12)


class TransformInputTests(unittest.TestCase):
    vector_operations = (dft1d, idft1d, fft_radix2, ifft_radix2, fft1d, ifft1d)
    plane_operations = (transform_2d, inverse_2d)

    def test_vector_inputs_unchanged_and_outputs_are_complex128_copies(self):
        for operation in self.vector_operations:
            for dtype in (np.uint8, np.float32, np.float64, np.complex64, np.complex128):
                with self.subTest(operation=operation.__name__, dtype=dtype):
                    values = np.arange(4, dtype=dtype)
                    if np.iscomplexobj(values):
                        values += 2j
                    original = values.copy()
                    result = operation(values)
                    self.assertEqual(result.dtype, np.dtype(np.complex128))
                    self.assertFalse(np.shares_memory(result, values))
                    np.testing.assert_array_equal(values, original)
                    result[0] = -100
                    np.testing.assert_array_equal(values, original)

    def test_readonly_noncontiguous_vectors_and_planes_are_supported(self):
        backing = np.arange(48, dtype=complex).reshape(6, 8) + 2j
        original = backing.copy()
        vector = backing[0, ::2]
        plane = backing[::2, ::2]
        vector.setflags(write=False)
        plane.setflags(write=False)
        for values, operations in ((vector, self.vector_operations), (plane, self.plane_operations)):
            for operation in operations:
                with self.subTest(operation=operation.__name__):
                    result = operation(values)
                    self.assertEqual(result.shape, values.shape)
                    self.assertFalse(np.shares_memory(result, backing))
                    np.testing.assert_array_equal(backing, original)

    def test_singleton_complex_value_is_an_unchanged_copy(self):
        for operations, values in (
            (self.vector_operations, np.array([2 - 3j])),
            (self.plane_operations, np.array([[2 - 3j]])),
        ):
            for operation in operations:
                with self.subTest(operation=operation.__name__):
                    result = operation(values)
                    np.testing.assert_array_equal(result, values)
                    self.assertEqual(result.dtype, np.dtype(np.complex128))
                    self.assertFalse(np.shares_memory(result, values))

    def test_empty_and_wrong_dimensional_vectors_are_rejected(self):
        for operation in self.vector_operations:
            for values in ([], 5, [[1, 2]], np.zeros((1, 1, 1)), [[1], [2, 3]]):
                with self.subTest(operation=operation.__name__, values=repr(values)):
                    with self.assertRaisesRegex(ValueError, "1D"):
                        operation(values)

    def test_empty_and_wrong_dimensional_planes_are_rejected(self):
        for operation in self.plane_operations:
            for values in ([], [[]], np.empty((0, 3)), 5, [1, 2], np.zeros((2, 2, 3)), [[1], [2, 3]]):
                with self.subTest(operation=operation.__name__, values=repr(values)):
                    with self.assertRaisesRegex(ValueError, "2D"):
                        operation(values)

    def test_nonnumeric_boolean_and_object_inputs_are_rejected(self):
        for values in (["1"], [True], [None], np.array([1], dtype=object)):
            for operation in self.vector_operations:
                with self.subTest(operation=operation.__name__, values=repr(values)):
                    with self.assertRaisesRegex(TypeError, "numeric"):
                        operation(values)
            for operation in self.plane_operations:
                with self.subTest(operation=operation.__name__, values=repr(values)):
                    with self.assertRaisesRegex(TypeError, "numeric"):
                        operation([values])

    def test_nonfinite_real_and_imaginary_components_are_rejected(self):
        for value in (np.nan, np.inf, -np.inf, complex(1, np.nan), complex(1, np.inf)):
            for operation in self.vector_operations:
                with self.subTest(operation=operation.__name__, value=value):
                    with self.assertRaisesRegex(ValueError, "finite"):
                        operation([value])
            for operation in self.plane_operations:
                with self.subTest(operation=operation.__name__, value=value):
                    with self.assertRaisesRegex(ValueError, "finite"):
                        operation([[value]])


if __name__ == "__main__":
    unittest.main()
