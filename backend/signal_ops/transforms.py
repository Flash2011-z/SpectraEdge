"""Manual DFT/FFT calculations for finite real or complex arrays.

Adapted from the supplied Jan2026_CSE220_Offline_DFT_FFT assignment:
DFTAnalyzer's defining sums, FFTTransformer's iterative bit reversal and
butterflies, ArbitraryLengthFFT's Bluestein chirps, and image_conv.py's
row/column passes. Assignment applications and file handling are not included.

All public helpers return new complex128 arrays. Forward outputs and inverse
inputs use unshifted frequency order.
Vectors must be nonempty and 1D; planes must be nonempty and 2D. Invalid shapes
or non-finite values raise ValueError; nonnumeric, boolean, and object dtypes
raise TypeError. Unlike the image-facing fft2d(), these helpers accept complex
data, which is necessary for intermediate transforms and reconstruction.

Plans hold only one length and live within a calculation: there is no global
cache. The direct DFT is an independent O(N^2) reference; the image path always
uses the O(N log N) FFT algorithms below.
"""

import numpy as np


def _as_complex_array(values, dimensions: int, name: str) -> np.ndarray:
    """Validate before casting, so numeric strings cannot slip through."""
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a rectangular {dimensions}D numeric array") from error

    if array.ndim != dimensions or array.size == 0:
        raise ValueError(f"{name} must be a nonempty {dimensions}D array")
    if array.dtype.kind not in "iufc":
        raise TypeError(f"{name} must contain real or complex numeric values, not booleans")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")

    # A wider dtype might contain finite numbers too large for complex128.
    # Reject that conversion clearly instead of feeding infinities to the FFT.
    with np.errstate(over="ignore", invalid="ignore"):
        converted = array.astype(np.complex128, copy=False)
    if not np.isfinite(converted).all():
        raise ValueError(f"{name} values must remain finite when converted to complex128")
    return converted


def _direct_dft(values: np.ndarray, inverse: bool) -> np.ndarray:
    """Evaluate every defining sum independently of any FFT machinery."""
    length = values.size
    indices = np.arange(length, dtype=np.int64)
    sign = 1 if inverse else -1
    roots = np.exp(sign * 2j * np.pi * indices / length)
    result = np.empty(length, dtype=np.complex128)

    # Adapted from DFTAnalyzer: roots repeat every N samples, so (k*n) % N
    # indexes a table of N roots without storing an N-by-N matrix.
    for frequency in range(length):
        result[frequency] = np.sum(values * roots[(frequency * indices) % length])
    if inverse:
        result /= length
    return result


def dft1d(signal) -> np.ndarray:
    """Compute X[k] = sum_n x[n] exp(-2*pi*j*k*n/N) directly, in O(N^2)."""
    return _direct_dft(_as_complex_array(signal, 1, "signal"), inverse=False)


def idft1d(spectrum) -> np.ndarray:
    """Compute x[n] = sum_k X[k] exp(+2*pi*j*k*n/N) / N directly."""
    return _direct_dft(_as_complex_array(spectrum, 1, "spectrum"), inverse=True)


class _Radix2Plan:
    """One length's bit-reversal indices and reusable stage twiddle factors."""

    def __init__(self, length: int):
        if length < 1 or length & (length - 1):
            raise ValueError("radix-2 FFT length must be a positive power of two")

        # Adapted from FFTTransformer._plan. Reversing the index bits puts
        # samples in the order needed to combine length-2, length-4, ... DFTs.
        source = np.arange(length, dtype=np.int64)
        self.reversed_indices = np.zeros(length, dtype=np.int64)
        for _ in range(length.bit_length() - 1):
            self.reversed_indices = (self.reversed_indices << 1) | (source & 1)
            source >>= 1

        self.stages = []
        width = 2
        while width <= length:
            twiddles = np.exp(-2j * np.pi * np.arange(width // 2) / width)
            self.stages.append((width, twiddles))
            width *= 2

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Apply the plan to an already validated vector of this length."""
        result = values[self.reversed_indices].copy()

        # Adapted from FFTTransformer.transform. Each butterfly produces
        # E + W*O and E - W*O from the even/odd partial transforms.
        # The reshaped blocks share the same twiddle vector for each stage.
        for width, twiddles in self.stages:
            half = width // 2
            blocks = result.reshape(-1, width)
            even = blocks[:, :half].copy()
            odd_twiddled = blocks[:, half:] * twiddles
            blocks[:, :half] = even + odd_twiddled
            blocks[:, half:] = even - odd_twiddled
        return result


def _apply_plan(plan, values: np.ndarray, inverse: bool) -> np.ndarray:
    """Share exactly the same forward butterflies for both directions."""
    if inverse:
        # Conjugation changes the exponent's sign; 1/N restores the scale.
        return np.conjugate(plan.transform(np.conjugate(values))) / values.size
    return plan.transform(values)


class _BluesteinPlan:
    """One arbitrary length's chirp and pretransformed convolution kernel."""

    def __init__(self, length: int):
        self.length = length
        required_length = 2 * length - 1
        self.convolution_length = 1 << (required_length - 1).bit_length()
        self.radix_plan = _Radix2Plan(self.convolution_length)

        # Adapted from ArbitraryLengthFFT: 2*k*n = k^2 + n^2 - (k-n)^2.
        # Splitting the DFT exponential with this identity leaves a
        # convolution between x[n]*exp(-j*pi*n^2/N) and exp(+j*pi*n^2/N).
        indices = np.arange(length, dtype=np.int64)
        # The chirp repeats in its squared phase modulo 2N. Reducing before
        # exp avoids large angles and improves accuracy for long vectors.
        squared_phase = (indices * indices) % (2 * length)
        self.chirp = np.exp(-1j * np.pi * squared_phase / length)

        kernel = np.zeros(self.convolution_length, dtype=np.complex128)
        kernel[:length] = np.conjugate(self.chirp)
        # Differences k-n can be negative. Put those chirp samples at the
        # end of the circular buffer. M >= 2N-1 keeps required lags distinct.
        kernel[-(length - 1) :] = np.conjugate(self.chirp[1:][::-1])
        self.kernel_transform = self.radix_plan.transform(kernel)

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Return the exact N-point DFT, despite the larger internal buffer."""
        padded = np.zeros(self.convolution_length, dtype=np.complex128)
        padded[: self.length] = values * self.chirp
        product = self.radix_plan.transform(padded) * self.kernel_transform
        convolution = _apply_plan(self.radix_plan, product, inverse=True)
        return self.chirp * convolution[: self.length]


def _make_plan(length: int):
    if length & (length - 1) == 0:
        return _Radix2Plan(length)
    return _BluesteinPlan(length)


def fft_radix2(signal) -> np.ndarray:
    """Manual radix-2 FFT of a vector; reject lengths other than 1, 2, 4, ..."""
    values = _as_complex_array(signal, 1, "signal")
    return _apply_plan(_Radix2Plan(values.size), values, inverse=False)


def ifft_radix2(spectrum) -> np.ndarray:
    """Inverse radix-2 FFT with 1/N scaling and the same length restriction."""
    values = _as_complex_array(spectrum, 1, "spectrum")
    return _apply_plan(_Radix2Plan(values.size), values, inverse=True)


def fft1d(signal) -> np.ndarray:
    """Exact-length manual FFT: radix-2 for powers of two, Bluestein otherwise."""
    values = _as_complex_array(signal, 1, "signal")
    return _apply_plan(_make_plan(values.size), values, inverse=False)


def ifft1d(spectrum) -> np.ndarray:
    """Exact-length inverse FFT, retaining complex values and dividing by N."""
    values = _as_complex_array(spectrum, 1, "spectrum")
    return _apply_plan(_make_plan(values.size), values, inverse=True)


def _transform_2d(values: np.ndarray, inverse: bool) -> np.ndarray:
    height, width = values.shape
    # At most two top-level plans, reused for every row/column of this call.
    # A square image shares one plan for both axes. No state survives the call.
    plans = {length: _make_plan(length) for length in set(values.shape)}

    # Adapted from image_conv.transform_2d / inverse_2d: the 2D exponential
    # separates into horizontal and vertical factors. Keep both passes
    # complex, because a real image usually has complex row transforms.
    rows = np.empty(values.shape, dtype=np.complex128)
    for row in range(height):
        rows[row, :] = _apply_plan(plans[width], values[row, :], inverse)

    result = np.empty(values.shape, dtype=np.complex128)
    for column in range(width):
        result[:, column] = _apply_plan(plans[height], rows[:, column], inverse)
    return result


def transform_2d(plane) -> np.ndarray:
    """Unshifted manual 2D FFT of a real/complex plane, with no shape changes."""
    return _transform_2d(_as_complex_array(plane, 2, "plane"), inverse=False)


def inverse_2d(spectrum) -> np.ndarray:
    """Inverse 2D FFT; row/column factors give total scaling 1/(height*width).

    Accepts unshifted coefficients. The result stays complex: the caller
    decides whether dropping small roundoff imaginary parts is appropriate.
    """
    return _transform_2d(_as_complex_array(spectrum, 2, "spectrum"), inverse=True)
