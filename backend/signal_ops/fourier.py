"""Array-only Fourier transforms and centred log-magnitude spectra."""

import numpy as np

from .convolution import _as_float_2d
from .transforms import transform_2d


def fft2d(image) -> np.ndarray:
    """Return the unshifted complex 2D transform using our manual FFT.

    Accepts a nonempty, real, finite 2D numeric array or rectangular numeric
    list, using the same validation as convolve2d(). Boolean, complex, and
    object arrays are rejected. Calculations use floating-point input.

    The result has the original dimensions, including rectangular shapes and
    sizes that are not powers of two. The image is never modified or resized;
    only Bluestein's internal convolution buffers use padding.
    The forward transform is unnormalised: [0, 0] is the sum of
    the image values (the DC coefficient), not their mean.

    Raises:
        TypeError: The image contains a non-real or non-numeric data type.
        ValueError: The image is empty, not 2D, or contains non-finite values.
    """
    image_array = _as_float_2d(image, "image")

    # For height H and width W, each frequency coefficient is
    # F[u, v] = sum(image[y, x] * exp(-2*pi*j*(u*y/H + v*x/W))).
    # Row/column passes use radix-2 or Bluestein at each exact axis length.
    # Keep complex coefficients and the original frequency ordering here.
    return transform_2d(image_array)


def fft_spectrum(image) -> np.ndarray:
    """Return a centred floating-point log-magnitude spectrum of image.

    Calls fft2d(), so its input requirements and errors are the same. The
    output has the input shape and contains log(1 + magnitude), with no
    uint8 conversion, clipping, or display normalisation. Image encoding
    and shared display limits belong to the caller, not this calculation.
    """
    transform = fft2d(image)

    # Reorder bins so DC is at [height // 2, width // 2], even for odd sizes.
    # This changes only their positions, not their frequency coefficients.
    height, width = transform.shape
    shifted_transform = np.roll(transform, (height // 2, width // 2), axis=(0, 1))

    # Absolute value measures strength, discarding phase. log1p compresses
    # large magnitude differences and maps zero to zero without log(0).
    return np.log1p(np.abs(shifted_transform))
