"""Gaussian kernel construction and smoothing using manual convolution."""

from numbers import Real

import numpy as np

from .convolution import _as_float_2d, convolve2d


def _validate_kernel_size(kernel_size) -> int:
    if isinstance(kernel_size, (bool, np.bool_)) or not isinstance(kernel_size, (int, np.integer)):
        raise TypeError("kernel_size must be a positive odd integer")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    return int(kernel_size)


def _validate_sigma(sigma, allow_zero: bool = False) -> float:
    if isinstance(sigma, (bool, np.bool_)) or not isinstance(sigma, Real):
        raise TypeError("sigma must be a real numeric scalar")
    try:
        sigma = float(sigma)
    except OverflowError as error:
        raise ValueError("sigma must be finite") from error

    if not np.isfinite(sigma):
        raise ValueError("sigma must be finite")
    if sigma < 0 or (sigma == 0 and not allow_zero):
        requirement = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"sigma must be {requirement}")
    return sigma


def gaussian_kernel(kernel_size, sigma) -> np.ndarray:
    """Return a square, centred Gaussian kernel whose coefficients sum to 1.

    kernel_size must be a positive odd integer, and sigma must be a positive
    finite real scalar. Sigma is measured in pixel units. The returned array
    uses float64 coefficients.

    Raises:
        TypeError: A parameter has the wrong type (including boolean values).
        ValueError: The size is not positive and odd, or sigma is not positive and finite.
    """
    kernel_size = _validate_kernel_size(kernel_size)
    sigma = _validate_sigma(sigma)

    radius = kernel_size // 2
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    x, y = np.meshgrid(coordinates, coordinates)

    # Sample exp(-(x^2 + y^2) / (2*sigma^2)) about (0, 0). The continuous
    # prefactor 1/(2*pi*sigma^2) cancels when we normalise the discrete weights.
    # Dividing coordinates by sigma first avoids underflow of sigma^2 for
    # tiny sigma. In that limit, off-centre weights may safely round to zero;
    # the centre remains exp(0) = 1, so the sum cannot become zero.
    with np.errstate(over="ignore", under="ignore"):
        kernel = np.exp(-0.5 * ((x / sigma) ** 2 + (y / sigma) ** 2))

    # A unit sum gives a weighted average and preserves constant brightness.
    kernel /= kernel.sum()
    return kernel


def gaussian_blur(image, sigma, kernel_size) -> np.ndarray:
    """Smooth a real, finite 2D image, or return a float copy when sigma is 0.

    The image requirements are the same as convolve2d(). kernel_size must be
    a positive odd integer even when smoothing is disabled. Negative or
    non-finite sigma is rejected. Neither input values nor output responses
    are clipped to the 0..255 display range.

    Raises:
        TypeError: An input has an unsupported type.
        ValueError: The image, size, or sigma has an invalid value or shape.
    """
    kernel_size = _validate_kernel_size(kernel_size)
    sigma = _validate_sigma(sigma, allow_zero=True)

    if sigma == 0:
        # Validate here too: disabling smoothing must not bypass input checks.
        return _as_float_2d(image, "image").copy()

    kernel = gaussian_kernel(kernel_size, sigma)
    return convolve2d(image, kernel)
