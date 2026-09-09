"""Manual Sobel derivatives of a real 2D intensity image."""

import numpy as np

from backend.signal_ops import convolve2d
from backend.signal_ops.convolution import _as_float_2d


def sobel(image) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return signed Gx, signed Gy, and sqrt(Gx**2 + Gy**2), in raw units.

    Accept the same nonempty, real, finite 2D arrays as convolve2d(). Outputs
    are floating point, have the input shape, and do not share input storage.
    No intensity range is imposed and no responses are clipped or normalized.
    """
    # These are CONVOLUTION kernels. convolve2d reverses both axes exactly
    # once, so Gx measures right minus left, and Gy bottom minus top.
    # The [1, 2, 1] weights smooth in the perpendicular direction.
    kernel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=float)
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=float)

    values = _as_float_2d(image, "image")
    # Both kernels sum to zero, so a constant brightness offset contributes
    # nothing. Removing it first makes a flat fractional Gaussian output give
    # EXACT zero, without a tolerance that could erase genuine weak gradients.
    centered = values - values[0, 0]
    gx = convolve2d(centered, kernel_x)
    gy = convolve2d(centered, kernel_y)
    # The Euclidean length combines horizontal and vertical change without
    # cancelling opposite signs. Keep raw values for subsequent thresholding.
    magnitude = np.sqrt(gx**2 + gy**2)
    return gx, gy, magnitude
