"""Manual Prewitt derivatives of a real 2D intensity image."""

import numpy as np

from backend.signal_ops import convolve2d
from backend.signal_ops.convolution import _as_float_2d


def prewitt(image) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return signed Gx, signed Gy, and sqrt(Gx**2 + Gy**2), in raw units.

    Convolution kernels are [[1,0,-1],[1,0,-1],[1,0,-1]] for X and
    [[1,1,1],[0,0,0],[-1,-1,-1]] for Y. convolve2d flips them once:
    Gx measures right minus left, Gy bottom minus top, as in Sobel.
    Uniform perpendicular weights give a response of 6 to a unit ramp.

    Accept nonempty, real, finite 2D arrays. Reflect padding preserves shape.
    All three outputs are new floating-point arrays; input is never modified.
    No clipping, normalization, thresholding, or display conversion occurs.
    """
    kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=float)
    kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=float)
    values = _as_float_2d(image, "image")
    # Zero-sum kernels ignore a constant offset; centering ensures exact zero
    # for flat fractional inputs without suppressing weak gradients.
    centered = values - values[0, 0]
    gx = convolve2d(centered, kernel_x)
    gy = convolve2d(centered, kernel_y)
    return gx, gy, np.sqrt(gx**2 + gy**2)
