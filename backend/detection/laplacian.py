"""Signed second derivative and a separate raw-response zero-crossing decision."""

from numbers import Real

import numpy as np

from backend.signal_ops import convolve2d
from backend.signal_ops.convolution import _as_float_2d


def laplacian(image) -> np.ndarray:
    """Return the signed four-neighbour Laplacian in raw intensity units.

    Kernel: [[0,1,0],[1,-4,1],[0,1,0]], i.e. neighbour sum minus four
    times the centre. Positive responses indicate a centre darker than its
    neighbours; negative responses indicate a brighter centre. Uses existing
    manual convolution with reflect padding. Accepts nonempty, real, finite
    2D arrays and returns a new same-shape float array without modifying input,
    clipping, taking absolute values, thresholding, or display conversion.
    """
    values = _as_float_2d(image, "image")
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=float)
    return convolve2d(values - values[0, 0], kernel)


def zero_crossing_edges(response, contrast_threshold) -> np.ndarray:
    """Return a new same-shape uint8 mask (0 or 255) from signed response.

    Use four-connectivity: horizontal and vertical neighbours, no diagonals
    and no wrapping/padding at borders. Mark BOTH endpoints of each strictly
    opposite-sign pair if abs(a-b) > contrast_threshold. Equality is rejected.
    Exact zeros have no sign: mark a zero centre only if its immediate left/
    right or up/down neighbours have opposite signs and exceed the threshold.
    Wider zero plateaus are not bridged. No tolerance rounds weak signs to zero.

    The threshold is a finite nonnegative real scalar in raw Laplacian response
    difference units, with no upper bound. It is not a display-brightness cutoff.
    Response must be nonempty, real, finite and 2D; input remains unchanged.
    This decision does not perform thinning, contour tracing or object analysis.
    """
    values = _as_float_2d(response, "response")
    if isinstance(contrast_threshold, (bool, np.bool_)) or not isinstance(contrast_threshold, Real):
        raise TypeError("contrast_threshold must be a real numeric scalar")
    try:
        contrast_threshold = float(contrast_threshold)
    except OverflowError as error:
        raise ValueError("contrast_threshold must be finite") from error
    if not np.isfinite(contrast_threshold):
        raise ValueError("contrast_threshold must be finite")
    if contrast_threshold < 0:
        raise ValueError("contrast_threshold must be nonnegative")

    def crosses(a, b):
        opposite = ((a < 0) & (b > 0)) | ((a > 0) & (b < 0))
        return opposite & (np.abs(a - b) > contrast_threshold)

    edges = np.zeros(values.shape, dtype=bool)
    horizontal = crosses(values[:, :-1], values[:, 1:])
    edges[:, :-1] |= horizontal
    edges[:, 1:] |= horizontal
    vertical = crosses(values[:-1, :], values[1:, :])
    edges[:-1, :] |= vertical
    edges[1:, :] |= vertical
    edges[:, 1:-1] |= (values[:, 1:-1] == 0) & crosses(values[:, :-2], values[:, 2:])
    edges[1:-1, :] |= (values[1:-1, :] == 0) & crosses(values[:-2, :], values[2:, :])
    return edges.astype(np.uint8) * 255
