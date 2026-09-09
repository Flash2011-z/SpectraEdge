"""Binary selection in raw gradient-magnitude units, not display values."""

from numbers import Real

import numpy as np

from backend.signal_ops.convolution import _as_float_2d


def threshold_edges(magnitude, threshold) -> np.ndarray:
    """Return 255 where magnitude > threshold, otherwise 0 (a new uint8 mask).

    Magnitude must be a nonempty, finite, nonnegative real 2D array. Threshold
    must be a finite, nonnegative real scalar. The numerical helper has no
    upper bound: the website's 0..1443 range is specific to 8-bit Sobel input.
    """
    values = _as_float_2d(magnitude, "magnitude")
    if np.any(values < 0):
        raise ValueError("magnitude must be nonnegative")
    if isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, Real):
        raise TypeError("threshold must be a real numeric scalar")
    try:
        threshold = float(threshold)
    except OverflowError as error:
        raise ValueError("threshold must be finite") from error
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if threshold < 0:
        raise ValueError("threshold must be nonnegative")

    # Strict comparison rejects equality, including zero gradients when the
    # threshold is zero. Quantization here encodes a binary decision only;
    # the signed derivatives and magnitude are never converted to uint8.
    return (values > threshold).astype(np.uint8) * 255
