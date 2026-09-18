"""Independent Gaussian/detector passes and persistence-based edge fusion.

Every scale starts from the same original grayscale array. Gaussian support is
chosen as 2*ceil(3*sigma)+1 (at least 3, at most 31), matching the project's UI
guidance. Sigma zero uses the existing no-blur path with a validated 3x3 size.

Persistence is the integer number of scale edge masks that select each pixel.
The fused edge map is 255 exactly where persistence >= support_count, otherwise
0. The default support is 2 when at least two scales are supplied, and 1 for a
single scale. Duplicate sigmas are rejected because repeated identical scales
would inflate persistence without adding evidence.

This is pixel-aligned fusion, not scale-space non-maximum suppression. Gaussian
smoothing can shift, widen, or remove responses, so nearby detections at
different pixels do not support each other. Threshold units remain detector
specific, and one raw threshold is used independently at every scale.
"""

from dataclasses import dataclass
from math import ceil
from numbers import Real

import numpy as np

from backend.detection import laplacian, prewitt, sobel, threshold_edges, zero_crossing_edges
from backend.signal_ops import gaussian_blur
from backend.signal_ops.convolution import _as_float_2d


SUPPORTED_DETECTORS = ("Sobel", "Prewitt", "Laplacian")
MAX_SIGMA = 5.0
MAX_SCALES = 5


@dataclass(frozen=True)
class ScaleResult:
    """Raw numerical result for one independently smoothed scale."""

    sigma: float
    kernel_size: int
    filtered: np.ndarray
    gx: np.ndarray | None
    gy: np.ndarray | None
    magnitude: np.ndarray | None
    laplacian_response: np.ndarray | None
    edge_map: np.ndarray


@dataclass(frozen=True)
class MultiScaleResult:
    """Per-scale results plus integer persistence and a binary fused mask."""

    sigma_values: tuple[float, ...]
    support_count: int
    scales: tuple[ScaleResult, ...]
    persistence_map: np.ndarray
    fused_edge_map: np.ndarray


def _validate_sigmas(sigmas) -> tuple[float, ...]:
    if isinstance(sigmas, (str, bytes)):
        raise TypeError("sigmas must be a sequence of real numeric values")
    try:
        supplied = tuple(sigmas)
    except TypeError as error:
        raise TypeError("sigmas must be a sequence of real numeric values") from error
    if not supplied or len(supplied) > MAX_SCALES:
        raise ValueError(f"sigmas must contain between 1 and {MAX_SCALES} values")

    validated = []
    for sigma in supplied:
        if isinstance(sigma, (bool, np.bool_)) or not isinstance(sigma, Real):
            raise TypeError("every sigma must be a real numeric scalar")
        try:
            value = float(sigma)
        except OverflowError as error:
            raise ValueError("every sigma must be finite") from error
        if not np.isfinite(value) or value < 0 or value > MAX_SIGMA:
            raise ValueError(f"every sigma must be finite and between 0 and {MAX_SIGMA:g}")
        validated.append(value)
    if len(set(validated)) != len(validated):
        raise ValueError("sigma values must be unique")
    return tuple(validated)


def _validate_threshold(threshold) -> float:
    if isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, Real):
        raise TypeError("threshold must be a real numeric scalar")
    try:
        value = float(threshold)
    except OverflowError as error:
        raise ValueError("threshold must be finite") from error
    if not np.isfinite(value) or value < 0:
        raise ValueError("threshold must be finite and nonnegative")
    return value


def _validate_support(support_count, scale_count: int) -> int:
    if support_count is None:
        return 1 if scale_count == 1 else 2
    if isinstance(support_count, (bool, np.bool_)) or not isinstance(support_count, (int, np.integer)):
        raise TypeError("support_count must be an integer")
    if support_count < 1 or support_count > scale_count:
        raise ValueError("support_count must be between 1 and the number of scales")
    return int(support_count)


def _kernel_size(sigma: float) -> int:
    return 3 if sigma == 0 else min(31, 2 * ceil(3 * sigma) + 1)


def multi_scale_edges(
    grayscale,
    detector: str,
    sigmas,
    threshold,
    support_count=None,
) -> MultiScaleResult:
    """Run independent scale passes and fuse coincident edge persistence.

    grayscale must be a nonempty finite real 2D array. detector is exactly
    ``Sobel``, ``Prewitt``, or ``Laplacian``. sigmas contains 1..5 unique values
    in [0, 5]. threshold is a finite nonnegative raw detector threshold.
    support_count is an integer in [1, number of scales], or uses the default
    described in this module's documentation.

    All returned arrays preserve input dimensions, use numerical (not display)
    values, and do not share input storage. The input is never modified.
    """
    values = _as_float_2d(grayscale, "grayscale")
    if detector not in SUPPORTED_DETECTORS:
        raise ValueError(f"detector must be one of {', '.join(SUPPORTED_DETECTORS)}")
    sigma_values = _validate_sigmas(sigmas)
    threshold = _validate_threshold(threshold)
    support_count = _validate_support(support_count, len(sigma_values))

    scale_results = []
    persistence = np.zeros(values.shape, dtype=np.uint8)
    for sigma in sigma_values:
        kernel_size = _kernel_size(sigma)
        filtered = gaussian_blur(values, sigma, kernel_size)
        if detector in ("Sobel", "Prewitt"):
            detector_function = sobel if detector == "Sobel" else prewitt
            gx, gy, magnitude = detector_function(filtered)
            response = None
            edges = threshold_edges(magnitude, threshold)
        else:
            response = laplacian(filtered)
            gx = gy = magnitude = None
            edges = zero_crossing_edges(response, threshold)
        persistence += (edges != 0).astype(np.uint8)
        scale_results.append(ScaleResult(
            sigma, kernel_size, filtered, gx, gy, magnitude, response, edges,
        ))

    fused = (persistence >= support_count).astype(np.uint8) * 255
    return MultiScaleResult(
        sigma_values, support_count, tuple(scale_results), persistence, fused,
    )
