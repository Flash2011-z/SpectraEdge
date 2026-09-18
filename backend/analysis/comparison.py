"""Shared-preprocessing comparison of the three implemented edge detectors.

The original grayscale signal is validated once and Gaussian-filtered once.
Sobel, Prewitt, and Laplacian then receive that same floating-point filtered
array. Their existing decision functions produce binary masks, and the existing
connected-component analysis measures each mask. This module contains no
detector, threshold, zero-crossing, or component-labeling implementation.

Sobel and Prewitt each receive a raw gradient-magnitude threshold. Laplacian
receives an independent raw zero-crossing response-difference threshold.

Per-detector processing time covers the detector response, edge decision, and
object analysis. It excludes shared Gaussian filtering and HTTP/PNG encoding.
"""

from dataclasses import dataclass
from numbers import Real
from time import perf_counter

import numpy as np

from backend.detection import laplacian, prewitt, sobel, threshold_edges, zero_crossing_edges
from backend.signal_ops import gaussian_blur
from backend.signal_ops.convolution import _as_float_2d

from .components import analyze_objects, filter_small_components


DEFAULT_LAPLACIAN_MIN_COMPONENT_AREA = 2


@dataclass(frozen=True)
class DetectorComparison:
    """Numerical edge mask, object measurements, timing, and decision metadata."""

    detector: str
    edge_map: np.ndarray
    object_list: list[dict]
    edge_pixel_count: int
    object_count: int
    average_object_area: float
    processing_time: float
    decision: str
    threshold_type: str
    threshold: float
    threshold_units: str
    threshold_label: str
    minimum_component_area: int | None


@dataclass(frozen=True)
class ComparisonResult:
    """One shared filtered signal and results for every implemented detector."""

    filtered: np.ndarray
    sobel: DetectorComparison
    prewitt: DetectorComparison
    laplacian: DetectorComparison


def _threshold_value(threshold) -> float:
    if isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, Real):
        raise TypeError("threshold must be a real numeric scalar")
    try:
        value = float(threshold)
    except OverflowError as error:
        raise ValueError("threshold must be finite") from error
    if not np.isfinite(value) or value < 0:
        raise ValueError("threshold must be finite and nonnegative")
    return value


def _measure(
    detector,
    filtered: np.ndarray,
    threshold: float,
    laplacian_min_component_area: int,
) -> DetectorComparison:
    started = perf_counter()
    if detector == "Sobel":
        magnitude = sobel(filtered)[2]
        edges = threshold_edges(magnitude, threshold)
        decision = "magnitude_threshold"
        threshold_type = "gradient_magnitude"
        units = "raw_gradient_magnitude"
        label = "Gradient magnitude threshold"
        minimum_component_area = None
    elif detector == "Prewitt":
        magnitude = prewitt(filtered)[2]
        edges = threshold_edges(magnitude, threshold)
        decision = "magnitude_threshold"
        threshold_type = "gradient_magnitude"
        units = "raw_gradient_magnitude"
        label = "Gradient magnitude threshold"
        minimum_component_area = None
    else:
        response = laplacian(filtered)
        edges = zero_crossing_edges(response, threshold)
        decision = "zero_crossing"
        threshold_type = "zero_crossing_contrast"
        units = "raw_response_difference"
        label = "Zero-crossing contrast threshold"
        minimum_component_area = laplacian_min_component_area
    analysis_edges = (filter_small_components(edges, laplacian_min_component_area)
                      if detector == "Laplacian" else edges)
    objects = analyze_objects(analysis_edges).objects
    edge_pixel_count = int(np.count_nonzero(edges))
    average_object_area = (sum(item["area"] for item in objects) / len(objects)
                           if objects else 0.0)
    elapsed = (perf_counter() - started) * 1000
    return DetectorComparison(
        detector, edges, objects, edge_pixel_count, len(objects), average_object_area,
        elapsed, decision, threshold_type, threshold, units, label,
        minimum_component_area,
    )


def compare_detectors(
    grayscale,
    sigma,
    kernel_size,
    sobel_threshold,
    prewitt_threshold,
    laplacian_contrast_threshold,
    laplacian_min_component_area=DEFAULT_LAPLACIAN_MIN_COMPONENT_AREA,
) -> ComparisonResult:
    """Filter once, then independently run and measure all three detectors.

    grayscale is a nonempty, finite, real 2D signal. Gaussian parameter rules
    are supplied by the existing ``gaussian_blur`` implementation. threshold is
    a finite nonnegative raw decision threshold. Arrays preserve the source
    shape, stay numerical, and never share writable storage with the input.
    Sobel and Prewitt thresholds are raw gradient magnitudes; the Laplacian
    threshold is a raw response difference across a zero crossing.
    """
    source = _as_float_2d(grayscale, "grayscale")
    sobel_threshold = _threshold_value(sobel_threshold)
    prewitt_threshold = _threshold_value(prewitt_threshold)
    laplacian_contrast_threshold = _threshold_value(laplacian_contrast_threshold)
    if (isinstance(laplacian_min_component_area, (bool, np.bool_)) or
            not isinstance(laplacian_min_component_area, (int, np.integer)) or
            laplacian_min_component_area < 1):
        raise ValueError("laplacian_min_component_area must be a positive integer")
    laplacian_min_component_area = int(laplacian_min_component_area)
    filtered = gaussian_blur(source, sigma, kernel_size)
    return ComparisonResult(
        filtered=filtered,
        sobel=_measure("Sobel", filtered, sobel_threshold, laplacian_min_component_area),
        prewitt=_measure("Prewitt", filtered, prewitt_threshold, laplacian_min_component_area),
        laplacian=_measure(
            "Laplacian", filtered, laplacian_contrast_threshold,
            laplacian_min_component_area,
        ),
    )
