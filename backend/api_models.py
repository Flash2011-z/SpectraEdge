"""Pydantic response contracts for the public analysis APIs.

These models describe existing JSON output for OpenAPI and response validation.
They intentionally allow compatible internal additions while keeping all public
fields typed. Numerical arrays remain internal and never enter these models.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict


DetectorName = Literal["Sobel", "Prewitt", "Laplacian"]


class ResponseModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class Dimensions(ResponseModel):
    width: int
    height: int


class BoundingBox(ResponseModel):
    x: int
    y: int
    width: int
    height: int


class ObjectMeasurement(ResponseModel):
    id: int
    area: int
    perimeter: int
    centroid: list[float]
    bounding_box: BoundingBox


class DetectorMetadata(ResponseModel):
    detector: DetectorName
    decision: Literal["magnitude_threshold", "zero_crossing"]
    threshold_type: Literal["gradient_magnitude", "zero_crossing_contrast"]
    threshold_label: Literal["Gradient magnitude threshold", "Zero-crossing contrast threshold"]
    threshold: float
    threshold_units: Literal["raw_gradient_magnitude", "raw_response_difference"]
    minimum_component_area: int | None = None


class NoiseMetadata(ResponseModel):
    model: Literal["None", "Gaussian", "Salt & Pepper"]
    strength: float
    units: Literal["none", "intensity standard deviation", "pixel corruption probability"]
    seed: int | None


class SpectrumScale(ResponseModel):
    min: float
    max: float
    mapping: Literal["linear_grayscale"]


class AnalysisParameters(ResponseModel):
    sigma: float
    kernel_size: int
    detector: DetectorName | None = None
    threshold: float | None = None
    multi_scale: bool | None = None
    scale_sigmas: list[float] | None = None
    scale_support: int | None = None
    noise_model: Literal["Gaussian", "Salt & Pepper"] | None = None
    noise_strength: float | None = None
    noise_seed: int | None = None
    laplacian_min_component_area: int | None = None


class MultiScaleItem(ResponseModel):
    sigma: float
    kernel_size: int
    edge_map: str


class PersistenceScale(ResponseModel):
    min: int
    max: int
    mapping: Literal["linear_grayscale"]


class MultiScaleOutput(ResponseModel):
    sigmas: list[float]
    support_count: int
    scales: list[MultiScaleItem]
    persistence_map: str
    persistence_scale: PersistenceScale
    fused_edge_map: str


class AnalysisResponse(ResponseModel):
    provenance: Literal["computed"]
    request_id: str
    original_image: str
    grayscale_image: str
    noisy_image: str | None
    filtered_image: str
    fft_image: str
    filtered_fft_image: str
    spectrum_scale: SpectrumScale
    parameters_used: AnalysisParameters
    source_dimensions: Dimensions
    analyzed_dimensions: Dimensions
    completed_stages: list[str]
    detection_status: Literal["not_run", "edges_computed"]
    detector_metadata: DetectorMetadata | None
    gx: str | None
    gy: str | None
    gradient_magnitude: str | None
    laplacian_response: str | None
    edge_map: str | None
    contour_image: str | None
    object_list: list[ObjectMeasurement] | None
    fps: float | None
    multi_scale: MultiScaleOutput | None
    noise: NoiseMetadata
    processing_time: float


class ComparisonParameters(ResponseModel):
    sigma: float
    kernel_size: int
    sobel_threshold: float
    prewitt_threshold: float
    laplacian_contrast_threshold: float


class ComparisonDetectorOutput(ResponseModel):
    edge_map: str
    object_list: list[ObjectMeasurement]
    edge_pixel_count: int
    object_count: int
    average_object_area: float
    processing_time: float
    metadata: DetectorMetadata


class ComparisonResponse(ResponseModel):
    provenance: Literal["computed"]
    request_id: str
    parameters_used: ComparisonParameters
    source_dimensions: Dimensions
    analyzed_dimensions: Dimensions
    sobel: ComparisonDetectorOutput
    prewitt: ComparisonDetectorOutput
    laplacian: ComparisonDetectorOutput
    processing_time: float
