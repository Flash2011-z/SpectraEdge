"""API handoff: HTTP/image handling; mathematics stays in array-only modules."""

import base64
from io import BytesIO
import logging
import os
from threading import Lock
from time import perf_counter
from typing import Literal
from uuid import UUID
import warnings

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser

from backend.signal_ops import (
    add_gaussian_noise,
    add_salt_pepper_noise,
    fft_spectrum,
    gaussian_blur,
)
from backend.detection import laplacian, prewitt, sobel, threshold_edges, zero_crossing_edges
from backend.analysis import analyze_objects, compare_detectors, multi_scale_edges
from backend.analysis import filter_small_components
from backend.api_models import AnalysisResponse, ComparisonResponse
from backend.cutout_api import create_cutout_router
from backend.live_api import create_live_router

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_REQUEST_BYTES = MAX_UPLOAD_BYTES + 64 * 1024
MAX_DECODED_PIXELS = 20_000_000
MAX_PREVIEW_SIDE = 512
KERNEL_SIZES = tuple(range(3, 32, 2))
# Fixed DISPLAY bounds for 0..255 grayscale inputs, not numerical clipping.
SOBEL_COMPONENT_LIMIT = 4 * 255
SOBEL_MAGNITUDE_LIMIT = np.sqrt(2) * SOBEL_COMPONENT_LIMIT
PREWITT_COMPONENT_LIMIT = 3 * 255
LAPLACIAN_RESPONSE_LIMIT = 4 * 255
MAX_THRESHOLD = 1443
DEFAULT_LAPLACIAN_MIN_COMPONENT_AREA = 2
_processing_slot = Lock()
logger = logging.getLogger(__name__)
ALLOWED_ORIGINS = [origin.strip() for origin in os.getenv(
    "SPECTRAEDGE_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
).split(",") if origin.strip()]

app = FastAPI(title="SpectraEdge", version="0.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)
app.include_router(create_cutout_router(ALLOWED_ORIGINS))
app.include_router(create_live_router(ALLOWED_ORIGINS))


class AnalyzeForm(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sigma: float = Field(ge=0, le=5, multiple_of=0.1, allow_inf_nan=False)
    kernel_size: int
    request_id: UUID
    detector: Literal["Sobel", "Prewitt", "Laplacian"] | None = None
    threshold: float | None = Field(default=None, ge=0, le=MAX_THRESHOLD, allow_inf_nan=False)
    multi_scale: bool = False
    scale_sigmas: list[float] | None = Field(default=None, min_length=1, max_length=5)
    scale_support: int | None = Field(default=None, ge=1, le=5)
    noise_model: Literal["Gaussian", "Salt & Pepper"] | None = None
    noise_strength: float | None = Field(default=None, ge=0, le=100, allow_inf_nan=False)
    noise_seed: int | None = Field(default=None, ge=0, le=4294967295)
    laplacian_min_component_area: int | None = Field(default=None, ge=1, le=10000)

    @field_validator("kernel_size")
    @classmethod
    def supported_kernel(cls, value):
        if value not in KERNEL_SIZES:
            raise ValueError("choose an odd kernel size from 3 through 31")
        return value

    @field_validator("scale_sigmas")
    @classmethod
    def supported_sigmas(cls, values):
        if values is None:
            return values
        if any(not np.isfinite(value) or value < 0 or value > 5 or
               abs(value * 10 - round(value * 10)) > 1e-8 for value in values):
            raise ValueError("scale sigmas must be unique values from 0 through 5 in steps of 0.1")
        if len(set(values)) != len(values):
            raise ValueError("scale sigmas must be unique")
        return values

    @model_validator(mode="after")
    def detection_settings_agree(self):
        if self.detector is not None and self.threshold is None:
            raise ValueError("threshold is required when a detector is selected")
        if self.detector is None and self.threshold is not None:
            raise ValueError("threshold requires a detector; omit both for Gaussian/Fourier only")
        if self.multi_scale:
            if self.detector is None:
                raise ValueError("multi_scale requires a detector")
            if self.scale_sigmas is None:
                raise ValueError("scale_sigmas is required when multi_scale is enabled")
            if self.scale_support is not None and self.scale_support > len(self.scale_sigmas):
                raise ValueError("scale_support cannot exceed the number of scale sigmas")
        elif self.scale_sigmas is not None or self.scale_support is not None:
            raise ValueError("scale_sigmas and scale_support require multi_scale=true")
        noise_values = (self.noise_strength, self.noise_seed)
        if self.noise_model is None and any(value is not None for value in noise_values):
            raise ValueError("noise_strength and noise_seed require noise_model")
        if self.noise_model is not None and any(value is None for value in noise_values):
            raise ValueError("noise_strength and noise_seed are required when noise_model is selected")
        if (self.noise_model == "Salt & Pepper" and self.noise_strength is not None and
                self.noise_strength > 1):
            raise ValueError("Salt & Pepper noise_strength must be a probability from 0 through 1")
        if self.laplacian_min_component_area is not None and self.detector != "Laplacian":
            raise ValueError("laplacian_min_component_area requires detector=Laplacian")
        return self


class CompareForm(BaseModel):
    """Shared settings applied to all three comparison detectors."""

    model_config = ConfigDict(extra="forbid")
    sigma: float = Field(ge=0, le=5, multiple_of=0.1, allow_inf_nan=False)
    kernel_size: int
    sobel_threshold: float = Field(ge=0, le=MAX_THRESHOLD, allow_inf_nan=False)
    prewitt_threshold: float = Field(ge=0, le=MAX_THRESHOLD, allow_inf_nan=False)
    laplacian_contrast_threshold: float = Field(ge=0, le=MAX_THRESHOLD, allow_inf_nan=False)
    request_id: UUID

    @field_validator("kernel_size")
    @classmethod
    def supported_kernel(cls, value):
        if value not in KERNEL_SIZES:
            raise ValueError("choose an odd kernel size from 3 through 31")
        return value


class _BodyTooLarge(MultiPartException):
    pass


class _MemoryMultipartParser(MultiPartParser):
    # The stream is capped BEFORE reaching this threshold, so uploaded files
    # cannot roll from the parser's memory buffer onto a temporary disk file.
    spool_max_size = MAX_REQUEST_BYTES


async def _bounded_stream(request):
    received = 0
    async for chunk in request.stream():
        received += len(chunk)
        if received > MAX_REQUEST_BYTES:
            raise _BodyTooLarge("Upload exceeds the 20 MB limit.")
        yield chunk


def _png_data_url(values):
    """Quantize only at this display boundary; never change signal arrays."""
    pixels = np.rint(np.clip(values, 0, 255)).astype(np.uint8)
    buffer = BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _decode_preview(payload):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(payload)) as image:
                if image.format not in {"PNG", "JPEG", "WEBP", "GIF"}:
                    raise HTTPException(415, "Choose a PNG, JPG, WebP, or GIF image.")
                if image.width * image.height > MAX_DECODED_PIXELS:
                    raise HTTPException(413, "Choose an image at or below 20 megapixels.")
                image.load()  # Fully decode frame zero; truncated data is rejected.
                oriented = ImageOps.exif_transpose(image)
                source_dimensions = {"width": oriented.width, "height": oriented.height}
                # Resizing is input preparation, not the Gaussian stage. Nearest
                # neighbour introduces no hidden smoothing into this demonstration.
                oriented.thumbnail((MAX_PREVIEW_SIDE, MAX_PREVIEW_SIDE), Image.Resampling.NEAREST)
                rgba = oriented.convert("RGBA")
                background = Image.new("RGBA", rgba.size, "white")
                original = Image.alpha_composite(background, rgba).convert("RGB")
                grayscale = np.asarray(original.convert("L"), dtype=np.float64)
                return np.asarray(original, dtype=np.float64), grayscale, source_dimensions
    except (Image.DecompressionBombWarning, Image.DecompressionBombError) as error:
        raise HTTPException(413, "Choose an image at or below 20 megapixels.") from error
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as error:
        raise HTTPException(400, "This file could not be decoded as a valid image.") from error


def _analyze_payload(payload, parameters):
    if not _processing_slot.acquire(blocking=False):
        raise HTTPException(429, "The analysis engine is busy. Try Process again shortly.")
    try:
        started = perf_counter()
        original, grayscale, source_dimensions = _decode_preview(payload)
        analysis_input = grayscale
        if parameters.noise_model == "Gaussian":
            analysis_input = add_gaussian_noise(
                grayscale, parameters.noise_strength, parameters.noise_seed,
            )
        elif parameters.noise_model == "Salt & Pepper":
            analysis_input = add_salt_pepper_noise(
                grayscale, parameters.noise_strength, parameters.noise_seed,
            )
        filtered = gaussian_blur(analysis_input, parameters.sigma, parameters.kernel_size)
        if parameters.detector in ("Sobel", "Prewitt"):
            detector = sobel if parameters.detector == "Sobel" else prewitt
            gx, gy, magnitude = detector(filtered)
            # Threshold the SAME float signal, before any display conversion.
            edges = threshold_edges(magnitude, parameters.threshold)
        elif parameters.detector == "Laplacian":
            response = laplacian(filtered)
            edges = zero_crossing_edges(response, parameters.threshold)
        if parameters.detector is not None:
            multi_scale = None
            analysis_edges = edges
            if parameters.multi_scale:
                multi_scale = multi_scale_edges(
                    analysis_input, parameters.detector, parameters.scale_sigmas,
                    parameters.threshold, parameters.scale_support,
                )
                analysis_edges = multi_scale.fused_edge_map
            if parameters.detector == "Laplacian":
                analysis_edges = filter_small_components(
                    analysis_edges,
                    parameters.laplacian_min_component_area or
                    DEFAULT_LAPLACIAN_MIN_COMPONENT_AREA,
                )
            object_analysis = analyze_objects(analysis_edges)
            # Outer contours are white, hole contours gray, and pixels belonging
            # to both are light gray. Numerical masks remain separate.
            contours = np.where(
                object_analysis.outer_boundaries & object_analysis.hole_boundaries, 192,
                np.where(object_analysis.outer_boundaries, 255,
                         np.where(object_analysis.hole_boundaries, 128, 0)),
            )
        original_spectrum = fft_spectrum(analysis_input)
        filtered_spectrum = fft_spectrum(filtered)
        shared_max = max(float(original_spectrum.max()), float(filtered_spectrum.max()))
        # Both spectra use the SAME zero-to-maximum scale, including the
        # all-black case. Raw floating-point spectra remain untouched.
        denominator = shared_max if shared_max > 0 else 1.0
        result = {
            "provenance": "computed",
            "request_id": str(parameters.request_id),
            "original_image": _png_data_url(original),
            "grayscale_image": _png_data_url(grayscale),
            "noisy_image": _png_data_url(analysis_input) if parameters.noise_model else None,
            "filtered_image": _png_data_url(filtered),
            "fft_image": _png_data_url(original_spectrum / denominator * 255),
            "filtered_fft_image": _png_data_url(filtered_spectrum / denominator * 255),
            "spectrum_scale": {"min": 0.0, "max": shared_max, "mapping": "linear_grayscale"},
            "parameters_used": {"sigma": parameters.sigma, "kernel_size": parameters.kernel_size},
            "source_dimensions": source_dimensions,
            "analyzed_dimensions": {"width": grayscale.shape[1], "height": grayscale.shape[0]},
            "completed_stages": ["input", "grayscale", "smooth", "fourier"],
            "detection_status": "not_run",
            "detector_metadata": None,
            "gx": None, "gy": None, "gradient_magnitude": None, "laplacian_response": None,
            "edge_map": None, "contour_image": None, "object_list": None, "fps": None,
            "multi_scale": None,
            "noise": {
                "model": parameters.noise_model or "None",
                "strength": parameters.noise_strength if parameters.noise_model else 0,
                "units": (
                    "intensity standard deviation" if parameters.noise_model == "Gaussian"
                    else "pixel corruption probability" if parameters.noise_model == "Salt & Pepper"
                    else "none"
                ),
                "seed": parameters.noise_seed if parameters.noise_model else None,
            },
        }
        if parameters.noise_model:
            result["parameters_used"].update(
                noise_model=parameters.noise_model,
                noise_strength=parameters.noise_strength,
                noise_seed=parameters.noise_seed,
            )
            result["completed_stages"].insert(2, "noise")
        if parameters.detector in ("Sobel", "Prewitt"):
            # Fixed signed display bounds: +/-1020 for Sobel, +/-765 for Prewitt.
            # Zero is middle gray; magnitude uses sqrt(2) times that bound.
            component_limit = SOBEL_COMPONENT_LIMIT if parameters.detector == "Sobel" else PREWITT_COMPONENT_LIMIT
            magnitude_limit = SOBEL_MAGNITUDE_LIMIT if parameters.detector == "Sobel" else np.sqrt(2) * component_limit
            result.update({
                "gx": _png_data_url((gx / component_limit + 1) * 127.5),
                "gy": _png_data_url((gy / component_limit + 1) * 127.5),
                "gradient_magnitude": _png_data_url(magnitude / magnitude_limit * 255),
                "edge_map": _png_data_url(edges),
                "contour_image": _png_data_url(contours),
                "object_list": object_analysis.objects,
                "detection_status": "edges_computed",
                "completed_stages": ["input", "grayscale", "smooth", parameters.detector.lower(),
                                     "threshold", "contours", "objects", "fourier"],
            })
        elif parameters.detector == "Laplacian":
            result.update({
                "laplacian_response": _png_data_url((response / LAPLACIAN_RESPONSE_LIMIT + 1) * 127.5),
                "edge_map": _png_data_url(edges),
                "contour_image": _png_data_url(contours),
                "object_list": object_analysis.objects,
                "detection_status": "edges_computed",
                "completed_stages": ["input", "grayscale", "smooth", "laplacian",
                                     "zero_crossing", "contours", "objects", "fourier"],
            })
        if parameters.detector is not None and parameters.noise_model:
            result["completed_stages"].insert(2, "noise")
        if parameters.detector is not None:
            result["parameters_used"].update(detector=parameters.detector, threshold=parameters.threshold)
            if parameters.detector == "Laplacian":
                minimum_area = (parameters.laplacian_min_component_area or
                                DEFAULT_LAPLACIAN_MIN_COMPONENT_AREA)
                if parameters.laplacian_min_component_area is not None:
                    result["parameters_used"]["laplacian_min_component_area"] = minimum_area
                result["detector_metadata"] = {
                    "detector": "Laplacian",
                    "decision": "zero_crossing",
                    "threshold_type": "zero_crossing_contrast",
                    "threshold_label": "Zero-crossing contrast threshold",
                    "threshold": parameters.threshold,
                    "threshold_units": "raw_response_difference",
                    "minimum_component_area": minimum_area,
                }
            else:
                result["detector_metadata"] = {
                    "detector": parameters.detector,
                    "decision": "magnitude_threshold",
                    "threshold_type": "gradient_magnitude",
                    "threshold_label": "Gradient magnitude threshold",
                    "threshold": parameters.threshold,
                    "threshold_units": "raw_gradient_magnitude",
                    "minimum_component_area": None,
                }
            if multi_scale is not None:
                decision_stage = "zero_crossing" if parameters.detector == "Laplacian" else "threshold"
                result["completed_stages"] = [
                    "input", "grayscale", *(["noise"] if parameters.noise_model else []),
                    "smooth", parameters.detector.lower(), decision_stage,
                    "multi_scale", "contours", "objects", "fourier",
                ]
                scale_count = len(multi_scale.scales)
                result["multi_scale"] = {
                    "sigmas": list(multi_scale.sigma_values),
                    "support_count": multi_scale.support_count,
                    "scales": [{
                        "sigma": scale.sigma,
                        "kernel_size": scale.kernel_size,
                        "edge_map": _png_data_url(scale.edge_map),
                    } for scale in multi_scale.scales],
                    "persistence_map": _png_data_url(
                        multi_scale.persistence_map.astype(float) / scale_count * 255
                    ),
                    "persistence_scale": {
                        "min": 0, "max": scale_count, "mapping": "linear_grayscale",
                    },
                    "fused_edge_map": _png_data_url(multi_scale.fused_edge_map),
                }
                result["parameters_used"].update(
                    multi_scale=True,
                    scale_sigmas=list(multi_scale.sigma_values),
                    scale_support=multi_scale.support_count,
                )
        # Milliseconds: decode, preparation, numerical work, and PNG encoding.
        # Upload transfer and JSON serialization are deliberately excluded.
        result["processing_time"] = round((perf_counter() - started) * 1000, 3)
        return result
    finally:
        _processing_slot.release()


def _comparison_entry(entry):
    return {
        "edge_map": _png_data_url(entry.edge_map),
        "object_list": entry.object_list,
        "edge_pixel_count": entry.edge_pixel_count,
        "object_count": entry.object_count,
        "average_object_area": entry.average_object_area,
        "processing_time": round(entry.processing_time, 3),
        "metadata": {
            "detector": entry.detector,
            "decision": entry.decision,
            "threshold_type": entry.threshold_type,
            "threshold_label": entry.threshold_label,
            "threshold": entry.threshold,
            "threshold_units": entry.threshold_units,
            "minimum_component_area": entry.minimum_component_area,
        },
    }


def _compare_payload(payload, parameters):
    if not _processing_slot.acquire(blocking=False):
        raise HTTPException(429, "The analysis engine is busy. Try Compare again shortly.")
    try:
        started = perf_counter()
        _original, grayscale, source_dimensions = _decode_preview(payload)
        comparison = compare_detectors(
            grayscale, parameters.sigma, parameters.kernel_size,
            parameters.sobel_threshold, parameters.prewitt_threshold,
            parameters.laplacian_contrast_threshold,
        )
        return {
            "provenance": "computed",
            "request_id": str(parameters.request_id),
            "parameters_used": {
                "sigma": parameters.sigma,
                "kernel_size": parameters.kernel_size,
                "sobel_threshold": parameters.sobel_threshold,
                "prewitt_threshold": parameters.prewitt_threshold,
                "laplacian_contrast_threshold": parameters.laplacian_contrast_threshold,
            },
            "source_dimensions": source_dimensions,
            "analyzed_dimensions": {
                "width": grayscale.shape[1], "height": grayscale.shape[0],
            },
            "sobel": _comparison_entry(comparison.sobel),
            "prewitt": _comparison_entry(comparison.prewitt),
            "laplacian": _comparison_entry(comparison.laplacian),
            "processing_time": round((perf_counter() - started) * 1000, 3),
        }
    finally:
        _processing_slot.release()


@app.get("/health")
def health():
    return {"status": "ok", "milestone": "noise-experiments", "max_preview_side": MAX_PREVIEW_SIDE}


@app.post("/analyze", response_model=AnalysisResponse, response_model_exclude_unset=True,
          openapi_extra={
    "requestBody": {"required": True, "content": {"multipart/form-data": {"schema": {
        "type": "object", "required": ["image", "sigma", "kernel_size", "request_id"],
        "properties": {
            "image": {"type": "string", "format": "binary"},
            "sigma": {"type": "number", "minimum": 0, "maximum": 5, "multipleOf": 0.1},
            "kernel_size": {"type": "integer", "enum": list(KERNEL_SIZES)},
            "request_id": {"type": "string", "format": "uuid"},
            "detector": {"type": "string", "enum": ["Sobel", "Prewitt", "Laplacian"],
                         "description": "Optional. Omit detector and threshold for Gaussian/Fourier only."},
            "threshold": {"type": "number", "minimum": 0, "maximum": MAX_THRESHOLD,
                          "description": "Required with a detector; forbidden without one. Sobel/Prewitt use raw magnitude > threshold. Laplacian uses opposite-sign neighbours with raw response difference > threshold."},
            "multi_scale": {"type": "boolean", "description": "Optional; enable independent scale passes."},
            "scale_sigmas": {"type": "string", "description": "Required with multi_scale=true; 1-5 unique comma-separated values from 0 to 5 in steps of 0.1."},
            "scale_support": {"type": "integer", "minimum": 1, "maximum": 5,
                              "description": "Optional persistence count; defaults to 2 for multiple scales and 1 for one scale."},
            "noise_model": {"type": "string", "enum": ["Gaussian", "Salt & Pepper"],
                            "description": "Optional synthetic noise applied to grayscale before smoothing."},
            "noise_strength": {"type": "number", "minimum": 0, "maximum": 100,
                               "description": "Gaussian intensity standard deviation, or Salt & Pepper probability from 0 to 1."},
            "noise_seed": {"type": "integer", "minimum": 0, "maximum": 4294967295},
            "laplacian_min_component_area": {"type": "integer", "minimum": 1, "maximum": 10000,
                                               "default": DEFAULT_LAPLACIAN_MIN_COMPONENT_AREA,
                                               "description": "Optional for Laplacian only. Removes smaller eight-connected components before contour/object analysis; the raw edge_map is unchanged."},
        },
        "additionalProperties": False,
    }}}},
})
async def analyze(request: Request):
    """Process one in-memory upload; see backend/README.md for the handoff contract."""
    if request.headers.get("origin") and request.headers["origin"] not in ALLOWED_ORIGINS:
        raise HTTPException(403, "This website origin is not allowed to submit images.")
    if request.headers.get("content-type", "").split(";")[0].lower() != "multipart/form-data":
        raise HTTPException(415, "Send an image and analysis settings as multipart/form-data.")
    try:
        form = await _MemoryMultipartParser(
            request.headers, _bounded_stream(request), max_files=1, max_fields=12, max_part_size=4096
        ).parse()
    except _BodyTooLarge as error:
        raise HTTPException(413, error.message) from error
    except (MultiPartException, ValueError) as error:
        raise HTTPException(400, "Invalid multipart upload. Send one image and analysis settings.") from error

    try:
        required = {"image", "sigma", "kernel_size", "request_id"}
        allowed = required | {"detector", "threshold", "multi_scale", "scale_sigmas", "scale_support",
                              "noise_model", "noise_strength", "noise_seed"}
        allowed.add("laplacian_min_component_area")
        if len(form.multi_items()) != len(form) or not required <= set(form) <= allowed:
            raise HTTPException(422, "Provide image, sigma, kernel_size, request_id and optionally detector plus threshold, without duplicates or extra fields.")
        image = form["image"]
        if not isinstance(image, UploadFile):
            raise HTTPException(422, "The image field must contain an uploaded file.")
        try:
            parameter_values = {key: form[key] for key in form if key != "image"}
            if "scale_sigmas" in parameter_values and isinstance(parameter_values["scale_sigmas"], str):
                parameter_values["scale_sigmas"] = parameter_values["scale_sigmas"].split(",")
            parameters = AnalyzeForm.model_validate(parameter_values)
        except ValidationError as error:
            detail = "; ".join(f"{item['loc'][0] if item['loc'] else 'settings'}: {item['msg']}" for item in error.errors())
            raise HTTPException(422, detail) from error
        payload = await image.read(MAX_UPLOAD_BYTES + 1)
        if not payload:
            raise HTTPException(400, "This image is empty. Choose another file.")
        if len(payload) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "This image exceeds the 20 MB limit.")
        return await run_in_threadpool(_analyze_payload, payload, parameters)
    except HTTPException:
        raise
    except Exception as error:
        logger.exception("Image analysis failed")
        raise HTTPException(500, "Analysis failed. Try another image or restart the backend.") from error
    finally:
        await form.close()


@app.post("/compare", response_model=ComparisonResponse, response_model_exclude_unset=True,
          openapi_extra={
    "requestBody": {"required": True, "content": {"multipart/form-data": {"schema": {
        "type": "object",
        "required": ["image", "sigma", "kernel_size", "sobel_threshold",
                     "prewitt_threshold", "laplacian_contrast_threshold", "request_id"],
        "properties": {
            "image": {"type": "string", "format": "binary"},
            "sigma": {"type": "number", "minimum": 0, "maximum": 5, "multipleOf": 0.1},
            "kernel_size": {"type": "integer", "enum": list(KERNEL_SIZES)},
            "sobel_threshold": {"type": "number", "minimum": 0, "maximum": MAX_THRESHOLD,
                                "description": "Raw Sobel gradient-magnitude threshold."},
            "prewitt_threshold": {"type": "number", "minimum": 0, "maximum": MAX_THRESHOLD,
                                  "description": "Raw Prewitt gradient-magnitude threshold."},
            "laplacian_contrast_threshold": {"type": "number", "minimum": 0, "maximum": MAX_THRESHOLD,
                                             "description": "Raw response difference required across a Laplacian zero crossing."},
            "request_id": {"type": "string", "format": "uuid"},
        },
        "additionalProperties": False,
    }}}},
})
async def compare(request: Request):
    """Prepare once and compare all implemented detectors on one filtered signal."""
    if request.headers.get("origin") and request.headers["origin"] not in ALLOWED_ORIGINS:
        raise HTTPException(403, "This website origin is not allowed to submit images.")
    if request.headers.get("content-type", "").split(";")[0].lower() != "multipart/form-data":
        raise HTTPException(415, "Send an image and comparison settings as multipart/form-data.")
    try:
        form = await _MemoryMultipartParser(
            request.headers, _bounded_stream(request), max_files=1, max_fields=7,
            max_part_size=4096,
        ).parse()
    except _BodyTooLarge as error:
        raise HTTPException(413, error.message) from error
    except (MultiPartException, ValueError) as error:
        raise HTTPException(400, "Invalid multipart upload. Send one image and comparison settings.") from error

    try:
        required = {"image", "sigma", "kernel_size", "sobel_threshold",
                    "prewitt_threshold", "laplacian_contrast_threshold", "request_id"}
        if len(form.multi_items()) != len(form) or set(form) != required:
            raise HTTPException(422, "Provide image, sigma, kernel_size, all three detector thresholds, and request_id without duplicates or extra fields.")
        image = form["image"]
        if not isinstance(image, UploadFile):
            raise HTTPException(422, "The image field must contain an uploaded file.")
        try:
            parameters = CompareForm.model_validate({key: form[key] for key in form if key != "image"})
        except ValidationError as error:
            detail = "; ".join(
                f"{item['loc'][0] if item['loc'] else 'settings'}: {item['msg']}"
                for item in error.errors()
            )
            raise HTTPException(422, detail) from error
        payload = await image.read(MAX_UPLOAD_BYTES + 1)
        if not payload:
            raise HTTPException(400, "This image is empty. Choose another file.")
        if len(payload) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "This image exceeds the 20 MB limit.")
        return await run_in_threadpool(_compare_payload, payload, parameters)
    except HTTPException:
        raise
    except Exception as error:
        logger.exception("Detector comparison failed")
        raise HTTPException(500, "Comparison failed. Try another image or restart the backend.") from error
    finally:
        await form.close()
