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

from backend.signal_ops import fft_spectrum, gaussian_blur
from backend.detection import sobel, threshold_edges
from backend.cutout_api import create_cutout_router

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_REQUEST_BYTES = MAX_UPLOAD_BYTES + 64 * 1024
MAX_DECODED_PIXELS = 20_000_000
MAX_PREVIEW_SIDE = 512
KERNEL_SIZES = tuple(range(3, 32, 2))
# Fixed DISPLAY bounds for 0..255 grayscale inputs, not numerical clipping.
SOBEL_COMPONENT_LIMIT = 4 * 255
SOBEL_MAGNITUDE_LIMIT = np.sqrt(2) * SOBEL_COMPONENT_LIMIT
MAX_THRESHOLD = 1443
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


class AnalyzeForm(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sigma: float = Field(ge=0, le=5, multiple_of=0.1, allow_inf_nan=False)
    kernel_size: int
    request_id: UUID
    detector: Literal["Sobel"] | None = None
    threshold: float | None = Field(default=None, ge=0, le=MAX_THRESHOLD, allow_inf_nan=False)

    @field_validator("kernel_size")
    @classmethod
    def supported_kernel(cls, value):
        if value not in KERNEL_SIZES:
            raise ValueError("choose an odd kernel size from 3 through 31")
        return value

    @model_validator(mode="after")
    def detection_settings_agree(self):
        if self.detector == "Sobel" and self.threshold is None:
            raise ValueError("threshold is required when detector is Sobel")
        if self.detector is None and self.threshold is not None:
            raise ValueError("threshold requires detector=Sobel; omit both for Gaussian/Fourier only")
        return self


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
        filtered = gaussian_blur(grayscale, parameters.sigma, parameters.kernel_size)
        if parameters.detector == "Sobel":
            gx, gy, magnitude = sobel(filtered)
            # Threshold the SAME float signal, before any display conversion.
            edges = threshold_edges(magnitude, parameters.threshold)
        original_spectrum = fft_spectrum(grayscale)
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
            "filtered_image": _png_data_url(filtered),
            "fft_image": _png_data_url(original_spectrum / denominator * 255),
            "filtered_fft_image": _png_data_url(filtered_spectrum / denominator * 255),
            "spectrum_scale": {"min": 0.0, "max": shared_max, "mapping": "linear_grayscale"},
            "parameters_used": {"sigma": parameters.sigma, "kernel_size": parameters.kernel_size},
            "source_dimensions": source_dimensions,
            "analyzed_dimensions": {"width": grayscale.shape[1], "height": grayscale.shape[0]},
            "completed_stages": ["input", "grayscale", "smooth", "fourier"],
            "detection_status": "not_run",
            "gx": None, "gy": None, "gradient_magnitude": None,
            "edge_map": None, "contour_image": None, "object_list": None, "fps": None,
        }
        if parameters.detector == "Sobel":
            # Signed responses: -1020 -> black, 0 -> middle gray, +1020 -> white.
            # Magnitude: 0 -> black, 1020*sqrt(2) -> white, independent of image.
            result.update({
                "gx": _png_data_url((gx / SOBEL_COMPONENT_LIMIT + 1) * 127.5),
                "gy": _png_data_url((gy / SOBEL_COMPONENT_LIMIT + 1) * 127.5),
                "gradient_magnitude": _png_data_url(magnitude / SOBEL_MAGNITUDE_LIMIT * 255),
                "edge_map": _png_data_url(edges),
                "detection_status": "edges_computed",
                "completed_stages": ["input", "grayscale", "smooth", "sobel", "threshold", "fourier"],
            })
            result["parameters_used"].update(detector="Sobel", threshold=parameters.threshold)
        # Milliseconds: decode, preparation, numerical work, and PNG encoding.
        # Upload transfer and JSON serialization are deliberately excluded.
        result["processing_time"] = round((perf_counter() - started) * 1000, 3)
        return result
    finally:
        _processing_slot.release()


@app.get("/health")
def health():
    return {"status": "ok", "milestone": "sobel-threshold", "max_preview_side": MAX_PREVIEW_SIDE}


@app.post("/analyze", openapi_extra={
    "requestBody": {"required": True, "content": {"multipart/form-data": {"schema": {
        "type": "object", "required": ["image", "sigma", "kernel_size", "request_id"],
        "properties": {
            "image": {"type": "string", "format": "binary"},
            "sigma": {"type": "number", "minimum": 0, "maximum": 5, "multipleOf": 0.1},
            "kernel_size": {"type": "integer", "enum": list(KERNEL_SIZES)},
            "request_id": {"type": "string", "format": "uuid"},
            "detector": {"type": "string", "enum": ["Sobel"],
                         "description": "Optional. Omit detector and threshold for Gaussian/Fourier only."},
            "threshold": {"type": "number", "minimum": 0, "maximum": MAX_THRESHOLD,
                          "description": "Required with Sobel; forbidden without detector. Edges use raw magnitude > threshold."},
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
            request.headers, _bounded_stream(request), max_files=1, max_fields=5, max_part_size=4096
        ).parse()
    except _BodyTooLarge as error:
        raise HTTPException(413, error.message) from error
    except (MultiPartException, ValueError) as error:
        raise HTTPException(400, "Invalid multipart upload. Send one image and analysis settings.") from error

    try:
        required = {"image", "sigma", "kernel_size", "request_id"}
        allowed = required | {"detector", "threshold"}
        if len(form.multi_items()) != len(form) or not required <= set(form) <= allowed:
            raise HTTPException(422, "Provide image, sigma, kernel_size, request_id and optionally detector plus threshold, without duplicates or extra fields.")
        image = form["image"]
        if not isinstance(image, UploadFile):
            raise HTTPException(422, "The image field must contain an uploaded file.")
        try:
            parameters = AnalyzeForm.model_validate({key: form[key] for key in form if key != "image"})
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
