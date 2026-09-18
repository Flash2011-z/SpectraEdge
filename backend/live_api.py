"""Bounded, in-memory live frames; spectrum uses the grayscale input.

Multipart fields: image, detector, sigma, kernel_size, threshold, and optional
frame_id (echoed; otherwise a UUID is generated). Images are PNG data URLs.
Frames fit within 256 pixels per side without upscaling. Timing is in ms and
includes decoding, signal processing, and PNG encoding, but not transport.
"""

import base64
from io import BytesIO
import logging
from threading import BoundedSemaphore
from time import perf_counter
from typing import Literal
from uuid import uuid4
import warnings

from fastapi import APIRouter, HTTPException, Request
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser

from backend.signal_ops import gaussian_blur, fft_spectrum
from backend.detection import sobel, prewitt, laplacian, threshold_edges, zero_crossing_edges

MAX_UPLOAD_BYTES = 2 * 1024 * 1024
MAX_REQUEST_BYTES = MAX_UPLOAD_BYTES + 64 * 1024
MAX_DECODED_PIXELS = 20_000_000
MAX_SIDE = 256
_processing_slot = BoundedSemaphore(1)
logger = logging.getLogger(__name__)


class LiveSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    detector: Literal["Sobel", "Prewitt", "Laplacian"]
    sigma: float = Field(ge=0, le=5, multiple_of=0.1, allow_inf_nan=False)
    kernel_size: int = Field(ge=3, le=31)
    threshold: float = Field(ge=0, le=1443, allow_inf_nan=False)

    @field_validator("kernel_size")
    @classmethod
    def odd_kernel(cls, value):
        if value % 2 == 0:
            raise ValueError("kernel_size must be odd")
        return value


class LiveForm(LiveSettings):
    frame_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1, max_length=128)


class LiveResponse(BaseModel):
    frame_id: str
    grayscale_image: str
    filtered_image: str
    edge_image: str
    spectrum_image: str
    processing_time: float
    settings_used: LiveSettings


class _BodyTooLarge(MultiPartException):
    pass


class _MemoryParser(MultiPartParser):
    spool_max_size = MAX_REQUEST_BYTES


async def _bounded_stream(request):
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_REQUEST_BYTES:
            raise _BodyTooLarge("Live frame request exceeds the 2 MB upload allowance.")
        yield chunk


def _decode(payload):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(payload)) as image:
                if image.format not in {"PNG", "JPEG", "WEBP", "GIF"}:
                    raise HTTPException(415, "Use PNG, JPEG, WebP, or GIF frames.")
                if image.width * image.height > MAX_DECODED_PIXELS:
                    raise HTTPException(413, "Frame exceeds 20 megapixels.")
                image.load()
                oriented = ImageOps.exif_transpose(image)
                oriented.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.NEAREST)
                rgba = oriented.convert("RGBA")
                rgb = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba)
                return np.asarray(rgb.convert("RGB").convert("L"), dtype=np.float64)
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise HTTPException(413, "Frame exceeds the decoded image limit.") from error
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as error:
        raise HTTPException(400, "Cannot decode image frame.") from error


def _png(values):
    output = BytesIO()
    Image.fromarray(np.rint(np.clip(values, 0, 255)).astype(np.uint8)).save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def _process_frame(payload, settings):
    if not _processing_slot.acquire(blocking=False):
        raise HTTPException(429, "Live processor is busy; submit the next frame later.")
    try:
        started = perf_counter()
        grayscale = _decode(payload)
        filtered = gaussian_blur(grayscale, settings.sigma, settings.kernel_size)
        if settings.detector == "Laplacian":
            edges = zero_crossing_edges(laplacian(filtered), settings.threshold)
        else:
            detector = sobel if settings.detector == "Sobel" else prewitt
            _, _, magnitude = detector(filtered)
            edges = threshold_edges(magnitude, settings.threshold)
        spectrum = fft_spectrum(grayscale)
        maximum = float(spectrum.max())
        result = {
            "frame_id": settings.frame_id,
            "grayscale_image": _png(grayscale),
            "filtered_image": _png(filtered),
            "edge_image": _png(edges),
            "spectrum_image": _png(spectrum / (maximum if maximum > 0 else 1) * 255),
            "settings_used": settings.model_dump(exclude={"frame_id"}),
        }
        result["processing_time"] = (perf_counter() - started) * 1000
        return result
    finally:
        _processing_slot.release()


def create_live_router(allowed_origins):
    router = APIRouter()
    schema = LiveForm.model_json_schema()
    schema["properties"]["image"] = {"type": "string", "format": "binary"}
    schema["required"].append("image")

    @router.post("/live/frame", response_model=LiveResponse, openapi_extra={
        "requestBody": {"required": True, "content": {
            "multipart/form-data": {"schema": schema},
        }},
    })
    async def live_frame(request: Request):
        """Process one frame; return grayscale, Gaussian, edges, and grayscale FFT."""
        origin = request.headers.get("origin")
        if origin and origin not in allowed_origins:
            raise HTTPException(403, "This website origin is not allowed to submit frames.")
        if request.headers.get("content-type", "").split(";")[0].lower() != "multipart/form-data":
            raise HTTPException(415, "Send multipart/form-data with an image and settings.")
        try:
            form = await _MemoryParser(request.headers, _bounded_stream(request),
                                       max_files=1, max_fields=6, max_part_size=4096).parse()
        except _BodyTooLarge as error:
            raise HTTPException(413, error.message) from error
        except (MultiPartException, ValueError) as error:
            raise HTTPException(400, "Invalid multipart frame upload.") from error
        try:
            if len(form.multi_items()) != len(form):
                raise HTTPException(422, "Duplicate fields are not allowed.")
            image = form.get("image")
            if not isinstance(image, UploadFile):
                raise HTTPException(422, "image must be an uploaded file.")
            try:
                settings = LiveForm.model_validate({key: form[key] for key in form if key != "image"})
            except ValidationError as error:
                detail = "; ".join(f"{item['loc'][0]}: {item['msg']}" for item in error.errors())
                raise HTTPException(422, detail) from error
            payload = await image.read(MAX_UPLOAD_BYTES + 1)
            if not payload:
                raise HTTPException(400, "Image frame is empty.")
            if len(payload) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, "Image frame exceeds 2 MB.")
            return await run_in_threadpool(_process_frame, payload, settings)
        except HTTPException:
            raise
        except Exception as error:
            logger.exception("Live frame processing failed")
            raise HTTPException(500, "Live processing failed. Retry or restart the backend.") from error
        finally:
            await form.close()

    return router
