"""Stateless, bounded upload endpoints for both Object Cutout methods."""

import base64
from hashlib import sha256
from io import BytesIO
import logging
from threading import BoundedSemaphore
from time import perf_counter
from typing import Literal
from uuid import UUID
import warnings

from fastapi import APIRouter, Depends, HTTPException, Request
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser

from backend.cutout import grabcut_rgba
from backend.edge_cutout import edge_guided_rgba
from backend.ai_cutout import AIUnavailable, MODEL_NAME, ai_cutout_rgba

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_PREPARED_BYTES = 6 * 1024 * 1024
MAX_SETTINGS_BYTES = 1024 * 1024
MAX_DECODED_PIXELS = 20_000_000
# Benchmarked manual Gaussian + Sobel: about 2 seconds at 512, 9 at 1024.
# Both methods share this prepared grid; extraction NEVER resizes it.
MAX_SIDE = 512
MAX_MARKS = 200
MAX_POINTS = 20_000
_cutout_slot = BoundedSemaphore(1)
logger = logging.getLogger(__name__)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Point(StrictModel):
    x: int = Field(strict=True, ge=0, lt=MAX_SIDE)
    y: int = Field(strict=True, ge=0, lt=MAX_SIDE)


class Rectangle(Point):
    width: int = Field(strict=True, ge=2, le=MAX_SIDE)
    height: int = Field(strict=True, ge=2, le=MAX_SIDE)


class BrushMark(StrictModel):
    mode: Literal["keep", "remove"]
    size: int = Field(strict=True, ge=1, le=128)
    points: list[Point] = Field(min_length=1, max_length=2000)


class ExtractSettings(StrictModel):
    request_id: UUID
    image_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    width: int = Field(strict=True, ge=1, le=MAX_SIDE)
    height: int = Field(strict=True, ge=1, le=MAX_SIDE)
    rectangle: Rectangle
    marks: list[BrushMark] = Field(max_length=MAX_MARKS)
    # Omitted method retains the old API contract. The editor explicitly sends
    # edge-watershed by default; GrabCut never silently uses Gaussian settings.
    method: Literal["edge-watershed", "grabcut", "ai-assisted"] = "grabcut"
    sigma: float | None = Field(default=None, strict=True, ge=0, le=5, multiple_of=0.1, allow_inf_nan=False)
    kernel_size: int | None = Field(default=None, strict=True, ge=3, le=31)

    @model_validator(mode="after")
    def method_parameters(self):
        if self.method == "edge-watershed":
            if self.sigma is None or self.kernel_size is None:
                raise ValueError("edge-watershed requires sigma and kernel_size")
            if self.kernel_size % 2 == 0:
                raise ValueError("kernel_size must be odd (3 through 31)")
        elif "sigma" in self.model_fields_set or "kernel_size" in self.model_fields_set:
            raise ValueError("Gaussian parameters apply only to edge-watershed; omit them for GrabCut and AI-assisted cutout")
        return self

    @model_validator(mode="after")
    def coordinates_match_image(self):
        r = self.rectangle
        if r.x + r.width > self.width or r.y + r.height > self.height:
            raise ValueError("rectangle must stay within the prepared image")
        if sum(len(mark.points) for mark in self.marks) > MAX_POINTS:
            raise ValueError(f"use at most {MAX_POINTS} brush points in total")
        for mark in self.marks:
            if any(p.x >= self.width or p.y >= self.height for p in mark.points):
                raise ValueError("brush coordinates must stay within the prepared image")
        return self


def _png_bytes(pixels):
    output = BytesIO()
    Image.fromarray(pixels).save(output, format="PNG")
    return output.getvalue()


def _data_url(payload):
    return "data:image/png;base64," + base64.b64encode(payload).decode("ascii")


def _decode(payload, prepared=False):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(payload)) as image:
                if image.format not in ({"PNG"} if prepared else {"PNG", "JPEG", "WEBP", "GIF"}):
                    raise HTTPException(415, "Use the prepared PNG." if prepared else "Choose a PNG, JPEG, WebP, or GIF image.")
                if image.width * image.height > MAX_DECODED_PIXELS:
                    raise HTTPException(413, "Choose an image at or below 20 megapixels.")
                if prepared and (max(image.size) > MAX_SIDE or image.mode != "RGBA"):
                    raise HTTPException(422, f"Upload again and use the exact prepared RGBA PNG (maximum side {MAX_SIDE}).")
                image.seek(0)
                image.load()
                if prepared:
                    return np.array(image), {"width": image.width, "height": image.height}
                oriented = ImageOps.exif_transpose(image)
                source_dimensions = {"width": oriented.width, "height": oriented.height}
                rgba = oriented.convert("RGBA")
                # Nearest sampling preserves sampled RGB and alpha, including
                # fully transparent pixels. No background is composited here.
                rgba.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.NEAREST)
                return np.array(rgba), source_dimensions
    except (Image.DecompressionBombWarning, Image.DecompressionBombError) as error:
        raise HTTPException(413, "Choose an image at or below 20 megapixels.") from error
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as error:
        raise HTTPException(400, "This file could not be decoded as a valid image.") from error


def _prepare(payload, request_id):
    started = perf_counter()
    rgba, source_dimensions = _decode(payload)
    png = _png_bytes(rgba)
    return {"request_id": str(request_id), "image_id": sha256(png).hexdigest(),
            "prepared_image": _data_url(png), "width": rgba.shape[1], "height": rgba.shape[0],
            "source_dimensions": source_dimensions, "processing_time": round((perf_counter() - started) * 1000, 3)}


def _extract(payload, settings):
    started = perf_counter()
    if sha256(payload).hexdigest() != settings.image_id:
        raise HTTPException(422, "Prepared image changed. Upload again and redraw the selection.")
    rgba, dimensions = _decode(payload, prepared=True)
    if dimensions != {"width": settings.width, "height": settings.height}:
        raise HTTPException(422, "Selection dimensions do not match the prepared image. Upload again.")
    magnitude = None
    parameters = {}
    rectangle = settings.rectangle.model_dump()
    marks = [mark.model_dump() for mark in settings.marks]
    try:
        if settings.method == "edge-watershed":
            parameters = {"sigma": settings.sigma, "kernel_size": settings.kernel_size}
            cutout, mask, bounds, magnitude = edge_guided_rgba(rgba, rectangle, marks, **parameters)
        elif settings.method == "ai-assisted":
            parameters = {"model": MODEL_NAME}
            cutout, mask, bounds = ai_cutout_rgba(rgba, rectangle, marks)
        else:
            cutout, mask, bounds = grabcut_rgba(rgba, rectangle, marks)
    except AIUnavailable as error:
        raise HTTPException(503, str(error)) from error
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    x, y, w, h = (bounds[key] for key in ("x", "y", "width", "height"))
    result = {"request_id": str(settings.request_id), "image_id": settings.image_id,
              "width": settings.width, "height": settings.height,
              "cutout_image": _data_url(_png_bytes(cutout)),
              "cropped_image": _data_url(_png_bytes(cutout[y:y + h, x:x + w])),
              "mask_image": _data_url(_png_bytes(mask)), "foreground_bounds": bounds,
              "cropped_dimensions": {"width": w, "height": h},
              "method": settings.method, "parameters_used": parameters,
              "seed_mode": ("brush" if any(mark["mode"] == "keep" for mark in marks) else "automatic") if magnitude is not None else None,
              "algorithm": "rembg-onnx" if settings.method == "ai-assisted" else "skimage-watershed" if magnitude is not None else "opencv-grabcut",
              "guidance_image": None, "guidance_scale": None}
    if magnitude is not None:
        # Encode the SAME elevation only after segmentation. This linear
        # display mapping loses precision; the algorithm received full floats.
        maximum = float(magnitude.max())
        guidance = np.rint(magnitude / (maximum if maximum > 0 else 1) * 255).astype(np.uint8)
        result["guidance_image"] = _data_url(_png_bytes(guidance))
        result["guidance_scale"] = {"min": 0, "max": maximum, "mapping": "linear_grayscale"}
    result["processing_time"] = round((perf_counter() - started) * 1000, 3)
    return result


class _BodyTooLarge(MultiPartException):
    pass


async def _read_upload(request, fields, file_limit, settings_limit):
    if request.headers.get("content-type", "").split(";")[0].lower() != "multipart/form-data":
        raise HTTPException(415, "Send one image and settings as multipart/form-data.")
    body_limit = file_limit + settings_limit + 64 * 1024

    async def bounded_stream():
        received = 0
        async for chunk in request.stream():
            received += len(chunk)
            if received > body_limit:
                raise _BodyTooLarge("Cutout request exceeds the upload limit.")
            yield chunk

    parser = MultiPartParser(request.headers, bounded_stream(), max_files=1, max_fields=1, max_part_size=settings_limit)
    # Never roll onto disk: bounded_stream rejects the body before this size.
    parser.spool_max_size = body_limit + 1
    try:
        form = await parser.parse()
    except _BodyTooLarge as error:
        raise HTTPException(413, error.message) from error
    except (MultiPartException, ValueError) as error:
        raise HTTPException(400, "Invalid multipart upload or oversized settings. Send one image and the required settings.") from error
    try:
        if len(form.multi_items()) != 2 or set(form) != fields:
            raise HTTPException(422, "Provide exactly " + ", ".join(sorted(fields)) + ".")
        image = form["image"]
        if not isinstance(image, UploadFile):
            raise HTTPException(422, "The image field must contain an uploaded file.")
        text = form[next(key for key in fields if key != "image")]
        if not isinstance(text, str):
            raise HTTPException(422, "Settings must be text, not another uploaded file.")
        payload = await image.read(file_limit + 1)
        if not payload:
            raise HTTPException(400, "This image is empty. Choose another file.")
        if len(payload) > file_limit:
            raise HTTPException(413, "This image exceeds the upload limit.")
        return payload, text
    finally:
        await form.close()


def create_cutout_router(allowed_origins):
    async def admit(request: Request):
        if request.headers.get("origin") and request.headers["origin"] not in allowed_origins:
            raise HTTPException(403, "This website origin is not allowed to submit images.")
        # Admit before parsing, bounding concurrent upload buffers as well as
        # expensive segmentation. Existing /analyze has its own unchanged lock.
        if not _cutout_slot.acquire(blocking=False):
            raise HTTPException(429, "The cutout engine is busy. Try again shortly.")
        try:
            yield
        finally:
            _cutout_slot.release()

    router = APIRouter(prefix="/cutout", tags=["Object Cutout"], dependencies=[Depends(admit)])

    def multipart_schema(setting, description):
        return {"requestBody": {"required": True, "content": {"multipart/form-data": {"schema": {
            "type": "object", "required": ["image", setting], "additionalProperties": False,
            "properties": {"image": {"type": "string", "format": "binary"},
                           setting: {"type": "string", "description": description}},
        }}}}}

    @router.post("/prepare", openapi_extra=multipart_schema("request_id", "Client-generated UUID."))
    async def prepare(request: Request):
        payload, identity = await _read_upload(request, {"image", "request_id"}, MAX_UPLOAD_BYTES, 128)
        try:
            request_id = UUID(identity)
        except ValueError as error:
            raise HTTPException(422, "request_id must be a UUID.") from error
        return await execute(_prepare, payload, request_id)

    @router.post("/extract", openapi_extra=multipart_schema("settings", "JSON: request_id, image_id, width, height, method (edge-watershed|grabcut|ai-assisted), sigma and kernel_size for edge-watershed only, rectangle {x,y,width,height}, marks [{mode: keep|remove, size: 1..128, points: [{x,y}]}]. AI uses a local pretrained portrait model and direct opacity corrections; send a full-image rectangle for unrestricted output. Omitted method means legacy GrabCut. See docs/OBJECT_CUTOUT.md. Send the exact prepared RGBA PNG as image."))
    async def extract(request: Request):
        payload, text = await _read_upload(request, {"image", "settings"}, MAX_PREPARED_BYTES, MAX_SETTINGS_BYTES)
        try:
            settings = ExtractSettings.model_validate_json(text)
        except ValidationError as error:
            first = error.errors()[0]
            field = ".".join(map(str, first["loc"])) or "settings"
            raise HTTPException(422, f"{field}: {first['msg']}") from error
        return await execute(_extract, payload, settings)

    async def execute(function, *args):
        try:
            return await run_in_threadpool(function, *args)
        except HTTPException:
            raise
        except Exception as error:
            logger.exception("Cutout processing failed")
            raise HTTPException(500, "Cutout processing failed. Try another selection or restart the backend.") from error

    return router
