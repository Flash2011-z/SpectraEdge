"""Actual GrabCut/HTTP tests on deterministic colour pixels, never mock cutouts."""

import base64
from hashlib import sha256
from io import BytesIO
import json
import tempfile
import unittest
from unittest.mock import patch

import httpx
import numpy as np
from PIL import Image

from backend.app import app
from backend import cutout_api as api
from backend.cutout import grabcut_rgba

REQUEST_ID = "12345678-1234-1234-1234-123456789abc"
RECTANGLE = {"x": 20, "y": 8, "width": 52, "height": 55}


def colour_fixture():
    y, x = np.indices((72, 96))
    pixels = np.empty((72, 96, 4), dtype=np.uint8)
    pixels[:, :, :3] = np.stack([20 + x % 5, 60 + y % 5, 120 + (x + y) % 5], axis=-1)
    pixels[:, :, 3] = 255
    pixels[14:58, 28:65, :3] = [220, 90, 35]
    return pixels


def encoded(pixels, format="PNG", **kwargs):
    buffer = BytesIO()
    Image.fromarray(pixels).save(buffer, format=format, **kwargs)
    return buffer.getvalue()


def decoded(url):
    with Image.open(BytesIO(base64.b64decode(url.split(",")[1]))) as image:
        return np.array(image), image.mode


def mark(mode, x, y, size=5):
    return {"mode": mode, "size": size, "points": [{"x": x, "y": y}]}


class CutoutTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
        self.pixels = colour_fixture()

    async def asyncTearDown(self):
        await self.client.aclose()

    async def prepare(self, payload=None, **data):
        return await self.client.post("/cutout/prepare", files={"image": ("photo.png", encoded(self.pixels) if payload is None else payload)},
                                      data={"request_id": REQUEST_ID, **data})

    async def extract(self, prepared, marks=None, rectangle=None, **changes):
        settings = {"request_id": REQUEST_ID, "image_id": prepared["image_id"], "width": prepared["width"], "height": prepared["height"],
                    "rectangle": RECTANGLE if rectangle is None else rectangle, "marks": [] if marks is None else marks, **changes}
        return await self.client.post("/cutout/extract", files={"image": ("prepared.png", base64.b64decode(prepared["prepared_image"].split(",")[1]))},
                                      data={"settings": json.dumps(settings)})

    async def test_prepare_preserves_rgb_alpha_and_dimensions_without_upscaling(self):
        self.pixels[30, 40, 3] = 0
        self.pixels[31, 41, 3] = 100
        response = await self.prepare()
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual((result["width"], result["height"]), (96, 72))
        self.assertEqual(result["source_dimensions"], {"width": 96, "height": 72})
        self.assertEqual(result["request_id"], REQUEST_ID)
        pixels, mode = decoded(result["prepared_image"])
        self.assertEqual(mode, "RGBA")
        np.testing.assert_array_equal(pixels, self.pixels)
        self.assertEqual(result["image_id"], sha256(base64.b64decode(result["prepared_image"].split(",")[1])).hexdigest())

    async def test_actual_grabcut_returns_transparency_preserved_rgb_and_consistent_crop(self):
        prepared = (await self.prepare()).json()
        with patch.object(api, "grabcut_rgba", wraps=grabcut_rgba) as segment:
            response = await self.extract(prepared)
        self.assertEqual(response.status_code, 200, response.text)
        segment.assert_called_once()
        np.testing.assert_array_equal(segment.call_args.args[0], self.pixels)
        result = response.json()
        rgba, mode = decoded(result["cutout_image"])
        mask, mask_mode = decoded(result["mask_image"])
        cropped, cropped_mode = decoded(result["cropped_image"])
        self.assertEqual((mode, cropped_mode, mask_mode), ("RGBA", "RGBA", "L"))
        self.assertEqual(rgba[0, 0, 3], 0)
        self.assertEqual(rgba[30, 45, 3], 255)
        np.testing.assert_array_equal(rgba[:, :, :3], self.pixels[:, :, :3])
        np.testing.assert_array_equal(mask > 0, rgba[:, :, 3] > 0)
        box = result["foreground_bounds"]
        x, y, w, h = (box[key] for key in ("x", "y", "width", "height"))
        np.testing.assert_array_equal(cropped, rgba[y:y + h, x:x + w])
        self.assertEqual(result["cropped_dimensions"], {"width": w, "height": h})
        self.assertEqual(result["request_id"], REQUEST_ID)
        self.assertEqual(result["image_id"], prepared["image_id"])
        self.assertEqual(result["algorithm"], "opencv-grabcut")
        self.assertGreater(result["processing_time"], 0)

    async def test_keep_restores_pixels_outside_the_initial_rectangle(self):
        self.pixels[28:36, 7:15, :3] = [220, 90, 35]
        prepared = (await self.prepare()).json()
        before = (await self.extract(prepared)).json()
        after = (await self.extract(prepared, [mark("keep", 10, 31)])).json()
        self.assertEqual(decoded(before["cutout_image"])[0][31, 10, 3], 0)
        self.assertEqual(decoded(after["cutout_image"])[0][31, 10, 3], 255)

    async def test_later_keep_remove_marks_override_previous_marks(self):
        prepared = (await self.prepare()).json()
        for modes, expected in ((["keep", "remove"], 0), (["remove", "keep"], 255)):
            result = (await self.extract(prepared, [mark(mode, 40, 30) for mode in modes])).json()
            self.assertEqual(decoded(result["cutout_image"])[0][30, 40, 3], expected)

    async def test_source_transparency_survives_keep_marks_and_partial_alpha_is_preserved(self):
        self.pixels[30, 40, 3] = 0
        self.pixels[31, 41, 3] = 100
        prepared = (await self.prepare()).json()
        result = (await self.extract(prepared, [mark("keep", 40, 30, 9)])).json()
        rgba, _ = decoded(result["cutout_image"])
        self.assertEqual(rgba[30, 40, 3], 0)
        self.assertEqual(rgba[31, 41, 3], 100)
        np.testing.assert_array_equal(rgba[:, :, :3], self.pixels[:, :, :3])

    async def test_brush_lines_are_continuous_and_input_is_unchanged(self):
        before = self.pixels.copy()
        stroke = {"mode": "remove", "size": 3, "points": [{"x": 35, "y": 25}, {"x": 55, "y": 25}]}
        rgba, _, _ = grabcut_rgba(self.pixels, RECTANGLE, [mark("keep", 50, 45), stroke])
        np.testing.assert_array_equal(rgba[25, 35:56, 3], 0)
        np.testing.assert_array_equal(self.pixels, before)

    async def test_repeated_stateless_requests_are_deterministic(self):
        prepared = (await self.prepare()).json()
        first = (await self.extract(prepared, [mark("remove", 40, 30)])).json()
        second = (await self.extract(prepared, [mark("remove", 40, 30)])).json()
        self.assertEqual(first["cutout_image"], second["cutout_image"])

    async def test_preparation_orients_and_resizes_and_extraction_uses_exact_dimensions(self):
        image = np.zeros((40, 1600, 3), dtype=np.uint8)
        image[:] = [15, 90, 150]
        image[10:30, 400:900] = [220, 90, 35]
        prepared = (await self.prepare(encoded(image))).json()
        self.assertEqual((prepared["width"], prepared["height"]), (512, 13))
        response = await self.extract(prepared, rectangle={"x": 115, "y": 1, "width": 190, "height": 11})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(decoded(response.json()["cutout_image"])[0].shape, (13, 512, 4))
        exif = Image.Exif()
        exif[274] = 6
        oriented = (await self.prepare(encoded(self.pixels[:, :, :3], "JPEG", exif=exif))).json()
        self.assertEqual(oriented["source_dimensions"], {"width": 72, "height": 96})
        self.assertEqual((oriented["width"], oriented["height"]), (72, 96))

    async def test_jpeg_webp_and_gif_first_frame_are_accepted(self):
        for format in ("JPEG", "WEBP"):
            response = await self.prepare(encoded(self.pixels[:, :, :3], format))
            self.assertEqual(response.status_code, 200, response.text)
        first = Image.fromarray(self.pixels[:, :, :3])
        second = Image.new("RGB", first.size, "white")
        buffer = BytesIO()
        first.save(buffer, format="GIF", save_all=True, append_images=[second])
        result = (await self.prepare(buffer.getvalue())).json()
        actual, _ = decoded(result["prepared_image"])
        with Image.open(BytesIO(buffer.getvalue())) as gif:
            np.testing.assert_array_equal(actual, np.asarray(gif.convert("RGBA")))

    async def test_insufficient_foreground_or_background_fails_clearly(self):
        prepared = (await self.prepare()).json()
        for rectangle in ({"x": 0, "y": 0, "width": 96, "height": 72}, {"x": 30, "y": 30, "width": 2, "height": 2}):
            response = await self.extract(prepared, rectangle=rectangle)
            self.assertEqual(response.status_code, 422)
            self.assertIn("foreground", response.json()["detail"])
            self.assertIn("background", response.json()["detail"])

    async def test_invalid_rectangles_brushes_dimensions_and_identifiers(self):
        prepared = (await self.prepare()).json()
        changes = [{"rectangle": {**RECTANGLE, key: value}} for key, value in (("x", -1), ("width", 100), ("height", 0), ("x", 1.5), ("x", True))]
        changes += [{"marks": [mark("invalid", 30, 30)]}, {"marks": [mark("keep", 96, 30)]},
                    {"marks": [mark("keep", 30, 72)]}, {"marks": [mark("keep", 30, 30, 0)]},
                    {"marks": [mark("keep", 30, 30, 129)]}, {"marks": [mark("keep", 30, 30, True)]},
                    {"marks": [{"mode": "keep", "size": 10, "points": []}]},
                    {"width": 95}, {"image_id": "0" * 64}, {"request_id": "invalid"}, {"unknown": 1}]
        for change in changes:
            with self.subTest(change=change):
                response = await self.extract(prepared, **change)
                self.assertEqual(response.status_code, 422, response.text)
                self.assertIsInstance(response.json()["detail"], str)

    async def test_mark_and_point_limits(self):
        prepared = (await self.prepare()).json()
        for marks in ([mark("keep", 30, 30)] * 201,
                      [{"mode": "keep", "size": 10, "points": [{"x": 30, "y": 30}] * 2001}],
                      [{"mode": "keep", "size": 10, "points": [{"x": 30, "y": 30}] * 2000}] * 11):
            self.assertEqual((await self.extract(prepared, marks)).status_code, 422)

    async def test_invalid_image_contents_and_unprepared_image_are_rejected(self):
        for payload in (b"", b"not an image", encoded(self.pixels)[:30]):
            self.assertEqual((await self.prepare(payload)).status_code, 400)
        self.assertEqual((await self.prepare(encoded(self.pixels[:, :, :3], "BMP"))).status_code, 415)
        payload = encoded(self.pixels[:, :, :3])
        fake_prepared = {"prepared_image": "data:image/png;base64," + base64.b64encode(payload).decode(),
                         "image_id": sha256(payload).hexdigest(), "width": 96, "height": 72}
        self.assertEqual((await self.extract(fake_prepared)).status_code, 422)

    async def test_upload_pixel_body_limits_and_no_disk_spooling(self):
        with patch.object(api, "MAX_UPLOAD_BYTES", 10):
            self.assertEqual((await self.prepare()).status_code, 413)
        with patch.object(api, "MAX_DECODED_PIXELS", 10):
            self.assertEqual((await self.prepare()).status_code, 413)
        pixels = np.random.default_rng(220).integers(0, 256, (700, 700, 3), dtype=np.uint8)
        payload = encoded(pixels)
        self.assertGreater(len(payload), 1024 * 1024)
        with patch.object(tempfile.SpooledTemporaryFile, "rollover", side_effect=AssertionError("disk write")):
            self.assertEqual((await self.prepare(payload)).status_code, 200)
        with patch.object(api, "MAX_UPLOAD_BYTES", 10):
            self.assertEqual((await self.prepare(payload)).status_code, 413)

    async def test_multipart_origin_and_json_validation(self):
        response = await self.client.post("/cutout/prepare", json={})
        self.assertEqual(response.status_code, 415)
        response = await self.client.post("/cutout/prepare", files={"image": ("a.png", encoded(self.pixels))})
        self.assertEqual(response.status_code, 422)
        response = await self.prepare(request_id="bad")
        self.assertEqual(response.status_code, 422)
        for route in ("prepare", "extract"):
            response = await self.client.post(f"/cutout/{route}", headers={"Origin": "https://untrusted.example"})
            self.assertEqual(response.status_code, 403)
        response = await self.client.post("/cutout/extract", files={"image": ("a.png", encoded(self.pixels))}, data={"settings": "{"})
        self.assertEqual(response.status_code, 422)

    async def test_busy_and_safe_failure_release_the_slot(self):
        api._cutout_slot.acquire()
        try:
            self.assertEqual((await self.prepare()).status_code, 429)
        finally:
            api._cutout_slot.release()
        with patch.object(api, "_prepare", side_effect=RuntimeError("private diagnostic")):
            with self.assertLogs(api.logger, level="ERROR"):
                response = await self.prepare()
        self.assertEqual(response.status_code, 500)
        self.assertNotIn("private diagnostic", response.text)
        self.assertEqual((await self.prepare()).status_code, 200)


if __name__ == "__main__":
    unittest.main()
