"""Live contract, signal parity, and bounded input integration tests."""

import base64
from io import BytesIO
import unittest
from unittest.mock import patch
from uuid import UUID

import httpx
import numpy as np
from PIL import Image

from backend.app import app
from backend import live_api as live
from backend.detection import sobel, prewitt, laplacian, threshold_edges, zero_crossing_edges
from backend.signal_ops import gaussian_blur, fft_spectrum


def png_bytes(pixels):
    output = BytesIO()
    Image.fromarray(np.asarray(pixels, dtype=np.uint8)).save(output, format="PNG")
    return output.getvalue()


def decode(url):
    prefix, encoded = url.split(",", 1)
    assert prefix == "data:image/png;base64"
    with Image.open(BytesIO(base64.b64decode(encoded, validate=True))) as image:
        assert image.format == "PNG"
        assert image.mode == "L"
        return np.array(image)


class LiveAPITests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
        y, x = np.indices((12, 16))
        self.pixels = (((x // 3 + y // 3) % 2) * 255).astype(np.uint8)

    async def asyncTearDown(self):
        await self.client.aclose()

    async def upload(self, payload=None, **settings):
        return await self.client.post("/live/frame", files={
            "image": ("frame.png", png_bytes(self.pixels) if payload is None else payload, "image/png"),
        }, data={"detector": "Sobel", "sigma": "1.2", "kernel_size": "7",
                 "threshold": "20", **settings})

    async def test_all_detectors_match_existing_signal_functions(self):
        filtered = gaussian_blur(self.pixels, 1.2, 7)
        spectrum = fft_spectrum(self.pixels)
        for name in ("Sobel", "Prewitt", "Laplacian"):
            with self.subTest(detector=name):
                with patch.object(live, "gaussian_blur", wraps=gaussian_blur) as blur, \
                     patch.object(live, "fft_spectrum", wraps=fft_spectrum) as fft:
                    response = await self.upload(detector=name, frame_id="frame-42")
                self.assertEqual(response.status_code, 200, response.text)
                result = response.json()
                self.assertEqual(set(result), {"frame_id", "grayscale_image", "filtered_image",
                                              "edge_image", "spectrum_image", "processing_time", "settings_used"})
                self.assertEqual(result["frame_id"], "frame-42")
                self.assertEqual(result["settings_used"], {
                    "detector": name, "sigma": 1.2, "kernel_size": 7, "threshold": 20,
                })
                self.assertGreater(result["processing_time"], 0)
                blur.assert_called_once()
                fft.assert_called_once()
                np.testing.assert_array_equal(fft.call_args.args[0], self.pixels)
                edges = zero_crossing_edges(laplacian(filtered), 20) if name == "Laplacian" else \
                    threshold_edges((sobel if name == "Sobel" else prewitt)(filtered)[2], 20)
                for key, expected in {
                    "grayscale_image": self.pixels,
                    "filtered_image": np.rint(np.clip(filtered, 0, 255)).astype(np.uint8),
                    "edge_image": edges,
                    "spectrum_image": np.rint(spectrum / spectrum.max() * 255).astype(np.uint8),
                }.items():
                    np.testing.assert_array_equal(decode(result[key]), expected)

    async def test_generated_frame_id_and_black_spectrum(self):
        response = await self.upload(png_bytes(np.zeros((8, 8), dtype=np.uint8)), sigma="0")
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        UUID(result["frame_id"])
        for key in ("grayscale_image", "filtered_image", "edge_image", "spectrum_image"):
            np.testing.assert_array_equal(decode(result[key]), np.zeros((8, 8)))

    async def test_invalid_settings_rejected_before_processing(self):
        for settings in ({"detector": "Canny"}, {"sigma": "nan"}, {"sigma": "-1"},
                         {"kernel_size": "4"}, {"kernel_size": "33"}, {"threshold": "inf"},
                         {"threshold": "-1"}, {"multi_scale": "true"}, {"frame_id": ""}):
            with self.subTest(settings=settings), patch.object(live, "_process_frame") as process:
                response = await self.upload(**settings)
                self.assertEqual(response.status_code, 422, response.text)
                process.assert_not_called()

    async def test_missing_settings_rejected(self):
        response = await self.client.post("/live/frame", files={"image": ("f.png", png_bytes(self.pixels))})
        self.assertEqual(response.status_code, 422)

    async def test_bad_and_oversized_frames(self):
        for payload, status in ((b"", 400), (b"not an image", 400),
                                (b"x" * (live.MAX_UPLOAD_BYTES + 1), 413),
                                (b"x" * (live.MAX_REQUEST_BYTES + 1), 413)):
            with self.subTest(size=len(payload)):
                self.assertEqual((await self.upload(payload)).status_code, status)
        with patch.object(live, "MAX_DECODED_PIXELS", 10):
            self.assertEqual((await self.upload()).status_code, 413)

    async def test_resize_and_rgb_grayscale(self):
        pixels = np.zeros((100, 400, 3), dtype=np.uint8)
        pixels[:, :, 0] = 255
        response = await self.upload(png_bytes(pixels), sigma="0")
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        expected_gray = np.asarray(Image.fromarray(pixels).convert("L"))[0, 0]
        for key in ("grayscale_image", "filtered_image", "edge_image", "spectrum_image"):
            self.assertEqual(decode(result[key]).shape, (64, 256))
        self.assertTrue(np.all(decode(result["grayscale_image"]) == expected_gray))

    async def test_busy_and_failure_release(self):
        live._processing_slot.acquire()
        try:
            self.assertEqual((await self.upload()).status_code, 429)
        finally:
            live._processing_slot.release()
        self.assertEqual((await self.upload(b"invalid")).status_code, 400)
        self.assertEqual((await self.upload()).status_code, 200)

    async def test_processing_failure_returns_safe_json_and_releases_slot(self):
        with self.assertLogs(live.logger, level="ERROR"), \
             patch.object(live, "gaussian_blur", side_effect=RuntimeError("internal diagnostic")):
            response = await self.upload()
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {
            "detail": "Live processing failed. Retry or restart the backend.",
        })
        self.assertEqual((await self.upload()).status_code, 200)

    async def test_content_type_and_origin(self):
        self.assertEqual((await self.client.post("/live/frame", json={})).status_code, 415)
        response = await self.client.post("/live/frame", headers={"Origin": "https://untrusted.test"})
        self.assertEqual(response.status_code, 403)
