"""Upload-to-PNG integration tests; no saved uploads or built-in signal filters."""

import base64
from io import BytesIO
import tempfile
import unittest
from unittest.mock import patch

import httpx
import numpy as np
from PIL import Image

from backend import app as api
from backend.signal_ops import fft_spectrum, gaussian_blur

REQUEST_ID = "12345678-1234-1234-1234-123456789abc"


def png_bytes(pixels):
    output = BytesIO()
    Image.fromarray(np.asarray(pixels, dtype=np.uint8)).save(output, format="PNG")
    return output.getvalue()


def decoded_png(url):
    prefix, encoded = url.split(",", 1)
    assert prefix == "data:image/png;base64"
    with Image.open(BytesIO(base64.b64decode(encoded))) as image:
        return np.array(image)


class AnalyzeAPITests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test")
        y, x = np.indices((24, 32))
        self.detail = (((x // 2 + y // 2) % 2) * 255).astype(np.uint8)

    async def asyncTearDown(self):
        await self.client.aclose()

    async def upload(self, payload=None, **parameters):
        data = {"sigma": "1.2", "kernel_size": "7", "request_id": REQUEST_ID, **parameters}
        return await self.client.post(
            "/analyze", files={"image": ("input.png", png_bytes(self.detail) if payload is None else payload, "image/png")}, data=data,
        )

    async def test_upload_calls_manual_modules_and_returns_computed_contract(self):
        with patch.object(api, "gaussian_blur", wraps=gaussian_blur) as blur:
            with patch.object(api, "fft_spectrum", wraps=fft_spectrum) as spectrum:
                response = await self.upload()
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        blur.assert_called_once()
        self.assertEqual(spectrum.call_count, 2)
        self.assertEqual(blur.call_args.args[0].dtype, np.dtype(np.float64))
        self.assertEqual(result["request_id"], REQUEST_ID)
        self.assertEqual(result["provenance"], "computed")
        self.assertEqual(result["parameters_used"], {"sigma": 1.2, "kernel_size": 7})
        self.assertEqual(result["analyzed_dimensions"], {"width": 32, "height": 24})
        self.assertEqual(result["completed_stages"], ["input", "grayscale", "smooth", "fourier"])
        self.assertEqual(result["detection_status"], "not_run")
        self.assertGreater(result["processing_time"], 0)
        for key in ("gx", "gy", "edge_map", "gradient_magnitude", "contour_image", "object_list", "fps"):
            self.assertIsNone(result[key])
        np.testing.assert_array_equal(decoded_png(result["grayscale_image"]), self.detail)
        expected = np.rint(gaussian_blur(self.detail, 1.2, 7)).astype(np.uint8)
        actual = decoded_png(result["filtered_image"])
        np.testing.assert_array_equal(actual, expected)
        self.assertGreater(float(np.abs(actual.astype(float) - self.detail).mean()), 50)
        for key in ("fft_image", "filtered_fft_image"):
            self.assertEqual(decoded_png(result[key]).shape, self.detail.shape)

    async def test_sigma_zero_preserves_grayscale_and_spectra(self):
        result = (await self.upload(sigma="0")).json()
        self.assertEqual(result["filtered_image"], result["grayscale_image"])
        self.assertEqual(result["fft_image"], result["filtered_fft_image"])

    async def test_kernel_choice_affects_actual_blur(self):
        small = (await self.upload(kernel_size="3", sigma="2")).json()
        large = (await self.upload(kernel_size="13", sigma="2")).json()
        self.assertNotEqual(small["filtered_image"], large["filtered_image"])

    async def test_actual_spectra_share_combined_maximum_before_quantization(self):
        result = (await self.upload()).json()
        original = fft_spectrum(self.detail)
        smoothed = fft_spectrum(gaussian_blur(self.detail, 1.2, 7))
        shared_max = max(float(original.max()), float(smoothed.max()))
        self.assertAlmostEqual(result["spectrum_scale"]["max"], shared_max)
        for field, spectrum in (("fft_image", original), ("filtered_fft_image", smoothed)):
            expected = np.rint(spectrum / shared_max * 255).astype(np.uint8)
            np.testing.assert_array_equal(decoded_png(result[field]), expected)

    async def test_unequal_spectrum_maxima_do_not_get_independent_normalization(self):
        spectra = [np.array([[0., 1.], [2., 4.]]), np.array([[0., .25], [.5, 1.]])]
        with patch.object(api, "fft_spectrum", side_effect=spectra):
            result = (await self.upload(png_bytes(np.zeros((2, 2))))).json()
        np.testing.assert_array_equal(decoded_png(result["fft_image"]), [[0, 64], [128, 255]])
        np.testing.assert_array_equal(decoded_png(result["filtered_fft_image"]), [[0, 16], [32, 64]])

    async def test_zero_spectra_are_safely_black(self):
        result = (await self.upload(png_bytes(np.zeros((5, 7))))).json()
        self.assertEqual(result["spectrum_scale"]["max"], 0)
        np.testing.assert_array_equal(decoded_png(result["fft_image"]), np.zeros((5, 7)))
        self.assertEqual(result["fft_image"], result["filtered_fft_image"])

    async def test_preview_bound_preserves_aspect_ratio_without_upscaling(self):
        for height, width, expected in ((300, 900, (512, 171)), (9, 5, (5, 9)), (1, 900, (512, 1))):
            with self.subTest(shape=(height, width)):
                result = (await self.upload(png_bytes(np.zeros((height, width))), sigma="0")).json()
                self.assertEqual(result["source_dimensions"], {"width": width, "height": height})
                self.assertEqual(result["analyzed_dimensions"], {"width": expected[0], "height": expected[1]})
                self.assertEqual(decoded_png(result["original_image"]).shape, (expected[1], expected[0], 3))

    async def test_transparency_is_composited_on_white(self):
        pixels = np.zeros((2, 3, 4), dtype=np.uint8)
        result = (await self.upload(png_bytes(pixels), sigma="0")).json()
        np.testing.assert_array_equal(decoded_png(result["grayscale_image"]), np.full((2, 3), 255))

    async def test_invalid_parameters_and_request_identity(self):
        for name, values in {
            "sigma": ["-1", "5.1", "0.25", "nan", "inf", "abc"],
            "kernel_size": ["0", "2", "33", "3.5", "true"],
            "request_id": ["", "not-a-uuid"],
        }.items():
            for value in values:
                with self.subTest(name=name, value=value):
                    response = await self.upload(**{name: value})
                    self.assertEqual(response.status_code, 422)
                    self.assertIn(name, response.json()["detail"])
        self.assertEqual((await self.upload(sigma="5", kernel_size="31")).status_code, 200)

    async def test_invalid_empty_and_truncated_images(self):
        for payload in (b"", b"not a PNG", png_bytes(self.detail)[:35]):
            with self.subTest(payload=payload[:10]):
                response = await self.upload(payload)
                self.assertEqual(response.status_code, 400)
                self.assertIn("image", response.json()["detail"].lower())

    async def test_file_size_decoded_pixels_and_total_body_are_bounded(self):
        with patch.object(api, "MAX_UPLOAD_BYTES", 10):
            self.assertEqual((await self.upload()).status_code, 413)
        with patch.object(api, "MAX_DECODED_PIXELS", 10):
            self.assertEqual((await self.upload()).status_code, 413)
        with patch.object(api, "MAX_REQUEST_BYTES", 100):
            self.assertEqual((await self.upload()).status_code, 413)

    async def test_multipart_does_not_spool_uploads_to_disk(self):
        generator = np.random.default_rng(220)
        payload = png_bytes(generator.integers(0, 256, (650, 650, 3), dtype=np.uint8))
        self.assertGreater(len(payload), 1024 * 1024)
        with patch.object(tempfile.SpooledTemporaryFile, "rollover", side_effect=AssertionError("disk write")):
            response = await self.upload(payload, sigma="0")
        self.assertEqual(response.status_code, 200, response.text)

    async def test_missing_fields_and_malformed_multipart(self):
        response = await self.client.post("/analyze", files={"image": ("a.png", png_bytes(self.detail))})
        self.assertEqual(response.status_code, 422)
        response = await self.client.post("/analyze", content=b"broken", headers={"Content-Type": "multipart/form-data"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual((await self.client.post("/analyze", json={})).status_code, 415)

    async def test_backend_failure_has_safe_actionable_error(self):
        with patch.object(api, "gaussian_blur", side_effect=RuntimeError("private diagnostic")):
            with self.assertLogs(api.logger, level="ERROR"):
                response = await self.upload()
        self.assertEqual(response.status_code, 500)
        self.assertIn("restart", response.json()["detail"])
        self.assertNotIn("private diagnostic", response.text)

    async def test_busy_engine_rejects_extra_processing(self):
        api._processing_slot.acquire()
        try:
            response = await self.upload()
        finally:
            api._processing_slot.release()
        self.assertEqual(response.status_code, 429)
        self.assertIn("busy", response.json()["detail"])

    async def test_local_cors_and_health(self):
        for origin, expected in (("http://localhost:3000", 200), ("http://127.0.0.1:3000", 200), ("https://untrusted.example", 400)):
            response = await self.client.options("/analyze", headers={"Origin": origin, "Access-Control-Request-Method": "POST"})
            self.assertEqual(response.status_code, expected)
            if expected == 200:
                self.assertEqual(response.headers["access-control-allow-origin"], origin)
        self.assertEqual((await self.client.get("/health")).json()["status"], "ok")
        response = await self.client.post("/analyze", headers={"Origin": "https://untrusted.example"})
        self.assertEqual(response.status_code, 403)


if __name__ == "__main__":
    unittest.main()
