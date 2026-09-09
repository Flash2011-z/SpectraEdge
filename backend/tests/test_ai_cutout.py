"""Optional AI isolation/contract tests. Stub masks do NOT measure model quality."""

import base64
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import httpx
import numpy as np
from PIL import Image

from backend import ai_cutout as ai, cutout_api as api
from backend.app import app
from backend.tests.test_cutout import REQUEST_ID, RECTANGLE, colour_fixture, decoded, encoded, mark


class AICutoutTests(unittest.TestCase):
    def setUp(self):
        self.rgba = colour_fixture()
        self.mask = np.zeros(self.rgba.shape[:2], np.uint8)
        self.mask[14:58, 28:65] = 255
        self.mask[35, 45] = 128

    def test_base_app_import_does_not_import_or_load_optional_ai(self):
        code = "import sys; import backend.app; assert 'rembg' not in sys.modules; assert 'onnxruntime' not in sys.modules"
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=Path(__file__).resolve().parents[2],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_session_is_lazy_cached_and_failed_initialization_can_retry(self):
        session = Mock()
        with patch.object(ai, "_session", None), patch.object(ai, "_load_session", return_value=session) as load:
            load.assert_not_called()
            self.assertIs(ai._get_session(), session)
            self.assertIs(ai._get_session(), session)
            load.assert_called_once()
        with patch.object(ai, "_session", None), patch.object(ai, "_load_session", side_effect=[ai.AIUnavailable("offline"), session]):
            with self.assertRaises(ai.AIUnavailable):
                ai._get_session()
            self.assertIs(ai._get_session(), session)

    def test_missing_optional_packages_have_actionable_error(self):
        with patch.object(ai, "_verified_model_path", return_value="already-checked.onnx"), \
                patch.dict(sys.modules, {"onnxruntime": None}):
            with self.assertRaisesRegex(ai.AIUnavailable, "requirements-ai.txt"):
                ai._load_session()

    def test_runtime_session_uses_verified_local_path_without_download_hook(self):
        class FakePortraitSession:
            def __init__(self, name, options, providers):
                self.path = self.download_models()
                self.name, self.options, self.providers = name, options, providers

            @classmethod
            def download_models(cls):
                raise AssertionError("Upstream network download hook must not run")

        modules = {"onnxruntime": SimpleNamespace(SessionOptions=SimpleNamespace),
                   "rembg.sessions.birefnet_portrait": SimpleNamespace(BiRefNetSessionPortrait=FakePortraitSession)}
        with patch.object(ai, "_verified_model_path", return_value="verified-local.onnx"), patch.dict(sys.modules, modules):
            session = ai._load_session()
        self.assertEqual(session.path, "verified-local.onnx")
        self.assertEqual(session.name, ai.MODEL_NAME)
        self.assertEqual(session.providers, ["CPUExecutionProvider"])
        self.assertEqual(session.options.intra_op_num_threads, 4)
        self.assertEqual(session.options.inter_op_num_threads, 1)
        self.assertFalse(session.options.enable_cpu_mem_arena)

    def test_missing_and_corrupt_model_are_rejected_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(ai, "MODEL_PATH", Path(directory) / "absent.onnx"):
                with self.assertRaisesRegex(ai.AIUnavailable, "prepare_ai"):
                    ai._verified_model_path()
        # An existing source file is deliberately not a valid model; never load it.
        with patch.object(ai, "MODEL_PATH", Path(__file__)):
            with self.assertRaisesRegex(ai.AIUnavailable, "verification failed"):
                ai._verified_model_path()

    def test_soft_mask_preserves_rgb_and_multiplies_existing_alpha(self):
        self.rgba[35, 45, 3] = 100
        self.rgba[36, 45, 3] = 0
        before = self.rgba.copy()
        original_mask = self.mask.copy()
        with patch.object(ai, "_predict_mask", return_value=self.mask):
            cutout, mask, bounds = ai.ai_cutout_rgba(self.rgba, RECTANGLE, [])
        np.testing.assert_array_equal(cutout[:, :, :3], before[:, :, :3])
        np.testing.assert_array_equal(self.rgba, before)
        np.testing.assert_array_equal(self.mask, original_mask)
        self.assertEqual(cutout[35, 45, 3], 50)
        self.assertEqual(mask[35, 45], 128)
        self.assertEqual(cutout[36, 45, 3], 0)
        self.assertEqual(mask[36, 45], 0)
        self.assertEqual(bounds, {"x": 28, "y": 14, "width": 37, "height": 44})

    def test_rectangle_clips_output_and_ordered_brushes_only_edit_opacity(self):
        self.mask[:] = 128
        marks = [mark("keep", 5, 5, 1), mark("remove", 45, 35, 1),
                 mark("keep", 45, 35, 1), mark("remove", 50, 40, 1)]
        with patch.object(ai, "_predict_mask", return_value=self.mask) as predict:
            cutout, mask, _ = ai.ai_cutout_rgba(self.rgba, RECTANGLE, marks)
        predict.assert_called_once_with(self.rgba)
        self.assertEqual(mask[0, 0], 0)
        self.assertEqual(mask[5, 5], 255)
        self.assertEqual(mask[35, 45], 255)
        self.assertEqual(mask[40, 50], 0)
        self.assertEqual(cutout[30, 40, 3], 128)

    def test_empty_output_fails_clearly(self):
        with patch.object(ai, "_predict_mask", return_value=np.zeros_like(self.mask)):
            with self.assertRaisesRegex(ValueError, "AI found no visible foreground"):
                ai.ai_cutout_rgba(self.rgba, RECTANGLE, [])

    def test_model_receives_full_rgb_and_requires_matching_mask(self):
        session = Mock()
        session.predict.return_value = [Image.fromarray(self.mask)]
        with patch.object(ai, "_get_session", return_value=session):
            np.testing.assert_array_equal(ai._predict_mask(self.rgba), self.mask)
            np.testing.assert_array_equal(np.asarray(session.predict.call_args.args[0]), self.rgba[:, :, :3])
            for invalid in ([], [Image.new("L", (2, 2))], [Image.new("RGB", (96, 72))]):
                session.predict.return_value = invalid
                with self.assertRaises(RuntimeError):
                    ai._predict_mask(self.rgba)


class AICutoutApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
        self.pixels = colour_fixture()
        response = await self.client.post("/cutout/prepare", files={"image": ("fixture.png", encoded(self.pixels))},
                                          data={"request_id": REQUEST_ID})
        self.assertEqual(response.status_code, 200, response.text)
        self.prepared = response.json()

    async def asyncTearDown(self):
        await self.client.aclose()

    async def extract(self, **changes):
        settings = {"request_id": REQUEST_ID, "image_id": self.prepared["image_id"],
                    "width": 96, "height": 72, "method": "ai-assisted",
                    "rectangle": RECTANGLE, "marks": [], **changes}
        return await self.client.post("/cutout/extract",
                                      files={"image": ("prepared.png", base64.b64decode(self.prepared["prepared_image"].split(",")[1]))},
                                      data={"settings": json.dumps(settings)})

    async def test_ai_contract_exact_rgb_soft_mask_crop_and_provenance(self):
        mask = np.zeros((72, 96), np.uint8)
        mask[20:50, 30:60] = 128
        with patch.object(ai, "_predict_mask", return_value=mask), \
                patch.object(api, "edge_guided_rgba", side_effect=AssertionError("No hidden edge mode")), \
                patch.object(api, "grabcut_rgba", side_effect=AssertionError("No hidden GrabCut")):
            response = await self.extract()
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["method"], "ai-assisted")
        self.assertEqual(result["algorithm"], "rembg-onnx")
        self.assertEqual(result["parameters_used"], {"model": "birefnet-portrait"})
        for key in ("guidance_image", "guidance_scale", "seed_mode"):
            self.assertIsNone(result[key])
        output, mode = decoded(result["cutout_image"])
        self.assertEqual(mode, "RGBA")
        np.testing.assert_array_equal(output[:, :, :3], self.pixels[:, :, :3])
        np.testing.assert_array_equal(output[:, :, 3], mask)
        np.testing.assert_array_equal(decoded(result["mask_image"])[0], mask)
        np.testing.assert_array_equal(decoded(result["cropped_image"])[0], output[20:50, 30:60])

    async def test_ai_unavailable_returns_503_without_fallback_and_releases_worker(self):
        with patch.object(api, "ai_cutout_rgba", side_effect=ai.AIUnavailable("AI model is not installed.")), \
                patch.object(api, "grabcut_rgba", side_effect=AssertionError("No fallback")):
            response = await self.extract()
            self.assertEqual(response.status_code, 503)
            self.assertIn("AI model", response.json()["detail"])
        with patch.object(api, "ai_cutout_rgba", side_effect=AssertionError("AI must be opt-in")):
            response = await self.extract(method="edge-watershed", sigma=1.2, kernel_size=5)
            self.assertEqual(response.status_code, 200, response.text)
            response = await self.extract(method="grabcut")
            self.assertEqual(response.status_code, 200, response.text)

    async def test_unexpected_ai_error_is_safe_and_releases_worker(self):
        with patch.object(api, "ai_cutout_rgba", side_effect=RuntimeError("private detail")):
            with self.assertLogs(api.logger, level="ERROR"):
                response = await self.extract()
        self.assertEqual(response.status_code, 500)
        self.assertNotIn("private detail", response.text)
        self.assertEqual((await self.extract(method="grabcut")).status_code, 200)

    async def test_ai_rejects_unused_parameters_or_client_selected_model(self):
        for change in ({"sigma": 1.2}, {"kernel_size": 5}, {"model": "withoutbg"}):
            with self.subTest(change=change), patch.object(api, "ai_cutout_rgba") as run:
                self.assertEqual((await self.extract(**change)).status_code, 422)
                run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
