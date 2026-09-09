"""Synthetic regressions, optional portrait evidence and the HTTP contract."""

import base64
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import httpx
import numpy as np

from backend.app import app
from backend import cutout_api as api, edge_cutout as edge
from backend.tests.test_cutout import (
    REQUEST_ID, RECTANGLE, colour_fixture, decoded, encoded, mark,
)


class EdgeCutoutTests(unittest.TestCase):
    def setUp(self):
        self.rgba = colour_fixture()
        self.marks = [mark("keep", 45, 35), mark("remove", 24, 20)]

    def test_manual_gaussian_and_sobel_magnitude_reach_watershed_unchanged(self):
        original = self.rgba.copy()
        with patch.object(edge, "gaussian_blur", wraps=edge.gaussian_blur) as gaussian:
            with patch.object(edge, "sobel", wraps=edge.sobel) as sobel:
                with patch.object(edge, "watershed", wraps=edge.watershed) as watershed:
                    cutout, mask, _, elevation = edge.edge_guided_rgba(self.rgba, RECTANGLE, self.marks, 1.2, 5)
        gaussian.assert_called_once()
        sobel.assert_called_once()
        watershed.assert_called_once()
        rgb = self.rgba[:, :, :3].astype(float)
        gray = rgb @ np.array([0.299, 0.587, 0.114])
        np.testing.assert_allclose(gaussian.call_args.args[0], gray)
        self.assertEqual(gaussian.call_args.args[1:], (1.2, 5))
        expected_filtered = edge.gaussian_blur(gray, 1.2, 5)
        np.testing.assert_allclose(sobel.call_args.args[0], expected_filtered)
        np.testing.assert_allclose(elevation, edge.sobel(expected_filtered)[2], atol=1e-12)
        self.assertIs(watershed.call_args.args[0], elevation)
        self.assertEqual(watershed.call_args.kwargs["connectivity"], 1)
        self.assertEqual(watershed.call_args.kwargs["compactness"], 0)
        self.assertFalse(watershed.call_args.kwargs["watershed_line"])
        self.assertEqual(elevation.dtype, np.float64)
        self.assertGreater(np.unique(elevation).size, 2)
        self.assertEqual((elevation.shape, mask.shape, cutout.shape), ((72, 96), (72, 96), (72, 96, 4)))
        np.testing.assert_array_equal(self.rgba, original)

    def test_moving_only_the_elevation_barrier_changes_segmentation(self):
        # Fixed seeds in two valleys. Moving the ridge moves the catchment
        # boundary; a marker-only or colour-only implementation fails this.
        markers = np.zeros((7, 13), dtype=np.int32)
        markers[3, 1], markers[3, 11] = edge.FOREGROUND, edge.BACKGROUND
        left_ridge = np.zeros((7, 13))
        right_ridge = np.zeros((7, 13))
        left_ridge[:, 4] = 100
        right_ridge[:, 8] = 100
        first = edge.watershed_foreground(left_ridge, markers)
        second = edge.watershed_foreground(right_ridge, markers)
        self.assertFalse(first[3, 6])
        self.assertTrue(second[3, 6])
        self.assertFalse(np.array_equal(first, second))
        self.assertTrue(first[3, 1] and second[3, 1])
        self.assertFalse(first[3, 11] or second[3, 11])

    def test_rectangle_interior_and_centre_are_unknown_not_foreground(self):
        markers = edge.build_markers(self.rgba, RECTANGLE, [mark("keep", 30, 16, 1)])
        self.assertEqual(markers[35, 46], edge.UNKNOWN)
        self.assertEqual(markers[10, 22], edge.UNKNOWN)
        self.assertEqual(markers[0, 0], edge.BACKGROUND)
        self.assertEqual(markers[16, 30], edge.FOREGROUND)

    def test_seed_constraints_outside_keep_and_later_strokes(self):
        for modes, expected in ((["keep", "remove"], 0), (["remove", "keep"], 255)):
            marks = self.marks + [mark(mode, 40, 30, 1) for mode in modes] + [mark("keep", 10, 31, 1)]
            _, mask, _, _ = edge.edge_guided_rgba(self.rgba, RECTANGLE, marks, 0, 3)
            self.assertEqual(mask[30, 40], expected)
            self.assertEqual(mask[31, 10], 255)
            self.assertEqual(mask[0, 0], 0)
            self.assertEqual(mask[35, 45], 255)
            self.assertEqual(mask[20, 24], 0)

    def test_remove_stroke_is_continuous(self):
        stroke = {"mode": "remove", "size": 1, "points": [{"x": 35, "y": 25}, {"x": 55, "y": 25}]}
        _, mask, _, _ = edge.edge_guided_rgba(self.rgba, RECTANGLE, self.marks + [stroke], 0, 3)
        np.testing.assert_array_equal(mask[25, 35:56], 0)

    def test_overwritten_or_transparent_brush_seeds_are_not_recreated_automatically(self):
        for marks in ([mark("keep", 30, 30), mark("remove", 30, 30)],):
            with self.assertRaisesRegex(ValueError, "No Keep pixels remain"):
                edge.build_markers(self.rgba, RECTANGLE, marks)
        self.rgba[:, :, 3] = 0
        with self.assertRaisesRegex(ValueError, "No Keep pixels remain"):
            edge.build_markers(self.rgba, RECTANGLE, self.marks)

    def test_no_brush_uses_manual_magnitude_for_automatic_seeds_and_watershed(self):
        original = self.rgba.copy()
        with patch.object(edge, "automatic_markers", wraps=edge.automatic_markers) as infer:
            with patch.object(edge, "watershed", wraps=edge.watershed) as watershed:
                cutout, mask, _, magnitude = edge.edge_guided_rgba(self.rgba, RECTANGLE, [], 1.2, 5)
        self.assertIs(infer.call_args.args[0], magnitude)
        self.assertIs(watershed.call_args.args[0], magnitude)
        self.assertEqual(mask[35, 45], 255)
        self.assertEqual(mask[20, 24], 0)
        self.assertEqual(mask[0, 0], 0)
        np.testing.assert_array_equal(cutout[:, :, :3], original[:, :, :3])
        np.testing.assert_array_equal(self.rgba, original)
        # No blanket rectangle/filled contour becomes the final foreground.
        expected = np.zeros(mask.shape, bool)
        expected[14:58, 28:65] = True
        intersection = np.count_nonzero((mask > 0) & expected)
        union = np.count_nonzero((mask > 0) | expected)
        self.assertGreater(intersection / union, 0.95)

    def test_automatic_seed_is_not_assumed_to_be_at_rectangle_centre(self):
        self.rgba[:, :, :3] = 20
        self.rgba[15:35, 18:35, :3] = 230
        box = {"x": 5, "y": 5, "width": 85, "height": 62}
        first = edge.edge_guided_rgba(self.rgba, box, [], 0, 3)
        second = edge.edge_guided_rgba(self.rgba, box, [], 0, 3)
        self.assertEqual(first[1][25, 25], 255)
        self.assertEqual(first[1][36, 47], 0)
        np.testing.assert_array_equal(first[1], second[1])

    def test_automatic_holes_and_multiple_objects_are_preserved(self):
        self.rgba[:, :, :3] = 20
        self.rgba[12:60, 22:55, :3] = 230
        self.rgba[25:43, 32:44, :3] = 20
        self.rgba[20:50, 66:85, :3] = 230
        box = {"x": 15, "y": 5, "width": 75, "height": 60}
        _, mask, _, _ = edge.edge_guided_rgba(self.rgba, box, [], 0, 3)
        self.assertEqual(mask[40, 27], 255)
        self.assertEqual(mask[40, 75], 255)
        self.assertEqual(mask[34, 38], 0)
        self.assertEqual(mask[35, 60], 0)

    def test_remove_only_refines_automatic_selection_and_alpha_is_respected(self):
        self.rgba[35, 45, 3] = 100
        self.rgba[36, 45, 3] = 0
        cutout, mask, _, _ = edge.edge_guided_rgba(self.rgba, RECTANGLE, [mark("remove", 35, 25)], 1.2, 5)
        self.assertEqual(mask[25, 35], 0)
        self.assertEqual(cutout[35, 45, 3], 100)
        self.assertEqual(cutout[36, 45, 3], 0)

    def test_flat_or_ambiguous_backgrounds_do_not_invent_an_automatic_object(self):
        for image in (np.full((72, 96, 3), 100, np.uint8),
                      np.repeat((np.arange(96) >= 48)[None, :, None], 72, axis=0).repeat(3, axis=2).astype(np.uint8) * 255):
            self.rgba[:, :, :3] = image
            with self.assertRaisesRegex(ValueError, "No confident automatic selection"):
                edge.edge_guided_rgba(self.rgba, RECTANGLE, [], 1.2, 5)

    def test_multicolour_subject_can_continue_beyond_bottom_of_photo_without_brush(self):
        pixels = np.full((120, 100, 4), 255, dtype=np.uint8)
        pixels[:, :, :3] = [195, 193, 193]
        pixels[8:42, 38:62, :3] = [155, 100, 75]  # Head.
        pixels[38:, 25:75, :3] = [15, 45, 65]  # Jacket reaches photo bottom.
        pixels[42:100, 41:59, :3] = [235, 240, 245]  # Shirt within jacket edges.
        pixels[105:111, 25:75, :3] = [75, 45, 25]  # Another internal colour edge.
        box = {"x": 10, "y": 0, "width": 80, "height": 120}
        cutout, mask, bounds, magnitude = edge.edge_guided_rgba(pixels, box, [], 1.2, 5)
        markers = edge.build_markers(pixels, box, [], magnitude)
        for y, x in ((20, 50), (70, 50), (70, 30), (108, 50), (119, 50)):
            self.assertEqual(mask[y, x], 255)
            self.assertNotEqual(markers[y, x], edge.BACKGROUND)
        self.assertEqual(mask[20, 20], 0)
        self.assertEqual(mask[119, 15], 0)
        self.assertEqual(bounds["y"] + bounds["height"], 120)
        expected = np.zeros(mask.shape, dtype=bool)
        expected[8:42, 38:62] = True
        expected[38:, 25:75] = True
        intersection = np.count_nonzero((mask > 0) & expected)
        union = np.count_nonzero((mask > 0) | expected)
        self.assertGreater(intersection / union, 0.95)
        np.testing.assert_array_equal(cutout[:, :, :3], pixels[:, :, :3])

    def test_enclosed_internal_detail_is_not_automatically_a_background_hole(self):
        self.rgba[:, :, :3] = 20
        self.rgba[12:60, 22:70, :3] = 230
        self.rgba[24:48, 33:58, :3] = [45, 60, 80]  # Printed detail, not background.
        self.rgba[30:42, 39:52, :3] = [180, 90, 45]  # Nested detail.
        box = {"x": 10, "y": 5, "width": 75, "height": 62}
        _, mask, _, _ = edge.edge_guided_rgba(self.rgba, box, [], 1.2, 5)
        for y, x in ((18, 30), (27, 36), (36, 45)):
            self.assertEqual(mask[y, x], 255)
        self.assertEqual(mask[10, 15], 0)

    def test_nearby_background_colours_across_quantization_bins_are_one_model(self):
        y, x = np.indices(self.rgba.shape[:2])
        self.rgba[:, :, :3] = (190 + (x + y) % 5)[:, :, None]
        self.rgba[14:58, 28:65, :3] = [25, 60, 90]
        _, mask, _, _ = edge.edge_guided_rgba(self.rgba, RECTANGLE, [], 1.2, 5)
        np.testing.assert_array_equal(mask[30:40, 40:50], 255)
        np.testing.assert_array_equal(mask[10:12, 22:68], 0)

    def test_full_frame_automatic_selection_infers_background_margin(self):
        box = {"x": 0, "y": 0, "width": 96, "height": 72}
        _, mask, _, _ = edge.edge_guided_rgba(self.rgba, box, [], 1.2, 5)
        self.assertEqual(mask[0, 0], 0)
        self.assertEqual(mask[35, 45], 255)

    def test_full_frame_requires_explicit_background_and_does_not_invent_seeds(self):
        full_frame = {"x": 0, "y": 0, "width": 96, "height": 72}
        with self.assertRaisesRegex(ValueError, "Paint a Remove stroke"):
            edge.build_markers(self.rgba, full_frame, [mark("keep", 45, 35)])
        edge.edge_guided_rgba(self.rgba, full_frame, self.marks, 0, 3)

    def test_constant_zero_gradients_are_finite_and_deterministic(self):
        for colour in (0, 127, 255):
            self.rgba[:, :, :3] = colour
            for sigma in (0, 1.2):
                first = edge.edge_guided_rgba(self.rgba, RECTANGLE, self.marks, sigma, 5)
                second = edge.edge_guided_rgba(self.rgba, RECTANGLE, self.marks, sigma, 5)
                np.testing.assert_array_equal(first[3], 0)
                np.testing.assert_array_equal(first[1], second[1])
                self.assertEqual(first[1][35, 45], 255)
                self.assertEqual(first[1][20, 24], 0)

    def test_holes_and_multiple_foreground_components_are_preserved(self):
        # Two bright objects, with a dark hole in one. No largest-component
        # filter or blanket hole filling is permitted after watershed.
        self.rgba[:, :, :3] = 20
        self.rgba[12:60, 22:55, :3] = 230
        self.rgba[25:43, 32:44, :3] = 20
        self.rgba[20:50, 66:85, :3] = 230
        rectangle = {"x": 15, "y": 5, "width": 75, "height": 60}
        marks = [mark("keep", 26, 20, 1), mark("keep", 72, 25, 1), mark("remove", 37, 32, 1)]
        _, mask, _, _ = edge.edge_guided_rgba(self.rgba, rectangle, marks, 0, 3)
        self.assertEqual(mask[40, 27], 255)
        self.assertEqual(mask[40, 75], 255)
        self.assertEqual(mask[34, 38], 0)
        self.assertEqual(mask[35, 60], 0)

    def test_original_rgb_and_partial_alpha_are_preserved(self):
        self.rgba[35, 45, 3] = 100
        self.rgba[36, 45, 3] = 0
        before = self.rgba.copy()
        cutout, mask, _, _ = edge.edge_guided_rgba(self.rgba, RECTANGLE, self.marks, 1.2, 5)
        np.testing.assert_array_equal(cutout[:, :, :3], before[:, :, :3])
        np.testing.assert_array_equal(cutout[:, :, 3], np.where(mask > 0, before[:, :, 3], 0))
        self.assertEqual(cutout[35, 45, 3], 100)
        self.assertEqual(mask[36, 45], 0)
        np.testing.assert_array_equal(self.rgba, before)


class EdgeCutoutApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")

    async def asyncTearDown(self):
        await self.client.aclose()

    async def prepare(self, pixels):
        response = await self.client.post("/cutout/prepare", files={"image": ("photo.png", encoded(pixels))}, data={"request_id": REQUEST_ID})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    async def extract(self, prepared, **changes):
        settings = {"request_id": REQUEST_ID, "image_id": prepared["image_id"],
                    "width": prepared["width"], "height": prepared["height"],
                    "method": "edge-watershed", "sigma": 1.2, "kernel_size": 5,
                    "rectangle": RECTANGLE, "marks": [mark("keep", 45, 35)], **changes}
        return await self.client.post("/cutout/extract",
                                      files={"image": ("prepared.png", base64.b64decode(prepared["prepared_image"].split(",")[1]))},
                                      data={"settings": json.dumps(settings)})

    async def test_response_exact_guidance_parameters_rgba_mask_and_crop(self):
        pixels = colour_fixture()
        pixels[35, 45, 3], pixels[36, 45, 3] = 100, 0
        prepared = await self.prepare(pixels)
        with patch.object(edge, "watershed", wraps=edge.watershed) as watershed:
            response = await self.extract(prepared)
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["method"], "edge-watershed")
        self.assertEqual(result["algorithm"], "skimage-watershed")
        self.assertEqual(result["parameters_used"], {"sigma": 1.2, "kernel_size": 5})
        self.assertEqual(result["request_id"], REQUEST_ID)
        self.assertEqual(result["image_id"], prepared["image_id"])
        self.assertEqual((result["width"], result["height"]), (96, 72))
        self.assertGreater(result["processing_time"], 0)
        rgba, mode = decoded(result["cutout_image"])
        cropped, crop_mode = decoded(result["cropped_image"])
        mask, mask_mode = decoded(result["mask_image"])
        guidance, guidance_mode = decoded(result["guidance_image"])
        self.assertEqual((mode, crop_mode, mask_mode, guidance_mode), ("RGBA", "RGBA", "L", "L"))
        self.assertEqual(guidance.shape, mask.shape)
        np.testing.assert_array_equal(rgba[:, :, :3], pixels[:, :, :3])
        np.testing.assert_array_equal(rgba[:, :, 3], np.where(mask > 0, pixels[:, :, 3], 0))
        self.assertEqual(rgba[0, 0, 3], 0)
        self.assertEqual(rgba[35, 45, 3], 100)
        self.assertEqual(rgba[36, 45, 3], 0)
        x, y, width, height = (result["foreground_bounds"][key] for key in ("x", "y", "width", "height"))
        np.testing.assert_array_equal(cropped, rgba[y:y + height, x:x + width])
        self.assertEqual(result["cropped_dimensions"], {"width": width, "height": height})
        elevation = watershed.call_args.args[0]
        self.assertEqual(result["guidance_scale"], {"min": 0, "max": float(elevation.max()), "mapping": "linear_grayscale"})
        np.testing.assert_array_equal(guidance, np.rint(elevation / elevation.max() * 255).astype(np.uint8))

    async def test_zero_guidance_encoding_has_no_divide_by_zero(self):
        pixels = colour_fixture()
        pixels[:, :, :3] = 127
        prepared = await self.prepare(pixels)
        response = await self.extract(prepared)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["guidance_scale"]["max"], 0)
        np.testing.assert_array_equal(decoded(response.json()["guidance_image"])[0], 0)

    async def test_empty_brush_list_extracts_with_automatic_edge_seeds(self):
        prepared = await self.prepare(colour_fixture())
        with patch.object(edge, "gaussian_blur", wraps=edge.gaussian_blur) as gaussian:
            response = await self.extract(prepared, marks=[])
        self.assertEqual(response.status_code, 200, response.text)
        gaussian.assert_called_once()
        self.assertEqual(response.json()["seed_mode"], "automatic")
        self.assertEqual(decoded(response.json()["mask_image"])[0][35, 45], 255)
        manual = await self.extract(prepared)
        self.assertEqual(manual.json()["seed_mode"], "brush")

    async def test_ambiguous_brush_free_request_returns_actionable_error(self):
        pixels = colour_fixture()
        pixels[:, :, :3] = 100
        prepared = await self.prepare(pixels)
        response = await self.extract(prepared, marks=[])
        self.assertEqual(response.status_code, 422, response.text)
        self.assertIn("No confident automatic selection", response.json()["detail"])
        self.assertIn("optionally", response.json()["detail"])

    async def test_invalid_parameters_and_coordinates_are_rejected(self):
        prepared = await self.prepare(colour_fixture())
        changes = [{"method": "unknown"}, {"sigma": None}, {"kernel_size": None}, {"sigma": -0.1},
                   {"sigma": 5.1}, {"sigma": 0.15}, {"sigma": True}, {"sigma": "1.2"},
                   {"sigma": float("nan")}, {"sigma": float("inf")}, {"kernel_size": True},
                   {"kernel_size": 2}, {"kernel_size": 4}, {"kernel_size": 33}, {"kernel_size": 5.0},
                   {"method": "grabcut"}, {"rectangle": {**RECTANGLE, "x": -1}},
                   {"marks": [mark("keep", 96, 35)]}, {"marks": [mark("keep", 45, 72)]},
                   {"width": 95}, {"width": 513}, {"edge_strength": 2}, {"threshold": 96}]
        for change in changes:
            with self.subTest(change=change):
                response = await self.extract(prepared, **change)
                self.assertEqual(response.status_code, 422, response.text)
                self.assertIsInstance(response.json()["detail"], str)

    async def test_parameter_extremes_and_changes_use_the_requested_elevation(self):
        prepared = await self.prepare(colour_fixture())
        original = (await self.extract(prepared, sigma=0, kernel_size=3)).json()
        smoothed = (await self.extract(prepared, sigma=5, kernel_size=31)).json()
        self.assertEqual(original["parameters_used"], {"sigma": 0, "kernel_size": 3})
        self.assertEqual(smoothed["parameters_used"], {"sigma": 5, "kernel_size": 31})
        self.assertNotEqual(original["guidance_image"], smoothed["guidance_image"])
        self.assertGreater(original["guidance_scale"]["max"], smoothed["guidance_scale"]["max"])

    async def test_resizing_occurs_before_selection_and_all_outputs_match(self):
        pixels = np.zeros((64, 1024, 4), dtype=np.uint8)
        pixels[:, :, 3] = 255
        pixels[10:50, 400:600, :3] = 230
        prepared = await self.prepare(pixels)
        self.assertEqual((prepared["width"], prepared["height"]), (512, 32))
        response = await self.extract(prepared, rectangle={"x": 190, "y": 3, "width": 120, "height": 26},
                                      marks=[mark("keep", 250, 16)], sigma=0, kernel_size=3)
        self.assertEqual(response.status_code, 200, response.text)
        for key in ("cutout_image", "mask_image", "guidance_image"):
            self.assertEqual(decoded(response.json()[key])[0].shape[:2], (32, 512))

    async def test_watershed_failure_releases_shared_concurrency_slot(self):
        prepared = await self.prepare(colour_fixture())
        with patch.object(edge, "watershed", side_effect=RuntimeError("private failure")):
            with self.assertLogs(api.logger, level="ERROR"):
                response = await self.extract(prepared)
        self.assertEqual(response.status_code, 500)
        self.assertNotIn("private failure", response.text)
        self.assertEqual((await self.extract(prepared)).status_code, 200)

    @unittest.skipUnless(os.environ.get("SPECTRAEDGE_TEST_PORTRAIT"), "Optional original portrait is not supplied")
    async def test_original_portrait_regression_without_brush_through_api(self):
        # Opt-in local fixture: never copy a user's photograph into the repo.
        photo = Path(os.environ["SPECTRAEDGE_TEST_PORTRAIT"]).read_bytes()
        response = await self.client.post("/cutout/prepare", files={"image": ("portrait.jpg", photo)},
                                          data={"request_id": REQUEST_ID})
        self.assertEqual(response.status_code, 200, response.text)
        prepared = response.json()
        self.assertEqual(prepared["source_dimensions"], {"width": 547, "height": 365})
        self.assertEqual((prepared["width"], prepared["height"]), (512, 342))
        response = await self.extract(prepared, marks=[],
                                      rectangle={"x": 132, "y": 4, "width": 243, "height": 338})
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        pixels, _ = decoded(prepared["prepared_image"])
        cutout, _ = decoded(result["cutout_image"])
        mask, _ = decoded(result["mask_image"])
        self.assertEqual(result["seed_mode"], "automatic")
        self.assertEqual(mask.shape, (342, 512))
        # Manually inspected interior points, not a claim of pixel-perfect IoU.
        for name, x, y in (("face", 260, 60), ("shirt", 260, 200),
                           ("jacket", 190, 220), ("trousers", 260, 335),
                           ("bottom of subject", 260, 341)):
            with self.subTest(part=name):
                self.assertEqual(mask[y, x], 255)
        for x, y in ((145, 50), (365, 30), (145, 335), (368, 335), (333, 224)):
            self.assertEqual(mask[y, x], 0)
        self.assertGreater(np.count_nonzero(mask), 40000)
        self.assertLess(np.count_nonzero(mask), 55000)
        np.testing.assert_array_equal(cutout[:, :, :3], pixels[:, :, :3])
        np.testing.assert_array_equal(cutout[:, :, 3], np.where(mask > 0, pixels[:, :, 3], 0))


if __name__ == "__main__":
    unittest.main()
