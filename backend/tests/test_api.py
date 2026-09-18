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
from backend.signal_ops import add_gaussian_noise, add_salt_pepper_noise, fft_spectrum, gaussian_blur
from backend.detection import laplacian, prewitt, sobel, threshold_edges, zero_crossing_edges
from backend.analysis import analyze_objects, filter_small_components, multi_scale_edges

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
        self.assertIsNone(result["detector_metadata"])
        self.assertIsNone(result["noisy_image"])
        self.assertEqual(result["noise"], {"model": "None", "strength": 0, "units": "none", "seed": None})
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

    async def test_gaussian_noise_is_reproducible_and_drives_the_analysis_input(self):
        settings = {"noise_model": "Gaussian", "noise_strength": "12", "noise_seed": "220"}
        first = (await self.upload(**settings)).json()
        second = (await self.upload(**settings)).json()
        changed = (await self.upload(**{**settings, "noise_seed": "221"})).json()
        self.assertEqual(first["noisy_image"], second["noisy_image"])
        self.assertEqual(first["filtered_image"], second["filtered_image"])
        self.assertNotEqual(first["noisy_image"], changed["noisy_image"])
        self.assertNotEqual(first["grayscale_image"], first["noisy_image"])
        expected_noisy = add_gaussian_noise(self.detail.astype(float), 12, 220)
        # Analysis must smooth the unbounded float signal, not its clipped PNG.
        expected_filtered = gaussian_blur(expected_noisy, 1.2, 7)
        np.testing.assert_array_equal(
            decoded_png(first["filtered_image"]),
            np.rint(np.clip(expected_filtered, 0, 255)).astype(np.uint8),
        )
        np.testing.assert_array_equal(
            decoded_png(first["noisy_image"]),
            np.rint(np.clip(expected_noisy, 0, 255)).astype(np.uint8),
        )
        self.assertEqual(first["noise"], {
            "model": "Gaussian", "strength": 12, "units": "intensity standard deviation", "seed": 220,
        })
        self.assertEqual(first["parameters_used"], {
            "sigma": 1.2, "kernel_size": 7, "noise_model": "Gaussian",
            "noise_strength": 12, "noise_seed": 220,
        })
        self.assertEqual(first["completed_stages"], ["input", "grayscale", "noise", "smooth", "fourier"])
        np.testing.assert_array_equal(decoded_png(first["grayscale_image"]), self.detail)

    async def test_zero_strength_preserves_all_clean_analysis_outputs(self):
        for detector in ("Sobel", "Prewitt", "Laplacian"):
            settings = {"detector": detector, "threshold": "20", "multi_scale": "true",
                        "scale_sigmas": "0,1.2", "scale_support": "1"}
            clean = (await self.upload(**settings)).json()
            for model in ("Gaussian", "Salt & Pepper"):
                with self.subTest(detector=detector, model=model):
                    response = await self.upload(**settings, noise_model=model,
                                                 noise_strength="0", noise_seed="220")
                    self.assertEqual(response.status_code, 200, response.text)
                    noisy = response.json()
                    self.assertEqual(noisy["noisy_image"], clean["grayscale_image"])
                    for key in ("original_image", "grayscale_image", "filtered_image",
                                "fft_image", "filtered_fft_image", "gx", "gy",
                                "gradient_magnitude", "laplacian_response", "edge_map",
                                "contour_image", "object_list", "multi_scale"):
                        self.assertEqual(noisy.get(key), clean.get(key), key)

    async def test_salt_pepper_noise_enters_detector_and_multiscale_from_same_array(self):
        settings = {"noise_model": "Salt & Pepper", "noise_strength": "0.2", "noise_seed": "17",
                    "detector": "Sobel", "threshold": "96", "multi_scale": "true",
                    "scale_sigmas": "0,1.2", "scale_support": "1"}
        noisy = add_salt_pepper_noise(self.detail.astype(float), 0.2, 17)
        with patch.object(api, "multi_scale_edges", wraps=multi_scale_edges) as scales:
            response = await self.upload(**settings)
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        np.testing.assert_array_equal(scales.call_args.args[0], noisy)
        np.testing.assert_array_equal(decoded_png(result["noisy_image"]), noisy)
        self.assertEqual(result["noise"], {
            "model": "Salt & Pepper", "strength": 0.2,
            "units": "pixel corruption probability", "seed": 17,
        })
        self.assertEqual(result["completed_stages"], [
            "input", "grayscale", "noise", "smooth", "sobel", "threshold",
            "multi_scale", "contours", "objects", "fourier",
        ])

    async def test_single_scale_noise_detector_reports_noise_stage(self):
        for detector, decision in (("Sobel", "threshold"), ("Prewitt", "threshold"),
                                   ("Laplacian", "zero_crossing")):
            with self.subTest(detector=detector):
                result = (await self.upload(
                    detector=detector, threshold="20", noise_model="Gaussian",
                    noise_strength="8", noise_seed="4",
                )).json()
                self.assertEqual(result["completed_stages"], [
                    "input", "grayscale", "noise", "smooth", detector.lower(), decision,
                    "contours", "objects", "fourier",
                ])

    async def test_invalid_or_incomplete_noise_settings_are_rejected(self):
        cases = [
            {"noise_model": "Gaussian"},
            {"noise_model": "Gaussian", "noise_strength": "12"},
            {"noise_strength": "12", "noise_seed": "1"},
            {"noise_model": "Salt & Pepper", "noise_strength": "1.1", "noise_seed": "1"},
            {"noise_model": "Gaussian", "noise_strength": "-1", "noise_seed": "1"},
            {"noise_model": "Gaussian", "noise_strength": "nan", "noise_seed": "1"},
            {"noise_model": "Gaussian", "noise_strength": "1", "noise_seed": "-1"},
            {"noise_model": "Speckle", "noise_strength": "1", "noise_seed": "1"},
        ]
        for settings in cases:
            with self.subTest(settings=settings):
                response = await self.upload(**settings)
                self.assertEqual(response.status_code, 422, response.text)

    async def test_sobel_uses_same_float_filtered_signal_and_returns_actual_outputs(self):
        filtered = gaussian_blur(self.detail, 1.2, 7)
        with patch.object(api, "gaussian_blur", return_value=filtered):
            with patch.object(api, "sobel", wraps=sobel) as detector:
                with patch.object(api, "analyze_objects", wraps=analyze_objects) as objects:
                    with patch.object(api, "fft_spectrum", wraps=fft_spectrum) as spectrum:
                        response = await self.upload(detector="Sobel", threshold="96")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIs(detector.call_args.args[0], filtered)
        self.assertIs(spectrum.call_args_list[1].args[0], filtered)
        np.testing.assert_array_equal(objects.call_args.args[0],
                                      threshold_edges(sobel(filtered)[2], 96))
        self.assertEqual(spectrum.call_count, 2)
        result = response.json()
        self.assertEqual(result["detection_status"], "edges_computed")
        self.assertEqual(result["parameters_used"], {"sigma": 1.2, "kernel_size": 7, "detector": "Sobel", "threshold": 96})
        self.assertEqual(result["completed_stages"], ["input", "grayscale", "smooth", "sobel",
                                                       "threshold", "contours", "objects", "fourier"])
        self.assertEqual(result["request_id"], REQUEST_ID)
        self.assertIsNone(result["fps"])
        gx, gy, magnitude = sobel(filtered)
        expected = {
            "gx": (gx / 1020 + 1) * 127.5,
            "gy": (gy / 1020 + 1) * 127.5,
            "gradient_magnitude": magnitude / (1020 * np.sqrt(2)) * 255,
            "edge_map": threshold_edges(magnitude, 96),
        }
        for field, values in expected.items():
            np.testing.assert_array_equal(decoded_png(result[field]), np.rint(values).astype(np.uint8))
        analysis = analyze_objects(expected["edge_map"])
        expected_contours = np.where(
            analysis.outer_boundaries & analysis.hole_boundaries, 192,
            np.where(analysis.outer_boundaries, 255,
                     np.where(analysis.hole_boundaries, 128, 0)),
        )
        np.testing.assert_array_equal(decoded_png(result["contour_image"]), expected_contours)
        self.assertEqual(result["object_list"], analysis.objects)
        gaussian_only = (await self.upload()).json()
        for field in ("original_image", "grayscale_image", "filtered_image", "fft_image", "filtered_fft_image", "spectrum_scale", "analyzed_dimensions"):
            self.assertEqual(result[field], gaussian_only[field])

    async def test_known_step_has_fixed_signed_and_magnitude_display_scales(self):
        for row, expected in (([0, 0, 255, 255, 255], 255), ([255, 255, 0, 0, 0], 0)):
            result = (await self.upload(png_bytes(np.tile(row, (3, 1))), sigma="0", detector="Sobel", threshold="96")).json()
            np.testing.assert_array_equal(decoded_png(result["gx"]), np.tile([128, expected, expected, 128, 128], (3, 1)))
            np.testing.assert_array_equal(decoded_png(result["gy"]), np.full((3, 5), 128))
            # 1020/(1020*sqrt(2))*255 rounds to 180, not per-image white.
            np.testing.assert_array_equal(decoded_png(result["gradient_magnitude"]), np.tile([0, 180, 180, 0, 0], (3, 1)))
            np.testing.assert_array_equal(decoded_png(result["edge_map"]), np.tile([0, 255, 255, 0, 0], (3, 1)))

    async def test_flat_sobel_image_has_no_edges_even_at_zero_threshold(self):
        for sigma in ("0", "1.2"):
            result = (await self.upload(png_bytes(np.full((5, 7), 123)), sigma=sigma, detector="Sobel", threshold="0")).json()
            np.testing.assert_array_equal(decoded_png(result["edge_map"]), np.zeros((5, 7)))
            np.testing.assert_array_equal(decoded_png(result["gradient_magnitude"]), np.zeros((5, 7)))

    async def test_raw_threshold_equality_and_monotonic_selection(self):
        payload = png_bytes(np.tile([0, 0, 255, 255, 255], (3, 1)))
        previous = np.ones((3, 5), dtype=bool)
        for threshold in (0, 96, 1019, 1020, 1443):
            result = (await self.upload(payload, sigma="0", detector="Sobel", threshold=str(threshold))).json()
            selected = decoded_png(result["edge_map"]) != 0
            self.assertFalse(np.any(selected & ~previous))
            self.assertEqual(int(selected.sum()), 6 if threshold < 1020 else 0)
            previous = selected

    async def test_invalid_or_incomplete_detection_settings(self):
        cases = [{"detector": name, "threshold": "96"} for name in ("sobel", "prewitt", "laplacian", "", "unknown")]
        cases += [{"detector": "Sobel", "threshold": value} for value in ("-1", "1443.1", "nan", "inf", "true", "abc", "")]
        cases += [{"detector": name} for name in ("Sobel", "Prewitt", "Laplacian")]
        cases += [{"threshold": "96"}]
        for parameters in cases:
            with self.subTest(parameters=parameters):
                response = await self.upload(**parameters)
                self.assertEqual(response.status_code, 422, response.text)
                self.assertIsInstance(response.json()["detail"], str)
                self.assertTrue(any(name in response.json()["detail"] for name in ("detector", "threshold")))
        self.assertEqual((await self.upload(detector="Sobel", threshold="96.5")).status_code, 200)

    async def test_extended_multipart_rejects_duplicates_unknown_fields_and_extra_files(self):
        fields = [("image", ("a.png", png_bytes(self.detail), "image/png")),
                  ("sigma", (None, "0")), ("kernel_size", (None, "3")),
                  ("request_id", (None, REQUEST_ID))]
        for extra, expected in (([("sigma", (None, "1"))], 422),
                                ([("unexpected", (None, "1"))], 422),
                                ([("image", ("b.png", png_bytes(self.detail)))], 400),
                                ([("detector", (None, "Sobel")), ("threshold", (None, "96")), ("extra", (None, "1"))], 422)):
            response = await self.client.post("/analyze", files=fields + extra)
            self.assertEqual(response.status_code, expected, response.text)

    async def test_sobel_upload_retains_size_limit_and_openapi_contract(self):
        with patch.object(api, "MAX_REQUEST_BYTES", 100):
            self.assertEqual((await self.upload(detector="Sobel", threshold="96")).status_code, 413)
        schema = api.app.openapi()["paths"]["/analyze"]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]
        self.assertEqual(set(schema["required"]), {"image", "sigma", "kernel_size", "request_id"})
        self.assertEqual(schema["properties"]["detector"]["enum"], ["Sobel", "Prewitt", "Laplacian"])
        self.assertEqual(schema["properties"]["threshold"]["maximum"], 1443)
        self.assertEqual(schema["properties"]["laplacian_min_component_area"]["default"], 2)
        response = api.app.openapi()["paths"]["/analyze"]["post"]["responses"]["200"]
        self.assertIn("AnalysisResponse", response["content"]["application/json"]["schema"]["$ref"])

    async def test_new_detectors_use_float_pipeline_before_encoding(self):
        filtered = gaussian_blur(self.detail, 1.2, 7)
        for name, function in (("Prewitt", prewitt), ("Laplacian", laplacian)):
            with self.subTest(detector=name):
                with patch.object(api, "gaussian_blur", return_value=filtered):
                    with patch.object(api, name.lower(), wraps=function) as detector:
                        with patch.object(api, "fft_spectrum", wraps=fft_spectrum) as spectrum:
                            response = await self.upload(detector=name, threshold="0.5")
                self.assertEqual(response.status_code, 200, response.text)
                self.assertIs(detector.call_args.args[0], filtered)
                self.assertIs(spectrum.call_args_list[1].args[0], filtered)
                result = response.json()
                self.assertEqual(result["parameters_used"],
                                 {"sigma": 1.2, "kernel_size": 7, "detector": name, "threshold": 0.5})
                self.assertEqual(result["detection_status"], "edges_computed")
                decision = "threshold" if name == "Prewitt" else "zero_crossing"
                self.assertEqual(result["completed_stages"],
                                 ["input", "grayscale", "smooth", name.lower(), decision,
                                  "contours", "objects", "fourier"])
                if name == "Prewitt":
                    gx, gy, magnitude = prewitt(filtered)
                    expected = {"gx": (gx / 765 + 1) * 127.5,
                                "gy": (gy / 765 + 1) * 127.5,
                                "gradient_magnitude": magnitude / (765 * np.sqrt(2)) * 255,
                                "edge_map": threshold_edges(magnitude, 0.5)}
                else:
                    raw = laplacian(filtered)
                    expected = {"laplacian_response": (raw / 1020 + 1) * 127.5,
                                "edge_map": zero_crossing_edges(raw, 0.5)}
                    for field in ("gx", "gy", "gradient_magnitude"):
                        self.assertIsNone(result[field])
                for field, values in expected.items():
                    np.testing.assert_array_equal(decoded_png(result[field]),
                                                  np.rint(values).astype(np.uint8))
                analysis_mask = (filter_small_components(expected["edge_map"], 2)
                                 if name == "Laplacian" else expected["edge_map"])
                analysis = analyze_objects(analysis_mask)
                self.assertEqual(result["object_list"], analysis.objects)
                self.assertEqual(decoded_png(result["contour_image"]).shape, filtered.shape)

    async def test_detector_threshold_metadata_has_detector_specific_meaning(self):
        for detector, threshold_type, label, units in (
            ("Sobel", "gradient_magnitude", "Gradient magnitude threshold", "raw_gradient_magnitude"),
            ("Prewitt", "gradient_magnitude", "Gradient magnitude threshold", "raw_gradient_magnitude"),
            ("Laplacian", "zero_crossing_contrast", "Zero-crossing contrast threshold", "raw_response_difference"),
        ):
            with self.subTest(detector=detector):
                result = (await self.upload(detector=detector, threshold="20")).json()
                metadata = result["detector_metadata"]
                self.assertEqual(metadata["detector"], detector)
                self.assertEqual(metadata["threshold_type"], threshold_type)
                self.assertEqual(metadata["threshold_label"], label)
                self.assertEqual(metadata["threshold_units"], units)
                self.assertEqual(metadata["threshold"], 20)
                self.assertEqual(metadata["minimum_component_area"],
                                 2 if detector == "Laplacian" else None)

    async def test_laplacian_cleanup_preserves_raw_edge_map_and_filters_objects(self):
        raw = np.zeros(self.detail.shape, dtype=np.uint8)
        raw[0, 0] = 255
        raw[5, 5:8] = 255
        with patch.object(api, "zero_crossing_edges", return_value=raw):
            default = (await self.upload(detector="Laplacian", threshold="20")).json()
            strict = (await self.upload(
                detector="Laplacian", threshold="20", laplacian_min_component_area="4",
            )).json()
        np.testing.assert_array_equal(decoded_png(default["edge_map"]), raw)
        np.testing.assert_array_equal(decoded_png(strict["edge_map"]), raw)
        self.assertEqual([item["area"] for item in default["object_list"]], [3])
        self.assertEqual(strict["object_list"], [])
        self.assertEqual(strict["parameters_used"]["laplacian_min_component_area"], 4)

    async def test_new_detectors_flat_images_and_strict_step_thresholds(self):
        for name, cutoff in (("Prewitt", 765), ("Laplacian", 510)):
            for sigma in ("0", "1.2"):
                response = await self.upload(png_bytes(np.full((5, 7), 123)),
                                             sigma=sigma, detector=name, threshold="0")
                self.assertEqual(response.status_code, 200, response.text)
                np.testing.assert_array_equal(decoded_png(response.json()["edge_map"]), 0)
                np.testing.assert_array_equal(decoded_png(response.json()["contour_image"]), 0)
                self.assertEqual(response.json()["object_list"], [])
            payload = png_bytes(np.tile([0, 0, 255, 255, 255], (3, 1)))
            for threshold in (cutoff - 0.1, cutoff):
                response = await self.upload(payload, sigma="0", detector=name, threshold=str(threshold))
                self.assertEqual(response.status_code, 200, response.text)
                count = np.count_nonzero(decoded_png(response.json()["edge_map"]))
                self.assertEqual(count, 6 if threshold < cutoff else 0)

    async def test_multi_scale_preserves_single_scale_displays_and_returns_fusion(self):
        settings = {"detector": "Sobel", "threshold": "96", "multi_scale": "true",
                    "scale_sigmas": "0,1.2,2", "scale_support": "2"}
        result = (await self.upload(**settings)).json()
        single = (await self.upload(detector="Sobel", threshold="96")).json()
        for field in ("filtered_image", "gx", "gy", "gradient_magnitude", "edge_map",
                      "fft_image", "filtered_fft_image", "spectrum_scale"):
            self.assertEqual(result[field], single[field])

        expected = multi_scale_edges(self.detail.astype(float), "Sobel", [0, 1.2, 2], 96, 2)
        multi = result["multi_scale"]
        self.assertEqual(multi["sigmas"], [0, 1.2, 2])
        self.assertEqual(multi["support_count"], 2)
        self.assertEqual(multi["persistence_scale"],
                         {"min": 0, "max": 3, "mapping": "linear_grayscale"})
        self.assertEqual([scale["kernel_size"] for scale in multi["scales"]], [3, 9, 13])
        for actual, scale in zip(multi["scales"], expected.scales):
            self.assertEqual(actual["sigma"], scale.sigma)
            np.testing.assert_array_equal(decoded_png(actual["edge_map"]), scale.edge_map)
        np.testing.assert_array_equal(
            decoded_png(multi["persistence_map"]),
            np.rint(expected.persistence_map.astype(float) / 3 * 255).astype(np.uint8),
        )
        np.testing.assert_array_equal(decoded_png(multi["fused_edge_map"]), expected.fused_edge_map)
        self.assertEqual(result["object_list"], analyze_objects(expected.fused_edge_map).objects)
        self.assertEqual(result["parameters_used"], {
            "sigma": 1.2, "kernel_size": 7, "detector": "Sobel", "threshold": 96,
            "multi_scale": True, "scale_sigmas": [0, 1.2, 2], "scale_support": 2,
        })
        self.assertEqual(result["completed_stages"], [
            "input", "grayscale", "smooth", "sobel", "threshold", "multi_scale",
            "contours", "objects", "fourier",
        ])

    async def test_multi_scale_single_sigma_defaults_to_one_and_matches_scale_edge(self):
        result = (await self.upload(detector="Laplacian", threshold="20", multi_scale="true",
                                    scale_sigmas="1.2")).json()
        multi = result["multi_scale"]
        self.assertEqual(multi["support_count"], 1)
        self.assertEqual(multi["fused_edge_map"], multi["scales"][0]["edge_map"])
        self.assertEqual(result["parameters_used"]["scale_support"], 1)

    async def test_invalid_multi_scale_settings_are_rejected(self):
        cases = [
            {"detector": "Sobel", "threshold": "96", "multi_scale": "true"},
            {"detector": "Sobel", "threshold": "96", "scale_sigmas": "0,1"},
            {"detector": "Sobel", "threshold": "96", "multi_scale": "false", "scale_support": "1"},
            {"detector": "Sobel", "threshold": "96", "multi_scale": "true", "scale_sigmas": ""},
            {"detector": "Sobel", "threshold": "96", "multi_scale": "true", "scale_sigmas": "1,1"},
            {"detector": "Sobel", "threshold": "96", "multi_scale": "true", "scale_sigmas": "-1,1"},
            {"detector": "Sobel", "threshold": "96", "multi_scale": "true", "scale_sigmas": "0.25,1"},
            {"detector": "Sobel", "threshold": "96", "multi_scale": "true", "scale_sigmas": "0,1", "scale_support": "3"},
            {"detector": "Sobel", "threshold": "96", "multi_scale": "sometimes", "scale_sigmas": "0,1"},
        ]
        for settings in cases:
            with self.subTest(settings=settings):
                response = await self.upload(**settings)
                self.assertEqual(response.status_code, 422, response.text)

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


class CompareAPITests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test")
        y, x = np.indices((24, 32))
        self.detail = (((x // 3 + y // 3) % 2) * 255).astype(np.uint8)

    async def asyncTearDown(self):
        await self.client.aclose()

    async def upload(self, payload=None, **parameters):
        data = {"sigma": "1.2", "kernel_size": "7", "sobel_threshold": "96",
                "prewitt_threshold": "96", "laplacian_contrast_threshold": "20",
                "request_id": REQUEST_ID, **parameters}
        return await self.client.post(
            "/compare",
            files={"image": ("input.png", png_bytes(self.detail) if payload is None else payload, "image/png")},
            data=data,
        )

    async def test_compare_returns_all_three_detector_results(self):
        with patch.object(api, "_decode_preview", wraps=api._decode_preview) as decode:
            with patch.object(api, "compare_detectors", wraps=api.compare_detectors) as orchestrator:
                response = await self.upload()
        self.assertEqual(response.status_code, 200, response.text)
        decode.assert_called_once()
        orchestrator.assert_called_once()
        result = response.json()
        self.assertEqual(result["provenance"], "computed")
        self.assertEqual(result["request_id"], REQUEST_ID)
        self.assertEqual(result["parameters_used"], {
            "sigma": 1.2, "kernel_size": 7, "sobel_threshold": 96,
            "prewitt_threshold": 96, "laplacian_contrast_threshold": 20,
        })
        self.assertEqual(result["analyzed_dimensions"], {"width": 32, "height": 24})
        self.assertGreaterEqual(result["processing_time"], 0)
        for key, detector, decision, threshold_type, threshold, units in (
            ("sobel", "Sobel", "magnitude_threshold", "gradient_magnitude", 96, "raw_gradient_magnitude"),
            ("prewitt", "Prewitt", "magnitude_threshold", "gradient_magnitude", 96, "raw_gradient_magnitude"),
            ("laplacian", "Laplacian", "zero_crossing", "zero_crossing_contrast", 20, "raw_response_difference"),
        ):
            entry = result[key]
            self.assertEqual(decoded_png(entry["edge_map"]).shape, self.detail.shape)
            self.assertEqual(entry["object_count"], len(entry["object_list"]))
            self.assertEqual(entry["edge_pixel_count"],
                             int(np.count_nonzero(decoded_png(entry["edge_map"]))))
            expected_average = (sum(item["area"] for item in entry["object_list"]) /
                                len(entry["object_list"]) if entry["object_list"] else 0.0)
            self.assertEqual(entry["average_object_area"], expected_average)
            self.assertGreaterEqual(entry["processing_time"], 0)
            self.assertEqual(entry["metadata"], {
                "detector": detector, "decision": decision, "threshold_type": threshold_type,
                "threshold_label": ("Zero-crossing contrast threshold" if detector == "Laplacian"
                                    else "Gradient magnitude threshold"),
                "threshold": threshold,
                "threshold_units": units,
                "minimum_component_area": 2 if detector == "Laplacian" else None,
            })

    async def test_compare_does_not_change_analyze_contract(self):
        before = (await self.client.post(
            "/analyze", files={"image": ("input.png", png_bytes(self.detail), "image/png")},
            data={"sigma": "1.2", "kernel_size": "7", "request_id": REQUEST_ID,
                  "detector": "Sobel", "threshold": "96"},
        )).json()
        self.assertIsNone(before["multi_scale"])
        self.assertNotIn("sobel", before)
        self.assertEqual(before["parameters_used"]["detector"], "Sobel")

    async def test_laplacian_contrast_threshold_does_not_change_gradient_detectors(self):
        low = (await self.upload(laplacian_contrast_threshold="0")).json()
        high = (await self.upload(laplacian_contrast_threshold="500")).json()
        self.assertEqual(low["sobel"]["edge_map"], high["sobel"]["edge_map"])
        self.assertEqual(low["prewitt"]["edge_map"], high["prewitt"]["edge_map"])
        self.assertNotEqual(low["laplacian"]["edge_map"], high["laplacian"]["edge_map"])
        self.assertNotEqual(
            int(np.count_nonzero(decoded_png(low["laplacian"]["edge_map"]))),
            int(np.count_nonzero(decoded_png(high["laplacian"]["edge_map"]))),
        )

    async def test_compare_rejects_invalid_input_and_settings(self):
        cases = (
            ({"sobel_threshold": "-1"}, 422), ({"prewitt_threshold": "nan"}, 422),
            ({"laplacian_contrast_threshold": "-1"}, 422),
            ({"sigma": "0.25"}, 422), ({"kernel_size": "4"}, 422),
            ({"request_id": "bad"}, 422),
        )
        for parameters, status in cases:
            with self.subTest(parameters=parameters):
                response = await self.upload(**parameters)
                self.assertEqual(response.status_code, status, response.text)
        self.assertEqual((await self.upload(b"not an image")).status_code, 400)
        missing = await self.client.post(
            "/compare", files={"image": ("input.png", png_bytes(self.detail), "image/png")},
            data={"sigma": "1.2", "kernel_size": "7", "sobel_threshold": "96",
                  "prewitt_threshold": "96", "request_id": REQUEST_ID},
        )
        self.assertEqual(missing.status_code, 422)
        extra = await self.upload(unexpected="value")
        self.assertEqual(extra.status_code, 422)

    async def test_compare_openapi_contract(self):
        schema = api.app.openapi()["paths"]["/compare"]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]
        self.assertEqual(set(schema["required"]), {
            "image", "sigma", "kernel_size", "sobel_threshold", "prewitt_threshold",
            "laplacian_contrast_threshold", "request_id",
        })
        self.assertFalse(schema["additionalProperties"])
        response = api.app.openapi()["paths"]["/compare"]["post"]["responses"]["200"]
        self.assertIn("ComparisonResponse", response["content"]["application/json"]["schema"]["$ref"])


if __name__ == "__main__":
    unittest.main()
