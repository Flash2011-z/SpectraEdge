"""Opt-in regressions for the two supplied portraits against a running server.

Photo paths are supplied locally, never bundled in the repository. These sparse
interior checks are not a ground-truth segmentation score or a perfect-hair claim.
"""

import base64
import json
import os
from pathlib import Path
import unittest

import httpx
import numpy as np

from backend.tests.test_cutout import REQUEST_ID, decoded


class RealAIPhotoTests(unittest.TestCase):
    def check_photo(self, path, source_size, foreground, background):
        with httpx.Client(base_url="http://127.0.0.1:8000", timeout=180) as client:
            response = client.post("/cutout/prepare", files={"image": ("portrait.jpg", Path(path).read_bytes())},
                                   data={"request_id": REQUEST_ID})
            self.assertEqual(response.status_code, 200, response.text)
            prepared = response.json()
            self.assertEqual(prepared["source_dimensions"], source_size)
            settings = {"request_id": REQUEST_ID, "image_id": prepared["image_id"],
                        "width": prepared["width"], "height": prepared["height"], "method": "ai-assisted",
                        "rectangle": {"x": 0, "y": 0, "width": prepared["width"], "height": prepared["height"]}, "marks": []}
            response = client.post("/cutout/extract",
                                   files={"image": ("prepared.png", base64.b64decode(prepared["prepared_image"].split(",")[1]))},
                                   data={"settings": json.dumps(settings)})
            self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["method"], "ai-assisted")
        self.assertEqual(result["algorithm"], "rembg-onnx")
        source, _ = decoded(prepared["prepared_image"])
        cutout, _ = decoded(result["cutout_image"])
        mask, _ = decoded(result["mask_image"])
        self.assertEqual(cutout.shape, source.shape)
        np.testing.assert_array_equal(cutout[:, :, :3], source[:, :, :3])
        np.testing.assert_array_equal(cutout[:, :, 3], np.rint(source[:, :, 3].astype(float) * mask / 255).astype(np.uint8))
        for x, y in foreground:
            with self.subTest(foreground=(x, y)):
                self.assertGreater(cutout[y, x, 3], 230)
        for x, y in background:
            with self.subTest(background=(x, y)):
                self.assertLess(cutout[y, x, 3], 25)

    @unittest.skipUnless(os.environ.get("SPECTRAEDGE_TEST_AI_PORTRAIT"), "Original plain-background portrait not supplied")
    def test_plain_background_portrait(self):
        self.check_photo(os.environ["SPECTRAEDGE_TEST_AI_PORTRAIT"], {"width": 547, "height": 365},
                         [(260, 60), (260, 200), (190, 220), (260, 335)],
                         [(145, 50), (365, 30), (145, 335)])

    @unittest.skipUnless(os.environ.get("SPECTRAEDGE_TEST_AI_COMPLEX"), "Original patterned-background portrait not supplied")
    def test_patterned_background_portrait(self):
        self.check_photo(os.environ["SPECTRAEDGE_TEST_AI_COMPLEX"], {"width": 2048, "height": 2042},
                         [(250, 270), (265, 400), (140, 445), (390, 470)],
                         [(100, 80), (320, 170), (155, 265), (25, 400), (440, 338)])


if __name__ == "__main__":
    unittest.main()
