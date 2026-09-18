"""Analytical connected-component, contour, and measurement expectations."""

import unittest

import numpy as np

from backend.analysis import analyze_objects, filter_small_components, label_components


class ComponentAnalysisTests(unittest.TestCase):
    def test_two_rectangles_have_stable_ids_and_measurements(self):
        image = np.zeros((8, 12), dtype=np.uint8)
        image[1:4, 1:5] = 255
        image[4:7, 8:10] = 255
        result = analyze_objects(image)
        self.assertEqual(result.objects, [
            {"id": 1, "area": 12, "perimeter": 14, "centroid": [2.5, 2.0],
             "bounding_box": {"x": 1, "y": 1, "width": 4, "height": 3}},
            {"id": 2, "area": 6, "perimeter": 10, "centroid": [8.5, 5.0],
             "bounding_box": {"x": 8, "y": 4, "width": 2, "height": 3}},
        ])
        np.testing.assert_array_equal(np.unique(result.labels), [0, 1, 2])
        self.assertFalse(result.hole_boundaries.any())

    def test_empty_image_has_no_objects_or_contours(self):
        result = analyze_objects(np.zeros((5, 7), dtype=bool))
        self.assertEqual(result.objects, [])
        for output in (result.labels, result.outer_boundaries, result.hole_boundaries):
            np.testing.assert_array_equal(output, np.zeros((5, 7)))

    def test_shape_with_hole_separates_outer_and_internal_boundaries(self):
        image = np.zeros((9, 9), dtype=np.uint8)
        image[1:8, 1:8] = 1
        image[3:6, 3:6] = 0
        result = analyze_objects(image)
        self.assertEqual(len(result.objects), 1)
        self.assertEqual(result.objects[0]["area"], 40)
        self.assertEqual(result.objects[0]["perimeter"], 40)  # 28 outer + 12 hole.
        self.assertEqual(result.objects[0]["centroid"], [4.0, 4.0])
        self.assertEqual(result.objects[0]["bounding_box"],
                         {"x": 1, "y": 1, "width": 7, "height": 7})
        self.assertEqual(int(result.outer_boundaries.sum()), 24)
        self.assertEqual(int(result.hole_boundaries.sum()), 12)
        self.assertFalse(np.any(result.outer_boundaries & result.hole_boundaries))

    def test_touching_objects_follow_eight_connectivity(self):
        edge_touch = np.zeros((6, 6), dtype=np.uint8)
        edge_touch[1:3, 1:3] = 1
        edge_touch[1:3, 3:5] = 1
        diagonal_touch = np.zeros((6, 6), dtype=np.uint8)
        diagonal_touch[1:3, 1:3] = 1
        diagonal_touch[3:5, 3:5] = 1
        for image in (edge_touch, diagonal_touch):
            with self.subTest(image=image):
                labels, count = label_components(image)
                self.assertEqual(count, 1)
                self.assertEqual(set(labels[image != 0]), {1})

    def test_image_boundary_object_has_full_exposed_perimeter(self):
        image = np.zeros((6, 7), dtype=np.uint8)
        image[0:3, 0:4] = 255
        result = analyze_objects(image)
        self.assertEqual(result.objects, [
            {"id": 1, "area": 12, "perimeter": 14, "centroid": [1.5, 1.0],
             "bounding_box": {"x": 0, "y": 0, "width": 4, "height": 3}},
        ])
        self.assertTrue(result.outer_boundaries[0, 0])
        self.assertFalse(result.hole_boundaries.any())

    def test_small_component_filter_removes_isolated_noise_and_preserves_boundary(self):
        image = np.zeros((7, 9), dtype=np.uint8)
        image[0, 0] = 255
        image[2:5, 3] = 255
        image[4, 3:7] = 255
        expected = image.copy()
        expected[0, 0] = 0
        np.testing.assert_array_equal(filter_small_components(image, 2), expected)
        np.testing.assert_array_equal(filter_small_components(image, 1), image)
        self.assertEqual(image[0, 0], 255)

    def test_small_component_filter_validates_area(self):
        for area in (0, -1):
            with self.subTest(area=area), self.assertRaises(ValueError):
                filter_small_components([[0]], area)
        for area in (True, 1.5, "2", None):
            with self.subTest(area=repr(area)), self.assertRaises(TypeError):
                filter_small_components([[0]], area)

    def test_input_is_unchanged_and_outputs_do_not_share_storage(self):
        image = np.array([[0, 255], [255, 0]], dtype=np.uint8)
        before = image.copy()
        image.setflags(write=False)
        result = analyze_objects(image)
        np.testing.assert_array_equal(image, before)
        for output in (result.labels, result.outer_boundaries, result.hole_boundaries):
            self.assertFalse(np.shares_memory(output, image))

    def test_rejects_nonbinary_and_invalid_arrays(self):
        for image in ([], [0, 1], [[0], [1, 2]], [[np.nan]], [[np.inf]], [[0, 2]]):
            with self.subTest(image=image), self.assertRaises(ValueError):
                analyze_objects(image)
        for image in ([["0"]], [[1j]], np.array([[object()]], dtype=object)):
            with self.subTest(image=image), self.assertRaises(TypeError):
                analyze_objects(image)


if __name__ == "__main__":
    unittest.main()
