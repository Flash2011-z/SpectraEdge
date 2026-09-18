"""Manual multi-object analysis of a binary foreground image.

Foreground components use eight-connectivity, so pixels touching at a corner
belong to one object. Background topology uses four-connectivity: background
reachable from the padded image exterior is outside, while unreachable regions
are holes. This complementary rule preserves closed holes without allowing the
background to leak diagonally through a corner contact.
"""

from collections import deque
from dataclasses import dataclass
from numbers import Integral

import numpy as np


FOREGROUND_NEIGHBOURS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)
SIDE_NEIGHBOURS = ((-1, 0), (0, -1), (0, 1), (1, 0))


@dataclass(frozen=True)
class ComponentAnalysis:
    """Numerical outputs; boundary masks stay separate from display encoding.

    labels contains deterministic object IDs 1..N in row-major discovery order.
    outer_boundaries and hole_boundaries are same-shape boolean masks. A thin
    foreground pixel may occur in both masks when it borders outside and a hole.
    objects contains JSON-ready measurements for each connected foreground.
    """

    labels: np.ndarray
    outer_boundaries: np.ndarray
    hole_boundaries: np.ndarray
    objects: list[dict]


def _as_binary_2d(binary) -> np.ndarray:
    try:
        values = np.asarray(binary)
    except (TypeError, ValueError) as error:
        raise ValueError("binary must be a rectangular 2D array") from error
    if values.ndim != 2 or values.size == 0:
        raise ValueError("binary must be a nonempty 2D array")
    if values.dtype.kind not in "biuf":
        raise TypeError("binary must contain boolean or real numeric values")
    if not np.isfinite(values).all():
        raise ValueError("binary must contain only finite values")
    if not np.all((values == 0) | (values == 1) | (values == 255)):
        raise ValueError("binary values must be 0/1 or 0/255")
    return values != 0


def label_components(binary) -> tuple[np.ndarray, int]:
    """Label eight-connected nonzero regions with deterministic IDs 1..N.

    Accept a nonempty 2D boolean mask or numeric binary mask using 0/1 or
    0/255. Return a new int32 label image and the component count. The input is
    never modified. A breadth-first flood fill is used; no image library labels.
    """
    foreground = _as_binary_2d(binary)
    height, width = foreground.shape
    labels = np.zeros(foreground.shape, dtype=np.int32)
    component_id = 0

    for row in range(height):
        for column in range(width):
            if not foreground[row, column] or labels[row, column] != 0:
                continue
            component_id += 1
            labels[row, column] = component_id
            queue = deque([(row, column)])
            while queue:
                current_row, current_column = queue.popleft()
                for offset_row, offset_column in FOREGROUND_NEIGHBOURS:
                    neighbour_row = current_row + offset_row
                    neighbour_column = current_column + offset_column
                    if (0 <= neighbour_row < height and 0 <= neighbour_column < width
                            and foreground[neighbour_row, neighbour_column]
                            and labels[neighbour_row, neighbour_column] == 0):
                        labels[neighbour_row, neighbour_column] = component_id
                        queue.append((neighbour_row, neighbour_column))
    return labels, component_id


def filter_small_components(binary, minimum_area) -> np.ndarray:
    """Return a binary mask containing components at or above ``minimum_area``.

    Components use the same eight-connectivity as object analysis. A minimum
    area of one preserves every foreground component. The input is validated as
    a binary mask, is never modified, and the returned uint8 values are 0/255.
    """
    if (isinstance(minimum_area, (bool, np.bool_)) or
            not isinstance(minimum_area, (Integral, np.integer))):
        raise TypeError("minimum_area must be a positive integer")
    if minimum_area < 1:
        raise ValueError("minimum_area must be a positive integer")
    labels, count = label_components(binary)
    if count == 0:
        return np.zeros(labels.shape, dtype=np.uint8)
    areas = np.bincount(labels.ravel(), minlength=count + 1)
    keep = areas >= int(minimum_area)
    keep[0] = False
    return keep[labels].astype(np.uint8) * 255


def _exterior_background(foreground: np.ndarray) -> np.ndarray:
    """Return background connected to outside through four-neighbour steps."""
    padded_foreground = np.pad(foreground, 1, constant_values=False)
    exterior = np.zeros(padded_foreground.shape, dtype=bool)
    exterior[0, 0] = True
    queue = deque([(0, 0)])
    height, width = exterior.shape
    while queue:
        row, column = queue.popleft()
        for offset_row, offset_column in SIDE_NEIGHBOURS:
            neighbour_row = row + offset_row
            neighbour_column = column + offset_column
            if (0 <= neighbour_row < height and 0 <= neighbour_column < width
                    and not padded_foreground[neighbour_row, neighbour_column]
                    and not exterior[neighbour_row, neighbour_column]):
                exterior[neighbour_row, neighbour_column] = True
                queue.append((neighbour_row, neighbour_column))
    return exterior


def analyze_objects(binary) -> ComponentAnalysis:
    """Measure connected foreground regions and classify their boundaries.

    Each object's area is its foreground-pixel count. Centroid is the mean
    pixel-centre coordinate `[x, y]`. Bounding box uses inclusive occupied
    pixels and is `{x, y, width, height}`. Perimeter counts exposed unit pixel
    sides, including sides bordering holes and the image exterior.

    Outer boundaries touch four-connected exterior background. Hole boundaries
    touch enclosed four-connected background. These separate masks distinguish
    the external silhouette from closed internal details; open internal marks
    remain ordinary foreground components rather than being called holes.
    """
    foreground = _as_binary_2d(binary)
    labels, count = label_components(foreground)
    padded_foreground = np.pad(foreground, 1, constant_values=False)
    exterior = _exterior_background(foreground)
    holes = ~padded_foreground & ~exterior
    outer_boundaries = np.zeros(foreground.shape, dtype=bool)
    hole_boundaries = np.zeros(foreground.shape, dtype=bool)
    areas = np.zeros(count + 1, dtype=np.int64)
    perimeters = np.zeros(count + 1, dtype=np.int64)
    row_sums = np.zeros(count + 1, dtype=np.float64)
    column_sums = np.zeros(count + 1, dtype=np.float64)
    minimum_rows = np.full(count + 1, foreground.shape[0], dtype=np.int64)
    minimum_columns = np.full(count + 1, foreground.shape[1], dtype=np.int64)
    maximum_rows = np.full(count + 1, -1, dtype=np.int64)
    maximum_columns = np.full(count + 1, -1, dtype=np.int64)

    # Accumulate every component in one image pass. Re-scanning the whole label
    # image once per object would become quadratic on sparse, noisy masks.
    rows, columns = np.nonzero(foreground)
    for row, column in zip(rows, columns):
        component_id = labels[row, column]
        areas[component_id] += 1
        row_sums[component_id] += row
        column_sums[component_id] += column
        minimum_rows[component_id] = min(minimum_rows[component_id], row)
        minimum_columns[component_id] = min(minimum_columns[component_id], column)
        maximum_rows[component_id] = max(maximum_rows[component_id], row)
        maximum_columns[component_id] = max(maximum_columns[component_id], column)
        padded_row, padded_column = row + 1, column + 1
        for offset_row, offset_column in SIDE_NEIGHBOURS:
            neighbour_row = padded_row + offset_row
            neighbour_column = padded_column + offset_column
            if padded_foreground[neighbour_row, neighbour_column]:
                continue
            perimeters[component_id] += 1
            if exterior[neighbour_row, neighbour_column]:
                outer_boundaries[row, column] = True
            elif holes[neighbour_row, neighbour_column]:
                hole_boundaries[row, column] = True

    objects = []
    for component_id in range(1, count + 1):
        minimum_row, maximum_row = int(minimum_rows[component_id]), int(maximum_rows[component_id])
        minimum_column = int(minimum_columns[component_id])
        maximum_column = int(maximum_columns[component_id])
        area = int(areas[component_id])
        objects.append({
            "id": component_id,
            "area": area,
            "perimeter": int(perimeters[component_id]),
            "centroid": [float(column_sums[component_id] / area),
                         float(row_sums[component_id] / area)],
            "bounding_box": {
                "x": minimum_column,
                "y": minimum_row,
                "width": maximum_column - minimum_column + 1,
                "height": maximum_row - minimum_row + 1,
            },
        })

    return ComponentAnalysis(labels, outer_boundaries, hole_boundaries, objects)
