"""Library-backed colour segmentation, separate from the manual signal modules."""

import cv2
import numpy as np


def paint_marks(labels, marks, foreground_label, background_label):
    """Replay validated brush strokes in order, shared by both methods."""
    for mark in marks:
        label = foreground_label if mark["mode"] == "keep" else background_label
        points = np.array([[p["x"], p["y"]] for p in mark["points"]], dtype=np.int32)
        # Library drawing only rasterizes user input; it does not find edges.
        cv2.polylines(labels, [points], False, label, mark["size"], cv2.LINE_8)
        for point in (points[0], points[-1]):
            cv2.circle(labels, tuple(point), mark["size"] // 2, label, -1, cv2.LINE_8)


def cutout_from_mask(rgba, foreground):
    """Apply a binary selection to original colour/alpha, without recolouring."""
    foreground = foreground & (rgba[:, :, 3] > 0)
    rows, columns = np.nonzero(foreground)
    if rows.size == 0:
        raise ValueError("No foreground was found. Paint Keep over the object and extract again.")
    cutout = rgba.copy()
    # RGB is not premultiplied. Retained partial alpha is kept exactly.
    cutout[:, :, 3] = np.where(foreground, rgba[:, :, 3], 0)
    mask = foreground.astype(np.uint8) * 255
    left, top = int(columns.min()), int(rows.min())
    bounds = {"x": left, "y": top, "width": int(columns.max()) - left + 1,
              "height": int(rows.max()) - top + 1}
    return cutout, mask, bounds


def grabcut_rgba(rgba, rectangle, marks):
    """Return an unchanged-RGB RGBA cutout, binary mask, and foreground bounds.

    Coordinates and brush diameters are integer prepared-image pixels. Bounds
    use x/y/width/height with exclusive right/bottom edges. The API validates
    these before this function runs. Each request starts from the rectangle
    and replays ALL marks in order; no previous segmentation is required.
    """
    height, width = rgba.shape[:2]
    labels = np.full((height, width), cv2.GC_BGD, dtype=np.uint8)
    x, y, w, h = (rectangle[key] for key in ("x", "y", "width", "height"))
    labels[y:y + h, x:x + w] = cv2.GC_PR_FGD

    paint_marks(labels, marks, cv2.GC_FGD, cv2.GC_BGD)

    # User marks must never resurrect pixels that were already transparent.
    labels[rgba[:, :, 3] == 0] = cv2.GC_BGD
    foreground_samples = np.count_nonzero((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD))
    background_samples = np.count_nonzero(labels == cv2.GC_BGD)
    # OpenCV initializes five colour-mixture components per class.
    if foreground_samples < 5 or background_samples < 5:
        raise ValueError("Leave at least five foreground and five background pixels. Draw a larger rectangle with a background margin, or adjust Keep/Remove marks.")

    # OpenCV expects BGR, but the untouched RGBA source supplies output RGB.
    bgr = np.ascontiguousarray(rgba[:, :, :3][:, :, ::-1])
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    # The router serializes cutout work; a fixed seed makes repeated requests
    # with the same rectangle and marks reproducible within this OpenCV build.
    cv2.setRNGSeed(220)
    try:
        cv2.grabCut(bgr, labels, None, background_model, foreground_model, 5, cv2.GC_INIT_WITH_MASK)
    except cv2.error as error:
        raise ValueError("GrabCut could not separate this selection. Leave some background outside the rectangle and add Keep/Remove marks.") from error

    foreground = ((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD)) & (rgba[:, :, 3] > 0)
    return cutout_from_mask(rgba, foreground)
