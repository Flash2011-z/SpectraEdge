"""Manual Gaussian/Sobel elevation with library-backed marker watershed.

The cutout API validates RGBA pixels, coordinates and bounded parameters.
These functions do numerical segmentation only, with no image encoding or I/O.
"""

import cv2
import numpy as np
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed

from backend.cutout import cutout_from_mask, paint_marks
from backend.detection.sobel import sobel
from backend.signal_ops.gaussian import gaussian_blur

UNKNOWN = 0
FOREGROUND = 1
BACKGROUND = 2


def automatic_markers(magnitude, rectangle, eligible, rgba):
    """Find confident colour interiors; leave their boundaries to manual edges.

    A dominant, reasonably uniform background must occupy most of the selection
    rim. This is a background-colour heuristic, NOT semantic recognition. Edge
    nesting cannot tell a shirt detail from a hole, and a subject can continue
    beyond the image border, so neither implies definite background.
    """
    x, y, width, height = (rectangle[key] for key in ("x", "y", "width", "height"))
    elevation = magnitude[y:y + height, x:x + width]
    seeds = np.zeros(magnitude.shape, dtype=np.int32)
    if elevation.max() <= 0:
        return seeds

    rgb = rgba[y:y + height, x:x + width, :3].astype(np.float64)
    visible = rgba[y:y + height, x:x + width, 3] > 0
    quiet = elevation <= threshold_otsu(elevation)
    rim = np.zeros(elevation.shape, dtype=bool)
    thickness = max(1, min(5, min(width, height) // 20))
    rim[:thickness, :] = rim[-thickness:, :] = True
    rim[:, :thickness] = rim[:, -thickness:] = True
    samples = rgb[rim & quiet & visible]
    if len(samples) < 4:
        return seeds

    # Use the most frequent coarse RGB bin, not an average of background and
    # clothing at the frame edge. Nearby bins may contain the same background
    # under JPEG noise. All distances below are in 0..255 RGB channel units.
    bins = (samples // 16).astype(np.int32)
    ids = bins[:, 0] * 256 + bins[:, 1] * 16 + bins[:, 2]
    mode = np.bincount(ids, minlength=4096).argmax()
    background = np.median(samples[ids == mode], axis=0)
    inliers = samples[np.linalg.norm(samples - background, axis=1) <= 24]
    if len(inliers) / len(samples) < 0.55:
        return seeds  # No clear background model: do not guess a central object.
    background = np.median(inliers, axis=0)
    spread = np.percentile(np.linalg.norm(inliers - background, axis=1), 95)
    background_radius = max(8.0, float(spread) + 2.0)
    foreground_radius = background_radius + max(12.0, background_radius)
    distance = np.linalg.norm(rgb - background, axis=2)
    allowed = eligible[y:y + height, x:x + width] & visible & quiet
    local_seeds = seeds[y:y + height, x:x + width]

    # Erode confident colour regions by one pixel to avoid pinning the edge.
    # Seed all quiet interiors so strong facial/clothing edges cannot isolate
    # unseeded foreground islands. Border replication permits a frame-touching
    # subject; the rectangle rim is sampled, never blindly labelled background.
    for label, candidates in ((BACKGROUND, distance <= background_radius),
                              (FOREGROUND, distance >= foreground_radius)):
        core = cv2.erode(candidates.astype(np.uint8), np.ones((3, 3), np.uint8),
                         borderType=cv2.BORDER_REPLICATE) > 0
        local_seeds[core & allowed] = label
    return seeds


def build_markers(rgba, rectangle, marks, magnitude=None):
    """Use brush seeds when supplied; otherwise infer colour/edge interiors."""
    markers = np.full(rgba.shape[:2], BACKGROUND, dtype=np.int32)
    x, y, width, height = (rectangle[key] for key in ("x", "y", "width", "height"))
    markers[y:y + height, x:x + width] = UNKNOWN
    # A Keep mark may replace the outside background. Later strokes win.
    paint_marks(markers, marks, FOREGROUND, BACKGROUND)
    markers[rgba[:, :, 3] == 0] = BACKGROUND
    if not any(mark["mode"] == "keep" for mark in marks) and magnitude is not None:
        automatic = automatic_markers(magnitude, rectangle, markers == UNKNOWN, rgba)
        # Remove strokes and source transparency have priority over inference.
        unknown = markers == UNKNOWN
        markers[unknown] = automatic[unknown]
        if not np.any(markers == FOREGROUND):
            raise ValueError("No confident automatic selection was found. Include more uniform background around the object, or optionally add Keep and Remove marks.")
    elif not np.any(markers == FOREGROUND):
        raise ValueError("No Keep pixels remain. Undo the covering Remove mark or paint Keep on a visible part of the object.")
    if not np.any(markers == BACKGROUND):
        raise ValueError("Paint a Remove stroke on the background, or leave background outside the rectangle.")
    return markers


def watershed_foreground(magnitude, markers, visible=None):
    """Flood the continuous elevation using four-connected pixel neighbours."""
    # No binary threshold, uint8 conversion or scaling enters watershed.
    # Zero/constant elevations are valid: ties use deterministic queue order.
    # No compactness penalty or extra watershed-line pixels alter the seeds.
    # Transparent source holes are outside the flooding domain. They must not
    # spread a background basin through otherwise uniform visible foreground.
    labels = watershed(magnitude, markers=markers, mask=visible, connectivity=1,
                       compactness=0, watershed_line=False)
    foreground = labels == FOREGROUND
    foreground[markers == FOREGROUND] = True
    foreground[markers == BACKGROUND] = False
    # Do not fill holes or discard small/disconnected foreground components.
    return foreground


def edge_guided_rgba(rgba, rectangle, marks, sigma, kernel_size):
    """Return original-colour cutout, mask, bounds and the exact float elevation."""
    # BT.601 luminance weights; convert before arithmetic to avoid uint8 wrap.
    # Alpha never becomes a colour channel or a composited white background.
    rgb = rgba[:, :, :3].astype(np.float64)
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    # Existing true-convolution functions use reflection padding, preserve
    # dimensions, and keep signed gradients and magnitudes above 255.
    filtered = gaussian_blur(gray, sigma, kernel_size)
    _, _, magnitude = sobel(filtered)
    markers = build_markers(rgba, rectangle, marks, magnitude)
    foreground = watershed_foreground(magnitude, markers, visible=rgba[:, :, 3] > 0)
    cutout, mask, bounds = cutout_from_mask(rgba, foreground)
    return cutout, mask, bounds, magnitude
