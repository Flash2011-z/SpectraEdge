"""Array-only Sobel and basic thresholding; contours remain a future stage."""

from .sobel import sobel
from .threshold import threshold_edges

__all__ = ["sobel", "threshold_edges"]
