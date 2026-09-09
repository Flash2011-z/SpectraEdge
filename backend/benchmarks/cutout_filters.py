"""Reproduce the size-limit benchmark: python -m backend.benchmarks.cutout_filters.

One timed pass per size/setting; no libraries replace the manual filters.
This measures only Gaussian + Sobel, not HTTP, image encoding or watershed.
"""

from time import perf_counter

import numpy as np

from backend.detection.sobel import sobel
from backend.signal_ops.gaussian import gaussian_blur


def main():
    print("side sigma kernel gaussian_s sobel_s total_s", flush=True)
    rng = np.random.default_rng(220)
    for side in (256, 512, 1024):
        image = rng.uniform(0, 255, (side, side))
        for sigma, kernel_size in ((1.2, 5), (5.0, 31)):
            start = perf_counter()
            filtered = gaussian_blur(image, sigma, kernel_size)
            middle = perf_counter()
            sobel(filtered)
            end = perf_counter()
            print(side, sigma, kernel_size, round(middle - start, 3),
                  round(end - middle, 3), round(end - start, 3), flush=True)


if __name__ == "__main__":
    main()
