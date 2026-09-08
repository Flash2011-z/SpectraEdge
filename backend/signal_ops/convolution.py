"""Manual, same-sized 2D convolution for real-valued images."""

import numpy as np


def _as_float_2d(values, name: str) -> np.ndarray:
    """Validate an image or kernel before converting it to floating point."""
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a rectangular 2D numeric array") from error

    if array.ndim != 2 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty 2D array")

    # Check before converting: numeric strings and complex values are not images.
    # NumPy uses i/u/f for signed integers, unsigned integers, and real floats.
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric values, not booleans or complex values")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")

    # At least float64 avoids uint8 wraparound and retains fractional responses.
    # result_type also preserves a wider floating-point input, if available.
    float_dtype = np.result_type(array.dtype, np.float64)
    return array.astype(float_dtype, copy=False)


def convolve2d(image, kernel) -> np.ndarray:
    """Convolve a 2D image with an odd-by-odd kernel using reflect padding.

    Both arguments can be NumPy arrays or rectangular numeric lists. They must
    be nonempty, real, and finite. The kernel may be rectangular, but each of
    its dimensions must be odd. Boolean, complex, and object arrays are rejected.

    The result is a new floating-point array with the same shape as image.
    Responses are not clipped or converted to image-display values.

    Raises:
        TypeError: An array contains a non-real or non-numeric data type.
        ValueError: An array is empty, not 2D, non-finite, or the kernel is even-sized.
    """
    image_array = _as_float_2d(image, "image")
    kernel_array = _as_float_2d(kernel, "kernel")
    kernel_height, kernel_width = kernel_array.shape

    if kernel_height % 2 == 0 or kernel_width % 2 == 0:
        raise ValueError("kernel dimensions must both be odd")

    # Odd dimensions give one centre coefficient. Padding by each half-size
    # lets us centre the full kernel on every pixel, including the borders.
    pad_y = kernel_height // 2
    pad_x = kernel_width // 2
    padded_image = np.pad(image_array, ((pad_y, pad_y), (pad_x, pad_x)), mode="reflect")

    # Convolution reverses both axes. Without this flip, the same sliding
    # multiply-and-sum operation would be cross-correlation instead.
    flipped_kernel = kernel_array[::-1, ::-1]
    result_dtype = np.result_type(image_array.dtype, kernel_array.dtype)
    result = np.empty(image_array.shape, dtype=result_dtype)

    for row in range(image_array.shape[0]):
        for column in range(image_array.shape[1]):
            neighbourhood = padded_image[
                row : row + kernel_height,
                column : column + kernel_width,
            ]
            result[row, column] = np.sum(neighbourhood * flipped_kernel)

    return result
