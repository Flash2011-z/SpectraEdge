"""Array-only signal operations for the SpectraEdge backend."""

from .convolution import convolve2d
from .fourier import fft2d, fft_spectrum
from .gaussian import gaussian_blur, gaussian_kernel
from .noise import add_gaussian_noise, add_salt_pepper_noise
from .transforms import (
    dft1d,
    fft1d,
    fft_radix2,
    idft1d,
    ifft1d,
    ifft_radix2,
    inverse_2d,
    transform_2d,
)

__all__ = [
    "convolve2d",
    "gaussian_kernel",
    "gaussian_blur",
    "add_gaussian_noise",
    "add_salt_pepper_noise",
    "fft2d",
    "fft_spectrum",
    "dft1d",
    "idft1d",
    "fft_radix2",
    "ifft_radix2",
    "fft1d",
    "ifft1d",
    "transform_2d",
    "inverse_2d",
]
