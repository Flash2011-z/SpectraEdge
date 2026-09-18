"""Reproducible noise models for grayscale image signals."""

from numbers import Integral, Real

import numpy as np

from .convolution import _as_float_2d


def _validate_seed(seed) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (Integral, np.integer)):
        raise TypeError("seed must be an integer")
    seed = int(seed)
    if seed < 0 or seed > np.iinfo(np.uint32).max:
        raise ValueError("seed must be from 0 through 4294967295")
    return seed


def _validate_strength(strength, maximum: float, model: str) -> float:
    if isinstance(strength, (bool, np.bool_)) or not isinstance(strength, Real):
        raise TypeError(f"{model} strength must be a real numeric scalar")
    try:
        strength = float(strength)
    except OverflowError as error:
        raise ValueError(f"{model} strength must be finite") from error
    if not np.isfinite(strength) or strength < 0 or strength > maximum:
        raise ValueError(f"{model} strength must be from 0 through {maximum:g}")
    return strength


def add_gaussian_noise(image, strength, seed) -> np.ndarray:
    """Add zero-mean Gaussian noise with ``strength`` intensity deviation.

    ``image`` must be a finite real 2D grayscale signal. ``strength`` is the
    standard deviation in grayscale intensity units and must be nonnegative.
    ``seed`` initializes an independent NumPy generator. The result is a new
    float64 array; values are deliberately not clipped to display bounds.
    """
    values = _as_float_2d(image, "image")
    strength = _validate_strength(strength, np.finfo(np.float64).max, "Gaussian")
    generator = np.random.default_rng(_validate_seed(seed))
    return values + generator.normal(0.0, strength, size=values.shape)


def add_salt_pepper_noise(image, strength, seed) -> np.ndarray:
    """Replace pixels with black or white at the requested corruption density.

    ``strength`` is the probability in [0, 1] that each pixel is replaced.
    Corrupted pixels independently become the grayscale-domain minimum 0 or
    maximum 255 with equal probability. The explicit seed makes selection
    reproducible. The input is never modified and the result is float64.
    """
    values = _as_float_2d(image, "image")
    strength = _validate_strength(strength, 1.0, "Salt & Pepper")
    generator = np.random.default_rng(_validate_seed(seed))
    decisions = generator.random(values.shape)
    result = values.copy()
    result[decisions < strength / 2] = 0.0
    result[(decisions >= strength / 2) & (decisions < strength)] = 255.0
    return result
