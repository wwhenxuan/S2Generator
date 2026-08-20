# -*- coding: utf-8 -*-
"""
Created on 2026/03/05 15:54:19
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/S2Generator
"""

import numpy as np

# Supported spike kernel shapes (TiRex-2 Section 3.4: Gaussian, triangular, or
# rectangular kernels).
_SPIKE_KERNELS = ["gaussian", "triangular", "rectangular"]


def spike_injection(
    time_series: np.ndarray,
    num_spikes: int = 2,
    amplitude_range: tuple = (1.0, 3.0),
    width_range: tuple = (1.0, 5.0),
    kernel: str = None,
    rng: np.random.RandomState = None,
    seed: int = 42,
) -> np.ndarray:
    """
    Perform spike injection augmentation on the input time series.

    This augmentation randomly injects synthetic spikes into the input time
    series to simulate sudden and extreme events, which can help models learn to
    handle such anomalies. Each spike is a localized perturbation whose shape is
    sampled from Gaussian, triangular, or rectangular kernels (or fixed via the
    ``kernel`` argument), added on top of the original signal with a random sign
    and amplitude.

    :param time_series: Input time series, a 1D numpy array.
    :param num_spikes: Number of spikes to inject, default is 2.
    :param amplitude_range: Tuple (min, max) for the spike amplitude, default (1.0, 3.0).
    :param width_range: Tuple (min, max) for the spike kernel width, default (1.0, 5.0).
    :param kernel: Kernel shape, one of "gaussian", "triangular", "rectangular".
                   If None, a random kernel is sampled per spike.
    :param rng: Optional random number generator for reproducibility.
                If None, a new RNG will be created using the provided seed.
    :param seed: Random seed for reproducibility if rng is not provided.

    :return: Time series with injected spikes, a 1D numpy array of the same
             length as the input series.
    """
    # Validate the input time series
    time_series = np.asarray(time_series, dtype=float)
    if time_series.ndim != 1:
        raise ValueError("Input time_series must be a 1D array.")

    # Validate the kernel argument
    if kernel is not None and kernel not in _SPIKE_KERNELS:
        raise ValueError(
            f"kernel must be one of {_SPIKE_KERNELS}, got {kernel!r}."
        )

    # Get the length of the time series
    length = time_series.shape[0]

    # Initialize random number generator
    if rng is None:
        rng = np.random.RandomState(seed)

    # Work on a copy to avoid mutating the caller's array
    result = time_series.copy()
    t = np.arange(length)

    for _ in range(num_spikes):
        # Sample a kernel shape if not fixed
        k = kernel if kernel is not None else rng.choice(_SPIKE_KERNELS)

        center = rng.randint(0, length)
        amplitude = rng.uniform(*amplitude_range)
        sign = rng.choice([-1.0, 1.0])
        width = rng.uniform(*width_range)

        # Build the localized spike kernel
        if k == "gaussian":
            spike = amplitude * np.exp(-0.5 * ((t - center) / width) ** 2)
        elif k == "triangular":
            spike = amplitude * np.maximum(0.0, 1.0 - np.abs(t - center) / width)
        else:  # rectangular
            spike = amplitude * (np.abs(t - center) <= width).astype(float)

        result += sign * spike

    return result
