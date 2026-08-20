# -*- coding: utf-8 -*-
"""
Created on 2026/03/05 00:42:51
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/S2Generator
"""

import numpy as np


def censor_augmentation(
    time_series: np.ndarray,
    upper_quantile: float = 0.65,
    lower_quantile: float = 0.35,
    bernoulli_p: float = 0.8,
    rng: np.random.RandomState = None,
    seed: int = 42,
) -> np.ndarray:
    """
    Perform censoring augmentation on the input time series.

    This augmentation censors (clips) the input signal either from below or
    above, depending on a randomly sampled direction. The clipping threshold is determined by drawing
    a quantile uniformly from the empirical distribution of the signal.

    :param time_series: Input time series, a 1D numpy array
    :param upper_quantile: Upper quantile threshold for censoring, default is 0.65
    :param lower_quantile: Lower quantile threshold for censoring, default is 0.35
    :param bernoulli_p: Probability of censoring direction (0 for lower censoring, 1 for upper censoring),
                        the default is 0.5, meaning equal probability for both directions.
    :param rng: Optional random number generator for reproducibility.
                If None, a new RNG will be created using the provided seed.
    :param seed: Random seed for reproducibility if rng is not provided.

    :return: Censored time series, a 1D numpy array of the same length as the input series.
    """

    # Validate the input time series (copy to avoid mutating the caller's array,
    # matching the behavior of the other augmentation functions).
    time_series = np.asarray(time_series, dtype=float).copy()
    if time_series.ndim != 1:
        raise ValueError("Input time_series must be a 1D array.")

    # Validate bernoulli_p
    if not (0 <= bernoulli_p <= 1):
        raise ValueError("bernoulli_p must be in the range [0, 1].")

    # Get the length of the time series
    length = time_series.shape[0]

    # Set random seed for reproducibility
    if rng is None:
        rng = np.random.RandomState(seed)

    # Randomly sample quantile thresholds for each time step
    quantile_threshold = rng.uniform(lower_quantile, upper_quantile, size=length)

    # Compute the threshold value based on the quantile of the time series
    threshold_value = np.quantile(time_series, quantile_threshold)

    # Sample the censor direction from bernoulli distribution (0.5)
    censor_direction = rng.binomial(
        n=1, p=bernoulli_p, size=length
    )  # 0 for lower censoring, 1 for upper censoring

    for t in range(length):
        if censor_direction[t] == 1:
            # Lower censoring
            time_series[t] = max(time_series[t], threshold_value[t])
        else:
            # Upper censoring
            time_series[t] = min(time_series[t], threshold_value[t])

    return time_series
