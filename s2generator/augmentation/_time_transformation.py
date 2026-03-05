# -*- coding: utf-8 -*-
"""
Created on 2026/03/05 16:19:59
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/S2Generator
"""
import numpy as np


def add_linear_trend(
    time_series: np.ndarray, trend_strength: float = 1.0, direction: str = "upward"
) -> np.ndarray:
    """
    Perform linear trend augmentation on the input time series.
    This augmentation adds a linear trend to the input time series,
    which can help models learn to handle non-stationary data and improve their robustness to trends.

    :param time_series: Input time series, a 1D numpy array
    :param trend_strength: The strength of the linear trend to be added, default is 1.0.
    :param direction: The direction of the linear trend, either "upward" or "downward", default is "upward".

    :return: Augmented time series with a linear trend, a 1D numpy array of the same length as the input series.
    """

    # Get the length of the time series
    seq_length = len(time_series)

    # Calculate the the energy of the original time series
    original_energy = np.mean(time_series**2)

    # Create a linear trend
    if direction == "upward":
        trend = np.linspace(0, trend_strength * seq_length, seq_length)
    elif direction == "downward":
        trend = np.linspace(0, -trend_strength * seq_length, seq_length)
    else:
        raise ValueError("direction must be either 'upward' or 'downward'")

    # Scale the trend to have the same energy as the original time series
    trend_energy = np.mean(trend**2)

    if trend_energy > 0:
        trend = trend * np.sqrt(original_energy / trend_energy)

    # Average the original signal and the trend to maintain the overall scale
    return (time_series + trend) / 2


def time_series_mixup(a: np.ndarray, b: np.ndarray, alpha: float = 0.7) -> np.ndarray:
    """
    Mixup Enhancement: Weighted mixing of two time series to create a new augmented signal.
    This method combines two time series by taking a weighted average of them,
    where the weights are determined by a mixing parameter alpha.
    This can help models learn to generalize better by exposing them to a wider variety of signal combinations.

    :param a: First input time series, a 1D numpy array.
    :param b: Second input time series, a 1D numpy array of the same length as a.
    :param alpha: The mixing parameter that controls the weight of each time series in the mixup,
    default is 0.7. A value of alpha close to 1 gives more weight to the first time series (a),
    while a value close to 0 gives more weight to the second time series (b).

    :return: Mixed time series, a 1D numpy array of the same length as the input series.
    """
    assert a.shape == b.shape, "Input time series must have the same shape"

    # Calculate the mixed signal as a weighted average of the two input signals
    return alpha * a + (1 - alpha) * b
