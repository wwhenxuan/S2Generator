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


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt

    # Create a sample time series (sine wave)
    t = np.linspace(0, 10, 500)
    original_series = np.sin(t)

    # Add linear trend
    augmented_series = add_linear_trend(
        original_series, trend_strength=1, direction="downward"
    )

    # Plot the original and augmented time series
    plt.figure(figsize=(12, 6))
    plt.plot(t, original_series, label="Original Time Series")
    plt.plot(t, augmented_series, label="Augmented Time Series with Linear Trend")
    plt.legend()
    plt.title("Linear Trend Augmentation")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid()
    plt.show()
