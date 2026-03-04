# -*- coding: utf-8 -*-
"""
Created on 2026/03/04 22:52:40
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/S2Generator
"""

import numpy as np


def amplitude_modulation(
    time_series: np.ndarray,
) -> np.ndarray:
    """
    Perform amplitude modulation on the input time series.
    This method applies a random amplitude modulation to the time series, which can help to enhance the diversity of the data and improve the robustness of models trained on it.

    :param time_series: Input time series, a 1D numpy array

    :return: Amplitude modulated time series, a 1D numpy array of the same length as the input series.
    """
    # Generate a random amplitude modulation signal
    modulation_signal = np.random.uniform(0.5, 1.5, size=len(time_series))

    # Apply amplitude modulation to the input time series
    modulated_series = time_series * modulation_signal

    return modulated_series
