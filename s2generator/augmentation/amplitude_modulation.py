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
    num_changepoints: int = 5,
    mean_amplitude: float = 1.0,
    amplitude_variation: float = 1.0,
) -> np.ndarray:
    """
    Perform amplitude modulation on the input time series.
    This method applies a random amplitude modulation to the time series, which can help to enhance the diversity of the data and improve the robustness of models trained on it.

    :param time_series: Input time series, a 1D numpy array

    :return: Amplitude modulated time series, a 1D numpy array of the same length as the input series.
    """
