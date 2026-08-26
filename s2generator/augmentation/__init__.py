# -*- coding: utf-8 -*-
"""
Created on 2026/03/02 12:15:45
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/S2Generator
"""

__all__ = [
    "amplitude_modulation",
    "censor_augmentation",
    "empirical_mode_modulation",
    "frequency_perturbation",
    "time_series_upsampling",
    "time_series_downsampling",
    "spike_injection",
    "wiener_filter",
    "add_linear_trend",
    "add_piecewise_linear_trend",
    "add_nonlinear_trend",
    "value_flipping",
    "time_series_mixup",
]

# Import the amplitude modulation function
from ._amplitude_modulation import amplitude_modulation

# Import the censoring augmentation function
from ._censor_augmentation import censor_augmentation

# Import the empirical mode modulation function
from ._empirical_mode_modulation import empirical_mode_modulation

# Import the frequency perturbation function
from ._frequency_perturbation import frequency_perturbation

# Import the resampling functions
from ._resampling import time_series_upsampling, time_series_downsampling

# Import the spike injection function
from ._spike_injection import spike_injection

# Import the wiener filter function
from ._wiener_filter import wiener_filter

# Import the time transformation functions
from ._time_transformation import (
    add_linear_trend,
    add_piecewise_linear_trend,
    add_nonlinear_trend,
    value_flipping,
    time_series_mixup,
)
