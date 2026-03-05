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
    "empirical_model_modulation",
    "frequency_perturbation",
    "spike_injection",
    "wiener_filter",
]

# Import the amplitude modulation function
from ._amplitude_modulation import amplitude_modulation

# Import the censoring augmentation function
from ._censor_augmentation import censor_augmentation

# Import the empirical model modulation function
from ._empirical_model_modulation import empirical_model_modulation

# Import the frequency perturbation function
from ._frequency_perturbation import frequency_perturbation

# Import the spike injection function
from ._spike_injection import spike_injection

# Import the wiener filter function
from ._wiener_filter import wiener_filter

# Import the time transformation functions
from ._time_transformation import add_linear_trend, time_series_mixup
