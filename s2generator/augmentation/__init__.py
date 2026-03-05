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
]

# Import the amplitude modulation function
from ._amplitude_modulation import amplitude_modulation

# Import the censoring augmentation function
from ._censor_augmentation import censor_augmentation

# Import the empirical model modulation function
from ._empirical_model_modulation import empirical_model_modulation

# Import the frequency perturbation function
from ._frequency_perturbation import frequency_perturbation
