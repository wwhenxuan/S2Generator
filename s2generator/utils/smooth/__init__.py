# -*- coding: utf-8 -*-
"""Smoothing filters for 1D time series."""

from ._smooth import (
    exponential_smoothing,
    gaussian_smoothing,
    savgol_smoothing,
    simple_moving_average,
    smooth_show_info,
    weighted_moving_average,
)

__all__ = [
    "simple_moving_average",
    "weighted_moving_average",
    "gaussian_smoothing",
    "savgol_smoothing",
    "exponential_smoothing",
    "smooth_show_info",
]
