# -*- coding: utf-8 -*-
"""
Created on 2026/03/14 22:48:10
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/S2Generator
"""
import numpy as np

from s2generator.utils import (
    linear_interpolation,
    cubic_spline_interpolation,
    lagrange_interpolation,
)


def upsample_time_series(
    time_series: np.ndarray,
    target_length: int,
    interpolation_method: str = "linear",
) -> np.ndarray:
    """
    Upsample the input time series to a specified target length using interpolation.

    :param time_series: Input time series, a 1D numpy array
    :param target_length: Desired length of the upsampled time series
    :param interpolation_method: Method of interpolation to use. Options are "linear", "cubic", or "lagrange".

    :return: Upsampled time series, a 1D numpy array of length equal to target_length.
    """
    # Validate the input time series
    time_series = np.asarray(time_series)
    if time_series.ndim != 1:
        raise ValueError("Input time_series must be a 1D array.")

    # Get the length of the input time series
    seq_length = len(time_series)

    # Validate target_length
    if target_length <= seq_length:
        raise ValueError(
            "target_length must be greater than the length of the input time series."
        )

    # Validate interpolation method
    if interpolation_method not in ["linear", "cubic", "lagrange"]:
        raise ValueError(
            "interpolation_method must be one of 'linear', 'cubic', or 'lagrange'."
        )

    # Original indices and new indices for interpolation
    original_indices = np.arange(seq_length)
    new_indices = np.linspace(0, seq_length - 1, target_length)

    # Perform interpolation based on the specified method
    if interpolation_method == "linear":
        upsampled_series = linear_interpolation(
            original_indices, time_series, new_indices
        )
    elif interpolation_method == "cubic":
        upsampled_series = cubic_spline_interpolation(
            original_indices, time_series, new_indices
        )
    elif interpolation_method == "lagrange":
        upsampled_series = lagrange_interpolation(
            original_indices, time_series, new_indices
        )

    return upsampled_series
