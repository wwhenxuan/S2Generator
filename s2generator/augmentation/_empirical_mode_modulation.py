# -*- coding: utf-8 -*-
"""
Created on 2026/03/05 11:05:45
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/S2Generator
"""
from typing import Optional

import numpy as np

from pysdkit import EMD


def empirical_mode_modulation(
    time_series: np.ndarray,
    min_scale_factor: float = 0.5,
    max_scale_factor: float = 2.0,
    low_frequency_enhancement: bool = True,
    spline_kind: str = "cubic",
    extrema_detection: str = "parabol",
    max_imfs: Optional[int] = None,
    rng: Optional[np.random.RandomState] = None,
    seed: int = 42,
) -> np.ndarray:
    """
    Perform empirical mode modulation on the input time series.
    This augmentation decomposes the time series into intrinsic mode functions (IMFs)
    using Empirical Mode Decomposition (EMD) and then reconstructs the signal
    by randomly modifying the IMFs to introduce non-linear trends and variations.

    :param time_series: Input time series, a 1D numpy array.
    :param min_scale_factor: The minimum scaling factor to apply to each IMF.
    :param max_scale_factor: The maximum scaling factor to apply to each IMF.
    :param low_frequency_enhancement: Whether to enhance the low-frequency components.
                                      If True, the method will suppress the high-frequency noise in the
                                      empirical mode decomposition results and focus on enhancing
                                      the characterization of the low-frequency part.
    :param spline_kind: The kind of spline to use for interpolation,
                        options are "akima", "cubic", "pchip", "cubic_hermite", "slinear", "quadratic", "linear"
    :param extrema_detection: The method for detecting extrema in the EMD process, options are "parabol" or "simple"
    :param max_imfs: The maximum number of IMFs to extract,
                     if None, it will extract all possible IMFs until the residue is a monotonic function.

    :return: Empirical mode modulated time series, a 1D numpy array of the same length as the input series.
    """
    # Validate the input time series
    time_series = np.asarray(time_series)
    if time_series.ndim != 1:
        raise ValueError("Input time_series must be a 1D array.")

    # Normalize the input time series to have zero mean and unit variance
    mean, std = np.mean(time_series), np.std(time_series)
    time_series = (time_series - mean) / (
        std + 1e-8
    )  # Add a small value to avoid division by zero

    # Check if max_imfs is valid
    if max_imfs is None:
        # 表示会完整的分解所有的IMF分量
        # 直到剩余的分量不再满足IMF的定义为止
        max_imfs = -1

    # Validate the spline kind
    assert spline_kind in [
        "akima",
        "cubic",
        "pchip",
        "cubic_hermite",
        "slinear",
        "quadratic",
        "linear",
    ], "spline_kind must be one of 'akima', 'cubic', 'pchip', 'cubic_hermite', 'slinear', 'quadratic', 'linear'."

    # Validate the extrema detection method
    assert extrema_detection in [
        "parabol",
        "simple",
    ], "extrema_detection must be one of 'parabol' or 'simple'."

    # Initialize random number generator
    if rng is None:
        rng = np.random.RandomState(seed=seed)

    # Perform Empirical Mode Decomposition
    emd = EMD(
        max_imfs=max_imfs, spline_kind=spline_kind, extrema_detection=extrema_detection
    )
    print(type(time_series))
    imfs = emd.fit_transform(signal=time_series)

    # Get the number of IMFs extracted
    num_imfs = imfs.shape[0]

    # Randomly select a scaling factor for modulation
    scale_factor = rng.uniform(
        low=min_scale_factor, high=max_scale_factor, size=num_imfs
    )

    # Validate the low_frequency_enhancement parameter
    assert isinstance(
        low_frequency_enhancement, bool
    ), "low_frequency_enhancement must be a boolean value."
    if low_frequency_enhancement is True:
        # If low-frequency enhancement is enabled, we can apply a stronger scaling to the lower frequency IMFs
        scale_factor = np.sort(
            scale_factor
        )  # Sort the scale factors to enhance low-frequency components more than high-frequency ones

    # Randomly modify the IMFs to create modulation
    modified_imfs = []
    for index, imf in enumerate(imfs):
        # Randomly scale each IMF by a factor between min_scale_factor and max_scale_factor
        scaled_imf = imf * scale_factor[index]
        modified_imfs.append(scaled_imf)

    # Reconstruct the signal from the modified IMFs
    modulated_time_series = np.sum(modified_imfs, axis=0)

    # Denormalize the modulated time series to restore the original scale
    modulated_time_series = (modulated_time_series - np.mean(modulated_time_series)) / (
        np.std(modulated_time_series) + 1e-8
    )
    modulated_time_series = modulated_time_series * (std + 1e-8) + mean

    return modulated_time_series
