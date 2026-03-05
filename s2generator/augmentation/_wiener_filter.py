# -*- coding: utf-8 -*-
"""
Created on 2026/03/05 16:12:41
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/S2Generator
"""

from typing import Optional

import numpy as np


def wiener_filter(
    time_series: np.ndarray,
    noise_variance: float = 1.0,
    window_size: Optional[int] = None,
) -> np.ndarray:
    """
    Implement Wiener filter to remove white noise from signals.

    :param time_series: The input time series, a 1D numpy array representing the noisy signal.
    :param noise_variance: The known variance of the white noise to be removed, default is 1.0
    :param window_size: The window size for power spectral density estimation, optional
        if None, it will use the entire signal length as the window size.

    :return: The filtered time series, a 1D numpy array of the same length as the input series.
    """
    # Validate the input time series
    time_series = np.asarray(time_series, dtype=np.float64)

    # If window_size is not provided, use the length of the time series
    if window_size is None:
        window_size = len(time_series)

    # Calculate the Fourier transform of the signal
    signal_fft = np.fft.fft(time_series)

    # Calculate the power spectral density (PSD) of the signal.
    signal_psd = np.abs(signal_fft) ** 2 / len(time_series)

    # Estimate the power spectrum of the original signal (noisy PSD - noise PSD)
    # The power spectral density of noise is constant for white noise: noise variance
    noise_psd = noise_variance
    original_signal_psd_estimate = np.maximum(
        signal_psd - noise_psd, 1e-10
    )  # Avoid negative values

    # Calculate the frequency domain response of the Wiener filter.
    wiener_filter_freq = original_signal_psd_estimate / (
        original_signal_psd_estimate + noise_psd
    )

    # Apply a filter and perform an inverse Fourier transform to return to the time domain.
    filtered_fft = signal_fft * wiener_filter_freq

    # Remove numerical error by taking the real part
    filtered_signal = np.fft.ifft(filtered_fft).real

    return filtered_signal
