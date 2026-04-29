# -*- coding: utf-8 -*-
"""
Created on 2025/08/23 17:09:18
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/S2Generator
"""
from typing import Union, Tuple

import numpy as np


def generate_arma_samples(
    num_samples: int,
    seq_len: int,
    phi1: float = 0.6,
    theta1: float = -0.4,
    sigma: float = 0.5,
    return_params: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, float, float, float]]:
    """
    Generate ARMA(1,1) stationary time series samples of shape [num_samples, seq_len].

    :param num_samples: Number of samples to generate.
    :param seq_len: Length of each time series sample.
    :param phi1: AR(1) coefficient.
    :param theta1: MA(1) coefficient.
    :param sigma: Standard deviation of the white noise.
    :param return_params: Whether to return the parameters used for generation.

    :return: Generated samples of shape [num_samples, seq_len]. If return_params is True, also returns the parameters (phi1, theta1, sigma).
    """
    samples = []
    for _ in range(num_samples):
        # Initialize white noise (excitation source) and time series
        eps = np.random.normal(0, sigma, seq_len)  # White noise sequence
        x = np.zeros(seq_len)
        x[0] = eps[0]  # Initial value

        # Recursively generate ARMA(1,1) sequence: Xt = phi1*Xt-1 + eps_t - theta1*eps_t-1
        for t in range(1, seq_len):
            x[t] = phi1 * x[t - 1] + eps[t] - theta1 * eps[t - 1]
        samples.append(x)

    # Return generated samples and parameters
    if return_params:
        return np.array(samples), (phi1, theta1, sigma)
    return np.array(samples)


def generate_nonstationary_sine(
    num_samples: int,
    seq_len: int,
    freq: float = 2.0,
    sample_rate: Union[int, float] = 100,
    amp: float = 1.5,
    return_params: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, float, Union[int, float]]]:
    """Generate non-stationary sine signals with linear trend of shape [num_samples, seq_len].

    :param num_samples: Number of samples to generate.
    :param seq_len: Length of each time series sample.
    :param freq: Frequency of the sine wave.
    :param sample_rate: Sampling rate of the time series.
    :param amp: Amplitude of the sine wave.
    :param return_params: Whether to return the parameters used for generation.

    :return: Generated non-stationary sine wave samples of shape [num_samples, seq_len], frequency, and sample rate.
    """
    t = np.linspace(0, seq_len / sample_rate, seq_len, endpoint=False)
    nonstationary_samples = []

    for _ in range(num_samples):
        phase = np.random.uniform(0, 2 * np.pi)

        # Sine signal + linear trend (causing non-stationarity) + small noise
        sine_seq = amp * np.sin(2 * np.pi * freq * t + phase)

        # Linear trend: increases over time, core source of non-stationarity
        trend = 0.1 * t
        noise = np.random.normal(0, 0.05, seq_len)
        nonstationary_seq = sine_seq + trend + noise
        nonstationary_samples.append(nonstationary_seq)

    # Return generated samples and parameters
    if return_params:
        return np.array(nonstationary_samples), freq, sample_rate
    return np.array(nonstationary_samples)
