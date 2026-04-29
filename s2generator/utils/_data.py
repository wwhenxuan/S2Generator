# -*- coding: utf-8 -*-
"""
Created on 2025/08/23 17:09:18
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/S2Generator
"""
__all__ = [
    "generate_arma_samples",
    "generate_nonstationary_sine",
    "generate_variable_frequency_sine",
    "generate_sine_with_local_frequency_changes",
    "generate_triangle_wave",
]

from typing import Union, Tuple, Optional, Sequence

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


def generate_variable_frequency_sine(
    num_samples: int,
    seq_len: int,
    start_freq: float = 1.0,
    end_freq: float = 5.0,
    sample_rate: Union[int, float] = 100,
    amp: float = 1.5,
    noise_std: float = 0.05,
    return_params: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, float, float, Union[int, float]]]:
    """Generate sine signals with time-varying frequency of shape [num_samples, seq_len].

    :param num_samples: Number of samples to generate.
    :param seq_len: Length of each time series sample.
    :param start_freq: Initial frequency of the sine wave.
    :param end_freq: Final frequency of the sine wave.
    :param sample_rate: Sampling rate of the time series.
    :param amp: Amplitude of the sine wave.
    :param noise_std: Standard deviation of the Gaussian noise added to the signal.
    :param return_params: Whether to return the parameters used for generation.

    :return: Generated variable-frequency sine wave samples of shape [num_samples, seq_len], start frequency, end frequency, and sample rate.
    """
    freq_progress = np.linspace(0, 1, seq_len)
    instantaneous_freq = start_freq + (end_freq - start_freq) * freq_progress
    variable_frequency_samples = []

    for _ in range(num_samples):
        phase = np.random.uniform(0, 2 * np.pi)

        # Integrate instantaneous frequency to keep the sine signal continuous
        instantaneous_phase = (
            phase + 2 * np.pi * np.cumsum(instantaneous_freq) / sample_rate
        )
        sine_seq = amp * np.sin(instantaneous_phase)

        # Add small noise to make generated samples more realistic
        noise = np.random.normal(0, noise_std, seq_len)
        variable_frequency_seq = sine_seq + noise
        variable_frequency_samples.append(variable_frequency_seq)

    # Return generated samples and parameters
    if return_params:
        return np.array(variable_frequency_samples), start_freq, end_freq, sample_rate
    return np.array(variable_frequency_samples)


def generate_sine_with_local_frequency_changes(
    num_samples: int,
    seq_len: int,
    base_freq: float = 2.0,
    sample_rate: Union[int, float] = 100,
    amp: float = 1.5,
    change_positions: Optional[Sequence[float]] = None,
    change_ranges: Optional[Union[float, Sequence[float]]] = None,
    change_percents: Optional[Union[float, Sequence[float]]] = None,
    directions: Optional[Union[str, Sequence[str]]] = None,
    noise_std: float = 0.05,
    return_params: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, float, Union[int, float]],]:
    """Generate sine signals with local frequency changes of shape [num_samples, seq_len].

    The frequency starts from ``base_freq``. Users can specify several sequence
    positions by percentile values, and the frequency will be increased or decreased
    within a local range around each specified position.

    :param num_samples: Number of samples to generate.
    :param seq_len: Length of each time series sample.
    :param base_freq: Base frequency of the sine wave.
    :param sample_rate: Sampling rate of the time series.
    :param amp: Amplitude of the sine wave.
    :param change_positions: Percentile positions where frequency changes occur.
                             Each value should be in [0, 1], for example 0.3 means
                             30% of the sequence length.
    :param change_ranges: Width of each local change range as a percentage of the
                          sequence length. A scalar applies to all positions.
    :param change_percents: Frequency change percentage for each local range.
                            A scalar applies to all positions. For example, 0.2
                            means a 20% frequency change.
    :param directions: Change direction for each local range. Supported values are
                       ``"increase"`` and ``"decrease"``. A scalar applies to all
                       positions.
    :param noise_std: Standard deviation of the Gaussian noise added to the signal.
    :param return_params: Whether to return the instantaneous frequency sequence
                          and generation parameters.

    :return: Generated sine wave samples of shape [num_samples, seq_len]. If
             return_params is True, also returns the instantaneous frequency
             sequence, base frequency, and sample rate.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be a positive integer")
    if seq_len <= 0:
        raise ValueError("seq_len must be a positive integer")
    if base_freq < 0:
        raise ValueError("base_freq must be non-negative")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if amp < 0:
        raise ValueError("amp must be non-negative")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")

    instantaneous_freq = np.full(seq_len, base_freq, dtype=float)

    if change_positions is not None:
        positions = list(change_positions)
        num_changes = len(positions)

        if change_ranges is None:
            change_ranges_list = [0.1] * num_changes
        elif isinstance(change_ranges, (int, float)):
            change_ranges_list = [float(change_ranges)] * num_changes
        else:
            change_ranges_list = list(change_ranges)

        if change_percents is None:
            change_percents_list = [0.2] * num_changes
        elif isinstance(change_percents, (int, float)):
            change_percents_list = [float(change_percents)] * num_changes
        else:
            change_percents_list = list(change_percents)

        if directions is None:
            directions_list = ["increase"] * num_changes
        elif isinstance(directions, str):
            directions_list = [directions] * num_changes
        else:
            directions_list = list(directions)

        if not (
            len(change_ranges_list)
            == len(change_percents_list)
            == len(directions_list)
            == num_changes
        ):
            raise ValueError(
                "change_positions, change_ranges, change_percents, and directions must have the same length"
            )

        for position, range_percent, change_percent, direction in zip(
            positions, change_ranges_list, change_percents_list, directions_list
        ):
            if not 0 <= position <= 1:
                raise ValueError("each value in change_positions must be in [0, 1]")
            if not 0 < range_percent <= 1:
                raise ValueError("each value in change_ranges must be in (0, 1]")
            if change_percent < 0:
                raise ValueError("each value in change_percents must be non-negative")
            if direction not in {"increase", "decrease"}:
                raise ValueError(
                    "directions must contain only 'increase' or 'decrease'"
                )

            center_idx = int(round(position * (seq_len - 1)))
            half_width = max(1, int(round(range_percent * seq_len / 2)))
            start_idx = max(0, center_idx - half_width)
            end_idx = min(seq_len, center_idx + half_width + 1)

            factor = (
                1.0 + change_percent
                if direction == "increase"
                else 1.0 - change_percent
            )
            factor = max(0.0, factor)
            instantaneous_freq[start_idx:end_idx] *= factor

    samples = []
    for _ in range(num_samples):
        phase = np.random.uniform(0, 2 * np.pi)

        # Integrate instantaneous frequency to keep the sine signal continuous
        instantaneous_phase = (
            phase + 2 * np.pi * np.cumsum(instantaneous_freq) / sample_rate
        )
        sine_seq = amp * np.sin(instantaneous_phase)

        # Add small noise to make generated samples more realistic
        noise = np.random.normal(0, noise_std, seq_len)
        samples.append(sine_seq + noise)

    # Return generated samples and parameters
    if return_params:
        return np.array(samples), instantaneous_freq, base_freq, sample_rate
    return np.array(samples)


def generate_triangle_wave(
    num_samples: int,
    seq_len: int,
    freq: float = 2.0,
    sample_rate: Union[int, float] = 100,
    amp: float = 1.5,
    noise_std: float = 0.05,
    return_params: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, float, Union[int, float]]]:
    """Generate triangle wave signals of shape [num_samples, seq_len].

    :param num_samples: Number of samples to generate.
    :param seq_len: Length of each time series sample.
    :param freq: Frequency of the triangle wave.
    :param sample_rate: Sampling rate of the time series.
    :param amp: Amplitude of the triangle wave.
    :param noise_std: Standard deviation of the Gaussian noise added to the signal.
    :param return_params: Whether to return the parameters used for generation.

    :return: Generated triangle wave samples of shape [num_samples, seq_len], frequency, and sample rate.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be a positive integer")
    if seq_len <= 0:
        raise ValueError("seq_len must be a positive integer")
    if freq < 0:
        raise ValueError("freq must be non-negative")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if amp < 0:
        raise ValueError("amp must be non-negative")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")

    t = np.linspace(0, seq_len / sample_rate, seq_len, endpoint=False)
    triangle_samples = []

    for _ in range(num_samples):
        phase = np.random.uniform(0, 1)

        # Generate a triangle wave in the range [-amp, amp]
        cycle_position = (freq * t + phase) % 1
        triangle_seq = amp * (4 * np.abs(cycle_position - 0.5) - 1)

        # Add small noise to make generated samples more realistic
        noise = np.random.normal(0, noise_std, seq_len)
        triangle_samples.append(triangle_seq + noise)

    # Return generated samples and parameters
    if return_params:
        return np.array(triangle_samples), freq, sample_rate
    return np.array(triangle_samples)



if __name__ == "__main__":
    from matplotlib import pyplot as plt

    samples = generate_sine_with_local_frequency_changes(
        2,
        1000,
        base_freq=12.0,
        change_positions=[0.3, 0.7],
        change_ranges=[0.1, 0.1],
        change_percents=[0.9, 0.9],
        directions=["increase", "decrease"],
    )
    fig, ax = plt.subplots(2, 1, figsize=(10, 5), dpi=160, sharex=True)
    ax[0].plot(samples[0])
    ax[1].plot(samples[1])
    plt.show()
