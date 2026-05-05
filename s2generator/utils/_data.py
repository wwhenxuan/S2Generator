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
    "generate_square_wave",
    "generate_sawtooth_wave",
    "generate_damped_oscillation",
]

from typing import Union, Tuple, Optional, Sequence

import numpy as np


def generate_arma_samples(
    seq_length: int,
    phi1: float = 0.6,
    theta1: float = -0.4,
    sigma: float = 0.5,
    return_params: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Tuple[float, float, float]]]:
    """
    Generate an ARMA(1,1) stationary one-dimensional time series.

    :param seq_length: Length of the time series.
    :param phi1: AR(1) coefficient.
    :param theta1: MA(1) coefficient.
    :param sigma: Standard deviation of the white noise.
    :param return_params: Whether to return the parameters used for generation.

    :return: Generated one-dimensional time series. If return_params is True, also returns the parameters (phi1, theta1, sigma).
    """
    if seq_length <= 0:
        raise ValueError("seq_length must be a positive integer")
    if sigma < 0:
        raise ValueError("sigma must be non-negative")

    # Initialize white noise (excitation source) and time series
    eps = np.random.normal(0, sigma, seq_length)  # White noise sequence
    x = np.zeros(seq_length)
    x[0] = eps[0]  # Initial value

    # Recursively generate ARMA(1,1) sequence: Xt = phi1*Xt-1 + eps_t - theta1*eps_t-1
    for t in range(1, seq_length):
        x[t] = phi1 * x[t - 1] + eps[t] - theta1 * eps[t - 1]

    # Return generated sample and parameters
    if return_params:
        return x, (phi1, theta1, sigma)
    return x


def generate_nonstationary_sine(
    seq_length: int,
    freq: float = 2.0,
    sample_rate: Union[int, float] = 100,
    amp: float = 1.5,
    return_params: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, float, Union[int, float]]]:
    """Generate a non-stationary sine signal with linear trend.

    :param seq_length: Length of the time series.
    :param freq: Frequency of the sine wave.
    :param sample_rate: Sampling rate of the time series.
    :param amp: Amplitude of the sine wave.
    :param return_params: Whether to return the parameters used for generation.

    :return: Generated one-dimensional non-stationary sine wave, frequency, and sample rate.
    """
    if seq_length <= 0:
        raise ValueError("seq_length must be a positive integer")
    if freq < 0:
        raise ValueError("freq must be non-negative")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if amp < 0:
        raise ValueError("amp must be non-negative")

    t = np.linspace(0, seq_length / sample_rate, seq_length, endpoint=False)
    phase = np.random.uniform(0, 2 * np.pi)

    # Sine signal + linear trend (causing non-stationarity) + small noise
    sine_seq = amp * np.sin(2 * np.pi * freq * t + phase)

    # Linear trend: increases over time, core source of non-stationarity
    trend = 0.1 * t
    noise = np.random.normal(0, 0.05, seq_length)
    nonstationary_seq = sine_seq + trend + noise

    # Return generated sample and parameters
    if return_params:
        return nonstationary_seq, freq, sample_rate
    return nonstationary_seq


def generate_variable_frequency_sine(
    seq_length: int,
    start_freq: float = 1.0,
    end_freq: float = 5.0,
    sample_rate: Union[int, float] = 100,
    amp: float = 1.5,
    noise_std: float = 0.05,
    return_params: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, float, float, Union[int, float]]]:
    """Generate a sine signal with time-varying frequency.

    :param seq_length: Length of the time series.
    :param start_freq: Initial frequency of the sine wave.
    :param end_freq: Final frequency of the sine wave.
    :param sample_rate: Sampling rate of the time series.
    :param amp: Amplitude of the sine wave.
    :param noise_std: Standard deviation of the Gaussian noise added to the signal.
    :param return_params: Whether to return the parameters used for generation.

    :return: Generated one-dimensional variable-frequency sine wave, start frequency, end frequency, and sample rate.
    """
    if seq_length <= 0:
        raise ValueError("seq_length must be a positive integer")
    if start_freq < 0 or end_freq < 0:
        raise ValueError("start_freq and end_freq must be non-negative")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if amp < 0:
        raise ValueError("amp must be non-negative")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")

    freq_progress = np.linspace(0, 1, seq_length)
    instantaneous_freq = start_freq + (end_freq - start_freq) * freq_progress
    phase = np.random.uniform(0, 2 * np.pi)

    # Integrate instantaneous frequency to keep the sine signal continuous
    instantaneous_phase = (
        phase + 2 * np.pi * np.cumsum(instantaneous_freq) / sample_rate
    )
    sine_seq = amp * np.sin(instantaneous_phase)

    # Add small noise to make generated sample more realistic
    noise = np.random.normal(0, noise_std, seq_length)
    variable_frequency_seq = sine_seq + noise

    # Return generated sample and parameters
    if return_params:
        return variable_frequency_seq, start_freq, end_freq, sample_rate
    return variable_frequency_seq


def generate_sine_with_local_frequency_changes(
    seq_length: int,
    base_freq: float = 2.0,
    sample_rate: Union[int, float] = 100,
    amp: float = 1.5,
    change_positions: Optional[Sequence[float]] = None,
    change_ranges: Optional[Union[float, Sequence[float]]] = None,
    change_percents: Optional[Union[float, Sequence[float]]] = None,
    directions: Optional[Union[str, Sequence[str]]] = None,
    noise_std: float = 0.05,
    return_params: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, float, Union[int, float]]]:
    """Generate a sine signal with local frequency changes.

    The frequency starts from ``base_freq``. Users can specify several sequence
    positions by percentile values, and the frequency will be increased or decreased
    within a local range around each specified position.

    :param seq_length: Length of the time series.
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

    :return: Generated one-dimensional sine wave. If return_params is True, also
             returns the instantaneous frequency sequence, base frequency, and sample rate.
    """
    if seq_length <= 0:
        raise ValueError("seq_length must be a positive integer")
    if base_freq < 0:
        raise ValueError("base_freq must be non-negative")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if amp < 0:
        raise ValueError("amp must be non-negative")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")

    instantaneous_freq = np.full(seq_length, base_freq, dtype=float)

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

            center_idx = int(round(position * (seq_length - 1)))
            half_width = max(1, int(round(range_percent * seq_length / 2)))
            start_idx = max(0, center_idx - half_width)
            end_idx = min(seq_length, center_idx + half_width + 1)

            factor = (
                1.0 + change_percent
                if direction == "increase"
                else 1.0 - change_percent
            )
            factor = max(0.0, factor)
            instantaneous_freq[start_idx:end_idx] *= factor

    phase = np.random.uniform(0, 2 * np.pi)

    # Integrate instantaneous frequency to keep the sine signal continuous
    instantaneous_phase = (
        phase + 2 * np.pi * np.cumsum(instantaneous_freq) / sample_rate
    )
    sine_seq = amp * np.sin(instantaneous_phase)

    # Add small noise to make generated sample more realistic
    noise = np.random.normal(0, noise_std, seq_length)
    sample = sine_seq + noise

    # Return generated sample and parameters
    if return_params:
        return sample, instantaneous_freq, base_freq, sample_rate
    return sample


def generate_triangle_wave(
    seq_length: int,
    freq: float = 2.0,
    sample_rate: Union[int, float] = 100,
    amp: float = 1.5,
    noise_std: float = 0.05,
    return_params: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, float, Union[int, float]]]:
    """Generate a one-dimensional triangle wave signal.

    :param seq_length: Length of the time series.
    :param freq: Frequency of the triangle wave.
    :param sample_rate: Sampling rate of the time series.
    :param amp: Amplitude of the triangle wave.
    :param noise_std: Standard deviation of the Gaussian noise added to the signal.
    :param return_params: Whether to return the parameters used for generation.

    :return: Generated one-dimensional triangle wave, frequency, and sample rate.
    """
    if seq_length <= 0:
        raise ValueError("seq_length must be a positive integer")
    if freq < 0:
        raise ValueError("freq must be non-negative")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if amp < 0:
        raise ValueError("amp must be non-negative")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")

    t = np.linspace(0, seq_length / sample_rate, seq_length, endpoint=False)
    phase = np.random.uniform(0, 1)

    # Generate a triangle wave in the range [-amp, amp]
    cycle_position = (freq * t + phase) % 1
    triangle_seq = amp * (4 * np.abs(cycle_position - 0.5) - 1)

    # Add small noise to make generated sample more realistic
    noise = np.random.normal(0, noise_std, seq_length)
    triangle_sample = triangle_seq + noise

    # Return generated sample and parameters
    if return_params:
        return triangle_sample, freq, sample_rate
    return triangle_sample


def generate_square_wave(
    seq_length: int,
    freq: float = 2.0,
    sample_rate: Union[int, float] = 100,
    amp: float = 1.5,
    duty_cycle: float = 0.5,
    noise_std: float = 0.05,
    return_params: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, float, Union[int, float]]]:
    """Generate a one-dimensional square wave signal.

    :param seq_length: Length of the time series.
    :param freq: Frequency of the square wave.
    :param sample_rate: Sampling rate of the time series.
    :param amp: Amplitude of the square wave.
    :param duty_cycle: Fraction of each cycle where the signal stays at the positive amplitude.
    :param noise_std: Standard deviation of the Gaussian noise added to the signal.
    :param return_params: Whether to return the parameters used for generation.

    :return: Generated one-dimensional square wave, frequency, and sample rate.
    """
    if seq_length <= 0:
        raise ValueError("seq_length must be a positive integer")
    if freq < 0:
        raise ValueError("freq must be non-negative")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if amp < 0:
        raise ValueError("amp must be non-negative")
    if not 0 < duty_cycle < 1:
        raise ValueError("duty_cycle must be in (0, 1)")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")

    t = np.linspace(0, seq_length / sample_rate, seq_length, endpoint=False)
    phase = np.random.uniform(0, 1)

    # Generate a square wave in the range [-amp, amp]
    cycle_position = (freq * t + phase) % 1
    square_seq = np.where(cycle_position < duty_cycle, amp, -amp)

    # Add small noise to make generated sample more realistic
    noise = np.random.normal(0, noise_std, seq_length)
    square_sample = square_seq + noise

    # Return generated sample and parameters
    if return_params:
        return square_sample, freq, sample_rate
    return square_sample


def generate_sawtooth_wave(
    seq_length: int,
    freq: float = 2.0,
    sample_rate: Union[int, float] = 100,
    amp: float = 1.5,
    width: float = 1.0,
    noise_std: float = 0.05,
    return_params: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, float, Union[int, float]]]:
    """Generate a one-dimensional sawtooth wave signal.

    :param seq_length: Length of the time series.
    :param freq: Frequency of the sawtooth wave.
    :param sample_rate: Sampling rate of the time series.
    :param amp: Amplitude of the sawtooth wave.
    :param width: Fraction of each cycle used for the rising ramp. A value of 1.0
                  produces a standard rising sawtooth wave, while 0.0 produces a
                  falling sawtooth wave.
    :param noise_std: Standard deviation of the Gaussian noise added to the signal.
    :param return_params: Whether to return the parameters used for generation.

    :return: Generated one-dimensional sawtooth wave, frequency, and sample rate.
    """
    if seq_length <= 0:
        raise ValueError("seq_length must be a positive integer")
    if freq < 0:
        raise ValueError("freq must be non-negative")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if amp < 0:
        raise ValueError("amp must be non-negative")
    if not 0 <= width <= 1:
        raise ValueError("width must be in [0, 1]")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")

    t = np.linspace(0, seq_length / sample_rate, seq_length, endpoint=False)
    phase = np.random.uniform(0, 1)

    # Generate a sawtooth wave in the range [-amp, amp]
    cycle_position = (freq * t + phase) % 1
    if width == 0:
        sawtooth_seq = amp * (1 - 2 * cycle_position)
    elif width == 1:
        sawtooth_seq = amp * (2 * cycle_position - 1)
    else:
        sawtooth_seq = amp * np.where(
            cycle_position < width,
            2 * cycle_position / width - 1,
            1 - 2 * (cycle_position - width) / (1 - width),
        )

    # Add small noise to make generated sample more realistic
    noise = np.random.normal(0, noise_std, seq_length)
    sawtooth_sample = sawtooth_seq + noise

    # Return generated sample and parameters
    if return_params:
        return sawtooth_sample, freq, sample_rate
    return sawtooth_sample


def generate_damped_oscillation(
    seq_length: int,
    freq: float = 2.0,
    damping_factor: float = 0.05,
    sample_rate: Union[int, float] = 100,
    amp: float = 1.5,
    noise_std: float = 0.05,
    flip: bool = False,
    return_params: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, float, float, Union[int, float], bool]]:
    """Generate a one-dimensional damped oscillation signal.

    A damped oscillation is formed by multiplying a sine wave by an exponential
    decay envelope, so the amplitude gradually decreases over time.

    :param seq_length: Length of the time series.
    :param freq: Frequency of the oscillation.
    :param damping_factor: Exponential damping factor controlling how quickly the amplitude decays.
    :param sample_rate: Sampling rate of the time series.
    :param amp: Initial amplitude of the oscillation.
    :param noise_std: Standard deviation of the Gaussian noise added to the signal.
    :param flip: Whether to flip the damping direction so the envelope grows over time.
    :param return_params: Whether to return the parameters used for generation.

    :return: Generated one-dimensional damped oscillation, frequency, damping factor, sample rate, and flip flag.
    """
    if seq_length <= 0:
        raise ValueError("seq_length must be a positive integer")
    if freq < 0:
        raise ValueError("freq must be non-negative")
    if damping_factor < 0:
        raise ValueError("damping_factor must be non-negative")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if amp < 0:
        raise ValueError("amp must be non-negative")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")

    t = np.linspace(0, seq_length / sample_rate, seq_length, endpoint=False)
    phase = np.random.uniform(0, 2 * np.pi)

    # Generate exponential damping envelope and optionally flip its direction
    envelope = np.exp(-damping_factor * t)
    damped_seq = amp * envelope * np.sin(2 * np.pi * freq * t + phase)

    # Add small noise to make generated sample more realistic
    noise = np.random.normal(0, noise_std, seq_length)
    damped_sample = damped_seq + noise

    # Return generated sample and parameters
    if return_params:
        if flip:
            return np.flip(damped_sample), freq, damping_factor, sample_rate, flip
        else:
            return damped_sample, freq, damping_factor, sample_rate, flip

    if flip:
        return np.flip(damped_sample)
    else:
        return damped_sample


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    a = np.array([1, 2, 3])
    print(a[::-1])

    sample = generate_damped_oscillation(
        1000, freq=4, damping_factor=0.1, noise_std=0, flip=True
    )
    plt.plot(sample)
    plt.show()
