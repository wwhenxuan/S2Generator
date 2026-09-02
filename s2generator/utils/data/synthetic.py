# -*- coding: utf-8 -*-
"""
Parametric (regular) univariate time-series generators.

These sit beside the bundled real-data loaders in ``s2generator.utils.data``
so examples can mix synthetic waveforms with ETT / weather / electricity
slices from one package.

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
    "generate_chirp_signal",
    "generate_impulse_signal",
    "generate_step_signal",
    "generate_ramp_signal",
    "generate_exponential_signal",
    "generate_logarithmic_signal",
    "generate_stock_price",
    "generate_electrocardiogram",
    "generate_electroencephalogram",
    "AVAILABLE_SYNTHETIC_GENERATORS",
    "generate",
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


def generate_chirp_signal(
    seq_length: int,
    start_freq: float = 1.0,
    end_freq: float = 10.0,
    sample_rate: Union[int, float] = 100,
    amp: float = 1.5,
    noise_std: float = 0.05,
    return_params: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, float, float, Union[int, float]]]:
    """Generate a one-dimensional chirp signal.

    :param seq_length: Length of the time series.
    :param start_freq: Starting frequency of the chirp signal.
    :param end_freq: Ending frequency of the chirp signal.
    :param sample_rate: Sampling rate of the time series.
    :param amp: Amplitude of the chirp signal.
    :param noise_std: Standard deviation of the Gaussian noise added to the signal.
    :param return_params: Whether to return the parameters used for generation.

    :return: Generated one-dimensional chirp signal. If return_params is True,
             also returns the start frequency, end frequency, and sample rate.
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

    t = np.linspace(0, seq_length / sample_rate, seq_length, endpoint=False)
    phase = np.random.uniform(0, 2 * np.pi)
    freq_slope = (end_freq - start_freq) / (seq_length / sample_rate)

    # Generate chirp signal with linearly varying instantaneous frequency
    chirp_phase = 2 * np.pi * (start_freq * t + 0.5 * freq_slope * t**2) + phase
    chirp_seq = amp * np.sin(chirp_phase)

    # Add small noise to make generated sample more realistic
    noise = np.random.normal(0, noise_std, seq_length)
    chirp_sample = chirp_seq + noise

    # Return generated sample and parameters
    if return_params:
        return chirp_sample, start_freq, end_freq, sample_rate
    return chirp_sample


def generate_impulse_signal(
    seq_length: int,
    impulse_position: Union[float, Sequence[float]] = 0.5,
    impulse_width: int = 2,
    impulse_amp: float = 1.5,
    background_value: float = 0.0,
    noise_std: float = 0.05,
    return_params: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Sequence[float], int, float]]:
    """Generate a one-dimensional impulse signal.

    :param seq_length: Length of the time series.
    :param impulse_position: Relative impulse center position or positions in [0, 1].
                             For example, 0.5 places the impulse near the middle,
                             and [0.2, 0.5, 0.8] places impulses at multiple positions.
    :param impulse_width: Width of each impulse in number of samples.
    :param impulse_amp: Amplitude added within each impulse region.
    :param background_value: Background value outside the impulse regions.
    :param noise_std: Standard deviation of the Gaussian noise added to the signal.
    :param return_params: Whether to return the parameters used for generation.

    :return: Generated one-dimensional impulse signal. If return_params is True,
             also returns the impulse positions, impulse width, and impulse amplitude.
    """
    if seq_length <= 0:
        raise ValueError("seq_length must be a positive integer")
    if isinstance(impulse_position, (int, float)):
        impulse_positions = [float(impulse_position)]
    else:
        impulse_positions = list(impulse_position)
    if len(impulse_positions) == 0:
        raise ValueError("impulse_position must contain at least one position")
    if any(not 0 <= position <= 1 for position in impulse_positions):
        raise ValueError("each value in impulse_position must be in [0, 1]")
    if impulse_width <= 0:
        raise ValueError("impulse_width must be a positive integer")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")

    impulse_sample = np.full(seq_length, background_value, dtype=float)
    half_width = impulse_width // 2

    for position in impulse_positions:
        center_idx = int(round(position * (seq_length - 1)))
        start_idx = max(0, center_idx - half_width)
        end_idx = min(seq_length, start_idx + impulse_width)
        start_idx = max(0, end_idx - impulse_width)

        # Add impulse in the selected local region
        impulse_sample[start_idx:end_idx] += impulse_amp

    # Add small noise to make generated sample more realistic
    noise = np.random.normal(0, noise_std, seq_length)
    impulse_sample = impulse_sample + noise

    # Return generated sample and parameters
    if return_params:
        return impulse_sample, impulse_positions, impulse_width, impulse_amp
    return impulse_sample


def generate_step_signal(
    seq_length: int,
    step_position: Union[float, Sequence[float]] = 0.5,
    step_height: Union[float, Sequence[float]] = 1.0,
    base_value: float = 0.0,
    noise_std: float = 0.05,
    flip: bool = False,
    return_params: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Sequence[float], Sequence[float]]]:
    """Generate a one-dimensional step signal.

    :param seq_length: Length of the time series.
    :param step_position: Relative step position or positions in [0, 1].
                          For example, 0.5 places a step near the middle,
                          and [0.2, 0.5, 0.8] creates multiple step changes.
    :param step_height: Height change applied at each step position. A scalar
                        applies the same step height to all positions.
    :param base_value: Initial background value before any step occurs.
    :param noise_std: Standard deviation of the Gaussian noise added to the signal.
    :param flip: Whether to flip the step signal so the step height increases over time.
    :param return_params: Whether to return the parameters used for generation.

    :return: Generated one-dimensional step signal. If return_params is True,
             also returns the step positions and step heights.
    """
    if seq_length <= 0:
        raise ValueError("seq_length must be a positive integer")
    if isinstance(step_position, (int, float)):
        step_positions = [float(step_position)]
    else:
        step_positions = list(step_position)
    if len(step_positions) == 0:
        raise ValueError("step_position must contain at least one position")
    if any(not 0 <= position <= 1 for position in step_positions):
        raise ValueError("each value in step_position must be in [0, 1]")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")

    if isinstance(step_height, (int, float)):
        step_heights = [float(step_height)] * len(step_positions)
    else:
        step_heights = list(step_height)
    if len(step_heights) != len(step_positions):
        raise ValueError("step_position and step_height must have the same length")

    step_sample = np.full(seq_length, base_value, dtype=float)
    sorted_steps = sorted(zip(step_positions, step_heights), key=lambda item: item[0])

    for position, height in sorted_steps:
        step_idx = int(round(position * (seq_length - 1)))
        step_sample[step_idx:] += height

    # Add small noise to make generated sample more realistic
    noise = np.random.normal(0, noise_std, seq_length)
    step_sample = step_sample + noise

    if flip:
        step_sample = np.flip(step_sample)

    # Return generated sample and parameters
    if return_params:
        return (
            step_sample,
            [item[0] for item in sorted_steps],
            [item[1] for item in sorted_steps],
        )
    return step_sample


def generate_ramp_signal(
    seq_length: int,
    start_position: Union[float, Sequence[float]] = 0.0,
    end_position: Union[float, Sequence[float]] = 1.0,
    ramp_height: Union[float, Sequence[float]] = 1.0,
    base_value: float = 0.0,
    noise_std: float = 0.05,
    flip: bool = False,
    return_params: bool = False,
) -> Union[
    np.ndarray,
    Tuple[np.ndarray, Sequence[float], Sequence[float], Sequence[float]],
]:
    """Generate a one-dimensional ramp signal.

    :param seq_length: Length of the time series.
    :param start_position: Relative ramp start position or positions in [0, 1].
                           A scalar creates one ramp, and a sequence creates
                           multiple ramp segments, such as [0.1, 0.5] creates two ramp segments.
    :param end_position: Relative ramp end position or positions in [0, 1].
                         A scalar applies to all ramp segments.
    :param ramp_height: Height change from the start to the end of each ramp.
                        A scalar applies to all ramp segments.
    :param base_value: Initial background value before any ramp is added.
    :param noise_std: Standard deviation of the Gaussian noise added to the signal.
    :param flip: Whether to reverse the generated signal along the time axis.
    :param return_params: Whether to return the parameters used for generation.

    :return: Generated one-dimensional ramp signal. If return_params is True,
             also returns the ramp start positions, end positions, and ramp heights.
    """
    if seq_length <= 0:
        raise ValueError("seq_length must be a positive integer")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")

    if isinstance(start_position, (int, float)):
        start_positions = [float(start_position)]
    else:
        start_positions = list(start_position)
    if len(start_positions) == 0:
        raise ValueError("start_position must contain at least one position")
    if any(not 0 <= position <= 1 for position in start_positions):
        raise ValueError("each value in start_position must be in [0, 1]")

    num_ramps = len(start_positions)

    if isinstance(end_position, (int, float)):
        end_positions = [float(end_position)] * num_ramps
    else:
        end_positions = list(end_position)
    if len(end_positions) != num_ramps:
        raise ValueError("start_position and end_position must have the same length")
    if any(not 0 <= position <= 1 for position in end_positions):
        raise ValueError("each value in end_position must be in [0, 1]")

    if isinstance(ramp_height, (int, float)):
        ramp_heights = [float(ramp_height)] * num_ramps
    else:
        ramp_heights = list(ramp_height)
    if len(ramp_heights) != num_ramps:
        raise ValueError("start_position and ramp_height must have the same length")

    ramp_sample = np.full(seq_length, base_value, dtype=float)

    for start, end, height in zip(start_positions, end_positions, ramp_heights):
        start_idx = int(round(start * (seq_length - 1)))
        end_idx = int(round(end * (seq_length - 1)))

        if start_idx == end_idx:
            ramp_sample[start_idx:] += height
            continue

        segment_start = min(start_idx, end_idx)
        segment_end = max(start_idx, end_idx)
        segment_length = segment_end - segment_start + 1
        ramp_values = np.linspace(0, height, segment_length)
        if end_idx < start_idx:
            ramp_values = ramp_values[::-1]

        ramp_sample[segment_start : segment_end + 1] += ramp_values
        ramp_sample[segment_end + 1 :] += height

    # Add small noise to make generated sample more realistic
    noise = np.random.normal(0, noise_std, seq_length)
    ramp_sample = ramp_sample + noise

    if flip:
        ramp_sample = np.flip(ramp_sample)

    # Return generated sample and parameters
    if return_params:
        return ramp_sample, start_positions, end_positions, ramp_heights
    return ramp_sample


def generate_exponential_signal(
    seq_length: int,
    growth_rate: float = 1.0,
    sample_rate: Union[int, float] = 100,
    amp: float = 1.0,
    offset: float = 0.0,
    decay: bool = False,
    normalize: bool = False,
    noise_std: float = 0.05,
    return_params: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, float, Union[int, float], float, float, bool]]:
    """Generate a one-dimensional exponential signal.

    :param seq_length: Length of the time series.
    :param growth_rate: Exponential growth or decay rate.
    :param sample_rate: Sampling rate of the time series.
    :param amp: Amplitude multiplier of the exponential signal.
    :param offset: Constant offset added to the signal.
    :param decay: Whether to generate an exponential decay signal instead of a growth signal.
    :param normalize: Whether to normalize the exponential component to [0, 1] before applying amp and offset.
    :param noise_std: Standard deviation of the Gaussian noise added to the signal.
    :param return_params: Whether to return the parameters used for generation.

    :return: Generated one-dimensional exponential signal. If return_params is True,
             also returns growth rate, sample rate, amplitude, offset, and decay flag.
    """
    if seq_length <= 0:
        raise ValueError("seq_length must be a positive integer")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if amp < 0:
        raise ValueError("amp must be non-negative")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")

    t = np.linspace(0, seq_length / sample_rate, seq_length, endpoint=False)
    exponent = -growth_rate * t if decay else growth_rate * t
    exponential_seq = np.exp(exponent)

    if normalize:
        seq_min = np.min(exponential_seq)
        seq_max = np.max(exponential_seq)
        if seq_max > seq_min:
            exponential_seq = (exponential_seq - seq_min) / (seq_max - seq_min)

    exponential_sample = amp * exponential_seq + offset

    # Add small noise to make generated sample more realistic
    noise = np.random.normal(0, noise_std, seq_length)
    exponential_sample = exponential_sample + noise

    # Return generated sample and parameters
    if return_params:
        return exponential_sample, growth_rate, sample_rate, amp, offset, decay
    return exponential_sample


def generate_logarithmic_signal(
    seq_length: int,
    sample_rate: Union[int, float] = 100,
    amp: float = 1.5,
    offset: float = 0.0,
    log_base: float = np.e,
    growth_rate: float = 1.0,
    shift: float = 1.0,
    normalize: bool = False,
    noise_std: float = 0.05,
    return_params: bool = False,
) -> Union[
    np.ndarray,
    Tuple[np.ndarray, Union[int, float], float, float, float, float],
]:
    """Generate a one-dimensional logarithmic signal.

    :param seq_length: Length of the time series.
    :param sample_rate: Sampling rate of the time series.
    :param amp: Amplitude multiplier of the logarithmic signal.
    :param offset: Constant offset added to the signal.
    :param log_base: Base of the logarithm. Must be positive and not equal to 1.
    :param growth_rate: Scaling factor applied to time before the logarithm.
    :param shift: Positive shift added inside the logarithm to avoid invalid values.
    :param normalize: Whether to normalize the logarithmic component to [0, 1] before applying amp and offset.
    :param noise_std: Standard deviation of the Gaussian noise added to the signal.
    :param return_params: Whether to return the parameters used for generation.

    :return: Generated one-dimensional logarithmic signal. If return_params is True,
             also returns sample rate, amplitude, offset, log base, and growth rate.
    """
    if seq_length <= 0:
        raise ValueError("seq_length must be a positive integer")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if amp < 0:
        raise ValueError("amp must be non-negative")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")
    if log_base <= 0 or log_base == 1:
        raise ValueError("log_base must be positive and not equal to 1")
    if growth_rate <= 0:
        raise ValueError("growth_rate must be positive")
    if shift <= 0:
        raise ValueError("shift must be positive")

    t = np.linspace(0, seq_length / sample_rate, seq_length, endpoint=False)
    log_argument = growth_rate * t + shift
    logarithmic_seq = np.log(log_argument) / np.log(log_base)

    if normalize:
        seq_min = np.min(logarithmic_seq)
        seq_max = np.max(logarithmic_seq)
        if seq_max > seq_min:
            logarithmic_seq = (logarithmic_seq - seq_min) / (seq_max - seq_min)

    logarithmic_sample = amp * logarithmic_seq + offset

    # Add small noise to make generated sample more realistic
    noise = np.random.normal(0, noise_std, seq_length)
    logarithmic_sample = logarithmic_sample + noise

    # Return generated sample and parameters
    if return_params:
        return logarithmic_sample, sample_rate, amp, offset, log_base, growth_rate
    return logarithmic_sample


def generate_stock_price(
    seq_length: int,
    initial_price: float = 100.0,
    drift: float = 0.0005,
    volatility: float = 0.02,
    jump_probability: float = 0.01,
    jump_scale: float = 0.08,
    trend_strength: float = 0.0,
    noise_std: float = 0.0,
    return_params: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, float, float, float, float]]:
    """Generate a one-dimensional simulated stock price series.

    The stock price is generated using a geometric random walk with optional
    jump events and a deterministic trend component, which can imitate common
    characteristics of stock price fluctuations.

    :param seq_length: Length of the time series.
    :param initial_price: Initial stock price.
    :param drift: Average log-return drift per time step.
    :param volatility: Standard deviation of the random log-return component.
    :param jump_probability: Probability of a jump event at each time step.
    :param jump_scale: Standard deviation of jump magnitudes in log-return space.
    :param trend_strength: Additional deterministic linear trend strength applied to returns.
    :param noise_std: Standard deviation of additive Gaussian observation noise.
    :param return_params: Whether to return the parameters used for generation.

    :return: Generated one-dimensional stock price series. If return_params is True,
             also returns the initial price, drift, volatility, and jump probability.
    """
    if seq_length <= 0:
        raise ValueError("seq_length must be a positive integer")
    if initial_price <= 0:
        raise ValueError("initial_price must be positive")
    if volatility < 0:
        raise ValueError("volatility must be non-negative")
    if not 0 <= jump_probability <= 1:
        raise ValueError("jump_probability must be in [0, 1]")
    if jump_scale < 0:
        raise ValueError("jump_scale must be non-negative")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")

    prices = np.zeros(seq_length, dtype=float)
    prices[0] = initial_price

    trend_component = np.linspace(0, trend_strength, seq_length)
    random_returns = np.random.normal(drift, volatility, seq_length - 1)
    jump_flags = np.random.rand(seq_length - 1) < jump_probability
    jump_returns = np.random.normal(0, jump_scale, seq_length - 1) * jump_flags
    total_returns = random_returns + jump_returns + trend_component[1:]

    for i in range(1, seq_length):
        prices[i] = prices[i - 1] * np.exp(total_returns[i - 1])

    if noise_std > 0:
        prices = prices + np.random.normal(0, noise_std, seq_length)
        prices = np.maximum(prices, 1e-8)

    # Return generated sample and parameters
    if return_params:
        return prices, initial_price, drift, volatility, jump_probability
    return prices


def generate_electrocardiogram(
    seq_length: int,
    heart_rate: float = 72.0,
    sample_rate: Union[int, float] = 250,
    amp: float = 1.0,
    noise_std: float = 0.01,
    baseline_wander_amp: float = 0.05,
    return_params: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, float, Union[int, float], float]]:
    """Generate a one-dimensional simulated electrocardiogram (ECG) signal.

    The ECG waveform is synthesized beat-by-beat using a sum of Gaussian
    components corresponding to the P wave, Q wave, R wave, S wave, and T wave,
    with optional baseline wander and additive Gaussian noise.

    :param seq_length: Length of the time series.
    :param heart_rate: Heart rate in beats per minute.
    :param sample_rate: Sampling rate of the time series.
    :param amp: Global amplitude scaling factor of the ECG waveform.
    :param noise_std: Standard deviation of additive Gaussian noise.
    :param baseline_wander_amp: Amplitude of low-frequency baseline wander.
    :param return_params: Whether to return the parameters used for generation.

    :return: Generated one-dimensional ECG signal. If return_params is True,
             also returns the heart rate, sample rate, and amplitude.
    """
    if seq_length <= 0:
        raise ValueError("seq_length must be a positive integer")
    if heart_rate <= 0:
        raise ValueError("heart_rate must be positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if amp < 0:
        raise ValueError("amp must be non-negative")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")
    if baseline_wander_amp < 0:
        raise ValueError("baseline_wander_amp must be non-negative")

    t = np.arange(seq_length) / sample_rate
    beat_period = 60.0 / heart_rate
    signal = np.zeros(seq_length, dtype=float)

    # A simple physiological template using Gaussian components.
    # Each tuple is (relative_position, relative_width, relative_amplitude).
    wave_components = [
        (0.18, 0.040, 0.12),  # P wave
        (0.36, 0.012, -0.15),  # Q wave
        (0.40, 0.010, 1.00),  # R wave
        (0.44, 0.014, -0.25),  # S wave
        (0.68, 0.080, 0.35),  # T wave
    ]

    beat_starts = np.arange(0.0, t[-1] + beat_period, beat_period)
    for beat_start in beat_starts:
        for rel_pos, rel_width, rel_amp in wave_components:
            center = beat_start + rel_pos * beat_period
            width = max(rel_width * beat_period, 1e-6)
            signal += rel_amp * np.exp(-0.5 * ((t - center) / width) ** 2)

    signal *= amp

    if baseline_wander_amp > 0:
        baseline_freq = np.random.uniform(0.15, 0.35)
        baseline_phase = np.random.uniform(0, 2 * np.pi)
        signal += baseline_wander_amp * np.sin(
            2 * np.pi * baseline_freq * t + baseline_phase
        )

    if noise_std > 0:
        signal += np.random.normal(0, noise_std, seq_length)

    # Return generated sample and parameters
    if return_params:
        return signal, heart_rate, sample_rate, amp
    return signal


def generate_electroencephalogram(
    seq_length: int,
    sample_rate: Union[int, float] = 256,
    amp: float = 1.0,
    alpha_weight: float = 1.0,
    beta_weight: float = 0.6,
    theta_weight: float = 0.4,
    delta_weight: float = 0.25,
    noise_std: float = 0.08,
    return_params: bool = False,
) -> Union[
    np.ndarray,
    Tuple[np.ndarray, Union[int, float], float, float, float, float, float],
]:
    """Generate a one-dimensional simulated electroencephalogram (EEG) signal.

    The EEG waveform is synthesized as a weighted combination of several
    canonical brain-rhythm bands, including delta, theta, alpha, and beta,
    with random phases, slow amplitude modulation, and additive Gaussian noise.

    :param seq_length: Length of the time series.
    :param sample_rate: Sampling rate of the time series.
    :param amp: Global amplitude scaling factor of the EEG waveform.
    :param alpha_weight: Weight of the alpha rhythm component.
    :param beta_weight: Weight of the beta rhythm component.
    :param theta_weight: Weight of the theta rhythm component.
    :param delta_weight: Weight of the delta rhythm component.
    :param noise_std: Standard deviation of additive Gaussian noise.
    :param return_params: Whether to return the parameters used for generation.

    :return: Generated one-dimensional EEG signal. If return_params is True,
             also returns the sample rate, amplitude, alpha weight, beta weight,
             theta weight, and delta weight.
    """
    if seq_length <= 0:
        raise ValueError("seq_length must be a positive integer")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if amp < 0:
        raise ValueError("amp must be non-negative")
    if alpha_weight < 0:
        raise ValueError("alpha_weight must be non-negative")
    if beta_weight < 0:
        raise ValueError("beta_weight must be non-negative")
    if theta_weight < 0:
        raise ValueError("theta_weight must be non-negative")
    if delta_weight < 0:
        raise ValueError("delta_weight must be non-negative")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")

    t = np.arange(seq_length) / sample_rate

    alpha_freq = np.random.uniform(8.0, 12.0)
    beta_freq = np.random.uniform(13.0, 30.0)
    theta_freq = np.random.uniform(4.0, 7.0)
    delta_freq = np.random.uniform(0.5, 3.5)

    alpha_phase = np.random.uniform(0, 2 * np.pi)
    beta_phase = np.random.uniform(0, 2 * np.pi)
    theta_phase = np.random.uniform(0, 2 * np.pi)
    delta_phase = np.random.uniform(0, 2 * np.pi)

    alpha_env = 1.0 + 0.20 * np.sin(2 * np.pi * np.random.uniform(0.1, 0.3) * t)
    beta_env = 1.0 + 0.15 * np.sin(2 * np.pi * np.random.uniform(0.2, 0.5) * t)
    theta_env = 1.0 + 0.18 * np.sin(2 * np.pi * np.random.uniform(0.05, 0.2) * t)
    delta_env = 1.0 + 0.12 * np.sin(2 * np.pi * np.random.uniform(0.03, 0.1) * t)

    signal = (
        alpha_weight * alpha_env * np.sin(2 * np.pi * alpha_freq * t + alpha_phase)
        + beta_weight * beta_env * np.sin(2 * np.pi * beta_freq * t + beta_phase)
        + theta_weight * theta_env * np.sin(2 * np.pi * theta_freq * t + theta_phase)
        + delta_weight * delta_env * np.sin(2 * np.pi * delta_freq * t + delta_phase)
    )

    signal *= amp

    if noise_std > 0:
        signal += np.random.normal(0, noise_std, seq_length)

    # Return generated sample and parameters
    if return_params:
        return (
            signal,
            sample_rate,
            amp,
            alpha_weight,
            beta_weight,
            theta_weight,
            delta_weight,
        )
    return signal


_GENERATOR_FUNCS = {
    "arma_samples": generate_arma_samples,
    "nonstationary_sine": generate_nonstationary_sine,
    "variable_frequency_sine": generate_variable_frequency_sine,
    "sine_with_local_frequency_changes": generate_sine_with_local_frequency_changes,
    "triangle_wave": generate_triangle_wave,
    "square_wave": generate_square_wave,
    "sawtooth_wave": generate_sawtooth_wave,
    "damped_oscillation": generate_damped_oscillation,
    "chirp_signal": generate_chirp_signal,
    "impulse_signal": generate_impulse_signal,
    "step_signal": generate_step_signal,
    "ramp_signal": generate_ramp_signal,
    "exponential_signal": generate_exponential_signal,
    "logarithmic_signal": generate_logarithmic_signal,
    "stock_price": generate_stock_price,
    "electrocardiogram": generate_electrocardiogram,
    "electroencephalogram": generate_electroencephalogram,
}

_GENERATOR_ALIASES = {
    "arma": "arma_samples",
    "ecg": "electrocardiogram",
    "eeg": "electroencephalogram",
    "chirp": "chirp_signal",
    "triangle": "triangle_wave",
    "square": "square_wave",
    "sawtooth": "sawtooth_wave",
    "impulse": "impulse_signal",
    "step": "step_signal",
    "ramp": "ramp_signal",
    "exponential": "exponential_signal",
    "logarithmic": "logarithmic_signal",
    "stock": "stock_price",
}

AVAILABLE_SYNTHETIC_GENERATORS: Tuple[str, ...] = tuple(_GENERATOR_FUNCS.keys())


def generate(
    name: str,
    seq_length: int,
    **kwargs,
) -> np.ndarray:
    """Generate a parametric series by catalog name.

    This is the synthetic counterpart of :func:`s2generator.utils.data.load_univariate`.

    :param name: Generator name (see :data:`AVAILABLE_SYNTHETIC_GENERATORS`)
                 or a short alias such as ``\"arma\"``, ``\"ecg\"``, ``\"eeg\"``.
    :param seq_length: Length of the generated series.
    :param kwargs: Forwarded to the underlying ``generate_*`` function.
    :return: One-dimensional ``ndarray``.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("generator name must be a non-empty string")
    key = name.strip()
    if key.startswith("generate_"):
        key = key[len("generate_") :]
    canonical = key if key in _GENERATOR_FUNCS else _GENERATOR_ALIASES.get(key.lower())
    if canonical is None or canonical not in _GENERATOR_FUNCS:
        raise ValueError(
            f"unknown synthetic generator {name!r}; "
            f"choose from {AVAILABLE_SYNTHETIC_GENERATORS}"
        )
    kwargs.pop("seq_length", None)
    return _GENERATOR_FUNCS[canonical](seq_length=seq_length, **kwargs)
