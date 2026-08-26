# -*- coding: utf-8 -*-
"""
Created on 2026/03/05 16:19:59
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/S2Generator
"""

__all__ = [
    "add_linear_trend",
    "add_piecewise_linear_trend",
    "value_flipping",
    "time_series_mixup",
]

from typing import Optional, Sequence, Tuple

import numpy as np


def add_linear_trend(
    time_series: np.ndarray,
    trend_strength: float = 1.0,
    direction: str = "upward",
    normalize: bool = True,
) -> np.ndarray:
    """
    Perform linear trend augmentation on the input time series.
    This augmentation adds a linear trend to the input time series,
    which can help models learn to handle non-stationary data and improve their robustness to trends.

    :param time_series: Input time series, a 1D numpy array
    :param trend_strength: The strength of the linear trend to be added, default is 1.0.
    :param direction: The direction of the linear trend, either "upward" or "downward", default is "upward".
    :param normalize: Whether to normalize the output time series to maintain the same scale as the input, default is True.

    :return: Augmented time series with a linear trend, a 1D numpy array of the same length as the input series.
    """

    # Get the length of the time series
    seq_length = len(time_series)

    # Calculate the the energy of the original time series
    original_energy = np.mean(time_series**2)

    # Create a linear trend
    if direction == "upward":
        trend = np.linspace(0, 1, seq_length)
    elif direction == "downward":
        trend = np.linspace(0, -1, seq_length)
    else:
        raise ValueError("direction must be either 'upward' or 'downward'")

    # Scale the trend to have the same energy as the original time series
    trend_energy = np.mean(trend**2)

    if trend_energy > 0:
        # Scale the trend to have the same energy as the original time series, and then apply the trend strength factor
        trend = trend * np.sqrt(original_energy / trend_energy) * trend_strength
    else:
        # If the trend energy is zero (which can happen if the trend is constant),
        # we set the trend to zero to avoid division by zero
        trend = np.zeros_like(trend)

    if normalize:
        augmented_series = time_series + trend
        # Normalize the augmented series to maintain the same energy as the original time series
        augmented_series = (augmented_series - np.mean(augmented_series)) / np.std(
            augmented_series
        ) * np.std(time_series) + np.mean(time_series)
        return augmented_series

    # Average the original signal and the trend to maintain the overall scale
    return (time_series + trend) / 2


def add_piecewise_linear_trend(
    time_series: np.ndarray,
    num_segments: int = 3,
    trend_strengths: Optional[Sequence[float]] = None,
    directions: Optional[Sequence[str]] = None,
    strength_range: Tuple[float, float] = (0.5, 2.0),
    normalize: bool = True,
    rng: Optional[np.random.RandomState] = None,
    seed: int = 42,
) -> np.ndarray:
    """
    Add a multi-segment (piecewise linear) trend to a 1-D time series.

    The series is split into ``num_segments`` contiguous pieces. Each piece
    gets its own linear ramp whose **range** (net change) is
    ``trend_strengths[i]`` and whose **sign** is ``directions[i]``
    (``"upward"`` or ``"downward"``). Adjacent pieces share an endpoint so
    the composite trend is continuous.

    If ``trend_strengths`` or ``directions`` is omitted, the missing values
    are drawn at random: strengths from ``U[strength_range]``, directions
    uniformly from ``{"upward", "downward"}``.

    :param time_series: Input time series, a 1D numpy array.
    :param num_segments: Number of trend pieces. Must be >= 1 and at most
                         the series length.
    :param trend_strengths: Per-segment trend ranges / magnitudes. Length
                            must equal ``num_segments``. If None, sampled
                            uniformly from ``strength_range``.
    :param directions: Per-segment trend directions, each ``"upward"`` or
                       ``"downward"``. Length must equal ``num_segments``.
                       If None, sampled uniformly.
    :param strength_range: ``(min, max)`` used when sampling random
                           ``trend_strengths``.
    :param normalize: If True, affine-rescale the output to the original
                      mean and std (same behaviour as ``add_linear_trend``).
                      If False, average the series with the trend.
    :param rng: Optional random number generator. Used only for entries
                that are not provided by the caller.
    :param seed: Seed used when ``rng`` is None.
    :return: Augmented series of the same length as the input.
    """
    time_series = np.asarray(time_series, dtype=float)
    if time_series.ndim != 1:
        raise ValueError("Input time_series must be a 1D array.")

    seq_length = len(time_series)
    if num_segments < 1:
        raise ValueError("num_segments must be >= 1")
    if num_segments > seq_length:
        raise ValueError(
            f"num_segments ({num_segments}) cannot exceed the series length "
            f"({seq_length})"
        )

    low, high = strength_range
    if high < low:
        raise ValueError(f"strength_range max ({high}) must be >= min ({low})")

    if rng is None:
        rng = np.random.RandomState(seed)

    strengths = _resolve_trend_strengths(
        trend_strengths, num_segments=num_segments, low=low, high=high, rng=rng
    )
    dirs = _resolve_trend_directions(directions, num_segments=num_segments, rng=rng)

    bounds = _segment_bounds(seq_length, num_segments)
    original_energy = np.mean(time_series**2)
    scale = np.sqrt(original_energy) if original_energy > 0 else 1.0

    trend = np.zeros(seq_length, dtype=float)
    cursor = 0.0
    for i in range(num_segments):
        start, end = int(bounds[i]), int(bounds[i + 1])
        length = end - start
        sign = 1.0 if dirs[i] == "upward" else -1.0
        delta = sign * float(strengths[i]) * scale
        if length == 1:
            trend[start] = cursor
            cursor = cursor + delta
        else:
            trend[start:end] = np.linspace(cursor, cursor + delta, length)
            cursor = trend[end - 1]

    if normalize:
        augmented_series = time_series + trend
        std = np.std(time_series)
        aug_std = np.std(augmented_series)
        if aug_std > 0 and std > 0:
            augmented_series = (
                (augmented_series - np.mean(augmented_series)) / aug_std
            ) * std + np.mean(time_series)
        return augmented_series

    return (time_series + trend) / 2


def _segment_bounds(seq_length: int, num_segments: int) -> np.ndarray:
    """Split ``[0, seq_length)`` into ``num_segments`` contiguous pieces."""
    sizes = np.full(num_segments, seq_length // num_segments, dtype=int)
    sizes[: seq_length % num_segments] += 1
    return np.concatenate(([0], np.cumsum(sizes)))


def _resolve_trend_strengths(
    trend_strengths: Optional[Sequence[float]],
    num_segments: int,
    low: float,
    high: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    if trend_strengths is None:
        return rng.uniform(low, high, size=num_segments)
    strengths = np.asarray(trend_strengths, dtype=float).reshape(-1)
    if strengths.size != num_segments:
        raise ValueError(
            f"trend_strengths must have length num_segments={num_segments}, "
            f"got {strengths.size}"
        )
    if np.any(strengths < 0):
        raise ValueError("trend_strengths must be non-negative")
    return strengths


def _resolve_trend_directions(
    directions: Optional[Sequence[str]],
    num_segments: int,
    rng: np.random.RandomState,
) -> list:
    if directions is None:
        return rng.choice(["upward", "downward"], size=num_segments).tolist()
    if len(directions) != num_segments:
        raise ValueError(
            f"directions must have length num_segments={num_segments}, "
            f"got {len(directions)}"
        )
    parsed = []
    for item in directions:
        key = str(item).strip().lower()
        if key in {"upward", "up", "1"}:
            parsed.append("upward")
        elif key in {"downward", "down", "-1"}:
            parsed.append("downward")
        else:
            raise ValueError(
                "each direction must be 'upward' or 'downward', " f"got {item!r}"
            )
    return parsed


def value_flipping(time_series: np.ndarray) -> np.ndarray:
    """
    Perform value flipping augmentation on the input time series.
    The input and output series are both multiplied by -1,
    thereby inverting their trends while preserving temporal dependencies.
    This simple operation counteracts the model's tendency to latch onto persistent directional trends

    :param time_series: Input time series, a 1D numpy array

    :return: Augmented time series with flipped values, a 1D numpy array of the same length as the input series.
    """
    return -time_series


def time_series_mixup(a: np.ndarray, b: np.ndarray, alpha: float = 0.7) -> np.ndarray:
    """
    Mixup Enhancement: Weighted mixing of two time series to create a new augmented signal.
    This method combines two time series by taking a weighted average of them,
    where the weights are determined by a mixing parameter alpha.
    This can help models learn to generalize better by exposing them to a wider variety of signal combinations.

    :param a: First input time series, a 1D numpy array.
    :param b: Second input time series, a 1D numpy array of the same length as a.
    :param alpha: The mixing parameter that controls the weight of each time series in the mixup,
    default is 0.7. A value of alpha close to 1 gives more weight to the first time series (a),
    while a value close to 0 gives more weight to the second time series (b).

    :return: Mixed time series, a 1D numpy array of the same length as the input series.
    """
    assert a.shape == b.shape, "Input time series must have the same shape"

    # Calculate the mixed signal as a weighted average of the two input signals
    return alpha * a + (1 - alpha) * b
