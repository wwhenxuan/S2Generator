# -*- coding: utf-8 -*-
"""
Post-processing transforms for synthetic multivariate coupling pipeline.

This module implements the observational layer (Stage 3) of the TiRex-2 coupling
pipeline. After coupling mechanisms impose cross-variate dependencies, these
transforms add realistic observational artefacts:

1. Variate permutation: random reordering of variates
2. Smooth time warping: Brownian-bridge lags for asynchronous sampling
3. Patch masking: contiguous NaN blocks (generalized from TiRex)
4. Partial future observability: truncating future portions of random covariates
5. Discretization: value-discretization (uniform, quantile, power-law) and
   time-discretization (freezes, staircases, duty cycles)

Reference:
    Podest, P., et al. (2026). TiRex-2: Generalizing TiRex to Multivariate Data
    and Streaming. arXiv:2607.01204v1, Section 3.4 & Appendix F.

Created on 2026/08/10 00:00:00
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/S2Generator
"""

from typing import Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d


def variate_permutation(
    rng: np.random.RandomState,
    series: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly permute the variate (column) order.

    This mimics the fact that variate ordering across datasets is arbitrary.

    :param rng: The random number generator.
    :param series: Input series of shape (T, Q).
    :return: Tuple of (permuted_series, permutation_indices).
    """
    T, Q = series.shape
    perm = rng.permutation(Q)
    return series[:, perm], perm


def smooth_time_warping(
    rng: np.random.RandomState,
    series: np.ndarray,
    max_lag: float = 0.05,
    n_bridges: int = 3,
) -> np.ndarray:
    """Apply smooth per-variate time warping via Brownian-bridge lags.

    Each variate is independently warped in time by a smooth, randomly generated
    lag function. This simulates asynchronous sampling across variates.

    The warping function maps the original time index t to a new index t' = t + lag(t),
    where lag(t) is a smooth Brownian bridge constrained to [-max_lag*T, max_lag*T].

    :param rng: The random number generator.
    :param series: Input series of shape (T, Q).
    :param max_lag: Maximum absolute lag as a fraction of sequence length.
    :param n_bridges: Number of Brownian bridge control points.
    :return: Time-warped series of shape (T, Q).
    """
    T, Q = series.shape
    max_shift = int(max_lag * T)
    result = np.zeros_like(series)

    for j in range(Q):
        # Generate a smooth lag function using Brownian bridge
        control_t = np.linspace(0, T - 1, n_bridges + 2)
        # Brownian increments
        increments = rng.normal(0, max_shift / np.sqrt(n_bridges), n_bridges)
        # Cumulative sum (Brownian motion), zero at both ends
        lag_at_controls = np.cumsum(increments)
        # Bridge: enforce zero at endpoints already (starts at 0 from cumsum)
        # Clamp to valid range
        lag_at_controls = np.clip(lag_at_controls, -max_shift, max_shift)
        # Add zero at start and end
        lag_at_controls = np.concatenate([[0], lag_at_controls, [0]])

        # Interpolate to full resolution
        lag_interp = interp1d(
            control_t,
            lag_at_controls,
            kind="cubic",
            fill_value="extrapolate",
        )
        lag = lag_interp(np.arange(T))

        # Apply warping via interpolation
        new_indices = np.arange(T) + lag
        new_indices = np.clip(new_indices, 0, T - 1)

        # Interpolate the original series at warped positions
        original_interp = interp1d(
            np.arange(T),
            series[:, j],
            kind="linear",
            fill_value="extrapolate",
        )
        result[:, j] = original_interp(new_indices)

    return result


def patch_masking(
    rng: np.random.RandomState,
    series: np.ndarray,
    patch_size: int = 32,
    mask_probability: float = 0.1,
    min_mask_patches: int = 1,
    max_mask_patches: int = 5,
    per_variate: bool = True,
    nan_value: float = np.nan,
) -> np.ndarray:
    """Apply contiguous patch masking to simulate missing observations.

    Generalized from the contiguous-patch scheme of TiRex (Auer et al., 2025b).
    Masks out contiguous blocks of patches with NaN values to simulate sensor
    dropouts, joint blackouts, or independent sensor faults.

    :param rng: The random number generator.
    :param series: Input series of shape (T, Q).
    :param patch_size: Size of each patch in time steps.
    :param mask_probability: Probability of masking any given patch.
    :param min_mask_patches: Minimum number of consecutive patches to mask.
    :param max_mask_patches: Maximum number of consecutive patches to mask.
    :param per_variate: If True, mask each variate independently.
                        If False, apply the same mask to all variates.
    :param nan_value: Value to use for masked positions (default: NaN).
    :return: Series with masked patches, shape (T, Q).
    """
    T, Q = series.shape
    result = series.copy()
    n_patches = T // patch_size

    if n_patches == 0:
        return result

    if per_variate:
        variates_to_mask = range(Q)
    else:
        variates_to_mask = [0]  # Generate once, apply to all

    for j in variates_to_mask:
        p = 0
        while p < n_patches:
            if rng.random() < mask_probability:
                # Determine the number of consecutive patches to mask
                n_mask = rng.randint(min_mask_patches, max_mask_patches + 1)
                end_p = min(p + n_mask, n_patches)

                start_idx = p * patch_size
                end_idx = min(end_p * patch_size, T)

                if per_variate:
                    result[start_idx:end_idx, j] = nan_value
                else:
                    result[start_idx:end_idx, :] = nan_value

                p = end_p
            else:
                p += 1

    return result


def partial_future_observability(
    rng: np.random.RandomState,
    series: np.ndarray,
    horizon: int,
    future_mask_probability: float = 0.3,
) -> np.ndarray:
    """Truncate future portions of random covariates.

    For each variate, with given probability, the future portion (beyond the
    forecast origin) is set to NaN. This prevents the model from becoming
    dependent on the future-covariate channel being fully populated.

    :param rng: The random number generator.
    :param series: Input series of shape (T, Q), where the first T - horizon
                   steps are historical context and the last `horizon` steps
                   are the future.
    :param horizon: Number of future time steps (prediction horizon).
    :param future_mask_probability: Probability of masking the future portion
                                    for each variate.
    :return: Series with partially masked future, shape (T, Q).
    """
    T, Q = series.shape
    if horizon <= 0 or horizon >= T:
        return series

    result = series.copy()
    origin = T - horizon

    for j in range(Q):
        if rng.random() < future_mask_probability:
            result[origin:, j] = np.nan

    return result


def value_discretization(
    rng: np.random.RandomState,
    series: np.ndarray,
    mode: Optional[str] = None,
    n_bins: Optional[int] = None,
) -> np.ndarray:
    """Apply value discretization (quantization) to simulate sensor precision limits.

    Supports three modes:
    - 'uniform': Equal-width bins
    - 'quantile': Equal-frequency bins (quantile-based)
    - 'power_law': Bin edges follow a power-law distribution

    :param rng: The random number generator.
    :param series: Input series of shape (T, Q).
    :param mode: Discretization mode. If None, randomly sampled.
    :param n_bins: Number of bins. If None, randomly sampled.
    :return: Discretized series, shape (T, Q).
    """
    T, Q = series.shape
    result = np.zeros_like(series)

    if mode is None:
        mode = rng.choice(["uniform", "quantile", "power_law"])

    for j in range(Q):
        col = series[:, j]
        valid = ~np.isnan(col)

        if not valid.any():
            result[:, j] = col
            continue

        v_min, v_max = col[valid].min(), col[valid].max()

        if n_bins is None:
            n_b = rng.randint(2, 20)
        else:
            n_b = n_bins

        if mode == "uniform":
            edges = np.linspace(v_min, v_max, n_b + 1)
        elif mode == "quantile":
            edges = np.quantile(
                col[valid], np.linspace(0, 1, n_b + 1)
            )
        elif mode == "power_law":
            # Power-law spaced bins: denser at one end
            exponent = rng.uniform(0.3, 3.0)
            t = np.linspace(0, 1, n_b + 1) ** exponent
            edges = v_min + t * (v_max - v_min)
        else:
            edges = np.linspace(v_min, v_max, n_b + 1)

        # Map each value to its bin center
        bin_centers = (edges[:-1] + edges[1:]) / 2.0
        indices = np.digitize(col, edges[1:-1], right=False)
        indices = np.clip(indices, 0, n_b - 1)

        result[:, j] = bin_centers[indices]
        # Preserve NaN values
        result[~valid, j] = np.nan

    return result


def time_discretization(
    rng: np.random.RandomState,
    series: np.ndarray,
    mode: Optional[str] = None,
    max_hold: int = 10,
) -> np.ndarray:
    """Apply time discretization to simulate irregular update intervals.

    Supports three modes:
    - 'freeze': Values repeat (hold last value) for random durations
    - 'staircase': Values change only at random intervals, with linear interpolation
    - 'duty_cycle': Values follow an on-off duty cycle pattern

    :param rng: The random number generator.
    :param series: Input series of shape (T, Q).
    :param mode: Time discretization mode. If None, randomly sampled.
    :param max_hold: Maximum number of steps to hold a value.
    :return: Time-discretized series, shape (T, Q).
    """
    T, Q = series.shape
    result = np.zeros_like(series)

    if mode is None:
        mode = rng.choice(["freeze", "staircase", "duty_cycle"])

    for j in range(Q):
        col = series[:, j]

        if mode == "freeze":
            result[:, j] = _apply_freeze(rng, col, max_hold)
        elif mode == "staircase":
            result[:, j] = _apply_staircase(rng, col, max_hold)
        elif mode == "duty_cycle":
            result[:, j] = _apply_duty_cycle(rng, col, max_hold)
        else:
            result[:, j] = col

    return result


def _apply_freeze(
    rng: np.random.RandomState,
    x: np.ndarray,
    max_hold: int,
) -> np.ndarray:
    """Apply freeze (sample-and-hold) discretization.

    :param rng: The random number generator.
    :param x: 1D input array of length T.
    :param max_hold: Maximum hold duration.
    :return: Freeze-discretized array.
    """
    T = len(x)
    result = np.zeros_like(x)

    t = 0
    while t < T:
        hold = rng.randint(1, max_hold + 1)
        end = min(t + hold, T)
        val = x[t]
        result[t:end] = val
        t = end

    return result


def _apply_staircase(
    rng: np.random.RandomState,
    x: np.ndarray,
    max_hold: int,
) -> np.ndarray:
    """Apply staircase discretization with linear interpolation between changes.

    :param rng: The random number generator.
    :param x: 1D input array of length T.
    :param max_hold: Maximum interval between value changes.
    :return: Staircase-discretized array.
    """
    T = len(x)
    result = np.zeros_like(x)

    t = 0
    last_val = x[0]
    while t < T:
        hold = rng.randint(1, max_hold + 1)
        end = min(t + hold, T)

        if end < T:
            next_val = x[end]
        else:
            next_val = last_val

        # Linear interpolation between last_val and next_val
        n_steps = end - t
        result[t:end] = np.linspace(last_val, next_val, n_steps)

        last_val = next_val
        t = end

    return result


def _apply_duty_cycle(
    rng: np.random.RandomState,
    x: np.ndarray,
    max_hold: int,
) -> np.ndarray:
    """Apply duty cycle discretization with on-off pattern.

    During 'off' periods, the value is held constant. During 'on' periods,
    the original signal passes through.

    :param rng: The random number generator.
    :param x: 1D input array of length T.
    :param max_hold: Maximum cycle duration.
    :return: Duty-cycle-discretized array.
    """
    T = len(x)
    result = np.zeros_like(x)

    t = 0
    is_on = True
    while t < T:
        duration = rng.randint(1, max_hold + 1)
        end = min(t + duration, T)

        if is_on:
            result[t:end] = x[t:end]
        else:
            # Hold last value during the off period
            if t > 0:
                result[t:end] = result[t - 1]
            else:
                result[t:end] = x[0]

        is_on = not is_on
        t = end

    return result


class PostProcessor:
    """Pipeline for applying observational transforms to coupled multivariate data.

    This combines all post-processing stages described in TiRex-2 Appendix F:
    variate permutation, smooth time warping, patch masking, partial future
    observability, and value/time discretization.
    """

    def __init__(
        self,
        patch_size: int = 32,
        horizon: int = 0,
        apply_permutation: bool = True,
        apply_warping: bool = True,
        apply_masking: bool = True,
        apply_future_mask: bool = True,
        apply_discretization: bool = True,
        dtype: np.dtype = np.float64,
    ) -> None:
        """Initialize the post-processing pipeline.

        :param patch_size: Patch size for mask operations.
        :param horizon: Forecast horizon for future masking.
        :param apply_permutation: Whether to apply variate permutation.
        :param apply_warping: Whether to apply smooth time warping.
        :param apply_masking: Whether to apply patch masking.
        :param apply_future_mask: Whether to apply future observability mask.
        :param apply_discretization: Whether to apply value/time discretization.
        :param dtype: The numpy data type.
        """
        self._patch_size = patch_size
        self._horizon = horizon
        self._apply_permutation = apply_permutation
        self._apply_warping = apply_warping
        self._apply_masking = apply_masking
        self._apply_future_mask = apply_future_mask
        self._apply_discretization = apply_discretization
        self._dtype = dtype

    def __str__(self) -> str:
        return "PostProcessor"

    def __call__(
        self,
        rng: np.random.RandomState,
        series: np.ndarray,
        horizon: Optional[int] = None,
    ) -> np.ndarray:
        """Apply the full post-processing pipeline.

        :param rng: The random number generator.
        :param series: Input series of shape (T, Q).
        :param horizon: Override for forecast horizon.
        :return: Post-processed series of shape (T, Q).
        """
        result = series.astype(self._dtype, copy=True)
        h = horizon if horizon is not None else self._horizon

        # Stage 1: Variate permutation
        if self._apply_permutation and rng.random() < 0.8:
            result, _ = variate_permutation(rng, result)

        # Stage 2: Smooth time warping
        if self._apply_warping and rng.random() < 0.5:
            result = smooth_time_warping(rng, result)

        # Stage 3: Patch masking
        if self._apply_masking and rng.random() < 0.7:
            result = patch_masking(
                rng,
                result,
                patch_size=self._patch_size,
                mask_probability=rng.uniform(0.02, 0.15),
            )

        # Stage 4: Partial future observability
        if self._apply_future_mask and h > 0 and rng.random() < 0.5:
            result = partial_future_observability(
                rng,
                result,
                horizon=h,
                future_mask_probability=rng.uniform(0.1, 0.5),
            )

        # Stage 5: Value discretization
        if self._apply_discretization and rng.random() < 0.4:
            result = value_discretization(rng, result)

        # Stage 6: Time discretization
        if self._apply_discretization and rng.random() < 0.3:
            result = time_discretization(rng, result)

        return result
