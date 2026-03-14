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


def time_series_upsampling(
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


def time_series_downsampling(
    ts_data: np.ndarray,
    target_length: int = None,
    scale_factor: float = None,
    method: str = "linear",
    custom_x_known: np.ndarray = None,
    window_agg_func: callable = None,
) -> np.ndarray:
    """
    Time series dedicated downsampling function (reduce resolution)
    Supports two ways to specify target length: directly specify target_length or calculate via scale_factor
    Supports two downsampling strategies: interpolation and sliding window aggregation

    :param ts_data: Input time series array, shape [num_samples, seq_length]
    :param target_length: Target sequence length (length after downsampling), must be less than original length
    :param scale_factor: Scaling factor (0 < scale_factor < 1), e.g, 0.5 means halving the length
                         If specified, target_length will be calculated automatically (choose one)
    :param method: Downsampling method:
                   - Interpolation: "linear", "cubic", "lagrange"
                   - Aggregation: "mean", "max", "min"
    :param custom_x_known: Custom known time coordinates (x-axis), only valid for interpolation methods
    :param window_agg_func: Custom window aggregation function (only valid for method="custom_agg"),
                             Inputs data in window, outputs single aggregated value

    :return: Downsampled array, shape [num_samples, target_length]

    :raises ValueError: If input parameters are invalid
    """
    # ===================== Input Validation =====================
    # 1. Dimension check
    if ts_data.ndim != 2:
        raise ValueError(
            f"Input must be 2D array [num_samples, seq_length], current dimension: {ts_data.ndim}"
        )
    num_samples, original_length = ts_data.shape

    # 2. Target length calculation (choose one, target_length has priority)
    if target_length is None and scale_factor is None:
        raise ValueError("Must specify target_length or scale_factor (choose one)")

    if scale_factor is not None:
        if not (0 < scale_factor < 1):
            raise ValueError(
                f"scale_factor must be between (0,1), current value: {scale_factor}"
            )
        target_length = int(original_length * scale_factor)

    # 3. Target length validity check
    if target_length >= original_length:
        raise ValueError(
            f"Target length {target_length} must be less than original length {original_length} (downsampling)"
        )
    if target_length <= 0:
        raise ValueError(f"Target length {target_length} must be greater than 0")

    # 4. Method validity check
    valid_methods = ["linear", "cubic", "lagrange", "mean", "max", "min", "custom_agg"]
    if method not in valid_methods:
        raise ValueError(
            f"Supported methods: {valid_methods}, current selection: {method}"
        )

    # 5. Custom aggregation function check
    if method == "custom_agg" and window_agg_func is None:
        raise ValueError("window_agg_func must be specified when method='custom_agg'")

    # ===================== Downsampling Logic =====================
    # Case 1: Interpolation-based downsampling (based on coordinate mapping)
    if method in ["linear", "cubic", "lagrange"]:
        # Generate time coordinates
        if custom_x_known is None:
            x_known = np.linspace(0, 1, original_length)  # Normalized coordinates
        else:
            if len(custom_x_known) != original_length:
                raise ValueError(
                    f"custom_x_known length {len(custom_x_known)} must equal original length {original_length}"
                )
            x_known = custom_x_known

        # Generate target coordinates after downsampling
        x_new = np.linspace(x_known.min(), x_known.max(), target_length)

        # Interpolate sample by sample
        downsampled_data = np.zeros((num_samples, target_length), dtype=ts_data.dtype)
        for i in range(num_samples):
            y_known = ts_data[i]
            try:
                if method == "linear":
                    downsampled_data[i] = linear_interpolation(x_known, y_known, x_new)
                elif method == "cubic":
                    # Fallback to linear if insufficient points for cubic spline
                    if original_length < 3:
                        downsampled_data[i] = linear_interpolation(
                            x_known, y_known, x_new
                        )
                    else:
                        downsampled_data[i] = cubic_spline_interpolation(
                            x_known, y_known, x_new
                        )
                elif method == "lagrange":
                    downsampled_data[i] = lagrange_interpolation(
                        x_known, y_known, x_new
                    )
            except Exception as e:
                raise RuntimeError(f"Sample {i} interpolation failed: {str(e)}")

    # Case 2: Window aggregation-based downsampling (based on sliding window/chunking)
    else:
        # Calculate window size (handle non-divisible case, last window may be smaller)
        window_size = original_length // target_length
        remainder = original_length % target_length

        downsampled_data = np.zeros((num_samples, target_length), dtype=ts_data.dtype)

        for i in range(num_samples):
            sample = ts_data[i]
            start_idx = 0

            for j in range(target_length):
                # Assign window size: first 'remainder' windows have 1 more point to ensure all points are covered
                current_window_size = window_size + 1 if j < remainder else window_size
                end_idx = start_idx + current_window_size

                # Extract data in window
                window_data = sample[start_idx:end_idx]

                # Perform aggregation
                if method == "mean":
                    downsampled_data[i, j] = np.mean(window_data)
                elif method == "max":
                    downsampled_data[i, j] = np.max(window_data)
                elif method == "min":
                    downsampled_data[i, j] = np.min(window_data)
                elif method == "custom_agg":
                    downsampled_data[i, j] = window_agg_func(window_data)

                start_idx = end_idx

    return downsampled_data
