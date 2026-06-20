# -*- coding: utf-8 -*-
"""
This module primarily houses tools and auxiliary functions or classes,
including functions for converting string notation into Markdown syntax,
normalizing time series data,
obtaining the current time,
saving and loading S2 data,
and calculating and measuring the similarity between two time series datasets.

Created on 2025/08/3 00:01:56
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""

__all__ = [
    "symbol_to_markdown",
    "z_score_normalization",
    "max_min_normalization",
    "get_time_now",
    "save_s2data",
    "load_s2data",
    "eacf_rlike",
    "yule_walker",
    "linear_interpolation",
    "cubic_spline_interpolation",
    "lagrange_interpolation",
    "fft",
    "fftshift",
    "ifft",
    "ifftshift",
    "wasserstein_distance",
    "wasserstein_distance_matrix",
    "plot_wasserstein_heatmap",
    "PrintStatus",
    "STL",
    "STLResult",
    "simple_moving_average",
    "weighted_moving_average",
    "gaussian_smoothing",
    "savgol_smoothing",
    "exponential_smoothing",
    "smooth_show_info",
    "MovingDecomp",
    "plot_time_series",
    "plot_series",
    "plot_symbol",
    "plot_shapiro_wilk",
    "plot_simulator_statistics",
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
]

# Transform the symbol from string to latex
from .print_symbol import symbol_to_markdown

# The z-score standardization
from ._tools import z_score_normalization

# The min-max normalization
from ._tools import max_min_normalization

# Get the datetime now
from ._tools import get_time_now

# The function to save and load the S2 data
from ._tools import save_s2data, load_s2data

# The Fast Fourier Transform
from ._tools import fft, fftshift, ifft, ifftshift

# The EACF function to determine the order of ARMA model
from ._tools import eacf_rlike

# The Yule-Walker method to estimate the parameters of AR model
from ._tools import yule_walker

# The interpolation methods for time series data, including linear interpolation, cubic spline interpolation, and Lagrange interpolation
from ._tools import (
    linear_interpolation,
    cubic_spline_interpolation,
    lagrange_interpolation,
)

# Print the Generation Status
from ._print_status import PrintStatus

# The Wasserstein distance used to measure the similarity between two datasets
from ._wasserstein_distance import wasserstein_distance

# Calculate the distance matrix between multiple time series data sets using the Wasserstein distance formula
from ._wasserstein_distance import wasserstein_distance_matrix

# Visualization the Wasserstein distance though the heatmap
from ._wasserstein_distance import plot_wasserstein_heatmap

# The smooth method for 1D time series data of signal
from ._smooth import (
    simple_moving_average,
    weighted_moving_average,
    gaussian_smoothing,
    savgol_smoothing,
    exponential_smoothing,
    smooth_show_info,
)

# The moving decomposition method for time series
from ._decomposition import MovingDecomp

# The Seasonal-Trend decomposition using LOESS (STL)
from ._decomposition import STL, STLResult

# The Shapiro-Wilk test for normality of the residuals
from .visualization import (
    plot_time_series,
    plot_series,
    plot_symbol,
    plot_shapiro_wilk,
    plot_simulator_statistics,
)

# The generate the time series data through the ARMA model
from ._data import (
    generate_arma_samples,
    generate_nonstationary_sine,
    generate_variable_frequency_sine,
    generate_sine_with_local_frequency_changes,
    generate_triangle_wave,
    generate_square_wave,
    generate_sawtooth_wave,
    generate_damped_oscillation,
    generate_chirp_signal,
    generate_impulse_signal,
    generate_step_signal,
    generate_ramp_signal,
    generate_exponential_signal,
    generate_logarithmic_signal,
    generate_stock_price,
    generate_electrocardiogram,
    generate_electroencephalogram,
)
