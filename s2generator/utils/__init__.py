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
]

# # Visualization the time series data in S2
# from .visualization import plot_series
#
# # Visualization the Symbol data in S2
# from .visualization import plot_symbol

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
