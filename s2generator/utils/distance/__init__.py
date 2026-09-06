# -*- coding: utf-8 -*-
"""Wasserstein distance between time-series datasets."""

from ._wasserstein_distance import (
    plot_wasserstein_heatmap,
    wasserstein_distance,
    wasserstein_distance_matrix,
)

__all__ = [
    "wasserstein_distance",
    "wasserstein_distance_matrix",
    "plot_wasserstein_heatmap",
]
