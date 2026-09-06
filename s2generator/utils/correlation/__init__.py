# -*- coding: utf-8 -*-
"""Pairwise correlation / similarity matrices for multivariate time series."""

from ._multivariate_correlation import (
    AVAILABLE_CORRELATION_MEASURES,
    autocorrelation_similarity_matrix,
    distribution_similarity_matrix,
    multivariate_correlation,
    parse_correlation_measures,
    pearson_correlation_matrix,
    power_spectrum_similarity_matrix,
    spearman_correlation_matrix,
    wasserstein_distance_correlation_matrix,
)

__all__ = [
    "AVAILABLE_CORRELATION_MEASURES",
    "parse_correlation_measures",
    "multivariate_correlation",
    "pearson_correlation_matrix",
    "spearman_correlation_matrix",
    "autocorrelation_similarity_matrix",
    "power_spectrum_similarity_matrix",
    "distribution_similarity_matrix",
    "wasserstein_distance_correlation_matrix",
]
