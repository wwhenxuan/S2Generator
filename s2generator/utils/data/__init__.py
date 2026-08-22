# -*- coding: utf-8 -*-
"""Bundled real time-series slices and loaders."""

from .loader import (
    AVAILABLE_MULTIVARIATE_DATASETS,
    AVAILABLE_UNIVARIATE_DATASETS,
    list_datasets,
    load_multivariate,
    load_univariate,
)

__all__ = [
    "AVAILABLE_MULTIVARIATE_DATASETS",
    "AVAILABLE_UNIVARIATE_DATASETS",
    "list_datasets",
    "load_multivariate",
    "load_univariate",
]
