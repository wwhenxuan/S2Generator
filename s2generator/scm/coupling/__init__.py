# -*- coding: utf-8 -*-
"""
TiRex-2 synthetic multivariate coupling subpackage.

This subpackage implements the dataset construction methods from:

    Podest, P., et al. (2026). TiRex-2: Generalizing TiRex to Multivariate
    Data and Streaming. arXiv:2607.01204v1.

It provides individual coupling mechanisms and a unified CouplingPipeline
for end-to-end dataset construction.

Created on 2026/08/10 00:00:00
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/S2Generator
"""

__all__ = [
    # Base class
    "BaseCoupling",
    # Coupling mechanisms
    "IdentityCoupling",
    "UnivariatePassThrough",
    "FunctionalCoupling",
    "LinearMixing",
    "Cointegration",
    "LinearSCM",
    "NonlinearSCM",
    # Post-processing
    "PostProcessor",
    "variate_permutation",
    "smooth_time_warping",
    "patch_masking",
    "partial_future_observability",
    "value_discretization",
    "time_discretization",
    # Unified interface
    "CouplingPipeline",
]

# --- Abstract base class ---
from .base_coupling import BaseCoupling

# --- Coupling mechanisms ---
from .identity import IdentityCoupling, UnivariatePassThrough
from .functional import FunctionalCoupling
from .linear_mixing import LinearMixing
from .cointegration import Cointegration
from .linear_scm import LinearSCM
from .nonlinear_scm import NonlinearSCM

# --- Post-processing ---
from .postprocessing import (
    PostProcessor,
    variate_permutation,
    smooth_time_warping,
    patch_masking,
    partial_future_observability,
    value_discretization,
    time_discretization,
)

# --- Unified pipeline interface ---
from ._interface import CouplingPipeline
