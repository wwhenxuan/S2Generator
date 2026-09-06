# -*- coding: utf-8 -*-
"""
Forecastable Component Analysis (ForeCA) utilities.

Spectral entropy, the Omega forecastability score, ZCA whitening, linear
Slow Feature Analysis, and EM-based ForeCA dimension reduction
(Goerg, JMLR 2013). Series are ``(T,)`` or ``(T, K)`` with time on axis 0.
"""

from ._entropy import discrete_entropy, omega, spectral_entropy
from ._foreca import ForeCA
from ._init_weights import initialize_weightvector
from ._sfa import SlowFeatureAnalysis
from ._spectrum import (
    mvspectrum,
    normalize_mvspectrum,
    spectrum_of_linear_combination,
    univariate_spectrum,
)
from ._whiten import WhitenResult, sqrt_matrix, whiten

__all__ = [
    "discrete_entropy",
    "spectral_entropy",
    "omega",
    "mvspectrum",
    "normalize_mvspectrum",
    "spectrum_of_linear_combination",
    "univariate_spectrum",
    "sqrt_matrix",
    "whiten",
    "WhitenResult",
    "SlowFeatureAnalysis",
    "ForeCA",
    "initialize_weightvector",
]
