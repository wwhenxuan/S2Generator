# -*- coding: utf-8 -*-
"""
Time-series data for S2Generator: bundled real slices and parametric generators.

Use :func:`load_univariate` / :func:`load_multivariate` for the packaged
ETT / FX / weather / electricity excerpts, and :func:`generate` (or the
individual ``generate_*`` helpers) for regular synthetic waveforms.

Created on 2026/08/22
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/S2Generator
"""

from typing import List

from ._print_status import PrintStatus
from .loader import (
    AVAILABLE_DEEPMIMO_DATASETS,
    AVAILABLE_MULTIVARIATE_DATASETS,
    AVAILABLE_UNIVARIATE_DATASETS,
    list_deepmimo_speeds,
    load_deepmimo_iq,
    load_multivariate,
    load_univariate,
)
from .loader import list_datasets as _list_real_datasets
from .synthetic import (
    AVAILABLE_SYNTHETIC_GENERATORS,
    generate,
    generate_arma_samples,
    generate_chirp_signal,
    generate_damped_oscillation,
    generate_electrocardiogram,
    generate_electroencephalogram,
    generate_exponential_signal,
    generate_impulse_signal,
    generate_logarithmic_signal,
    generate_nonstationary_sine,
    generate_ramp_signal,
    generate_sawtooth_wave,
    generate_sine_with_local_frequency_changes,
    generate_square_wave,
    generate_step_signal,
    generate_stock_price,
    generate_triangle_wave,
    generate_variable_frequency_sine,
)

__all__ = [
    "PrintStatus",
    "AVAILABLE_DEEPMIMO_DATASETS",
    "AVAILABLE_MULTIVARIATE_DATASETS",
    "AVAILABLE_UNIVARIATE_DATASETS",
    "AVAILABLE_SYNTHETIC_GENERATORS",
    "list_datasets",
    "list_deepmimo_speeds",
    "load_deepmimo_iq",
    "load_multivariate",
    "load_univariate",
    "generate",
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


def list_datasets(kind: str = "all") -> List[str]:
    """List available series sources.

    :param kind:
        - ``\"univariate\"`` / ``\"all\"``: bundled real univariate slices
          (ETT, exchange rate, weather, electricity).
        - ``\"multivariate\"``: bundled real multivariate CSVs.
        - ``\"real\"``: alias of ``\"univariate\"``.
        - ``\"synthetic\"``: parametric generator names
          (see :data:`AVAILABLE_SYNTHETIC_GENERATORS`).
        - ``\"deepmimo\"``: packaged DeepMIMO CSI subsets
          (see :data:`AVAILABLE_DEEPMIMO_DATASETS`).
    :return: Canonical names in a stable order.
    """
    if kind == "synthetic":
        return list(AVAILABLE_SYNTHETIC_GENERATORS)
    if kind == "deepmimo":
        return list(AVAILABLE_DEEPMIMO_DATASETS)
    if kind == "real":
        kind = "univariate"
    return _list_real_datasets(kind)
