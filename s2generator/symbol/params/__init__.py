# -*- coding: utf-8 -*-
"""
Parameter control for S2 (Series-Symbol) data generation,
including stimulus time series and symbolic expression settings.
"""

__all__ = ["SeriesParams", "SymbolParams", "check_inputs_probability"]

from .series_params import SeriesParams
from .symbol_params import SymbolParams, check_inputs_probability
