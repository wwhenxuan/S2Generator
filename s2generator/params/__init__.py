# -*- coding: utf-8 -*-
"""
This module is used for parameter control and management,
including parameter control for generating stimulus time series and
generating symbolic expressions (parameter control for complex systems).

Created on 2025/08/19 11:03:53"
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/S2Generator
"""
__all__ = ["SeriesParams", "SymbolParams"]

# Parameters used to control the generation of stimulus time series data
from .series_params import SeriesParams

# Parameters used to generate complex systems (symbolic expressions)
from .symbol_params import SymbolParams
