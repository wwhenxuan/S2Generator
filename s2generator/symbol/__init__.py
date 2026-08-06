# -*- coding: utf-8 -*-
"""
Symbolic expression (complex system) generation for S2 (Series-Symbol) data.

This subpackage contains the original Series-Symbol core:
expression trees, encoders, the S2 Generator, and parameters.
"""

__all__ = [
    "Node",
    "NodeList",
    "operators_real",
    "operators_extra",
    "math_constants",
    "all_operators",
    "SPECIAL_WORDS",
    "SeriesParams",
    "SymbolParams",
    "check_inputs_probability",
    "GeneralEncoder",
    "FloatSequences",
    "Equation",
    "Generator",
    "symbol_to_markdown",
    "params",
]

from .base import (
    Node,
    NodeList,
    operators_real,
    operators_extra,
    math_constants,
    all_operators,
    SPECIAL_WORDS,
)
from .params import SeriesParams, SymbolParams, check_inputs_probability
from .encoders import GeneralEncoder, FloatSequences, Equation
from .generators import Generator
from .print_symbol import symbol_to_markdown
from . import params
