# -*- coding: utf-8 -*-
"""
Symbolic expression (complex system) generation for S2 (Series-Symbol) data.

This subpackage contains the original Series-Symbol core:
expression trees, encoders, SeriesSymbolGenerator / CustomSymbolGenerator,
and parameters.
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
    "SeriesSymbolGenerator",
    "CustomSymbolGenerator",
    "Generator",
    "parse_symbol",
    "infer_input_dimension",
    "infer_output_dimension",
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
from .generators import (
    SeriesSymbolGenerator,
    CustomSymbolGenerator,
    Generator,
)
from .parse_symbol import (
    parse_symbol,
    infer_input_dimension,
    infer_output_dimension,
)
from .print_symbol import symbol_to_markdown
from . import params
