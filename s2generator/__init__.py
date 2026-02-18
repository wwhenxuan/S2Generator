# -*- coding: utf-8 -*-

__version__ = "0.0.6"

__all__ = [
    "Node",
    "NodeList",
    "SeriesParams",
    "SymbolParams",
    "Generator",
    "GeneralEncoder",
    "FloatSequences",
    "Equation",
    "plot_series",
    "plot_symbol",
    "print_ascii",
    "print_hello",
    "excitation",
    "simulator",
    "utils",
    "params",
]

# The basic data structure of symbolic expressions
from .base import Node, NodeList

# Parameter control of S2 data generation
from .params import SeriesParams, SymbolParams

# S2 Data Generator
from .generators import Generator

# The encoder for symbol and number
from .encoders import GeneralEncoder, FloatSequences, Equation

# Generic interface for generating stimulus time series data
from .excitation import Excitation

# # The simulators for generating time series data
# from .simulator import ARIMASimulator, WienerFilterSimulator

# Visualize the generated S2 object
from .utils.visualization import plot_series

# Visualize the symbol expression
from .utils.visualization import plot_symbol


def print_ascii() -> None:
    print(
        """
   _____   ___     _____                                        _                  
  / ____| |__ \   / ____|                                      | |                 
 | (___      ) | | |  __    ___   _ __     ___   _ __    __ _  | |_    ___    _ __ 
  \___ \    / /  | | |_ |  / _ \ | '_ \   / _ \ | '__|  / _` | | __|  / _ \  | '__|
  ____) |  / /_  | |__| | |  __/ | | | | |  __/ | |    | (_| | | |_  | (_) | | |   
 |_____/  |____|  \_____|  \___| |_| |_|  \___| |_|     \__,_|  \__|  \___/  |_|                                   
"""
    )


def print_hello() -> None:
    print("Hello, S2Generator!")
    print("=" * 30)
    print("Version:", __version__)
    print(
        "This is a Python package for generating time series data with symbolic representations."
    )
    print(
        "For more information, please visit: https://github.com/wwhenxuan/S2Generator"
    )
    print_ascii()
