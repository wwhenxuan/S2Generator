# -*- coding: utf-8 -*-

__version__ = "0.0.18"

__all__ = [
    "augmentation",
    "excitation",
    "simulator",
    "symbol",
    "utils",
    "print_ascii",
    "print_hello",
]

from . import symbol, excitation, simulator, augmentation, utils


def print_ascii() -> None:
    print(
        r"""
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
