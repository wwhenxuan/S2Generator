# -*- coding: utf-8 -*-
"""Time-series decomposition: STL and moving-average trend extraction."""

from ._moving import MovingDecomp
from ._stl import STL, STLResult

__all__ = [
    "MovingDecomp",
    "STL",
    "STLResult",
]
