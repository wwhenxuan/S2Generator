# -*- coding: utf-8 -*-
"""
Created on 2025/08/22 21:45:16
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/S2Generator
"""

import unittest
import numpy as np

from s2generator.symbol import SymbolParams
from s2generator.symbol.params import check_inputs_probability


class TestSymbolParams(unittest.TestCase):
    """Test parameter object used to generate the symbolic expressions"""

    def test_create(self) -> None:
        """Create a SymbolParams instance and verify the default object type."""
        pass

    def test_operators_to_downsample(self) -> None:
        """Verify the default operators_to_downsample string used during generation."""
        pass

    def test_fix_inputs_prob_rand(self) -> None:
        """fix_inputs_prob_rand should cap the random-leaf probability at 0.5."""
        pass

    def test_check_inputs_probability(self) -> None:
        """check_inputs_probability should accept [0, 1] and reject values outside it."""
        pass
