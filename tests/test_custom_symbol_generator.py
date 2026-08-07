# -*- coding: utf-8 -*-
"""
Unit tests for user-specified symbolic expression generation.

Created on 2026/08/07
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/S2Generator
"""

import unittest

import numpy as np

from s2generator.symbol import (
    SeriesSymbolGenerator,
    CustomSymbolGenerator,
    Generator,
    parse_symbol,
    infer_input_dimension,
    infer_output_dimension,
)
from s2generator.symbol.base import Node, NodeList
from s2generator.symbol.params import SymbolParams


class TestParseSymbol(unittest.TestCase):
    """Tests for parsing user-provided symbolic expressions."""

    def setUp(self) -> None:
        self.params = SymbolParams()
        self.generator = SeriesSymbolGenerator(symbol_params=self.params)

    def test_parse_infix_simple(self) -> None:
        trees = parse_symbol("(x_0 add sin(x_0))", self.params)
        self.assertIsInstance(trees, NodeList)
        self.assertEqual(trees.prefix(), "add,x_0,sin,x_0")
        self.assertEqual(infer_input_dimension(trees), 1)
        self.assertEqual(infer_output_dimension(trees), 1)

    def test_parse_prefix_string(self) -> None:
        trees = parse_symbol("mul,x_0,cos,x_1", self.params)
        self.assertEqual(trees.infix(), "(x_0 mul cos(x_1))")
        self.assertEqual(infer_input_dimension(trees), 2)

    def test_parse_multi_output_infix(self) -> None:
        trees = parse_symbol("(x_0 add 1) | sin(x_1)", self.params)
        self.assertEqual(infer_output_dimension(trees), 2)
        self.assertEqual(infer_input_dimension(trees), 2)

    def test_parse_pow2_suffix(self) -> None:
        trees = parse_symbol("((x_0 add 1))**2", self.params)
        self.assertEqual(trees.prefix(), "pow2,add,x_0,1")

    def test_parse_node_passthrough(self) -> None:
        node = Node("x_0", self.params)
        trees = parse_symbol(node, self.params)
        self.assertEqual(len(trees.nodes), 1)
        self.assertEqual(trees.nodes[0].value, "x_0")

    def test_roundtrip_random_expressions(self) -> None:
        for seed in range(8):
            rng = np.random.RandomState(seed)
            trees, _, _ = self.generator.run(
                rng,
                n_inputs_points=32,
                input_dimension=2,
                output_dimension=2,
            )
            if trees is None:
                continue
            parsed = parse_symbol(trees.infix(), self.params)
            self.assertEqual(parsed.prefix(), trees.prefix())


class TestCustomSymbolGenerator(unittest.TestCase):
    """Tests for CustomSymbolGenerator and SeriesSymbolGenerator.run_from_symbol."""

    def test_generator_alias(self) -> None:
        self.assertIs(Generator, SeriesSymbolGenerator)

    def test_run_from_symbol(self) -> None:
        generator = SeriesSymbolGenerator()
        rng = np.random.RandomState(0)
        symbol, x, y = generator.run_from_symbol(
            rng,
            symbol="(x_0 add sin(x_0))",
            n_inputs_points=64,
        )
        self.assertIsNotNone(symbol)
        self.assertEqual(x.shape, (64, 1))
        self.assertEqual(y.shape, (64, 1))
        self.assertFalse(np.isnan(y).any())

    def test_custom_symbol_generator(self) -> None:
        custom = CustomSymbolGenerator("(x_0 mul cos(x_0)) | (x_0 add 1)")
        self.assertEqual(custom.input_dimension, 1)
        self.assertEqual(custom.output_dimension, 2)

        rng = np.random.RandomState(7)
        symbol, x, y = custom.run(rng, n_inputs_points=48)
        self.assertIsNotNone(symbol)
        self.assertEqual(x.shape, (48, 1))
        self.assertEqual(y.shape, (48, 2))

    def test_custom_symbol_with_prefix(self) -> None:
        custom = CustomSymbolGenerator("add,x_0,mul,2,x_1")
        self.assertEqual(custom.input_dimension, 2)
        rng = np.random.RandomState(3)
        symbol, x, y = custom.run(rng, n_inputs_points=40, input_dimension=2)
        self.assertEqual(x.shape[1], 2)
        self.assertEqual(y.shape[1], 1)


if __name__ == "__main__":
    unittest.main()
