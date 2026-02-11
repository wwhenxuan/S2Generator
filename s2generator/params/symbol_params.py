# -*- coding: utf-8 -*-
"""
This module is mainly used to store parameters that control the generation of symbolic expressions.

Created on 2025/08/19 11:07:39
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/S2Generator
"""
import numpy as np

from typing import Optional


def check_inputs_probability(probability: float) -> float:
    """
    Check if probability is between 0 and 1.

    :param probability: The user inputs probability.
    :return: The legal probability.
    """
    if probability < 0 or probability > 1:
        raise ValueError("Probability must be between 0 and 1.")
    return probability


class SymbolParams(object):
    """Parameter Control The Generation of Complex Systems (Symbolic Expression) in S2 (Series-Symbol) Data Generation."""

    def __init__(
        self,
        max_trials: Optional[int] = 64,
        prob_rand: Optional[
            float
        ] = 0.25,  # TODO: It is necessary to add a measure to the symbolic expression so that it always has at least one leaf node no matter how it is expressed.
        prob_const: Optional[float] = 0.25,
        min_binary_ops_per_dim: Optional[int] = 0,
        max_binary_ops_per_dim: Optional[int] = 1,
        max_binary_ops_offset: Optional[int] = 4,
        min_unary_ops: Optional[int] = 0,
        max_unary_ops: Optional[int] = 5,
        float_precision: Optional[int] = 3,
        mantissa_len: Optional[int] = 1,
        max_int: Optional[int] = 10,
        max_exponent: Optional[int] = 3,
        max_exponent_prefactor: Optional[int] = 1,
        use_abs: Optional[bool] = True,
        operators_to_downsample: Optional[str] = None,
        max_unary_depth: Optional[int] = 6,
        required_operators: Optional[str] = "",
        extra_unary_operators: Optional[str] = "",
        extra_binary_operators: Optional[str] = "",
        extra_constants: Optional[str] = "",
        use_sympy: Optional[bool] = False,
        reduce_num_constants: Optional[bool] = True,
        solve_diff: Optional[int] = 0,
        decimals: Optional[int] = 6,
        min_input_dimension: Optional[int] = 1,
        max_input_dimension: Optional[int] = 6,
        min_output_dimension: Optional[int] = 1,
        max_output_dimension: Optional[int] = 12,
    ) -> None:
        """
        :param max_trials: Maximum number of trials to generate.
        :param prob_rand: Probability to generate n in leafs.
        :param prob_const: Probability to generate integer in leafs.
        :param min_binary_ops_per_dim: Min number of binary operators per input dimension.
        :param max_binary_ops_per_dim: Max number of binary operators per input dimension.
        :param max_binary_ops_offset: Offset for max number of binary operators.
        :param min_unary_ops: Min number of unary operators.
        :param max_unary_ops: Max number of unary operators.
        :param float_precision: Number of digits in the mantissa.
        :param mantissa_len: Number of tokens for the mantissa (must be a divisor or float_precision+1).
        :param max_int: Maximal integer in symbolic expressions.
        :param max_exponent: Maximal order of magnitude.
        :param max_exponent_prefactor: Maximal order of magnitude in prefactors.
        :param use_abs: Whether to replace log and sqrt by log(abs) and sqrt(abs).
        :param operators_to_downsample: Which operator to remove.
        :param max_unary_depth: Max number of operators inside unary.
        :param required_operators: Which operator to remove.
        :param extra_unary_operators: Extra unary operator to add to data generation.
        :param extra_binary_operators: Extra binary operator to add to data generation.
        :param extra_constants: Additional int constants floats instead of ints.
        :param use_sympy: Whether to use sympy parsing (basic simplification).
        :param reduce_num_constants: Use minimal amount of constants in eqs.
        :param solve_diff: Order of differential equation solving (0: no diff, 1: first order, etc.).
        :param decimals: Number of digits reserved for floating-point numbers in symbolic expressions.
        :param min_input_dimension: Minimum input dimension (minimum number of variables) for symbolic expressions
        :param max_input_dimension: Maximum input dimension of symbolic expressions (maximum number of variables)
        :param min_output_dimension: Minimum output dimension of multivariate symbolic expressions
        :param max_output_dimension: Maximum output dimension of multivariate symbolic expressions
        """
        # Handle overflow outside the domain of the generation attempt
        self.max_trials = max_trials

        # Parameters about leaf node generation constants and random number probabilities
        # The control of these two parameters is reflected in the `generate_leaf` method
        self.prob_const, self.prob_rand = (
            check_inputs_probability(probability=prob_const),
            check_inputs_probability(probability=prob_rand),
        )
        self.fix_inputs_prob_rand()

        # Maximum and minimum numbers of unary and binary operators
        self.min_binary_ops_per_dim, self.max_binary_ops_per_dim = (
            min_binary_ops_per_dim,
            max_binary_ops_per_dim,
        )
        self.max_binary_ops_offset = max_binary_ops_offset
        self.min_unary_ops, self.max_unary_ops = min_unary_ops, max_unary_ops

        # Numerical precision of floating-point numbers
        self.float_precision = float_precision
        self.mantissa_len = mantissa_len

        # Parameters that limit the value range
        self.max_int = max_int
        self.max_exponent = max_exponent
        self.max_exponent_prefactor = max_exponent_prefactor

        # For functions with nonzero domains, use
        self.use_abs = use_abs

        # Which operator to remove.
        self._operators_to_downsample = (
            "div_0,arcsin_0,arccos_0,tan_0.2,arctan_0.2,sqrt_5,pow2_3,inv_3"
            if operators_to_downsample is None
            else operators_to_downsample
        )

        # Maximum depth of symbolic expressions
        self.max_unary_depth = max_unary_depth
        self.required_operators = required_operators

        # Handling differential operations in generated data
        self.solve_diff = solve_diff

        # Force adjustment of floating point precision
        self.decimals = decimals
        self.extra_unary_operators, self.extra_binary_operators = (
            extra_unary_operators,
            extra_binary_operators,
        )
        self.extra_constants = extra_constants
        self.use_sympy = use_sympy

        # This traversal controls whether to perform radiation transformations
        # to further increase the diversity of symbolic expressions
        self.reduce_num_constants = reduce_num_constants

        self.min_input_dimension, self.max_input_dimension = (
            min_input_dimension,
            max_input_dimension,
        )
        self.min_output_dimension, self.max_output_dimension = (
            min_output_dimension,
            max_output_dimension,
        )

    @property
    def operators_to_downsample(self) -> str:
        """Get the string of operators to be removed."""
        return self._operators_to_downsample

    def fix_inputs_prob_rand(self) -> None:
        """Control the probability of random number generation for input not to be too large"""
        self.prob_rand = min(self.prob_rand, 0.5)
