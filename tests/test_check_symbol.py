# -*- coding: utf-8 -*-
"""
Unit tests for symbolic-expression validation.

Created on 2026/08/07
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/S2Generator
"""

import unittest

from s2generator.symbol import (
    check_symbol,
    is_valid_symbol,
    explain_symbol_error,
    suggest_fix,
    CustomSymbolGenerator,
    EmptySymbolError,
    UnbalancedParenthesesError,
    InvalidBinaryFormError,
    TrailingTokensError,
    PrefixArityError,
    UnknownOperatorError,
    MissingVariableError,
    EmptySegmentError,
)


class TestCheckSymbol(unittest.TestCase):
    """Tests for check_symbol precise diagnostics."""

    def test_valid_infix(self) -> None:
        """A well-formed infix expression should parse and validate successfully."""
        trees = check_symbol("(x_0 add sin(x_0))")
        self.assertEqual(trees.prefix(), "add,x_0,sin,x_0")
        self.assertTrue(is_valid_symbol("(x_0 add sin(x_0))"))
        self.assertIsNone(explain_symbol_error("(x_0 add sin(x_0))"))

    def test_empty_expression(self) -> None:
        """An empty or whitespace-only expression should raise EmptySymbolError."""
        with self.assertRaises(EmptySymbolError) as ctx:
            check_symbol("   ")
        self.assertIn("empty", str(ctx.exception).lower())
        self.assertIsNotNone(ctx.exception.hint)

    def test_unbalanced_parentheses(self) -> None:
        """Unbalanced parentheses should raise UnbalancedParenthesesError."""
        with self.assertRaises(UnbalancedParenthesesError) as ctx:
            check_symbol("(x_0 add sin(x_0)")
        self.assertEqual(ctx.exception.token, "(")

    def test_ascii_plus_rejected(self) -> None:
        """ASCII '+' is invalid and should suggest the 'add' operator."""
        with self.assertRaises(InvalidBinaryFormError) as ctx:
            check_symbol("(x_0 + 1)")
        self.assertEqual(ctx.exception.token, "+")
        self.assertIn("add", ctx.exception.hint)

    def test_trailing_tokens_without_outer_parens(self) -> None:
        """Trailing tokens without wrapping parentheses should raise TrailingTokensError."""
        with self.assertRaises(TrailingTokensError) as ctx:
            check_symbol("(x_0 mul 2) add sin(x_0)")
        self.assertIn("left op right", ctx.exception.hint)

    def test_prefix_arity_missing_arg(self) -> None:
        """A prefix operator missing an argument should raise PrefixArityError."""
        with self.assertRaises(PrefixArityError) as ctx:
            check_symbol("add,x_0")
        self.assertEqual(ctx.exception.token, "add")

    def test_unknown_operator(self) -> None:
        """An unknown operator should raise UnknownOperatorError and suggest a fix."""
        with self.assertRaises(UnknownOperatorError) as ctx:
            check_symbol("sinn(x_0)")
        self.assertEqual(ctx.exception.token, "sinn")
        self.assertEqual(suggest_fix("sinn"), "sin")

    def test_missing_variable(self) -> None:
        """An expression without variables should fail unless require_variable is False."""
        with self.assertRaises(MissingVariableError):
            check_symbol("sin(1)")
        # Allowed when require_variable=False
        trees = check_symbol("sin(1)", require_variable=False)
        self.assertEqual(trees.prefix(), "sin,1")

    def test_empty_multi_output_segment(self) -> None:
        """An empty multi-output segment should raise EmptySegmentError."""
        with self.assertRaises(EmptySegmentError):
            check_symbol("(x_0 add 1) | ")

    def test_explain_symbol_error(self) -> None:
        """explain_symbol_error should return a diagnostic for an invalid expression."""
        err = explain_symbol_error("(x_0 + 1)")
        self.assertIsInstance(err, InvalidBinaryFormError)
        self.assertFalse(is_valid_symbol("(x_0 + 1)"))

    def test_custom_generator_uses_checker(self) -> None:
        """CustomSymbolGenerator should reject invalid symbols through the checker."""
        with self.assertRaises(InvalidBinaryFormError):
            CustomSymbolGenerator("(x_0 * x_1)")


if __name__ == "__main__":
    unittest.main()
