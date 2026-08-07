# -*- coding: utf-8 -*-
"""
Precise exception types for symbolic-expression parsing and validation.

Created on 2026/08/07
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/S2Generator
"""

from __future__ import annotations

from typing import Optional


class SymbolExpressionError(ValueError):
    """Base class for all symbolic-expression validation / parsing errors."""

    def __init__(
        self,
        message: str,
        *,
        expression: Optional[str] = None,
        position: Optional[int] = None,
        token: Optional[str] = None,
        hint: Optional[str] = None,
    ) -> None:
        self.expression = expression
        self.position = position
        self.token = token
        self.hint = hint
        super().__init__(self._format_message(message))

    def _format_message(self, message: str) -> str:
        parts = [message]
        if self.token is not None:
            parts.append(f"token={self.token!r}")
        if self.position is not None:
            parts.append(f"position={self.position}")
        if self.expression is not None:
            preview = self.expression
            if len(preview) > 120:
                preview = preview[:117] + "..."
            parts.append(f"expression={preview!r}")
        if self.hint:
            parts.append(f"hint: {self.hint}")
        return " | ".join(parts)


class EmptySymbolError(SymbolExpressionError):
    """Raised when the expression is empty or only whitespace."""


class SymbolTypeError(SymbolExpressionError, TypeError):
    """Raised when the expression type is not supported."""


class UnbalancedParenthesesError(SymbolExpressionError):
    """Raised when parentheses are unbalanced."""


class UnrecognizedTokenError(SymbolExpressionError):
    """Raised when the tokenizer cannot consume part of the expression."""


class UnexpectedTokenError(SymbolExpressionError):
    """Raised when a token appears in an illegal position."""


class UnexpectedEndError(SymbolExpressionError):
    """Raised when the expression ends before a complete construct is finished."""


class TrailingTokensError(SymbolExpressionError):
    """Raised when extra tokens remain after a complete expression."""


class EmptySegmentError(SymbolExpressionError):
    """Raised when a multi-output segment separated by ``|`` is empty."""


class UnknownOperatorError(SymbolExpressionError):
    """Raised when an identifier is not a supported operator / leaf."""


class PrefixArityError(SymbolExpressionError):
    """Raised when a prefix operator does not receive the required arguments."""


class MissingVariableError(SymbolExpressionError):
    """Raised when the expression contains no input variable ``x_i``."""


class InvalidBinaryFormError(SymbolExpressionError):
    """Raised for common mistakes around binary-operator infix form."""
