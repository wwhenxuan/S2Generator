# -*- coding: utf-8 -*-
"""
Validate user-provided symbolic expressions with precise, actionable errors.

This module checks infix / prefix expressions before parsing and raises specific
exception subclasses so users can locate and fix mistakes quickly.

Created on 2026/08/07
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/S2Generator
"""

from __future__ import annotations

import re
from difflib import get_close_matches
from typing import List, Optional, Sequence, Union, Tuple

from s2generator.symbol.base import Node, NodeList, all_operators
from s2generator.symbol.params import SymbolParams
from s2generator.symbol.symbol_errors import (
    SymbolExpressionError,
    EmptySymbolError,
    SymbolTypeError,
    UnbalancedParenthesesError,
    UnrecognizedTokenError,
    UnexpectedTokenError,
    UnexpectedEndError,
    TrailingTokensError,
    EmptySegmentError,
    UnknownOperatorError,
    PrefixArityError,
    MissingVariableError,
    InvalidBinaryFormError,
)
from s2generator.symbol.parse_symbol import (
    parse_symbol,
    _BINARY_OPS,
    _UNARY_OPS,
    _LEAF_CONSTANTS,
    _TOKEN_RE,
    _looks_like_prefix,
)

__all__ = [
    "SymbolExpressionError",
    "EmptySymbolError",
    "SymbolTypeError",
    "UnbalancedParenthesesError",
    "UnrecognizedTokenError",
    "UnexpectedTokenError",
    "UnexpectedEndError",
    "TrailingTokensError",
    "EmptySegmentError",
    "UnknownOperatorError",
    "PrefixArityError",
    "MissingVariableError",
    "InvalidBinaryFormError",
    "check_symbol",
    "is_valid_symbol",
    "explain_symbol_error",
    "suggest_fix",
    "allowed_operators",
]

_ASCII_OP_HINTS = {
    "+": "use the word operator 'add' inside parentheses, e.g. '(x_0 add 1)'",
    "-": "use the word operator 'sub' inside parentheses, e.g. '(x_0 sub 1)'",
    "*": "use the word operator 'mul' inside parentheses, e.g. '(x_0 mul 2)'",
    "/": "use the word operator 'div' inside parentheses, e.g. '(x_0 div 2)'",
    "^": "use 'pow' as a binary operator, e.g. '(x_0 pow 2)', or '**2' / '**3' suffixes",
    "**": "only '**2' and '**3' are supported as postfix powers (pow2 / pow3)",
}


def check_symbol(
    symbol: Union[str, Node, NodeList, List[str]],
    symbol_params: Optional[SymbolParams] = None,
    *,
    require_variable: bool = True,
) -> NodeList:
    """
    Validate and parse a user-specified symbolic expression.

    Performs structural checks first, then parses into a ``NodeList``. On failure,
    raises a specific ``SymbolExpressionError`` subclass with a concrete hint.

    :param symbol: Expression as string / Node / NodeList / prefix token list.
    :param symbol_params: Optional parameter object attached to created nodes.
    :param require_variable: If True, require at least one ``x_i`` leaf.
    :return: Parsed multivariate symbolic expression.
    :raises SymbolExpressionError: When the expression is invalid.
    """
    params = SymbolParams() if symbol_params is None else symbol_params

    if isinstance(symbol, (Node, NodeList)):
        trees = parse_symbol(symbol, params)
        _check_semantics(
            trees, expression=str(symbol), require_variable=require_variable
        )
        return trees

    if isinstance(symbol, list):
        joined = ",".join(map(str, symbol))
        _check_prefix_tokens(symbol, expression=joined)
        trees = parse_symbol(symbol, params)
        _check_semantics(trees, expression=joined, require_variable=require_variable)
        return trees

    if not isinstance(symbol, str):
        raise SymbolTypeError(
            "Unsupported symbol type.",
            expression=repr(symbol),
            hint=(
                "pass a str (infix/prefix), Node, NodeList, or list of prefix tokens; "
                f"got {type(symbol).__name__}"
            ),
        )

    expression = symbol
    text = symbol.strip()
    if not text:
        raise EmptySymbolError(
            "Symbolic expression is empty.",
            expression=expression,
            hint="provide an infix string like '(x_0 add sin(x_0))' "
            "or a prefix string like 'add,x_0,sin,x_0'",
        )

    if _looks_like_prefix(text):
        tokens = [tok for tok in text.split(",") if tok != ""]
        _check_prefix_tokens(tokens, expression=expression)
    else:
        _check_infix_structure(text, expression=expression)

    trees = parse_symbol(text, params)
    _check_semantics(trees, expression=expression, require_variable=require_variable)
    return trees


def is_valid_symbol(
    symbol: Union[str, Node, NodeList, List[str]],
    symbol_params: Optional[SymbolParams] = None,
    *,
    require_variable: bool = True,
) -> bool:
    """Return whether ``symbol`` is valid without raising."""
    try:
        check_symbol(
            symbol, symbol_params=symbol_params, require_variable=require_variable
        )
        return True
    except SymbolExpressionError:
        return False


def explain_symbol_error(
    symbol: Union[str, Node, NodeList, List[str]],
    symbol_params: Optional[SymbolParams] = None,
    *,
    require_variable: bool = True,
) -> Optional[SymbolExpressionError]:
    """
    Validate ``symbol`` and return the error instance, or ``None`` if valid.

    Useful for UIs / notebooks that want diagnostics without try/except.
    """
    try:
        check_symbol(
            symbol, symbol_params=symbol_params, require_variable=require_variable
        )
        return None
    except SymbolExpressionError as err:
        return err


def suggest_fix(token: str) -> Optional[str]:
    """Suggest a close operator / leaf name for ``token``."""
    return _suggest_token(token)


def allowed_operators() -> Tuple[List[str], List[str]]:
    """Return ``(binary_operators, unary_operators)`` sorted lists."""
    return sorted(_BINARY_OPS), sorted(_UNARY_OPS)


def _check_semantics(
    trees: NodeList,
    *,
    expression: str,
    require_variable: bool,
) -> None:
    if not require_variable:
        return
    if not any(tok.startswith("x_") for tok in trees.prefix().split(",")):
        raise MissingVariableError(
            "Expression contains no input variable such as 'x_0'.",
            expression=expression,
            hint="include at least one leaf like 'x_0', 'x_1', ... so the system "
            "can be driven by an excitation series",
        )


def _check_infix_structure(text: str, *, expression: str) -> None:
    _check_ascii_operators(text, expression=expression)
    _check_balanced_parentheses(text, expression=expression)
    _check_empty_segments(text, expression=expression, kind="infix")
    _check_unrecognized_characters(text, expression=expression)


def _check_ascii_operators(text: str, *, expression: str) -> None:
    for match in re.finditer(r"\*\*(?!2|3)|[+\*/^]|-(?!\d)", text):
        op = match.group(0)
        if op == "-" and _is_likely_negative_number(text, match.start()):
            continue
        hint = _ASCII_OP_HINTS.get(
            op if not op.startswith("**") else "**",
            "binary operators must be words inside parentheses, "
            "e.g. '(x_0 add x_1)', '(x_0 mul 2)'",
        )
        raise InvalidBinaryFormError(
            f"Unsupported operator syntax {op!r}.",
            expression=expression,
            position=match.start(),
            token=op,
            hint=hint,
        )


def _is_likely_negative_number(text: str, index: int) -> bool:
    j = index + 1
    if j >= len(text):
        return False
    if text[j].isdigit() or text[j] == ".":
        if index == 0:
            return True
        return text[index - 1] in "(|, \t\n"
    return False


def _check_balanced_parentheses(text: str, *, expression: str) -> None:
    depth = 0
    for i, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                raise UnbalancedParenthesesError(
                    "Unmatched closing parenthesis ')'.",
                    expression=expression,
                    position=i,
                    token=")",
                    hint="remove the extra ')' or add a matching '(' earlier",
                )
    if depth > 0:
        raise UnbalancedParenthesesError(
            f"Unclosed parenthesis: {depth} '(' left open.",
            expression=expression,
            position=len(text) - 1,
            token="(",
            hint="add the missing ')' at the end of the corresponding sub-expression",
        )


def _check_empty_segments(text: str, *, expression: str, kind: str) -> None:
    depth = 0
    start = 0
    segments = []
    for i, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "|" and depth == 0:
            segments.append(text[start:i].strip())
            start = i + 1
    segments.append(text[start:].strip())

    for idx, segment in enumerate(segments):
        if segment == "":
            raise EmptySegmentError(
                f"Empty {kind} segment at output index {idx}.",
                expression=expression,
                hint="multi-output expressions use '|' between non-empty parts, "
                "e.g. '(x_0 add 1) | sin(x_1)'",
            )


def _check_unrecognized_characters(text: str, *, expression: str) -> None:
    pos = 0
    n = len(text)
    while pos < n:
        if text[pos].isspace():
            pos += 1
            continue
        match = _TOKEN_RE.match(text, pos)
        if match is None:
            bad = text[pos]
            raise UnrecognizedTokenError(
                f"Unrecognized character {bad!r} in symbolic expression.",
                expression=expression,
                position=pos,
                token=bad,
                hint=_hint_for_bad_character(bad, text, pos),
            )
        pos = match.end()


def _hint_for_bad_character(bad: str, text: str, pos: int) -> str:
    if bad in _ASCII_OP_HINTS:
        return _ASCII_OP_HINTS[bad]
    nearby = text[max(0, pos - 8) : pos + 8]
    return (
        "allowed tokens are variables (x_0), numbers, parentheses, '|', "
        f"'**2'/'**3', and operators {sorted(_BINARY_OPS | _UNARY_OPS)}; "
        f"context near error: {nearby!r}"
    )


def _check_prefix_tokens(tokens: Sequence[str], *, expression: str) -> None:
    if not tokens:
        raise EmptySymbolError(
            "Prefix token list is empty.",
            expression=expression,
            hint="example prefix: 'add,x_0,sin,x_0'",
        )

    groups: List[List[str]] = [[]]
    for tok in tokens:
        if tok == "|":
            groups.append([])
        else:
            groups[-1].append(str(tok))

    for g_idx, group in enumerate(groups):
        if not group:
            raise EmptySegmentError(
                f"Empty prefix segment at output index {g_idx}.",
                expression=expression,
                hint="do not place '|' at the ends or consecutively, "
                "e.g. 'add,x_0,1,|,sin,x_1'",
            )
        consumed = _validate_prefix_arity(group, expression=expression, start_index=0)
        if consumed != len(group):
            leftover = group[consumed:]
            raise TrailingTokensError(
                "Extra tokens remain after a complete prefix expression.",
                expression=expression,
                token=leftover[0] if leftover else None,
                hint=(
                    f"segment {g_idx} consumed {consumed}/{len(group)} tokens; "
                    f"leftover={leftover!r}. Check operator arity "
                    f"(binary ops need 2 arguments, unary ops need 1)."
                ),
            )


def _validate_prefix_arity(
    tokens: Sequence[str], *, expression: str, start_index: int
) -> int:
    if start_index >= len(tokens):
        raise UnexpectedEndError(
            "Prefix expression ended before a complete node was formed.",
            expression=expression,
            hint="a binary operator such as 'add' needs two arguments; "
            "a unary operator such as 'sin' needs one argument",
        )

    head = tokens[start_index]
    if head in all_operators:
        arity = all_operators[head]
        pos = start_index + 1
        for arg_i in range(arity):
            if pos >= len(tokens):
                raise PrefixArityError(
                    f"Operator {head!r} expects {arity} argument(s), "
                    f"but argument #{arg_i + 1} is missing.",
                    expression=expression,
                    token=head,
                    hint=_operator_arity_hint(head, arity),
                )
            child_consumed = _validate_prefix_arity(
                tokens, expression=expression, start_index=pos
            )
            pos += child_consumed
        return pos - start_index

    if (
        str(head).startswith("x_")
        or head in _LEAF_CONSTANTS
        or _is_number(head)
        or str(head).startswith("CONSTANT")
    ):
        return 1

    suggestion = _suggest_token(str(head))
    hint = (f"did you mean {suggestion!r}? " if suggestion else "") + (
        "allowed leaves: x_i / numbers / rand / e / pi; "
        f"operators: {sorted(all_operators)}"
    )
    raise UnknownOperatorError(
        f"Unknown prefix token {head!r}.",
        expression=expression,
        token=str(head),
        hint=hint,
    )


def _operator_arity_hint(op: str, arity: int) -> str:
    if arity == 1:
        return f"unary form examples: '{op},x_0' or infix '{op}(x_0)'"
    if arity == 2:
        return f"binary form examples: '{op},x_0,x_1' or infix '(x_0 {op} x_1)'"
    return f"operator {op!r} requires exactly {arity} argument(s)"


def _suggest_token(token: str) -> Optional[str]:
    candidates = sorted(
        set(all_operators)
        | _LEAF_CONSTANTS
        | {"x_0", "x_1", "x_2"}
        | set(_BINARY_OPS)
        | set(_UNARY_OPS)
    )
    matches = get_close_matches(token, candidates, n=1, cutoff=0.6)
    return matches[0] if matches else None


def _is_number(token: str) -> bool:
    try:
        float(token)
        return True
    except (TypeError, ValueError):
        return False
