# -*- coding: utf-8 -*-
"""
Parse user-provided symbolic expressions into Node / NodeList trees.

Supports:
- Node / NodeList instances (passed through)
- Prefix token lists or comma-separated prefix strings
- Infix strings in the same format as ``Node.infix()`` / ``NodeList.infix()``
- Multi-output expressions joined by ``|``

Created on 2026/08/07
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/S2Generator
"""

from __future__ import annotations

import re
from typing import List, Optional, Union, Tuple

from s2generator.symbol.base import (
    Node,
    NodeList,
    operators_real,
    operators_extra,
    math_constants,
    all_operators,
)
from s2generator.symbol.params import SymbolParams

# Binary operators that appear as infix words: (left op right)
_BINARY_OPS = {
    name for name, arity in {**operators_real, **operators_extra}.items() if arity == 2
}
# Unary operators that appear as f(arg), excluding pow2/pow3 (rendered as **2/**3)
_UNARY_OPS = {
    name
    for name, arity in operators_real.items()
    if arity == 1 and name not in {"pow2", "pow3"}
}
_LEAF_CONSTANTS = set(math_constants) | {"rand"}

_TOKEN_RE = re.compile(
    r"""
    \s*(
        x_\d+                                  # variables
        | \*\*2 | \*\*3                        # pow2 / pow3 suffixes
        | [A-Za-z_][A-Za-z0-9_]*               # operators / named constants
        | [+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?  # numbers
        | [()]                                 # parentheses
        | \|                                   # multi-output separator
    )
    """,
    re.VERBOSE,
)


def infer_input_dimension(symbol: Union[Node, NodeList]) -> int:
    """Infer the required input dimension from variable indices ``x_i``."""
    prefix = symbol.prefix() if isinstance(symbol, NodeList) else symbol.prefix()
    dims = []
    for token in prefix.split(","):
        if token.startswith("x_"):
            dims.append(int(token.split("_", 1)[1]))
    return (max(dims) + 1) if dims else 1


def infer_output_dimension(symbol: Union[Node, NodeList]) -> int:
    """Infer the output dimension from a parsed symbolic expression."""
    if isinstance(symbol, NodeList):
        return len(symbol.nodes)
    return 1


def parse_symbol(
    symbol: Union[str, Node, NodeList, List[str]],
    symbol_params: Optional[SymbolParams] = None,
) -> NodeList:
    """
    Parse a user-specified symbolic expression into a ``NodeList``.

    :param symbol: Expression as ``Node`` / ``NodeList``, prefix tokens / string,
                   or infix string matching the project's print format.
    :param symbol_params: Optional parameter object attached to created nodes.
    :return: Parsed multivariate symbolic expression.
    """
    params = SymbolParams() if symbol_params is None else symbol_params

    if isinstance(symbol, NodeList):
        return symbol
    if isinstance(symbol, Node):
        return NodeList([symbol])
    if isinstance(symbol, list):
        return _parse_prefix_tokens(symbol, params)
    if not isinstance(symbol, str):
        raise TypeError(
            "symbol must be str, Node, NodeList, or a list of prefix tokens, "
            f"got {type(symbol)!r}"
        )

    text = symbol.strip()
    if not text:
        raise ValueError("Empty symbolic expression.")

    # Prefix form is comma-separated (optionally with ",|," for multi-output).
    if _looks_like_prefix(text):
        tokens = [tok for tok in text.split(",") if tok != ""]
        return _parse_prefix_tokens(tokens, params)

    return _parse_infix(text, params)


def _looks_like_prefix(text: str) -> bool:
    """Heuristic: prefix encodings are comma-separated operator/token streams."""
    if "," not in text:
        return False
    # Infix expressions use spaces around binary ops and do not use commas.
    return " " not in text.replace(",", "")


def _parse_prefix_tokens(tokens: List[str], params: SymbolParams) -> NodeList:
    """Decode a prefix token list into a NodeList (supports ``|`` separators)."""
    if not tokens:
        raise ValueError("Empty prefix token list.")

    # Split multi-output expressions at "|"
    groups: List[List[str]] = [[]]
    for tok in tokens:
        if tok == "|":
            groups.append([])
        else:
            groups[-1].append(tok)

    nodes = []
    for group in groups:
        if not group:
            raise ValueError("Empty expression segment in prefix tokens.")
        node, consumed = _decode_prefix_node(group, params)
        if node is None or consumed != len(group):
            raise ValueError(
                f"Failed to parse prefix tokens: {group!r} "
                f"(consumed={consumed}, length={len(group)})"
            )
        nodes.append(node)
    return NodeList(nodes)


def _decode_prefix_node(
    tokens: List[str], params: SymbolParams
) -> Tuple[Optional[Node], int]:
    """Recursively decode one prefix expression into a Node."""
    if not tokens:
        return None, 0

    head = tokens[0]
    if head in all_operators:
        node = Node(head, params)
        arity = all_operators[head]
        pos = 1
        for _ in range(arity):
            child, length = _decode_prefix_node(tokens[pos:], params)
            if child is None:
                return None, pos
            node.push_child(child)
            pos += length
        return node, pos

    # Leaf: variable, named constant, or numeric literal
    if (
        head.startswith("x_")
        or head in _LEAF_CONSTANTS
        or _is_number(head)
        or head.startswith("CONSTANT")
    ):
        return Node(head, params), 1

    return None, 0


def _is_number(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def _tokenize(text: str) -> List[str]:
    tokens = [m.group(1) for m in _TOKEN_RE.finditer(text)]
    if not tokens:
        raise ValueError(f"Unable to tokenize symbolic expression: {text!r}")
    # Ensure the whole string was consumed (aside from whitespace)
    consumed = "".join(tokens)
    compact_src = re.sub(r"\s+", "", text)
    compact_tok = re.sub(r"\s+", "", consumed)
    if compact_src != compact_tok:
        raise ValueError(f"Unrecognized tokens in symbolic expression: {text!r}")
    return tokens


def _parse_infix(text: str, params: SymbolParams) -> NodeList:
    """Parse one or more infix expressions separated by ``|``."""
    # Split top-level multi-output expressions on "|"
    segments: List[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "|" and depth == 0:
            segments.append(text[start:i].strip())
            start = i + 1
    segments.append(text[start:].strip())

    nodes = []
    for segment in segments:
        if not segment:
            raise ValueError("Empty expression segment in infix string.")
        parser = _InfixParser(segment, params)
        nodes.append(parser.parse())
    return NodeList(nodes)


class _InfixParser:
    """Recursive-descent parser for the project's infix symbolic format."""

    def __init__(self, text: str, params: SymbolParams) -> None:
        self.tokens = _tokenize(text)
        self.pos = 0
        self.params = params

    def parse(self) -> Node:
        node = self._parse_expr()
        if self.pos != len(self.tokens):
            raise ValueError(
                f"Unexpected trailing tokens starting at {self.tokens[self.pos]!r}"
            )
        return node

    def _peek(self) -> Optional[str]:
        if self.pos >= len(self.tokens):
            return None
        return self.tokens[self.pos]

    def _consume(self, expected: Optional[str] = None) -> str:
        tok = self._peek()
        if tok is None:
            raise ValueError("Unexpected end of symbolic expression.")
        if expected is not None and tok != expected:
            raise ValueError(f"Expected {expected!r}, got {tok!r}")
        self.pos += 1
        return tok

    def _parse_expr(self) -> Node:
        node = self._parse_primary()
        # Handle trailing **2 / **3 produced by pow2 / pow3
        while self._peek() in {"**2", "**3"}:
            op = "pow2" if self._consume() == "**2" else "pow3"
            parent = Node(op, self.params)
            parent.push_child(node)
            node = parent
        return node

    def _parse_primary(self) -> Node:
        tok = self._peek()
        if tok is None:
            raise ValueError("Unexpected end of symbolic expression.")

        # Parenthesized binary expression or grouped sub-expression:
        # (left op right)  or  (expr)
        if tok == "(":
            self._consume("(")
            left = self._parse_expr()
            nxt = self._peek()
            if nxt in _BINARY_OPS:
                op = self._consume()
                right = self._parse_expr()
                self._consume(")")
                node = Node(op, self.params)
                node.push_child(left)
                node.push_child(right)
                return node
            # Grouping only: (expr)
            self._consume(")")
            return left

        # Unary operator: sin(arg), log(arg), ...
        if tok in _UNARY_OPS:
            op = self._consume()
            self._consume("(")
            child = self._parse_expr()
            self._consume(")")
            node = Node(op, self.params)
            node.push_child(child)
            return node

        # Leaf nodes
        if tok.startswith("x_") or tok in _LEAF_CONSTANTS or _is_number(tok):
            return Node(self._consume(), self.params)

        raise ValueError(f"Unexpected token in symbolic expression: {tok!r}")
