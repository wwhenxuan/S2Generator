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
from s2generator.symbol.symbol_errors import (
    EmptySymbolError,
    SymbolTypeError,
    UnrecognizedTokenError,
    UnexpectedTokenError,
    UnexpectedEndError,
    TrailingTokensError,
    EmptySegmentError,
    UnknownOperatorError,
    PrefixArityError,
)

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
        return _parse_prefix_tokens(
            symbol, params, expression=",".join(map(str, symbol))
        )
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

    # Prefix form is comma-separated (optionally with ",|," for multi-output).
    if _looks_like_prefix(text):
        tokens = [tok for tok in text.split(",") if tok != ""]
        return _parse_prefix_tokens(tokens, params, expression=expression)

    return _parse_infix(text, params, expression=expression)


def _looks_like_prefix(text: str) -> bool:
    """Heuristic: prefix encodings are comma-separated operator/token streams."""
    if "," not in text:
        return False
    # Infix expressions use spaces around binary ops and do not use commas.
    return " " not in text.replace(",", "")


def _parse_prefix_tokens(
    tokens: List[str],
    params: SymbolParams,
    *,
    expression: Optional[str] = None,
) -> NodeList:
    """Decode a prefix token list into a NodeList (supports ``|`` separators)."""
    expression = expression if expression is not None else ",".join(map(str, tokens))
    if not tokens:
        raise EmptySymbolError(
            "Prefix token list is empty.",
            expression=expression,
            hint="example prefix: 'add,x_0,sin,x_0'",
        )

    # Split multi-output expressions at "|"
    groups: List[List[str]] = [[]]
    for tok in tokens:
        if tok == "|":
            groups.append([])
        else:
            groups[-1].append(tok)

    nodes = []
    for g_idx, group in enumerate(groups):
        if not group:
            raise EmptySegmentError(
                f"Empty prefix segment at output index {g_idx}.",
                expression=expression,
                hint="do not place '|' at the ends or consecutively, "
                "e.g. 'add,x_0,1,|,sin,x_1'",
            )
        node, consumed = _decode_prefix_node(group, params, expression=expression)
        if node is None or consumed != len(group):
            bad = group[consumed] if consumed < len(group) else group[0]
            if (
                bad not in all_operators
                and not str(bad).startswith("x_")
                and not _is_number(str(bad))
            ):
                raise UnknownOperatorError(
                    f"Unknown prefix token {bad!r}.",
                    expression=expression,
                    token=str(bad),
                    hint=(
                        "allowed leaves: x_i / numbers / rand / e / pi; "
                        f"operators: {sorted(all_operators)}"
                    ),
                )
            raise PrefixArityError(
                f"Failed to parse prefix segment {g_idx}: {group!r} "
                f"(consumed={consumed}, length={len(group)}).",
                expression=expression,
                token=str(bad),
                hint="check operator arity: binary ops need 2 args, unary ops need 1",
            )
        nodes.append(node)
    return NodeList(nodes)


def _decode_prefix_node(
    tokens: List[str],
    params: SymbolParams,
    *,
    expression: Optional[str] = None,
) -> Tuple[Optional[Node], int]:
    """Recursively decode one prefix expression into a Node."""
    if not tokens:
        return None, 0

    head = tokens[0]
    if head in all_operators:
        node = Node(head, params)
        arity = all_operators[head]
        pos = 1
        for arg_i in range(arity):
            if pos >= len(tokens):
                raise PrefixArityError(
                    f"Operator {head!r} expects {arity} argument(s), "
                    f"but argument #{arg_i + 1} is missing.",
                    expression=expression,
                    token=str(head),
                    hint=(
                        f"binary form: '{head},x_0,x_1'; unary form: '{head},x_0'"
                        if arity == 2
                        else f"unary form: '{head},x_0' or infix '{head}(x_0)'"
                    ),
                )
            child, length = _decode_prefix_node(
                tokens[pos:], params, expression=expression
            )
            if child is None:
                bad = tokens[pos]
                raise UnknownOperatorError(
                    f"Cannot parse argument of operator {head!r}: {bad!r}.",
                    expression=expression,
                    token=str(bad),
                    hint=f"operators: {sorted(all_operators)}; leaves: x_i / numbers / rand",
                )
            node.push_child(child)
            pos += length
        return node, pos

    # Leaf: variable, named constant, or numeric literal
    if (
        str(head).startswith("x_")
        or head in _LEAF_CONSTANTS
        or _is_number(head)
        or str(head).startswith("CONSTANT")
    ):
        return Node(head, params), 1

    return None, 0


def _is_number(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def _tokenize(text: str, *, expression: Optional[str] = None) -> List[str]:
    expression = expression if expression is not None else text
    tokens = [m.group(1) for m in _TOKEN_RE.finditer(text)]
    if not tokens:
        raise UnrecognizedTokenError(
            "Unable to tokenize symbolic expression.",
            expression=expression,
            hint="example: '(x_0 add sin(x_0))'",
        )
    # Ensure the whole string was consumed (aside from whitespace)
    consumed = "".join(tokens)
    compact_src = re.sub(r"\s+", "", text)
    compact_tok = re.sub(r"\s+", "", consumed)
    if compact_src != compact_tok:
        # Locate first mismatch for a precise position
        pos = 0
        for ch in text:
            if ch.isspace():
                pos += 1
                continue
            break
        # Fallback scan
        scan = 0
        built = ""
        while scan < len(text):
            if text[scan].isspace():
                scan += 1
                continue
            m = _TOKEN_RE.match(text, scan)
            if m is None:
                raise UnrecognizedTokenError(
                    f"Unrecognized character {text[scan]!r} in symbolic expression.",
                    expression=expression,
                    position=scan,
                    token=text[scan],
                    hint=(
                        "allowed tokens: x_i, numbers, '(', ')', '|', '**2'/'**3', "
                        f"operators {sorted(_BINARY_OPS | _UNARY_OPS)}"
                    ),
                )
            built += m.group(1)
            scan = m.end()
        raise UnrecognizedTokenError(
            "Unrecognized tokens in symbolic expression.",
            expression=expression,
            hint="check for illegal characters or malformed numbers",
        )
    return tokens


def _parse_infix(
    text: str,
    params: SymbolParams,
    *,
    expression: Optional[str] = None,
) -> NodeList:
    """Parse one or more infix expressions separated by ``|``."""
    expression = expression if expression is not None else text
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
    for idx, segment in enumerate(segments):
        if not segment:
            raise EmptySegmentError(
                f"Empty infix segment at output index {idx}.",
                expression=expression,
                hint="multi-output expressions use '|' between non-empty parts, "
                "e.g. '(x_0 add 1) | sin(x_1)'",
            )
        parser = _InfixParser(segment, params, expression=expression)
        nodes.append(parser.parse())
    return NodeList(nodes)


class _InfixParser:
    """Recursive-descent parser for the project's infix symbolic format."""

    def __init__(
        self,
        text: str,
        params: SymbolParams,
        *,
        expression: Optional[str] = None,
    ) -> None:
        self.expression = expression if expression is not None else text
        self.tokens = _tokenize(text, expression=self.expression)
        self.pos = 0
        self.params = params

    def parse(self) -> Node:
        node = self._parse_expr()
        if self.pos != len(self.tokens):
            raise TrailingTokensError(
                "Unexpected trailing tokens after a complete expression.",
                expression=self.expression,
                token=self.tokens[self.pos],
                hint=(
                    "binary operators must be written as '(left op right)'; "
                    "for example write '((x_0 mul 2) add sin(x_0))' "
                    "instead of '(x_0 mul 2) add sin(x_0)'"
                ),
            )
        return node

    def _peek(self) -> Optional[str]:
        if self.pos >= len(self.tokens):
            return None
        return self.tokens[self.pos]

    def _consume(self, expected: Optional[str] = None) -> str:
        tok = self._peek()
        if tok is None:
            raise UnexpectedEndError(
                "Unexpected end of symbolic expression.",
                expression=self.expression,
                hint=(
                    f"expected {expected!r}"
                    if expected is not None
                    else "expression is incomplete"
                ),
            )
        if expected is not None and tok != expected:
            raise UnexpectedTokenError(
                f"Expected {expected!r}, got {tok!r}.",
                expression=self.expression,
                token=tok,
                hint=f"check parentheses and operator placement near {tok!r}",
            )
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
            raise UnexpectedEndError(
                "Unexpected end of symbolic expression.",
                expression=self.expression,
                hint="expression is incomplete",
            )

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

        # Unknown identifier: maybe a mistyped operator
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", tok):
            raise UnknownOperatorError(
                f"Unknown operator or leaf {tok!r}.",
                expression=self.expression,
                token=tok,
                hint=(
                    f"binary operators: {sorted(_BINARY_OPS)}; "
                    f"unary operators: {sorted(_UNARY_OPS)}; "
                    "leaves: x_i / numbers / rand / e / pi"
                ),
            )

        raise UnexpectedTokenError(
            f"Unexpected token in symbolic expression: {tok!r}.",
            expression=self.expression,
            token=tok,
            hint="check parentheses and operator words such as add/mul/sin",
        )
