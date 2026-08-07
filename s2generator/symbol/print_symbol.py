# -*- coding: utf-8 -*-
"""
Created on 2025/08/25 00:14:51
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""

import re

from typing import Union, Callable, List
from s2generator.symbol.base import Node, NodeList

# Create the chain of replacements data type
ReplaceFn = Callable[[str], str]


def power(text: str, power_name: str, power_member: str) -> str:
    """
    Adjustments are made to suffixes that have power forms.

    :param text: The input string containing 'power or inv' expressions.
    :param power_name: the name of the power expression.
    :param power_member: the value of the power expression.
    :return: return the new symbol replaced by power.
    """
    # Count the number of symbols
    number = text.count(power_name)

    # Loop pointer
    c = 0
    # Starting index pointer
    start = 0

    while c < number:
        # Determine the location of the symbol
        position = text[start:].find(power_name) + start
        start = position + 1

        # Record the number of left and right brackets
        left, right = 0, 0

        for i in range(position, len(text)):
            if text[i] == "(":
                left += 1
            elif text[i] == ")":
                right += 1

                if i + 1 < len(text):
                    # Try to access this index
                    if text[i + 1] == " " and left == right:
                        text = text[: i + 1] + f" ^ {power_member}" + text[i + 1 :]
                else:
                    if left == right:
                        text = text + f" ^ {power_member}"

        c += 1

    return text.replace(power_name, "")


def replace_add(text: str) -> str:
    """
    Replace occurrences of 'add' with the equivalent mathematical expression.

    :param text: The input string containing 'add' expressions.
    :return: The modified string with 'add' replaced by the equivalent expression.
    """
    return text.replace("add", "+")


def replace_sub(text: str) -> str:
    """
    Replace occurrences of 'sub' with the equivalent mathematical expression.

    :param text: The input string containing 'sub' expressions.
    :return: The modified string with 'sub' replaced by the equivalent expression.
    """
    return text.replace("sub", "-")


def replace_mul(text: str) -> str:
    """
    Replace occurrences of 'mul' with the equivalent mathematical expression.

    :param text: The input string containing 'mul' expressions.
    :return: The modified string with 'mul' replaced by the equivalent expression.
    """
    return text.replace("mul", r"\times")


def replace_div(text: str) -> str:
    """
    Replace occurrences of 'div' with the equivalent mathematical expression.

    :param text: The input string containing 'div' expressions.
    :return: The modified string with 'div' replaced by the equivalent expression.
    """
    return text.replace("div", r"\div")


def replace_pow2(text: str) -> str:
    """
    Replace occurrences of 'pow2' with the equivalent mathematical expression.

    :param text: The input string containing 'pow2' expressions.
    :return: The modified string with 'pow2' replaced by the equivalent expression.
    """
    # return re.sub(r"pow2\(([^()]*)\)", r"(\1)^2", text)
    return power(text, power_name="pow2", power_member="2")


def replace_pow3(text: str) -> str:
    """
    Replace occurrences of 'pow3' with the equivalent mathematical expression.

    :param text: The input string containing 'pow3' expressions.
    :return: The modified string with 'pow3' replaced by the equivalent expression.
    """
    # return re.sub(r"pow3\(([^()]*)\)", r"(\1)^2", text)
    return power(text, power_name="pow3", power_member="3")


def replace_sqrt(text: str) -> str:
    """
    Replace occurrences of 'sqrt' with the equivalent mathematical expression.

    :param text: The input string containing 'sqrt' expressions.
    :return: The modified string with 'sqrt' replaced by the equivalent expression.
    """
    text = re.sub(r"sqrt\(([^()]*)\)", r"sqrt({\1})", text)
    return text.replace("sqrt", "\sqrt")


def replace_log(text: str) -> str:
    """
    Replace occurrences of 'log' with the equivalent mathematical expression.

    :param text: The input string containing 'log' expressions.
    :return: The modified string with 'log' replaced by the equivalent expression.
    """
    return text.replace("log", "\mathrm{log}")


def replace_inv(text: str) -> str:
    """
    Replace occurrences of 'inv' with the equivalent mathematical expression.

    :param text: The input string containing 'inv' expressions.
    :return: The modified string with 'inv' replaced by the equivalent expression.
    """
    # return re.sub(r"inv\(([^()]*)\)", r"(\1)^{-1}", text)
    return power(text, power_name="inv", power_member="{-1}")


def replace_exp(text: str) -> str:
    """
    Replace occurrences of 'exp' with the equivalent mathematical expression.

    :param text: The input string containing 'exp' expressions.
    :return: The modified string with 'exp' replaced by the equivalent expression.
    """
    return text.replace("exp", r"\mathrm{exp}")


def replace_sin(text: str) -> str:
    """
    Replace occurrences of 'sin' with the equivalent mathematical expression.

    :param text: The input string containing 'sin' expressions.
    :return: The modified string with 'sin' replaced by the equivalent expression.
    """
    return text.replace("sin", r"\mathrm{sin}")


def replace_cos(text: str) -> str:
    """
    Replace occurrences of 'cos' with the equivalent mathematical expression.

    :param text: The input string containing 'cos' expressions.
    :return: The modified string with 'cos' replaced by the equivalent expression.
    """
    return text.replace("cos", r"\mathrm{cos}")


def replace_tan(text: str) -> str:
    """
    Replace occurrences of 'tan' with the equivalent mathematical expression.

    :param text: The input string containing 'tan' expressions.
    :return: The modified string with 'tan' replaced by the equivalent expression.
    """
    return text.replace("tan", r"\mathrm{tan}")


def replace_arcsin(text: str) -> str:
    """
    Replace occurrences of 'arcsin' with the equivalent mathematical expression.

    :param text: The input string containing 'arcsin' expressions.
    :return: The modified string with 'arcsin' replaced by the equivalent expression.
    """
    return text.replace("arcsin", r"\mathrm{arcsin}")


def replace_arccos(text: str) -> str:
    """
    Replace occurrences of 'arccos' with the equivalent mathematical expression.

    :param text: The input string containing 'arccos' expressions.
    :return: The modified string with 'arccos' replaced by the equivalent expression.
    """
    return text.replace("arccos", r"\mathrm{arccos}")


def replace_arctan(text: str) -> str:
    """
    Replace occurrences of 'arctan' with the equivalent mathematical expression.

    :param text: The input string containing 'arctan' expressions.
    :return: The modified string with 'arctan' replaced by the equivalent expression.
    """
    return text.replace("arctan", r"\mathrm{arctan}")


def replace_diff(text: str) -> str:
    """
    Replace occurrences of 'diff' with the equivalent mathematical expression.

    :param text: The input string containing 'diff' expressions.
    :return: The modified string with 'diff' replaced by the equivalent expression.
    """
    return text.replace("diff", r"\bigtriangledown")


def replace_extra_brackets(text: str) -> str:
    """
    Replace occurrences of extra brackets with the equivalent mathematical expression.

    :param text: The input string containing extra brackets.
    :return: The modified string with extra brackets replaced by the equivalent expression.
    """
    # Replace '()' with ''
    text = text.replace("()", "")

    # Replace '[]' with ''
    text = text.replace("[]", "")

    # 处理两个括号中仅有的那一个字符
    # if only_one_char_between(text):
    return re.sub(r"\(([^()])\)", r"\1", text)


def replace_brackets(text: str) -> str:
    """
    Replace occurrences of brackets with the equivalent mathematical expression.

    :param text: The input string containing brackets.
    :return: The modified string with brackets replaced by the equivalent expression.
    """
    left_bracket = text.replace("(", r" { \left ( ")
    right_bracket = left_bracket.replace(")", r" \right ) } ")

    return right_bracket


def relace_backslash(text: str) -> str:
    """
    Replace occurrences of backslash with the equivalent mathematical expression.

    :param text: The input string containing backslash.
    :return: The modified string with backslash replaced by the equivalent expression.
    """
    return text.replace("\\\\", "\\")


def get_replace_chain() -> List[ReplaceFn]:
    """
    Create the chain of replacements for list.

    We first replace the binary operators, then the unary operators, and finally the brackets.

    :return: the chain of replacements
    """
    replace_chain: List[ReplaceFn] = [
        # Binary operators
        replace_add,
        replace_sub,
        replace_mul,
        replace_div,
        # Unary operators
        replace_pow2,
        replace_pow3,
        replace_sqrt,
        replace_log,
        replace_inv,
        replace_exp,
        replace_sin,
        replace_cos,
        replace_tan,
        replace_arcsin,
        replace_arccos,
        replace_arctan,
        replace_diff,
        # Brackets
        replace_extra_brackets,
        replace_brackets,
        relace_backslash,
    ]

    return replace_chain


def string_to_markdown(string: str) -> str:
    """
    Convert a string to a markdown formatted string.

    :param string: The input string to be converted.
    :return: The markdown formatted string.
    """
    # If the input is a string, return it as a string in markdown format

    # Get the relace chains
    replace_chain = get_replace_chain()

    for relace_fun in replace_chain:
        string = relace_fun(string)

    return string


def symbol_to_markdown(symbol: Union[str, Node, NodeList]) -> List[str]:
    """
    Convert a string to a markdown formatted string.

    :param symbol: The input string or the s2 Node/NodeList to be converted.
    :return: The markdown formatted string.
    """
    symbol_list = []

    # First, we check the data type of the input
    if isinstance(symbol, str):
        # the input is string
        symbol_list.append(string_to_markdown(symbol))

    elif isinstance(symbol, Node):
        # the input is Node object
        symbol_list.append(string_to_markdown(string=str(symbol)))

    elif isinstance(symbol, NodeList):
        # the input is NodeList object, so we first split the string with ' | '
        split_symbol = str(symbol).split(" | ")

        # Traverse this partitioned list to process each string
        for symbol in split_symbol:
            symbol_list.append(string_to_markdown(string=str(symbol)))

    else:
        # Handling Exceptions
        raise TypeError("Invalid symbol type, please input str or Node or NodeList !")

    return symbol_list


# if __name__ == '__main__':
#     import numpy as np
#
#     # Importing data generators, parameter controllers and visualization functions
#     from s2generator.symbol import SeriesSymbolGenerator
#
#     generator = SeriesSymbolGenerator()  # Create an instance
#
#     rng = np.random.RandomState(0)  # Creating a random number object
#     # Start generating symbolic expressions, sampling and generating series
#     trees, x, y = generator.run(rng, input_dimension=3, output_dimension=4, n_points=256)
#     # Print the expressions
#     # print(trees)
#     s_list = symbol_to_markdown(symbol=trees)
#     for s in s_list:
#         print(s)
