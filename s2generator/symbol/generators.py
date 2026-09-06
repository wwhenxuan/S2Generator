# -*- coding: utf-8 -*-
"""
Series-Symbol generators: random S2 construction and user-specified expressions.

Created on 2025/01/23 17:37:24
@author: Whenxuan Wang, Yifan Wu
@email: wwhenxuan@gmail.com, wy3370868155@outlook.com
@url: https://github.com/wwhenxuan/S2Generator
"""

import copy
import numpy as np
from numpy import ndarray
from numpy.random import RandomState
from collections import defaultdict

from typing import Optional, Union, Tuple, List, TYPE_CHECKING

if TYPE_CHECKING:
    from s2generator.excitation import Excitation

from s2generator.symbol.params import SeriesParams, SymbolParams
from s2generator.symbol.base import (
    Node,
    NodeList,
    operators_real,
    math_constants,
    all_operators,
    SPECIAL_WORDS,
)
from s2generator.symbol.encoders import GeneralEncoder
from s2generator.symbol.parse_symbol import (
    parse_symbol,
    infer_input_dimension,
    infer_output_dimension,
)
from s2generator.symbol.check_symbol import check_symbol
from s2generator.utils.tools import (
    z_score_normalization,
    max_min_normalization,
    save_s2data,
)
from s2generator.utils.data import PrintStatus


class SeriesSymbolGenerator(object):
    """Interface for constructing symbolic expressions and sampling time series."""

    def __init__(
        self,
        series_params: Optional[SeriesParams] = None,
        symbol_params: Optional[SymbolParams] = None,
        print_status: Optional[bool] = False,
        logging_path: Optional[str] = None,
        special_words: Optional[dict] = None,
    ) -> None:
        """
        :param series_params: The parameters controlling the generation of the stimulus time series.
        :param symbol_params: The parameters controlling the generation of the symbolic expressions.
        :param print_status: Whether to print the status of data generation.
        :param special_words: The special words controlling the generation of the symbolic expressions.
        :param logging_path: The path of the logging folder or file, if is file please end with `.txt`.
        """
        self.series_params = series_params = (
            SeriesParams() if series_params is None else series_params
        )
        self.symbol_params = symbol_params = (
            SymbolParams() if symbol_params is None else symbol_params
        )

        self.print_status = print_status
        if print_status:
            self.status = PrintStatus(
                series_params=series_params,
                symbol_params=symbol_params,
                logging_path=logging_path,
            )

        special_words = SPECIAL_WORDS if special_words is None else special_words

        # Probability to generate integer in leafs
        self.prob_const = symbol_params.prob_const

        # Probability to generate n in leafs
        self.prob_rand = symbol_params.prob_rand
        self.max_int = symbol_params.max_int  # Maximal integer in symbolic expressions
        self.min_binary_ops_per_dim = (
            symbol_params.min_binary_ops_per_dim
        )  # Min number of binary operators per input dimension
        self.max_binary_ops_per_dim = (
            symbol_params.max_binary_ops_per_dim
        )  # Max number of binary operators per input dimension
        self.min_unary_ops = (
            symbol_params.min_unary_ops
        )  # Min number of unary operators
        self.max_unary_ops = (
            symbol_params.max_unary_ops
        )  # Max number of unary operators

        # Maximum and minimum input dimensions
        self.min_output_dimension = symbol_params.min_output_dimension
        self.min_input_dimension = symbol_params.min_input_dimension
        self.max_input_dimension = symbol_params.max_input_dimension
        self.max_output_dimension = symbol_params.max_output_dimension

        # Maximum numerical range
        self.max_number = 10**symbol_params.max_exponent

        # Operators that can be used with copy
        self.operators = copy.deepcopy(operators_real)
        self.operators_dowsample_ratio = defaultdict(float)
        if symbol_params.operators_to_downsample != "":
            # Some invalid operations need to be removed, such as div0
            for operator in self.symbol_params.operators_to_downsample.split(","):
                operator, ratio = operator.split("_")
                # Specify the probability of certain expressions appearing here
                ratio = float(ratio)
                self.operators_dowsample_ratio[operator] = ratio

        if symbol_params.required_operators != "":
            # Specify the symbolic expressions to be removed
            self.required_operators = self.symbol_params.required_operators.split(",")
        else:
            self.required_operators = []

        if symbol_params.extra_binary_operators != "":
            # Additional binary operators
            self.extra_binary_operators = (
                self.symbol_params.extra_binary_operators.split(",")
            )
        else:
            self.extra_binary_operators = []

        if symbol_params.extra_unary_operators != "":
            # Additional unary operators
            self.extra_unary_operators = self.symbol_params.extra_unary_operators.split(
                ","
            )
        else:
            self.extra_unary_operators = []

        # All unary operators that can be used when constructing expressions
        self.unaries = [
            o for o in self.operators.keys() if np.abs(self.operators[o]) == 1
        ] + self.extra_unary_operators
        # All binary operators that can be used when constructing expressions
        self.binaries = [
            o for o in self.operators.keys() if np.abs(self.operators[o]) == 2
        ] + self.extra_binary_operators

        # Adjust the probability of each unary operator appearing
        unaries_probabilities = []
        for op in self.unaries:
            # If the probability of this operator appearing is not specifically specified, default to 1
            if op not in self.operators_dowsample_ratio:
                unaries_probabilities.append(1.0)
            else:
                ratio = self.operators_dowsample_ratio[op]
                unaries_probabilities.append(ratio)
        # Normalize the probabilities
        self.unaries_probabilities = np.array(unaries_probabilities)
        self.unaries_probabilities /= self.unaries_probabilities.sum()

        # Adjust the probability of each binary operator appearing
        binaries_probabilities = []
        for op in self.binaries:
            if op not in self.operators_dowsample_ratio:
                binaries_probabilities.append(1.0)
            else:
                ratio = self.operators_dowsample_ratio[op]
                binaries_probabilities.append(ratio)
        self.binaries_probabilities = np.array(binaries_probabilities)
        self.binaries_probabilities /= self.binaries_probabilities.sum()

        self.unary = False  # len(self.unaries) > 0
        # Enumerate the possible number of unary binary trees that can be generated from an empty node
        self.distrib = self.generate_dist(
            2 * self.max_binary_ops_per_dim * self.max_input_dimension
        )

        # The numerical range of constants in leaf nodes
        self.constants = [
            str(i) for i in range(-self.max_int, self.max_int + 1) if i != 0
        ]
        self.constants += math_constants  # Add specific mathematical symbol constants
        # Initialize the number of variables
        self.variables = ["rand"] + [f"x_{i}" for i in range(self.max_input_dimension)]

        # Summarize all symbols that can be used when constructing symbolic expressions
        self.symbols = (
            list(self.operators)
            + self.constants
            + self.variables
            + ["|", "INT+", "INT-", "FLOAT+", "FLOAT-", "pow", "0"]
        )
        self.constants.remove("CONSTANT")
        if self.symbol_params.extra_constants is not None:
            self.extra_constants = self.symbol_params.extra_constants.split(",")
        else:
            self.extra_constants = []

        # Obtain the numerical encoder and symbol encoder
        self.general_encoder = GeneralEncoder(
            symbol_params, self.symbols, all_operators
        )
        # Encoder for input and output sequences
        self.float_encoder = self.general_encoder.float_encoder
        self.float_words = special_words + sorted(list(set(self.float_encoder.symbols)))
        # Encoder for symbolic expressions
        self.equation_encoder = self.general_encoder.equation_encoder
        self.equation_words = sorted(list(set(self.symbols)))
        self.equation_words = special_words + self.equation_words
        # breakpoint()

        # Number of decimal places for floating-point numbers in symbolic expressions
        self.decimals = self.symbol_params.decimals

        # Record how many dimensions of variables have been generated in the symbolic expression
        self._n_used_dims = 0

        # Create an interface for generating stimulus time series data
        self.excitation = self.create_excitation(series_params=series_params)

        # Handle overflow outside the domain of the generation attempt
        self.max_trials = symbol_params.max_trials

    def create_excitation(
        self, series_params: Optional[SeriesParams] = None
    ) -> "Excitation":
        """
        Create the base and general generator for the excitation time series.

        :param series_params: The parameters controlling the generation of the stimulus time series
        :return: The general generator for the excitation time series.
        """
        from s2generator.excitation import Excitation

        return Excitation(
            series_params=self.series_params if series_params is None else series_params
        )

    def generate_dist(self, max_ops: int) -> List:
        """
        `max_ops`: maximum number of operators
        Enumerate the number of possible unary-binary trees that can be generated from empty nodes.
        D[e][n] represents the number of different binary trees with n nodes that
        can be generated from e empty nodes, using the following recursion:

        .. math:
            D(n, 0) = 0
            D(0, e) = 1
            D(n, e) = D(n, e - 1) + p_1 * D(n- 1, e) + D(n - 1, e + 1)

        :math:`p_1` =  if binary trees, 1 if unary binary
        """
        p1 = 1 if self.unary else 0
        # enumerate possible trees
        D = [[0] + ([1 for i in range(1, 2 * max_ops + 1)])]
        for n in range(1, 2 * max_ops + 1):  # number of operators
            s = [0]
            for e in range(1, 2 * max_ops - n + 1):  # number of empty nodes
                s.append(s[e - 1] + p1 * D[n - 1][e] + D[n - 1][e + 1])
            D.append(s)
        assert all(
            len(D[i]) >= len(D[i + 1]) for i in range(len(D) - 1)
        ), "issue in generate_dist"
        return D

    def generate_float(self, rng: RandomState, exponent=None) -> str:
        """Generate a valid random floating-point number within a specified range"""
        # Generate the sign of the number
        sign = rng.choice([-1, 1])
        mantissa = float(rng.choice(range(1, 10**self.symbol_params.float_precision)))
        if not exponent:
            # Determine whether to generate the exponent
            min_power = (
                -self.symbol_params.max_exponent_prefactor
                - (self.symbol_params.float_precision + 1) // 2
            )
            max_power = (
                self.symbol_params.max_exponent_prefactor
                - (self.symbol_params.float_precision + 1) // 2
            )
            exponent = rng.randint(min_power, max_power + 1)
        constant = sign * (mantissa * 10**exponent)  # Sign bit + mantissa + exponent
        return str(np.round(constant, decimals=self.decimals))  # Return as a string

    def generate_int(self, rng: RandomState) -> str:
        """Generate a valid random integer within a specified range"""
        return str(rng.choice(self.constants + self.extra_constants))

    def generate_leaf(self, rng: RandomState, input_dimension: int) -> str:
        """
        Generate a leaf node in the sampling expression.

        The logic behind this code generation process first determines whether to generate a random number.
        If no random number is generated, leaf node variables are generated.
        If all leaf node variables have been traversed and there are still leaf nodes remaining,
        integer nodes or random leaf nodes are generated based on the specified probability.

        :param rng: The random number generator in NumPy with fixed seed.
        :param input_dimension: The dimension of the symbolic expression.
        :return: The leaf node, rand, x_{dimension} or int.
        """
        if self.n_used_dims == 0:
            # At least make sure there is a variable node
            dimension = rng.randint(0, input_dimension)
            self.add_used_dims(dims=1)
            return f"x_{dimension}"

        if rng.rand() < self.prob_rand:
            # Prioritize random number generation nodes
            return "rand"  # Generate a random number
        else:
            if self.n_used_dims < input_dimension:
                # When the number of used variables is less than the specified number
                dimension = self.n_used_dims
                # self.n_used_dims += 1
                # Add the already used dimension
                self.add_used_dims(dims=1)
                return f"x_{dimension}"
            else:
                # Generate an integer or a random symbolic variable
                draw = rng.rand()
                if draw < self.prob_const:
                    # Generate random constant leaf nodes
                    return self.generate_int(rng)
                else:
                    # When all dimensions have been traversed
                    # And if there are still free leaf nodes, return a random traversal node
                    dimension = rng.randint(0, input_dimension)
                    return f"x_{dimension}"

    def generate_ops(self, rng: RandomState, arity: int) -> str:
        """Select a specific operation for an operator node"""
        if arity == 1:
            # Handling unary operators
            ops = self.unaries
            probas = self.unaries_probabilities
        else:
            # Handling binary operators
            ops = self.binaries
            probas = self.binaries_probabilities
        return rng.choice(ops, p=probas)

    def sample_next_pos(self, rng: RandomState, nb_empty: int, nb_ops: int) -> Tuple:
        """
        Sample the position of the next node (binary case).
        Sample a position in {0, ..., `nb_empty` - 1}.
        """
        assert nb_empty > 0
        assert nb_ops > 0
        probs = []
        if self.unary:
            for i in range(nb_empty):
                probs.append(self.distrib[nb_ops - 1][nb_empty - i])
        for i in range(nb_empty):
            probs.append(self.distrib[nb_ops - 1][nb_empty - i + 1])
        probs = [p / self.distrib[nb_ops][nb_empty] for p in probs]
        probs = np.array(probs, dtype=np.float64)
        e = rng.choice(len(probs), p=probs)
        arity = 1 if self.unary and e < nb_empty else 2
        e %= nb_empty
        return e, arity

    @property
    def n_used_dims(self) -> int:
        """Get the number of nodes currently generated (the used dimensions)"""
        return self._n_used_dims

    def reset_used_dims(self) -> None:
        """Reset the used dimensions to zero"""
        self._n_used_dims = 0

    def add_used_dims(self, dims: Optional[int] = 1) -> None:
        """
        Add the used dimensions to the current used dimensions.

        :param dims: Number of used dimensions, defaults to 1.
        :return: None.
        """
        self._n_used_dims += dims

    def generate_tree(
        self, rng: RandomState, nb_binary_ops: int, input_dimension: int
    ) -> Node:
        """
        Function to generate a tree, which is essentially an expression.

        :param rng: The random number generator in NumPy with fixed seed.
        :param nb_binary_ops: Number of binary operators used in a binary expression.
        :param input_dimension: Number of dimensions used in a binary expression.
        :return: The generated tree (symbolic expression or complex system).
        """
        # Reset the pointer that currently records the number of generated nodes
        self.reset_used_dims()

        # Initialize the first root node of the tree
        tree = Node(0, self.symbol_params)
        empty_nodes = [tree]
        next_en = 0

        # Initially, there is only one empty node, which will gradually accumulate
        nb_empty = 1
        while nb_binary_ops > 0:
            # Sample to generate the basic framework of the tree; the basic framework is composed of binary operators
            next_pos, arity = self.sample_next_pos(rng, nb_empty, nb_binary_ops)
            next_en += next_pos
            op = self.generate_ops(rng, arity)
            empty_nodes[next_en].value = op
            for _ in range(arity):
                e = Node(0, self.symbol_params)
                empty_nodes[next_en].push_child(e)
                empty_nodes.append(e)
            next_en += 1
            nb_empty += arity - 1 - next_pos
            nb_binary_ops -= 1
        rng.shuffle(empty_nodes)  # Shuffle the sampled nodes
        for n in empty_nodes:
            if len(n.children) == 0:
                n.value = self.generate_leaf(rng, input_dimension)
        return tree

    def generate_symbolic_expression(
        self,
        rng: RandomState,
        input_dimension: Optional[int] = None,
        output_dimension: Optional[int] = None,
        nb_unary_ops: Optional[int] = None,
        nb_binary_ops: Optional[int] = None,
        return_all: bool = False,
    ):
        """
        Generates a multivariate set of symbolic expressions with channel
        dependencies based on input parameters and a random number generator.

        Specific steps for generating a tree:
        (1) Determine hyperparameters: First, determine the relevant hyperparameters. If the user specifies them, use the user-input parameters; otherwise, randomly generate the parameters.
        (2) Generate the trunk of the binary tree: Use binary operators to form the basic structure of the symbolic expression and add leaf nodes of random numbers and variables.
        (3) Add unary operators: Randomly insert unary operators into the constructed binary tree data to enrich its operations.
        (4) Final improvement: Add other variables or perform radial transformations to further improve its diversity.

        :param rng: The random number generator in NumPy with fixed seed.
        :param input_dimension: Number of input dimensions in the target symbolic expression to be generated.
        :param output_dimension: Number of output dimensions in the generated symbolic expression to be generated.
        :param nb_unary_ops: Number of binary operators used in a binary expression.
        :param nb_binary_ops: Number of binary operators used in a binary expression.
        :param return_all: Whether to return all information in the generation of the trees.
        :return: The generated trees (symbolic expression or complex system).
        """
        trees = []  # Initialize a list to store multiple symbolic expressions
        if input_dimension is None:
            # If the input dimension is not specified, initialize it randomly
            input_dimension = rng.randint(
                self.min_input_dimension, self.max_input_dimension + 1
            )

        if output_dimension is None:
            # If the output dimension is not specified, initialize it randomly
            output_dimension = rng.randint(
                self.min_output_dimension, self.max_output_dimension + 1
            )

        if nb_binary_ops is None:
            # If the number of binary operators is not specified, initialize it based on the input dimension
            min_binary_ops = self.min_binary_ops_per_dim * input_dimension
            max_binary_ops = self.max_binary_ops_per_dim * input_dimension

            # Initialize randomly within the range of minimum to maximum operators plus an offset
            nb_binary_ops_to_use = [
                rng.randint(
                    min_binary_ops,
                    self.symbol_params.max_binary_ops_offset + max_binary_ops,
                )
                for dim in range(output_dimension)
            ]  # Initialize for each dimension

        elif isinstance(nb_binary_ops, int):
            # If a specific number is provided, use that number for each dimension
            nb_binary_ops_to_use = [nb_binary_ops for _ in range(output_dimension)]

        else:
            # If it's not a number, it must be a list
            nb_binary_ops_to_use = nb_binary_ops

        if nb_unary_ops is None:
            # Initialize the number of unary operators
            nb_unary_ops_to_use = [
                rng.randint(self.min_unary_ops, self.max_unary_ops + 1)
                for dim in range(output_dimension)
            ]
        elif isinstance(nb_unary_ops, int):
            nb_unary_ops_to_use = [nb_unary_ops for _ in range(output_dimension)]
        else:
            nb_unary_ops_to_use = nb_unary_ops

        for i in range(output_dimension):
            # Iterate over the specified number of output dimensions to generate data
            # Generate a binary tree as the basic framework
            tree = self.generate_tree(rng, nb_binary_ops_to_use[i], input_dimension)

            # Insert unary operators into the binary tree
            tree = self.add_unaries(rng, tree, nb_unary_ops_to_use[i])

            if tree is None:
                if return_all:
                    return None, None, None, None, None
                else:
                    return None, None, None

            # Adding constants
            if self.symbol_params.reduce_num_constants:
                tree = self.add_prefactors(rng, tree)
            else:
                # Apply affine transformations
                tree = self.add_linear_transformations(rng, tree, target=self.variables)
                tree = self.add_linear_transformations(rng, tree, target=self.unaries)
            trees.append(tree)  # Add to the specified storage list

        # Construct a data structure to store multi-dimensional symbolic expressions
        tree = NodeList(trees)

        if return_all:
            # Iterate over the expressions to count the used symbols
            nb_unary_ops_to_use = [
                len([x for x in tree_i.prefix().split(",") if x in self.unaries])
                for tree_i in tree.nodes
            ]
            nb_binary_ops_to_use = [
                len([x for x in tree_i.prefix().split(",") if x in self.binaries])
                for tree_i in tree.nodes
            ]
        for op in self.required_operators:
            if op not in tree.infix():
                return self.generate_symbolic_expression(
                    rng, input_dimension, output_dimension, nb_unary_ops, nb_binary_ops
                )

        if return_all:
            return (
                tree,
                input_dimension,
                output_dimension,
                nb_unary_ops_to_use,
                nb_binary_ops_to_use,
            )
        else:
            return tree, input_dimension, output_dimension

    def add_unaries(
        self, rng: RandomState, tree: Node, nb_unaries: int
    ) -> Union[Node, None]:
        """Insert unary operators into a binary tree composed of binary operators and leaf nodes to increase diversity"""
        prefix = self._add_unaries(
            rng, tree
        )  # Get the traversal sequence after insertion
        prefix = prefix.split(",")  # Split the traversal sequence
        indices = []
        for i, x in enumerate(prefix):
            if x in self.unaries:
                indices.append(i)
        rng.shuffle(indices)
        if len(indices) > nb_unaries:
            to_remove = indices[: len(indices) - nb_unaries]
            for index in sorted(to_remove, reverse=True):
                del prefix[index]

        # Decode using the symbol encoder
        symbol = self.equation_encoder.decode(prefix)
        if symbol is None:
            return None
        tree = symbol.nodes[0]
        return tree

    def _add_unaries(self, rng: RandomState, tree: Node) -> str:
        """Insert unary operators into a symbolic expression and get the traversal sequence"""
        # Get the specific value of the current node
        s = str(tree.value)
        for c in tree.children:
            # Ensure the depth of unary operators meets the requirements
            if len(c.prefix().split(",")) < self.symbol_params.max_unary_depth:
                # Randomly select a unary operator to insert
                unary = rng.choice(self.unaries, p=self.unaries_probabilities)
                s += f",{unary}," + self._add_unaries(rng, c)
            else:
                s += f"," + self._add_unaries(rng, c)
        return s

    def add_prefactors(self, rng: RandomState, tree: Node) -> Node:
        """Insert prefactors into a symbolic expression"""
        transformed_prefix = self._add_prefactors(rng, tree)
        if transformed_prefix == tree.prefix():
            a = self.generate_float(rng)
            transformed_prefix = f"mul,{a}," + transformed_prefix
        a = self.generate_float(rng)
        transformed_prefix = f"add,{a}," + transformed_prefix
        tree = self.equation_encoder.decode(transformed_prefix.split(",")).nodes[0]
        return tree

    def _add_prefactors(self, rng, tree) -> str:
        """Add prefactors to a symbolic expression and get the traversal sequence"""
        s = str(tree.value)  # Get the value of the current node
        # Generate two random floating-point numbers
        a, b = self.generate_float(rng), self.generate_float(rng)
        if s in ["add", "sub"]:
            # Handle binary operators
            s += (
                "," if tree.children[0].value in ["add", "sub"] else f",mul,{a},"
            ) + self._add_prefactors(rng, tree.children[0])
            s += (
                "," if tree.children[1].value in ["add", "sub"] else f",mul,{b},"
            ) + self._add_prefactors(rng, tree.children[1])
        elif s in self.unaries and tree.children[0].value not in ["add", "sub"]:
            # Handle unary operators
            s += f",add,{a},mul,{b}," + self._add_prefactors(rng, tree.children[0])
        else:
            for c in tree.children:
                s += f"," + self._add_prefactors(rng, c)
        return s

    def add_linear_transformations(
        self,
        rng: RandomState,
        tree: Node,
        target: List[str],
        add_after: Optional[bool] = False,
    ) -> Node:
        """Apply affine transformations to the constructed symbolic expression to increase diversity"""
        prefix = tree.prefix().split(",")
        indices = []
        for i, x in enumerate(prefix):
            if x in target:
                indices.append(i)
        offset = 0
        for idx in indices:
            # Generate random floating-point numbers as weights and biases
            a, b = self.generate_float(rng), self.generate_float(rng)
            if add_after:
                prefix = (
                    prefix[: idx + offset + 1]
                    + ["add", a, "mul", b]
                    + prefix[idx + offset + 1 :]
                )
            else:
                prefix = (
                    prefix[: idx + offset]
                    + ["add", a, "mul", b]
                    + prefix[idx + offset :]
                )
            offset += 4
        tree = self.equation_encoder.decode(prefix).nodes[0]
        return tree

    @staticmethod
    def relabel_variables(tree: Node) -> int:
        """Count the number of leaf nodes in the tree and relabel them"""
        active_variables = []
        for elem in tree.prefix().split(","):
            if elem.startswith("x_"):
                active_variables.append(elem)
        active_variables = list(set(active_variables))
        input_dimension = len(active_variables)
        if input_dimension == 0:
            return 0
        active_variables.sort(key=lambda x: int(x[2:]))
        for j, xi in enumerate(active_variables):
            tree.replace_node_value(xi, "x_{}".format(j))
        return input_dimension

    def function_to_skeleton(
        self,
        tree: Union[Node, NodeList],
        skeletonize_integers: Optional[bool] = False,
        constants_with_idx: Optional[bool] = False,
    ) -> Tuple[Union[Node, NodeList], List]:
        """
        Obtain the basic framework of a symbolic expression.

        :param tree: The symbolic expression to be processed.
        :param skeletonize_integers: Whether to process integer values.
        :param constants_with_idx: Whether the output numerical operators should have indices
        """
        constants = []
        prefix = tree.prefix().split(",")  # Get the pre-order traversal of the symbols
        j = 0
        for i, pre in enumerate(prefix):
            # Use exception handling to determine if it is a number
            try:
                float(pre)
                is_float = True
                if pre.lstrip("-").isdigit():
                    is_float = False
            except ValueError:
                is_float = False

            if pre.startswith("CONSTANT"):
                # If the value is already CONSTANT
                constants.append("CONSTANT")
                if constants_with_idx:
                    # Mark each numerical floating-point number with an index
                    prefix[i] = "CONSTANT_{}".format(j)
                j += 1
            elif is_float or (pre in self.constants and skeletonize_integers):
                if constants_with_idx:
                    prefix[i] = "CONSTANT_{}".format(j)
                else:
                    prefix[i] = "CONSTANT"
                while i > 0 and prefix[i - 1] in self.unaries:
                    del prefix[i - 1]
                try:
                    value = float(pre)
                except:
                    value = getattr(np, pre)
                constants.append(value)
                j += 1
            else:
                continue
        new_tree = self.equation_encoder.decode(prefix)
        return new_tree, constants

    @staticmethod
    def order_datapoints(inputs: ndarray, outputs: ndarray) -> Tuple[ndarray, ndarray]:
        mean_input = inputs.mean(0)
        distance_to_mean = np.linalg.norm(inputs - mean_input, axis=-1)
        order_by_distance = np.argsort(distance_to_mean)
        return inputs[order_by_distance], outputs[order_by_distance]

    def get_rid(self, x: ndarray, y: ndarray) -> Tuple[ndarray, ndarray]:
        """
        Remove illegal values from the generated time series from the complex system.

        To ensure accurate sampling of complex systems, we discard values outside the
        domain of the symbolic expression $f(\cdot)$ or values that are too large.
        While this may reduce the sampling efficiency of data generation,
        it will improve the quality of data generation to a certain extent.

        :param x: The original excitation sampling time series.
        :param y: The generated time series from the complex system.
        :return: The input and output time series which removed the illegal values.
        """
        # Remove NaNs
        is_nan_idx = np.any(np.isnan(y), -1)

        # Remove values outside the domain
        x = x[~is_nan_idx, :]
        y = y[~is_nan_idx, :]

        # Remove very large numbers
        y[np.abs(y) >= self.max_number] = np.nan
        y[np.abs(y) == np.inf] = np.nan  # Infinity
        is_nan_idx = np.any(np.isnan(y), -1)
        x = x[~is_nan_idx, :]
        y = y[~is_nan_idx, :]

        return x, y

    def generate_excitation(
        self,
        rng: np.random.RandomState,
        seq_length: int,
        input_dimension: int = 1,
        normalize: Optional[bool] = False,
    ) -> np.ndarray:
        """
        Generate excitation time series data with different sampling strategies.

        This method primarily calls our highly encapsulated `Excitation` class to generate sampled time series data X.
        This sampled time series is then input into the constructed complex system
        (symbolic expression) to further generate a response time series.

        :param rng: The random number generator in NumPy with fixed seed.
        :param seq_length: The number of points of time series to generate.
        :param input_dimension: The number of dimensions of time series to generate.
        :param normalize: If True, normalize the output time series.
        :return: The generated excitation time series.
        """
        return self.excitation.generate(
            rng=rng,
            seq_length=seq_length,
            num_channels=input_dimension,
            normalization=normalize,
        )

    @staticmethod
    def save_s2data(
        symbol: Union[str, "Node", "NodeList"],
        excitation: np.ndarray,
        response: np.ndarray,
        save_path: str = None,
    ) -> bool:
        """
        Saves S2 data (symbolic expressions and time series) to the specified location.

        This function works with load_s2data to provide a complete save/load workflow.
        Supports multiple file formats (.npy, .npz) and handles directory creation.

        :param symbol: Symbolic expression data generated by SeriesSymbolGenerator
                       (Node, NodeList, or str).
        :param excitation: Input excitation time series data.
        :param response: Response time series data obtained from the system.
        :param save_path: Save location with two input options:
                         - Directory path: data will be saved as 's2data.npz' in this directory;
                         - Full file path: must end with .npy or .npz extension.
        :return: Boolean indicating success status of the save operation.
        """
        status = save_s2data(
            symbol=symbol, excitation=excitation, response=response, save_path=save_path
        )

        return status

    def parse_symbol(self, symbol: Union[str, Node, NodeList, List[str]]) -> NodeList:
        """
        Parse a user-specified symbolic expression into a ``NodeList``.

        Accepts infix / prefix strings, prefix token lists, ``Node``, or ``NodeList``.

        :param symbol: User-provided symbolic expression.
        :return: Parsed multivariate symbolic expression.
        """
        return parse_symbol(symbol=symbol, symbol_params=self.symbol_params)

    def evaluate_symbol(
        self,
        rng: np.random.RandomState,
        trees: Union[str, Node, NodeList, List[str]],
        seq_length: int,
        input_dimension: Optional[int] = None,
        output_dimension: Optional[int] = None,
        max_trials: Optional[int] = None,
        input_normalize: Optional[str] = "z-score",
        output_normalize: Optional[str] = "z-score",
        input_max_scale: Optional[float] = 16.0,
        output_max_scale: Optional[float] = 16.0,
        offset: Optional[Tuple[float, float]] = None,
        save_path: Optional[str] = None,
    ) -> Union[Tuple[None, None, None], Tuple[NodeList, ndarray, ndarray]]:
        """
        Generate excitation series and evaluate a (user-specified) symbolic system.

        :param rng: The random number generator in NumPy with fixed seed.
        :param trees: Symbolic expression (string / Node / NodeList / prefix tokens).
        :param seq_length: The number of points of time series to generate.
        :param input_dimension: Input channels; inferred from the symbol if omitted.
        :param output_dimension: Output channels; inferred from the symbol if omitted.
        :param max_trials: The maximum number of trials to generate and try.
        :param input_normalize: Normalize the input time series.
        :param output_normalize: Normalize the output time series.
        :param input_max_scale: The scaling factor of the input time series.
        :param output_max_scale: The scaling factor of the output time series.
        :param offset: The offset mean and std for the input time series.
        :param save_path: Optional path to save the generated S2 data.
        :return: ``(symbol, excitation, response)`` or ``(None, None, None)`` on failure.
        """
        trees = self.parse_symbol(trees)
        input_dimension = (
            infer_input_dimension(trees) if input_dimension is None else input_dimension
        )
        output_dimension = (
            infer_output_dimension(trees)
            if output_dimension is None
            else output_dimension
        )
        max_trials = self.max_trials if max_trials is None else max_trials

        if self.print_status:
            self.status.show_start(
                seq_length=seq_length,
                input_dimension=input_dimension,
                output_dimension=output_dimension,
                max_trials=max_trials,
                input_normalize=input_normalize,
                output_normalize=output_normalize,
                input_max_scale=input_max_scale,
                output_max_scale=output_max_scale,
                offset=offset,
            )
            self.status.update_symbol(status="success")

        inputs, outputs = [], []
        trials = 0
        remaining_points = seq_length

        while remaining_points > 0 and trials < max_trials:
            x = self.generate_excitation(
                rng=rng,
                seq_length=seq_length,
                input_dimension=input_dimension,
                normalize=input_normalize,
            )
            x *= rng.uniform(low=0, high=input_max_scale)

            if offset is not None:
                mean, std = offset
                x *= std
                x += mean

            if self.print_status:
                self.status.update_excitation(status="success")

            y = trees.val_router(x)
            x, y = self.get_rid(x, y)

            valid_points = y.shape[0]
            trials += 1
            remaining_points -= valid_points

            if valid_points == 0:
                if self.print_status:
                    self.status.update_response(status="failure")
                continue

            inputs.append(x)
            outputs.append(y)

            if self.print_status:
                if remaining_points > 0:
                    self.status.update_response(status="failure")
                else:
                    self.status.update_response(status="success")
                    break

        if remaining_points > 0:
            if self.print_status:
                self.status.reset()
            return None, None, None

        inputs = np.concatenate(inputs, axis=0)[:seq_length]
        outputs = np.concatenate(outputs, axis=0)[:seq_length]

        if output_normalize is None:
            pass
        elif output_normalize == "z-score":
            for dim in range(output_dimension):
                outputs[:, dim] = z_score_normalization(x=outputs[:, dim])
        elif output_normalize == "max-min":
            for dim in range(output_dimension):
                outputs[:, dim] = max_min_normalization(x=outputs[:, dim])
        else:
            raise ValueError(
                "The normalization option must be 'z-score' or 'max-min' or None!"
            )

        outputs *= rng.uniform(low=0, high=output_max_scale)

        if self.print_status:
            self.status.show_end(symbol=trees)
            self.status.reset()

        if save_path is not None:
            self.save_s2data(
                symbol=trees, excitation=inputs, response=outputs, save_path=save_path
            )

        return trees, inputs, outputs

    def run_from_symbol(
        self,
        rng: np.random.RandomState,
        symbol: Union[str, Node, NodeList, List[str]],
        seq_length: int,
        input_dimension: Optional[int] = None,
        output_dimension: Optional[int] = None,
        max_trials: Optional[int] = None,
        input_normalize: Optional[str] = "z-score",
        output_normalize: Optional[str] = "z-score",
        input_max_scale: Optional[float] = 16.0,
        output_max_scale: Optional[float] = 16.0,
        offset: Optional[Tuple[float, float]] = None,
        save_path: Optional[str] = None,
    ) -> Union[Tuple[None, None, None], Tuple[NodeList, ndarray, ndarray]]:
        """
        Read a user-specified symbolic expression and generate series via excitation.

        This is the dedicated entry for fixed complex systems ``f(.)``: sample
        excitation ``X`` and return ``Y = f(X)``.

        :param rng: The random number generator in NumPy with fixed seed.
        :param symbol: User-specified expression (infix / prefix / Node / NodeList).
        :param seq_length: The number of points of time series to generate.
        :param input_dimension: Input channels; inferred from ``symbol`` if omitted.
        :param output_dimension: Output channels; inferred from ``symbol`` if omitted.
        :param max_trials: The maximum number of trials to generate and try.
        :param input_normalize: Normalize the input time series.
        :param output_normalize: Normalize the output time series.
        :param input_max_scale: The scaling factor of the input time series.
        :param output_max_scale: The scaling factor of the output time series.
        :param offset: The offset mean and std for the input time series.
        :param save_path: Optional path to save the generated S2 data.
        :return: ``(symbol, excitation, response)`` or ``(None, None, None)`` on failure.
        """
        return self.evaluate_symbol(
            rng=rng,
            trees=symbol,
            seq_length=seq_length,
            input_dimension=input_dimension,
            output_dimension=output_dimension,
            max_trials=max_trials,
            input_normalize=input_normalize,
            output_normalize=output_normalize,
            input_max_scale=input_max_scale,
            output_max_scale=output_max_scale,
            offset=offset,
            save_path=save_path,
        )

    def run_with_state(
        self,
        rng: np.random.RandomState,
        seq_length: int,
        input_dimension: int = 1,
        output_dimension: int = 1,
        max_trials: Optional[int] = None,
        input_normalize: Optional[str] = "z-score",
        output_normalize: Optional[str] = "z-score",
        input_max_scale: Optional[float] = 16.0,
        output_max_scale: Optional[float] = 16.0,
        offset: Optional[Tuple[float, float]] = None,
        save_path: Optional[str] = None,
    ) -> Union[Tuple[None, None, None], Tuple[NodeList, ndarray, ndarray]]:
        """
        Generate a random symbolic expression and excitation series with status logging.

        :param rng: The random number generator in NumPy with fixed seed.
        :param seq_length: The number of points of time series to generate.
        :param input_dimension: The number of the input dimensions of time series to generate.
        :param output_dimension: The number of the output dimensions of time series or symbol expression to generate.
        :param max_trials: The maximum number of trials to generate and try.
        :param input_normalize: Normalize the input time series, choice of 'z-score' or 'max-min' or None, defaults to 'z-score'.
        :param output_normalize: Normalize the output time series, choice of 'z-score' or 'max-min' or None, defaults to 'z-score'.
        :param input_max_scale: The scaling factor of the input time series to generate.
        :param output_max_scale: The scaling factor of the output time series to generate.
        :param offset: The offset mean and std for the input time series.
        :param save_path: Whether to save the S2 data generated this time.
                          The default value is None. If the relevant address is passed in, the data will be saved.
        """
        trees, _, _ = self.generate_symbolic_expression(
            rng,
            input_dimension=input_dimension,
            output_dimension=output_dimension,
            return_all=False,
        )
        if trees is None:
            return None, None, None

        return self.evaluate_symbol(
            rng=rng,
            trees=trees,
            seq_length=seq_length,
            input_dimension=input_dimension,
            output_dimension=output_dimension,
            max_trials=max_trials,
            input_normalize=input_normalize,
            output_normalize=output_normalize,
            input_max_scale=input_max_scale,
            output_max_scale=output_max_scale,
            offset=offset,
            save_path=save_path,
        )

    def run(
        self,
        rng: np.random.RandomState,
        seq_length: int,
        input_dimension: int = 1,
        output_dimension: int = 1,
        max_trials: Optional[int] = None,
        input_normalize: Optional[str] = "z-score",
        output_normalize: Optional[str] = "z-score",
        input_max_scale: Optional[float] = 16.0,
        output_max_scale: Optional[float] = 16.0,
        offset: Optional[Tuple[float, float]] = None,
        save_path: Optional[str] = None,
    ) -> Union[Tuple[None, None, None], Tuple[NodeList, ndarray, ndarray]]:
        """
        Randomly generate a symbolic expression (complex system) and excitation time series.

        :param rng: The random number generator in NumPy with fixed seed.
        :param seq_length: The number of points of time series to generate.
        :param input_dimension: The number of the input dimensions of time series to generate.
        :param output_dimension: The number of the output dimensions of time series or symbol expression to generate.
        :param max_trials: The maximum number of trials to generate and try.
        :param input_normalize: Normalize the input time series, choice of 'z-score' or 'max-min' or None, defaults to 'z-score'.
        :param output_normalize: Normalize the output time series, choice of 'z-score' or 'max-min' or None, defaults to 'z-score'.
        :param input_max_scale: The scaling factor of the input time series to generate.
        :param output_max_scale: The scaling factor of the output time series to generate.
        :param offset: The offset mean and std for the input time series.
        :param save_path: The path to save the generated time series, defaults to None means not to save the data.
        """
        if self.print_status:
            return self.run_with_state(
                rng=rng,
                seq_length=seq_length,
                input_dimension=input_dimension,
                output_dimension=output_dimension,
                input_normalize=input_normalize,
                output_normalize=output_normalize,
                input_max_scale=input_max_scale,
                output_max_scale=output_max_scale,
                offset=offset,
                max_trials=max_trials,
                save_path=save_path,
            )

        trees, _, _ = self.generate_symbolic_expression(
            rng,
            input_dimension=input_dimension,
            output_dimension=output_dimension,
            return_all=False,
        )
        if trees is None:
            return None, None, None

        return self.evaluate_symbol(
            rng=rng,
            trees=trees,
            seq_length=seq_length,
            input_dimension=input_dimension,
            output_dimension=output_dimension,
            max_trials=max_trials,
            input_normalize=input_normalize,
            output_normalize=output_normalize,
            input_max_scale=input_max_scale,
            output_max_scale=output_max_scale,
            offset=offset,
            save_path=save_path,
        )


class CustomSymbolGenerator(object):
    """
    Dedicated interface for user-specified symbolic expressions (complex systems).

    Bind a fixed expression ``f(.)`` at construction time, then sample excitation
    series ``X`` via the excitation module and return the response ``Y = f(X)``.
    """

    def __init__(
        self,
        symbol: Union[str, Node, NodeList, List[str]],
        series_params: Optional[SeriesParams] = None,
        symbol_params: Optional[SymbolParams] = None,
        print_status: Optional[bool] = False,
        logging_path: Optional[str] = None,
        special_words: Optional[dict] = None,
    ) -> None:
        """
        :param symbol: User-specified symbolic expression (infix / prefix / Node / NodeList).
        :param series_params: Parameters controlling excitation time-series generation.
        :param symbol_params: Parameters attached to parsed expression nodes.
        :param print_status: Whether to print generation status.
        :param logging_path: Optional logging path used when ``print_status`` is True.
        :param special_words: Optional special words for the underlying generator.
        """
        self._generator = SeriesSymbolGenerator(
            series_params=series_params,
            symbol_params=symbol_params,
            print_status=print_status,
            logging_path=logging_path,
            special_words=special_words,
        )
        # Validate with precise exceptions before binding the expression.
        self.symbol = check_symbol(
            symbol,
            symbol_params=self._generator.symbol_params,
            require_variable=True,
        )
        self.input_dimension = infer_input_dimension(self.symbol)
        self.output_dimension = infer_output_dimension(self.symbol)

    @property
    def series_params(self) -> SeriesParams:
        return self._generator.series_params

    @property
    def symbol_params(self) -> SymbolParams:
        return self._generator.symbol_params

    @property
    def excitation(self) -> "Excitation":
        return self._generator.excitation

    def __str__(self) -> str:
        return f"CustomSymbolGenerator({self.symbol.infix()})"

    def __repr__(self) -> str:
        return str(self)

    def run(
        self,
        rng: np.random.RandomState,
        seq_length: int,
        input_dimension: Optional[int] = None,
        output_dimension: Optional[int] = None,
        max_trials: Optional[int] = None,
        input_normalize: Optional[str] = "z-score",
        output_normalize: Optional[str] = "z-score",
        input_max_scale: Optional[float] = 16.0,
        output_max_scale: Optional[float] = 16.0,
        offset: Optional[Tuple[float, float]] = None,
        save_path: Optional[str] = None,
    ) -> Union[Tuple[None, None, None], Tuple[NodeList, ndarray, ndarray]]:
        """
        Sample excitation with the excitation module and evaluate the bound symbol.

        :param rng: The random number generator in NumPy with fixed seed.
        :param seq_length: The number of points of time series to generate.
        :param input_dimension: Input channels; defaults to the dimension inferred
                                from the bound symbolic expression.
        :param output_dimension: Output channels; defaults to the bound expression
                                 output dimension.
        :param max_trials: The maximum number of trials to generate and try.
        :param input_normalize: Normalize the input time series.
        :param output_normalize: Normalize the output time series.
        :param input_max_scale: The scaling factor of the input time series.
        :param output_max_scale: The scaling factor of the output time series.
        :param offset: The offset mean and std for the input time series.
        :param save_path: Optional path to save the generated S2 data.
        :return: ``(symbol, excitation, response)`` or ``(None, None, None)`` on failure.
        """
        return self._generator.run_from_symbol(
            rng=rng,
            symbol=self.symbol,
            seq_length=seq_length,
            input_dimension=(
                self.input_dimension if input_dimension is None else input_dimension
            ),
            output_dimension=(
                self.output_dimension if output_dimension is None else output_dimension
            ),
            max_trials=max_trials,
            input_normalize=input_normalize,
            output_normalize=output_normalize,
            input_max_scale=input_max_scale,
            output_max_scale=output_max_scale,
            offset=offset,
            save_path=save_path,
        )


# Backward-compatible alias
Generator = SeriesSymbolGenerator


if __name__ == "__main__":
    generator = SeriesSymbolGenerator(print_status=True, logging_path="../data")

    rng = np.random.RandomState(0)
    trees, x, y = generator.run(
        rng, input_dimension=6, output_dimension=4, seq_length=1024
    )
    trees, x, y = generator.run(
        rng, input_dimension=6, output_dimension=4, seq_length=1024
    )
