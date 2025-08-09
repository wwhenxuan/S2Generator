# -*- coding: utf-8 -*-
"""
Created on 2025/01/23 17:37:24
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import copy
import numpy as np
from numpy import ndarray
from numpy.random import RandomState
from collections import defaultdict
from scipy.stats import special_ortho_group

from typing import Optional, Union, Tuple, List

from S2Generator.params import Params
from S2Generator.base import Node, NodeList
from S2Generator.base import operators_real
from S2Generator.base import math_constants, all_operators, SPECIAL_WORDS
from S2Generator.base import generate_KernelSynth
from S2Generator.encoders import GeneralEncoder


class Generator(object):
    """Interface for constructing symbolic expressions and sampling time series"""

    def __init__(
        self, params: Optional[Params] = None, special_words: Optional[dict] = None
    ) -> None:
        self.params = Params() if params is None else params
        special_words = SPECIAL_WORDS if special_words is None else special_words
        self.prob_const = params.prob_const  # Probability to generate integer in leafs
        self.prob_rand = params.prob_rand  # Probability to generate n in leafs
        self.max_int = params.max_int  # Maximal integer in symbolic expressions
        self.min_binary_ops_per_dim = (
            params.min_binary_ops_per_dim
        )  # Min number of binary operators per input dimension
        self.max_binary_ops_per_dim = (
            params.max_binary_ops_per_dim
        )  # Max number of binary operators per input dimension
        self.min_unary_ops = params.min_unary_ops  # Min number of unary operators
        self.max_unary_ops = params.max_unary_ops  # Max number of unary operators
        # Maximum and minimum input dimensions
        self.min_output_dimension = params.min_output_dimension
        self.min_input_dimension = params.min_input_dimension
        self.max_input_dimension = params.max_input_dimension
        self.max_output_dimension = params.max_output_dimension
        # Maximum numerical range
        self.max_number = 10**params.max_exponent
        # Operators that can be used with copy
        self.operators = copy.deepcopy(operators_real)

        self.operators_dowsample_ratio = defaultdict(float)
        if params.operators_to_downsample != "":
            # Some invalid operations need to be removed, such as div0
            for operator in self.params.operators_to_downsample.split(","):
                operator, ratio = operator.split("_")
                # Specify the probability of certain expressions appearing here
                ratio = float(ratio)
                self.operators_dowsample_ratio[operator] = ratio

        if params.required_operators != "":
            # Specify the symbolic expressions to be removed
            self.required_operators = self.params.required_operators.split(",")
        else:
            self.required_operators = []

        if params.extra_binary_operators != "":
            # Additional binary operators
            self.extra_binary_operators = self.params.extra_binary_operators.split(",")
        else:
            self.extra_binary_operators = []

        if params.extra_unary_operators != "":
            # Additional unary operators
            self.extra_unary_operators = self.params.extra_unary_operators.split(",")
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
        if self.params.extra_constants is not None:
            self.extra_constants = self.params.extra_constants.split(",")
        else:
            self.extra_constants = []

        # Obtain the numerical encoder and symbol encoder
        self.general_encoder = GeneralEncoder(params, self.symbols, all_operators)
        # Encoder for input and output sequences
        self.float_encoder = self.general_encoder.float_encoder
        self.float_words = special_words + sorted(list(set(self.float_encoder.symbols)))
        # Encoder for symbolic expressions
        self.equation_encoder = self.general_encoder.equation_encoder
        self.equation_words = sorted(list(set(self.symbols)))
        self.equation_words = special_words + self.equation_words
        # breakpoint()
        self.n_used_dims = None
        self.decimals = (
            self.params.decimals
        )  # Number of decimal places for floating-point numbers in symbolic expressions
        # List of sampling methods
        self.num_type = len(self.sampling_type)

        # Model order when generating ARMA sequences
        self.p_min, self.p_max = params.p_min, params.p_max
        self.q_min, self.q_max = params.q_min, params.q_max

        self.rotate = params.rotate

    def generate_dist(self, max_ops: int) -> List:
        """
        `max_ops`: maximum number of operators
        Enumerate the number of possible unary-binary trees that can be generated from empty nodes.
        D[e][n] represents the number of different binary trees with n nodes that
        can be generated from e empty nodes, using the following recursion:
            D(n, 0) = 0
            D(0, e) = 1
            D(n, e) = D(n, e - 1) + p_1 * D(n- 1, e) + D(n - 1, e + 1)
        p1 =  if binary trees, 1 if unary binary
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
        mantissa = float(rng.choice(range(1, 10**self.params.float_precision)))
        if not exponent:
            # Determine whether to generate the exponent
            min_power = (
                -self.params.max_exponent_prefactor
                - (self.params.float_precision + 1) // 2
            )
            max_power = (
                self.params.max_exponent_prefactor
                - (self.params.float_precision + 1) // 2
            )
            exponent = rng.randint(min_power, max_power + 1)
        constant = sign * (mantissa * 10**exponent)  # Sign bit + mantissa + exponent
        return str(np.round(constant, decimals=self.decimals))  # Return as a string

    def generate_int(self, rng: RandomState) -> str:
        """Generate a valid random integer within a specified range"""
        return str(rng.choice(self.constants + self.extra_constants))

    def generate_leaf(self, rng: RandomState, input_dimension: int) -> str:
        """Generate a leaf node in the sampling expression"""
        if rng.rand() < self.prob_rand:
            return "rand"  # Generate a random number
        else:
            if self.n_used_dims < input_dimension:
                # When the number of used variables is less than the specified number
                dimension = self.n_used_dims
                self.n_used_dims += 1
                return f"x_{dimension}"
            else:
                # Generate an integer or a random symbolic variable
                draw = rng.rand()
                if draw < self.prob_const:
                    return self.generate_int(rng)
                else:
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

    def generate_tree(
        self, rng: RandomState, nb_binary_ops: int, input_dimension: int
    ) -> Node:
        """Function to generate a tree, which is essentially an expression"""
        self.n_used_dims = 0
        tree = Node(0, self.params)  # Initialize the first root node of the tree
        empty_nodes = [tree]
        next_en = 0
        nb_empty = 1  # Initially, there is only one empty node, which will gradually accumulate
        while nb_binary_ops > 0:
            # Sample to generate the basic framework of the tree; the basic framework is composed of binary operators
            next_pos, arity = self.sample_next_pos(rng, nb_empty, nb_binary_ops)
            next_en += next_pos
            op = self.generate_ops(rng, arity)
            empty_nodes[next_en].value = op
            for _ in range(arity):
                e = Node(0, self.params)
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

    def generate_multi_dimensional_tree(
        self,
        rng: RandomState,
        input_dimension: Optional[int] = None,
        output_dimension: Optional[int] = None,
        nb_unary_ops: Optional[int] = None,
        nb_binary_ops: Optional[int] = None,
        return_all: bool = False,
    ):
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
                    min_binary_ops, self.params.max_binary_ops_offset + max_binary_ops
                )
                for dim in range(output_dimension)
            ]  # Initialize for each dimension
        # If a specific number is provided, use that number for each dimension
        elif isinstance(nb_binary_ops, int):
            nb_binary_ops_to_use = [nb_binary_ops for _ in range(output_dimension)]
        # If it's not a number, it must be a list
        else:
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

        for i in range(
            output_dimension
        ):  # Iterate over the specified number of output dimensions to generate data
            # Generate a binary tree as the basic framework
            tree = self.generate_tree(rng, nb_binary_ops_to_use[i], input_dimension)
            # Insert unary operators into the binary tree
            tree = self.add_unaries(rng, tree, nb_unary_ops_to_use[i])
            # Adding constants
            if self.params.reduce_num_constants:
                tree = self.add_prefactors(rng, tree)
            else:
                # Apply affine transformations
                tree = self.add_linear_transformations(rng, tree, target=self.variables)
                tree = self.add_linear_transformations(rng, tree, target=self.unaries)
            trees.append(tree)  # Add to the specified storage list
        # Construct a data structure to store multi-dimensional symbolic expressions
        tree = NodeList(trees)

        if return_all is True:
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
                return self.generate_multi_dimensional_tree(
                    rng, input_dimension, output_dimension, nb_unary_ops, nb_binary_ops
                )

        if return_all is True:
            return (
                tree,
                input_dimension,
                output_dimension,
                nb_unary_ops_to_use,
                nb_binary_ops_to_use,
            )
        else:
            return tree, input_dimension, output_dimension

    def add_unaries(self, rng: RandomState, tree: Node, nb_unaries: int) -> Node:
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
        tree = self.equation_encoder.decode(prefix).nodes[
            0
        ]  # Decode using the symbol encoder
        return tree

    def _add_unaries(self, rng: RandomState, tree: Node) -> str:
        """Insert unary operators into a symbolic expression and get the traversal sequence"""
        # Get the specific value of the current node
        s = str(tree.value)
        for c in tree.children:
            # Ensure the depth of unary operators meets the requirements
            if len(c.prefix().split(",")) < self.params.max_unary_depth:
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
        Obtain the basic framework of a symbolic expression
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

    @property
    def sampling_type(self) -> List:
        """Sampling method for data generation"""
        return self.params.sampling_type

    def type_sampling(self, rng: RandomState, n: int) -> Tuple[List[str], dict]:
        """Identify three specific sampling methods"""
        indices = rng.randint(0, self.num_type, size=n)
        type_list = [self.sampling_type[i] for i in indices]
        type_dict = {key: type_list.count(key) for key in self.sampling_type}
        return type_list, type_dict

    def generate_stats(
        self, rng: RandomState, input_dimension: int, n_centroids: int
    ) -> Tuple[ndarray, ndarray, List[ndarray]]:
        """Generate parameters required for sampling from a mixture distribution"""
        means = rng.randn(
            n_centroids, input_dimension
        )  # Means of the mixture distribution
        covariances = rng.uniform(
            0, 1, size=(n_centroids, input_dimension)
        )  # Variances of the mixture distribution
        if self.rotate:
            rotations = [
                (
                    special_ortho_group.rvs(input_dimension)
                    if input_dimension > 1
                    else np.identity(1)
                )
                for i in range(n_centroids)
            ]
        else:
            rotations = [np.identity(input_dimension) for i in range(n_centroids)]
        return means, covariances, rotations

    def generate_gaussian(
        self,
        rng: RandomState,
        input_dimension: int,
        n_centroids: int,
        n_points_comp: ndarray,
    ) -> ndarray:
        """Generate sequences of specified dimensions and lengths using a Gaussian mixture distribution"""
        means, covariances, rotations = self.generate_stats(
            rng, input_dimension, n_centroids
        )
        return np.vstack(
            [
                rng.multivariate_normal(mean, np.diag(covariance), int(sample))
                @ rotation
                for (mean, covariance, rotation, sample) in zip(
                    means, covariances, rotations, n_points_comp
                )
            ]
        )

    def generate_uniform(
        self,
        rng: RandomState,
        input_dimension: int,
        n_centroids: int,
        n_points_comp: ndarray,
    ) -> ndarray:
        """Generate sequences of specified dimensions and lengths using a uniform mixture distribution"""
        means, covariances, rotations = self.generate_stats(
            rng, input_dimension, n_centroids
        )
        return np.vstack(
            [
                (
                    mean
                    + rng.uniform(-1, 1, size=(sample, input_dimension))
                    * np.sqrt(covariance)
                )
                @ rotation
                for (mean, covariance, rotation, sample) in zip(
                    means, covariances, rotations, n_points_comp
                )
            ]
        )

    def generate_ARMA(
        self, rng: RandomState, n_inputs_points: int, input_dimension: int = 1
    ) -> ndarray:
        """Generate ARMA stationary time series based on the specified input points and dimensions"""
        x = np.zeros(shape=(n_inputs_points, input_dimension))
        # Generate clusters with numerical explosion through a while loop
        d = 0
        while d < input_dimension:
            # Get the number of clusters k through a uniform distribution
            p = rng.randint(low=self.p_min, high=self.p_max)
            q = rng.randint(low=self.q_min, high=self.q_max)

            # Generate AR(p) parameters
            p_last = rng.uniform(-1, 1)
            p_former = rng.uniform(-1, 1, p - 1)
            P = np.append(p_former / np.sum(p_former) * (1 - p_last), p_last)
            # Generate MA(q) parameters
            Q = rng.uniform(-1, 1, q)

            output = self.ARMA(rng=rng, ts=x[:, d], P=P, Q=Q)
            if np.max(np.abs(output)) <= 256:
                x[:, d] = output
                d += 1
        return x

    @staticmethod
    def ARMA(rng, ts: ndarray, P: ndarray, Q: ndarray) -> ndarray:
        """Generate an ARMA process based on the specified parameters"""
        for index in range(len(ts)):
            # Get the previous p AR values
            index_p = max(0, index - len(P))
            p_vector = np.flip(ts[index_p:index])
            # Compute the dot product of p values and model parameters
            p_value = np.dot(p_vector, P[0 : len(p_vector)])
            # Generate q values through a white noise sequence
            q_value = np.dot(rng.randn(len(Q)), Q)
            sum_value = p_value + rng.randn(1) + q_value
            if sum_value > 1024:
                sum_value = q_value
            ts[index] = sum_value
        return ts

    def generate_KernelSynth(
        self, rng: RandomState, n_inputs_points: int, input_dimension: int = 1
    ) -> ndarray:
        """Generate a time series from KernelSynth, which comes from Chronos"""
        return np.vstack(
            [
                generate_KernelSynth(
                    rng=rng, max_kernels=self.params.max_kernels, length=n_inputs_points
                )
                for _ in range(input_dimension)
            ]
        ).T

    def get_rid(self, x: ndarray, y: ndarray) -> Tuple[ndarray, ndarray]:
        """Remove illegal values from the generated sequences"""

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

    def run(
        self,
        rng: RandomState,
        n_points: int,
        input_dimension: Optional[int] = 1,
        output_dimension: Optional[int] = 1,
        scale: Optional[int] = 1,
        max_trials: Optional[int] = None,
        rotate: Optional[bool] = False,
        offset: Tuple[float, float] = None,
        output_norm: Optional[bool] = False,
        diff: Optional[bool] = False,
    ) -> tuple[None, None, None] | tuple[NodeList, ndarray, ndarray]:
        """Generate sampling sequences using a mixture distribution"""
        # Obtain the generated symbolic expressions
        trees, _, _ = self.generate_multi_dimensional_tree(
            rng,
            input_dimension=input_dimension,
            output_dimension=output_dimension,
            return_all=False,
        )
        # Store the generated sequence data
        inputs, outputs = [], []
        # Get the sampling distribution order
        type_list, type_dict = self.type_sampling(rng, n=input_dimension)

        # Statistical parameters for mixture distribution sampling
        n_centroids = rng.randint(low=5, high=self.params.max_centroids)
        # Randomly generate the weight values for each distribution
        weights = rng.uniform(0, 1, size=(n_centroids,))
        weights /= np.sum(weights)
        n_points_comp = rng.multinomial(n_points, weights)

        if rotate is not False:
            self.rotate = rotate

        # Start generating data from the mixture distribution
        trials = 0  # Current number of attempts
        max_trials = self.params.max_trials if max_trials is None else max_trials
        remaining_points = n_points  # Target length for sampling
        while remaining_points > 0 and trials < max_trials:
            # Sample using a Gaussian distribution
            x = []
            for sampling_type in type_list:
                if sampling_type == "gaussian":
                    # Sample using a Gaussian mixture distribution
                    x.append(
                        self.generate_gaussian(
                            rng=rng,
                            input_dimension=1,
                            n_centroids=n_centroids,
                            n_points_comp=n_points_comp,
                        )
                    )
                elif sampling_type == "uniform":
                    # Sample using a uniform mixture distribution
                    x.append(
                        self.generate_uniform(
                            rng=rng,
                            input_dimension=1,
                            n_centroids=n_centroids,
                            n_points_comp=n_points_comp,
                        )
                    )
                elif sampling_type == "ARMA":
                    # Sample using forward propagation of the ARMA model
                    x.append(
                        self.generate_ARMA(
                            rng=rng, n_inputs_points=n_points, input_dimension=1
                        )
                    )
                elif sampling_type == "KernelSynth":
                    # Sample using KernelSynth from Chronos
                    x.append(
                        self.generate_KernelSynth(
                            rng=rng, input_dimension=1, n_inputs_points=n_points
                        )
                    )
                else:
                    raise ValueError("Unknown sampling type!")

            # Standardize the multi-channels sequences obtained from sampling
            x = np.hstack(x)
            x = (x - np.mean(x, axis=0, keepdims=True)) / np.std(
                x, axis=0, keepdims=True
            )
            x *= scale
            # Add the specified distribution bias to the sampling sequence
            if offset is not None:
                mean, std = offset
                x *= std
                x += mean

            # Sample using the generated symbolic expressions
            y = trees.val_router(x, diff=diff)

            x, y = self.get_rid(x, y)

            # Number of valid values successfully retained in this sampling
            valid_points = y.shape[0]
            # Number of attempts this time
            trials += 1
            # Number of values still needed to be sampled
            remaining_points -= valid_points
            if valid_points == 0:
                continue
            inputs.append(x)
            outputs.append(y)

        if remaining_points > 0:
            # Sampling failed
            return None, None, None

        # Combine the results of all sampling attempts
        inputs = np.concatenate(inputs, axis=0)[:n_points]
        outputs = np.concatenate(outputs, axis=0)[:n_points]

        # Whether to normalize the output y
        if output_norm is True:
            outputs = (outputs - np.mean(outputs, axis=0, keepdims=True)) / np.std(
                outputs, axis=0, keepdims=True
            )

        return trees, inputs, outputs
