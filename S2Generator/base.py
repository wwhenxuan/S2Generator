# -*- coding: utf-8 -*-
"""
Created on 2025/01/23 18:25:07
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com

Edited on 2025/08/09 16:51:36
@author:Yifan Wu
@email: wy3370868155@outlook.com
"""
import functools

import numpy as np
from numpy import ndarray
from numpy.random import RandomState
import scipy.special
from scipy.integrate import cumulative_trapezoid

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    DotProduct,
    ExpSineSquared,
    Kernel,
    RationalQuadratic,
    WhiteKernel,
)

from typing import Optional, Union, List

from S2Generator.params import Params

operators_real = {
    "add": 2,
    "sub": 2,
    "mul": 2,
    "div": 2,
    "abs": 1,
    "inv": 1,
    "sqrt": 1,
    "log": 1,
    "exp": 1,
    "sin": 1,
    "arcsin": 1,
    "cos": 1,
    "arccos": 1,
    "tan": 1,
    "arctan": 1,
    "pow2": 1,
    "pow3": 1,
}

operators_extra = {"pow": 2}

math_constants = ["e", "pi", "euler_gamma", "CONSTANT"]
all_operators = {**operators_real, **operators_extra}

SPECIAL_WORDS = [
    "<EOS>",
    "<X>",
    "</X>",
    "<Y>",
    "</Y>",
    "</POINTS>",
    "<INPUT_PAD>",
    "<OUTPUT_PAD>",
    "<PAD>",
    "(",
    ")",
    "SPECIAL",
    "OOD_unary_op",
    "OOD_binary_op",
    "OOD_constant",
]


class Node(object):
    """Generate a node in the sampling tree"""

    def __init__(
        self, value: Union[str, int], params: Params, children: list = None
    ) -> None:
        # The specific value stored in the current node
        self.value = value
        # The list of child nodes that the current node points to
        self.children = children if children else []
        self.params = params

    def push_child(self, child: "Node") -> None:
        """Add a child node to the current node"""
        self.children.append(child)

    def prefix(self) -> str:
        """Get all the contents of this tree using a recursive traversal starting from the current root node"""
        s = str(self.value)
        for c in self.children:
            s += "," + c.prefix()
        return s

    def qtree_prefix(self) -> str:
        """Get all the contents of this tree using a recursive traversal starting from the current root node, storing the result in a list"""
        s = "[.$" + str(self.value) + "$ "
        for c in self.children:
            s += c.qtree_prefix()
        s += "]"
        return s

    def infix(self) -> str:
        """Output the entire symbolic expression using in-order traversal"""
        nb_children = len(self.children)  # Get the number of children
        if nb_children == 0:
            # If there are no children, the current node is a leaf node
            if self.value.lstrip("-").isdigit():
                return str(self.value)
            else:
                s = str(self.value)
                return s  # Output the content of the leaf node
        if nb_children == 1:
            # If there is only one child, it indicates a unary operator
            s = str(self.value)
            # Handle different types of unary operators
            if s == "pow2":
                s = "(" + self.children[0].infix() + ")**2"
            elif s == "pow3":
                s = "(" + self.children[0].infix() + ")**3"
            else:
                # Output in the form of f(x), where f is functions like sin and cos
                s = s + "(" + self.children[0].infix() + ")"
            return s
        # If the current node is a binary operator, combine using the intermediate terms
        s = "(" + self.children[0].infix()
        for c in self.children[1:]:
            s = s + " " + str(self.value) + " " + c.infix()
        return s + ")"

    def val(self, x: ndarray, deterministic: Optional[bool] = True) -> ndarray:
        """Evaluate the symbolic expression using specific numerical sequences"""
        if len(self.children) == 0:
            # If the node is a leaf node, it is a symbolic variable or a random constant
            if str(self.value).startswith("x_"):
                # Handle symbolic expressions
                _, dim = self.value.split("_")
                dim = int(dim)
                return x[:, dim]
            elif str(self.value) == "rand":
                # Handle random constants
                if deterministic:
                    return np.zeros((x.shape[0],))
                return np.random.randn(x.shape[0])
            elif str(self.value) in math_constants:
                return getattr(np, str(self.value)) * np.ones((x.shape[0],))
            else:
                return float(self.value) * np.ones((x.shape[0],))

        # Handle various binary operators and perform specific calculations recursively
        if self.value == "add":
            return self.children[0].val(x) + self.children[1].val(x)  # Addition
        if self.value == "sub":
            return self.children[0].val(x) - self.children[1].val(x)  # Subtraction
        if self.value == "mul":
            m1, m2 = self.children[0].val(x), self.children[1].val(x)  # Multiplication
            # Handle exceptions in penalized calculations
            try:
                return m1 * m2
            except Exception as e:
                nans = np.empty((m1.shape[0],))
                nans[:] = np.nan
                return nans
        if self.value == "pow":
            m1, m2 = self.children[0].val(x), self.children[1].val(x)  # Exponentiation
            try:
                return np.power(m1, m2)
            except Exception as e:
                nans = np.empty((m1.shape[0],))
                nans[:] = np.nan
                return nans
        if self.value == "max":
            return np.maximum(
                self.children[0].val(x), self.children[1].val(x)
            )  # Maximum
        if self.value == "min":
            return np.minimum(
                self.children[0].val(x), self.children[1].val(x)
            )  # Minimum
        if self.value == "div":
            # Ensure denominator is not zero
            denominator = self.children[1].val(x)
            denominator[denominator == 0.0] = np.nan
            try:
                return self.children[0].val(x) / denominator  # Division
            except Exception as e:
                nans = np.empty((denominator.shape[0],))
                nans[:] = np.nan
                return nans

        # Handle various unary operators
        if self.value == "inv":
            # Ensure denominator is not zero
            denominator = self.children[0].val(x)
            denominator[denominator == 0.0] = np.nan
            try:
                return 1 / denominator  # Reciprocal
            except Exception as e:
                nans = np.empty((denominator.shape[0],))
                nans[:] = np.nan
                return nans
        if self.value == "log":
            numerator = self.children[0].val(x)
            # Ensure logarithm inputs are not negative or zero
            if self.params.use_abs:
                # Use log(abs(.)) if specified
                numerator[numerator <= 0.0] *= -1
            else:
                numerator[numerator <= 0.0] = np.nan
            try:
                return np.log(numerator)  # Logarithm
            except Exception as e:
                nans = np.empty((numerator.shape[0],))
                nans[:] = np.nan
                return nans
        if self.value == "sqrt":
            numerator = self.children[0].val(x)
            # Ensure square root inputs are non-negative
            if self.params.use_abs:
                # Apply absolute value if specified
                numerator[numerator <= 0.0] *= -1
            else:
                numerator[numerator < 0.0] = np.nan
            try:
                return np.sqrt(numerator)  # Square root
            except Exception as e:
                nans = np.empty((numerator.shape[0],))
                nans[:] = np.nan
                return nans
        if self.value == "pow2":
            numerator = self.children[0].val(x)
            try:
                return numerator**2  # Square
            except Exception as e:
                nans = np.empty((numerator.shape[0],))
                nans[:] = np.nan
                return nans
        if self.value == "pow3":
            numerator = self.children[0].val(x)
            try:
                return numerator**3  # Cube
            except Exception as e:
                nans = np.empty((numerator.shape[0],))
                nans[:] = np.nan
                return nans
        if self.value == "abs":
            return np.abs(self.children[0].val(x))  # Absolute value
        if self.value == "sign":
            return (self.children[0].val(x) >= 0) * 2.0 - 1.0  # Sign function
        if self.value == "step":
            x = self.children[0].val(x)  # Step function
            return x if x > 0 else 0
        if self.value == "id":
            return self.children[0].val(x)  # Identity mapping
        if self.value == "fresnel":
            return scipy.special.fresnel(self.children[0].val(x))[0]
        if self.value.startswith("eval"):
            n = self.value[-1]
            return getattr(scipy.special, self.value[:-1])(n, self.children[0].val(x))[
                0
            ]
        else:
            fn = getattr(np, self.value, None)
            if fn is not None:
                try:
                    return fn(self.children[0].val(x))
                except Exception as e:
                    nans = np.empty((x.shape[0],))
                    nans[:] = np.nan
                    return nans
            fn = getattr(scipy.special, self.value, None)
            if fn is not None:
                return fn(self.children[0].val(x))
            assert False, "Could not find function"

    def get_recurrence_degree(self) -> int:
        """Get the maximum variable index for leaf nodes when the current node is the root"""
        recurrence_degree = 0
        if len(self.children) == 0:
            # If the current node is a leaf node
            if str(self.value).startswith("x_"):
                _, offset = self.value.split("_")
                offset = int(offset)
                if offset > recurrence_degree:
                    recurrence_degree = offset
            return recurrence_degree
        return max([child.get_recurrence_degree() for child in self.children])

    def replace_node_value(self, old_value: str, new_value: str) -> None:
        """Traverse the entire symbolic expression and replace it with a specific value"""
        if self.value == old_value:
            self.value = new_value
        for child in self.children:
            child.replace_node_value(old_value, new_value)

    def __len__(self) -> int:
        """Output the total length of the expression with the current node as the root node"""
        lenc = 1
        for c in self.children:
            lenc += len(c)
        return lenc

    def __str__(self) -> str:
        # infix a default print
        return self.infix()

    def __repr__(self) -> str:
        # infix a default print
        return str(self)


class NodeList(object):
    """A list that stores the entire multivariate symbolic expression"""

    def __init__(self, nodes: List[Node]) -> None:
        self.nodes = []  # Initialize the list to store root nodes
        for node in nodes:
            self.nodes.append(node)
        self.params = nodes[0].params

    def infix(self) -> str:
        """Connect all multivariate symbolic expressions with |"""
        return " | ".join(
            [node.infix() for node in self.nodes]
        )  # In-order traversal of the tree

    def prefix(self) -> str:
        """Connect all multivariate symbolic expressions with ,|,"""
        return ",|,".join([node.prefix() for node in self.nodes])

    def val_router(
        self, xs: ndarray, deterministic: Optional[bool] = True, diff: Optional[int] = 0
    ) -> ndarray:
        if diff == 0:
            return self.val(xs, deterministic=deterministic)
        elif diff == 1:
            return self.val_diff(xs, deterministic=deterministic)
        else:
            raise ValueError(f"Unsupported diff value: {diff}")

    def val(self, xs: ndarray, deterministic: Optional[bool] = True) -> ndarray:
        """Sample the entire multivariate symbolic expression to obtain a specific numerical sequence"""
        batch_vals = [
            np.expand_dims(node.val(np.copy(xs), deterministic=deterministic), -1)
            for node in self.nodes
        ]
        return np.concatenate(batch_vals, -1)

    def val_diff(self, xs: ndarray, deterministic: Optional[bool] = True) -> ndarray:
        """Solve differential equation dy/dx = f(x) to get time series y(x)"""
        # Extract x values for integration
        x_values = xs[:, 0] if xs.ndim > 1 else xs

        if len(x_values) <= 1:
            solutions = np.zeros_like(
                self.val(xs, deterministic=deterministic), dtype=np.float64
            )
            return solutions

        # Create a uniform grid for integration from min to max of x_values
        x_min, x_max = np.min(x_values), np.max(x_values)

        # Always ensure the integration grid includes x=0 as the starting point
        grid_min = min(0.0, x_min)
        grid_max = max(0.0, x_max)

        integration_step = 0.001  # Adjust to your needs
        n_integration_points = max(100, int((grid_max - grid_min) / integration_step))
        x_uniform = np.linspace(grid_min, grid_max, n_integration_points)

        # Create input array for uniform grid evaluation
        if xs.ndim > 1:
            # For multivariate case, keep other dimensions constant
            x_uniform_input = np.tile(np.mean(xs, axis=0), (n_integration_points, 1))
            x_uniform_input[:, 0] = (
                x_uniform  # Replace first dimension with uniform grid
            )
        else:
            x_uniform_input = x_uniform.reshape(-1, 1)  # Ensure 2D array for val method

        # Evaluate the symbolic expressions on uniform grid to get f'(x)
        derivatives_uniform = self.val(x_uniform_input, deterministic=deterministic)

        # Initialize result array
        solutions = np.zeros(
            (len(x_values), derivatives_uniform.shape[1]), dtype=np.float64
        )

        # For each equation in the multivariate system
        for i in range(derivatives_uniform.shape[1]):
            f_x_uniform = derivatives_uniform[:, i]

            # Find the index corresponding to x=0 in the uniform grid
            zero_idx = np.argmin(np.abs(x_uniform - 0.0))

            # Split the integration: from x=0 to positive side and from x=0 to negative side
            integrated_uniform = np.zeros_like(x_uniform)

            # Integrate from x=0 to the right (positive direction)
            if zero_idx < len(x_uniform) - 1:
                x_right = x_uniform[zero_idx:]
                f_right = f_x_uniform[zero_idx:]
                integ_right = cumulative_trapezoid(f_right, x_right, initial=0.0)
                integrated_uniform[zero_idx:] = integ_right

            # Integrate from x=0 to the left (negative direction)
            if zero_idx > 0:
                x_left = x_uniform[: zero_idx + 1][
                    ::-1
                ]  # Reverse for integration from 0 to left
                f_left = f_x_uniform[: zero_idx + 1][::-1]
                integ_left = cumulative_trapezoid(f_left, x_left, initial=0.0)
                integrated_uniform[: zero_idx + 1] = -integ_left[
                    ::-1
                ]  # Reverse back and negate

            # Interpolate the integrated values to match the original x_values
            solutions[:, i] = np.interp(x_values, x_uniform, integrated_uniform)

        return solutions

    def replace_node_value(self, old_value: str, new_value: str) -> None:
        """Traverse the entire symbolic expression to replace a specific value"""
        for node in self.nodes:
            node.replace_node_value(old_value, new_value)

    def __len__(self) -> int:
        # Get the length of the entire multivariate symbolic expression
        return sum([len(node) for node in self.nodes])

    def __str__(self) -> str:
        """Output the multivariate symbolic expression in string form"""
        return self.infix()

    def __repr__(self) -> str:
        return str(self)


def get_kernel_bank(length: Optional[int] = 256):
    """Get all kernel in the bank list"""
    kernel_bank = [
        ExpSineSquared(periodicity=24 / length),  # H
        ExpSineSquared(periodicity=48 / length),  # 0.5H
        ExpSineSquared(periodicity=96 / length),  # 0.25H
        ExpSineSquared(periodicity=24 * 7 / length),  # H
        ExpSineSquared(periodicity=48 * 7 / length),  # 0.5H
        ExpSineSquared(periodicity=96 * 7 / length),  # 0.25H
        ExpSineSquared(periodicity=7 / length),  # D
        ExpSineSquared(periodicity=14 / length),  # 0.5D
        ExpSineSquared(periodicity=30 / length),  # D
        ExpSineSquared(periodicity=60 / length),  # 0.5D
        ExpSineSquared(periodicity=365 / length),  # D
        ExpSineSquared(periodicity=365 * 2 / length),  # 0.5D
        ExpSineSquared(periodicity=4 / length),  # W
        ExpSineSquared(periodicity=26 / length),  # W
        ExpSineSquared(periodicity=52 / length),  # W
        ExpSineSquared(periodicity=4 / length),  # M
        ExpSineSquared(periodicity=6 / length),  # M
        ExpSineSquared(periodicity=12 / length),  # M
        ExpSineSquared(periodicity=4 / length),  # Q
        ExpSineSquared(periodicity=4 * 10 / length),  # Q
        ExpSineSquared(periodicity=10 / length),  # Y
        DotProduct(sigma_0=0.0),
        DotProduct(sigma_0=1.0),
        DotProduct(sigma_0=10.0),
        RBF(length_scale=0.1),
        RBF(length_scale=1.0),
        RBF(length_scale=10.0),
        RationalQuadratic(alpha=0.1),
        RationalQuadratic(alpha=1.0),
        RationalQuadratic(alpha=10.0),
        WhiteKernel(noise_level=0.1),
        WhiteKernel(noise_level=1.0),
        ConstantKernel(),
    ]
    return kernel_bank


def random_binary_map(a: Kernel, b: Kernel) -> ndarray:
    """
    Applies a random binary operator (+ or *) with equal probability
    on kernels ``a`` and ``b``.
    :param a: A GP kernel
    :param b: A GP kernel
    :return: The composite kernel `a + b` or `a * b`.
    """
    binary_maps = [lambda x, y: x + y, lambda x, y: x * y]
    return np.random.choice(binary_maps)(a, b)


def sample_from_gp_prior(
    kernel: Kernel, X: ndarray, random_seed: Optional[int] = None
) -> ndarray:
    """
    Draw a sample from a GP prior.
    :param kernel: The GP covaraince kernel
    :param X: The input "time" points
    :param random_seed: The random seed for sampling, by default None
    :return: A time series sampled from the GP prior
    """
    if X.ndim == 1:
        X = X[:, None]

    assert X.ndim == 2
    gpr = GaussianProcessRegressor(kernel=kernel)
    ts = gpr.sample_y(X, n_samples=1, random_state=random_seed)

    return ts


def sample_from_gp_prior_efficient(
    kernel: Kernel,
    X: ndarray,
    random_seed: Optional[int] = None,
    method: str = "eigh",
) -> ndarray:
    """
    Draw a sample from a GP prior. An efficient version that allows specification
    of the sampling method. The default sampling method used in GaussianProcessRegressor
    is based on SVD which is significantly slower that alternatives such as `eigh` and
    `cholesky`.
    :param kernel: The GP covaraince kernel
    :param X: The input "time" points
    :param random_seed: The random seed for sampling, by default None
    :param method: The sampling method for multivariate_normal, by default `eigh`
    :return: A time series sampled from the GP prior
    """
    if X.ndim == 1:
        X = X[:, None]

    assert X.ndim == 2

    cov = kernel(X)
    ts = np.random.default_rng(seed=random_seed).multivariate_normal(
        mean=np.zeros(X.shape[0]), cov=cov, method=method
    )

    return ts


def generate_KernelSynth(
    rng: RandomState, max_kernels: Optional[int] = 5, length: Optional[int] = 256
) -> ndarray:
    """
    Generate a synthetic time series from KernelSynth.
    :param rng: Random Number Generator
    :param max_kernels: The maximum number of base kernels to use for each time series, by default 5
    :param length: The length of the time series, by default 256
    :return: A time series generated by KernelSynth
    """
    while True:
        X = np.linspace(0, 1, length)

        # Randomly select upto max_kernels kernels from the KERNEL_BANK
        selected_kernels = rng.choice(
            get_kernel_bank(length), rng.randint(1, max_kernels + 1), replace=True
        )

        # Combine the sampled kernels using random binary operators
        kernel = functools.reduce(random_binary_map, selected_kernels)

        # Sample a time series from the GP prior
        try:
            ts = sample_from_gp_prior(kernel=kernel, X=X)
        except np.linalg.LinAlgError as err:
            print("Error caught:", err)
            continue

        # The timestamp is arbitrary
        return ts.squeeze()


if __name__ == "__main__":
    data = generate_KernelSynth(RandomState(42))
    print(data.shape)

    data = np.vstack([generate_KernelSynth(RandomState(i)) for i in range(1)]).T
    print(data.shape)
