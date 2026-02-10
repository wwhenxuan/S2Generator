# -*- coding: utf-8 -*-
"""
Created on 2025/01/23 18:25:07
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com

Edited on 2025/08/09 16:51:36
@author:Yifan Wu
@email: wy3370868155@outlook.com
"""
import numpy as np
from numpy import ndarray
import scipy.special
from typing import Optional, Union, List
from scipy.integrate import cumulative_trapezoid
from S2Generator.params import SymbolParams
from scipy.ndimage import gaussian_filter1d  # 用于平滑微分

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
    "diff": 1,
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
        self, value: Union[str, int], params: SymbolParams, children: list = None
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
        if self.value == "diff":
            child_vals = self.children[0].val(x)  # 获取子表达式的值
            # 使用数值差分法计算微分
            diff_vals = np.zeros_like(child_vals)
            diff_vals[:-1] = np.diff(child_vals)
            # 最后一个点使用向后差分
            diff_vals[-1] = diff_vals[-2]
            # 对结果进行轻微平滑以减少噪声
            diff_vals = gaussian_filter1d(diff_vals, sigma=0.5)
            return diff_vals
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
        self,
        xs: ndarray,
        deterministic: Optional[bool] = True,
    ) -> ndarray:
        if self.params.solve_diff == 0:
            return self.val(xs, deterministic=deterministic)
        elif self.params.solve_diff == 1:
            return self.val_diff(xs, deterministic=deterministic)
        else:
            raise ValueError(f"Unsupported diff value: {self.params.solve_diff}.")

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
                integrated_uniform[: zero_idx + 1] = integ_left[
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


#
# if __name__ == "__main__":
#     data = generate_KernelSynth(RandomState(42))
#     print(data.shape)
#
#     data = np.vstack([generate_KernelSynth(RandomState(i)) for i in range(1)]).T
#     print(data.shape)
