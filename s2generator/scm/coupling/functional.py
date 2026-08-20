# -*- coding: utf-8 -*-
"""
Functional coupling mechanism.

This mechanism implements direct pointwise functional dependencies between variates:
    x_j(t) = f_j(z_0(t)) + epsilon_j(t)

where f_j are randomly sampled transformations (monotone, compressive, discretizing,
or piecewise-linear). This represents deterministic covariate relationships such as
sensor redundancies, calendar features, or derived quantities.

Reference:
    Podest, P., et al. (2026). TiRex-2: Generalizing TiRex to Multivariate Data
    and Streaming. arXiv:2607.01204v1, Section 3.4, mechanism 2 & Appendix F.

Created on 2026/08/10 00:00:00
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/S2Generator
"""

from typing import Callable, Optional

import numpy as np

from .base_coupling import BaseCoupling


class FunctionalCoupling(BaseCoupling):
    """Functional coupling: x_j(t) = f_j(z_0(t)) + noise.

    Each output variate is a deterministic non-linear transformation of the first
    input series, plus additive noise. This models direct pointwise dependencies
    as in sensor redundancies.
    """

    # Available function types for random sampling
    _FUNCTION_TYPES = [
        "monotone",
        "compressive",
        "discretizing",
        "piecewise_linear",
        "polynomial",
    ]

    min_variates = 1
    min_length = 2

    def __init__(
        self,
        dtype: np.dtype = np.float64,
        noise_std: float = 0.05,
        function_types: Optional[list] = None,
    ) -> None:
        """Initialize the functional coupling mechanism.

        :param dtype: The numpy data type for generated data.
        :param noise_std: Standard deviation of additive noise epsilon_j.
        :param function_types: List of function types to sample from.
                              If None, all available types are used.
        """
        super().__init__(dtype=dtype)
        self._noise_std = noise_std
        self._function_types = (
            function_types
            if function_types is not None
            else self._FUNCTION_TYPES
        )

    def __str__(self) -> str:
        return "FunctionalCoupling"

    @property
    def noise_std(self) -> float:
        """Get the noise standard deviation."""
        return self._noise_std

    @property
    def function_types(self) -> list:
        """Get the available function types."""
        return self._function_types

    def couple(
        self,
        rng: np.random.RandomState,
        series: np.ndarray,
        noise_std: Optional[float] = None,
        **kwargs,
    ) -> np.ndarray:
        """Apply functional coupling to the input series.

        The first variate z_0 is used as the base, and each output variate x_j
        is a random transformation of z_0 plus additive Gaussian noise.

        :param rng: The random number generator with fixed seed.
        :param series: Input univariate series of shape (T, Q).
        :param noise_std: Override for the noise standard deviation.
        :param kwargs: Additional parameters.
        :return: Coupled multivariate series of shape (T, Q).
        """
        self._validate_series(series, min_variates=self.min_variates, min_length=self.min_length)

        T, Q = series.shape
        noise_std = noise_std if noise_std is not None else self._noise_std
        result = np.zeros((T, Q), dtype=self._data_type)

        # The first variate passes through unchanged (it is the base)
        result[:, 0] = series[:, 0]

        # Each subsequent variate is a random function of the first
        for j in range(1, Q):
            func_type = rng.choice(self._function_types)
            func = self._sample_function(rng, func_type)
            result[:, j] = func(series[:, 0])

            # Add noise
            if noise_std > 0:
                result[:, j] += rng.normal(0, noise_std, T)

        return result

    def _sample_function(
        self,
        rng: np.random.RandomState,
        func_type: str,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Sample a random transformation function of the specified type.

        :param rng: The random number generator.
        :param func_type: The type of function to sample.
        :return: A callable function that maps (T,) -> (T,).
        """
        if func_type == "monotone":
            return self._make_monotone(rng)
        elif func_type == "compressive":
            return self._make_compressive(rng)
        elif func_type == "discretizing":
            return self._make_discretizing(rng)
        elif func_type == "piecewise_linear":
            return self._make_piecewise_linear(rng)
        elif func_type == "polynomial":
            return self._make_polynomial(rng)
        else:
            return lambda x: x

    @staticmethod
    def _make_monotone(
        rng: np.random.RandomState,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Create a monotone function via sigmoid with random parameters.

        f(x) = a * sigmoid(b * (x - c)) + d * x
        """
        a = rng.uniform(0.5, 3.0)
        b = rng.uniform(0.5, 5.0)
        c = rng.uniform(-1.0, 1.0)
        d = rng.uniform(0.0, 1.0)

        def func(x: np.ndarray) -> np.ndarray:
            return a / (1.0 + np.exp(-b * (x - c))) + d * x

        return func

    @staticmethod
    def _make_compressive(
        rng: np.random.RandomState,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Create a compressive function using arcsinh or tanh.

        f(x) = a * arcsinh(b * x)   or   f(x) = a * tanh(b * x)
        """
        a = rng.uniform(0.5, 3.0)
        b = rng.uniform(0.5, 3.0)
        use_tanh = rng.choice([True, False])

        if use_tanh:

            def func(x: np.ndarray) -> np.ndarray:
                return a * np.tanh(b * x)
        else:

            def func(x: np.ndarray) -> np.ndarray:
                return a * np.arcsinh(b * x)

        return func

    @staticmethod
    def _make_discretizing(
        rng: np.random.RandomState,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Create a discretizing function via quantile binning.

        f(x) returns the bin-center values, producing a staircase-like
        transformation.
        """
        n_bins = rng.randint(3, 10)

        def func(x: np.ndarray) -> np.ndarray:
            # Normalize to [0, 1] via empirical CDF, then bin
            x_sorted = np.sort(x)
            bin_edges = np.linspace(0, 1, n_bins + 1)
            # Map to quantile bins
            bins = np.quantile(x, bin_edges)
            # Assign each value to its bin center
            indices = np.digitize(x, bins[1:-1], right=False)
            # Clip indices to valid range
            indices = np.clip(indices, 0, n_bins - 1)
            bin_centers = np.array([
                (bins[i] + bins[i + 1]) / 2 for i in range(n_bins)
            ])
            return bin_centers[indices]

        return func

    @staticmethod
    def _make_piecewise_linear(
        rng: np.random.RandomState,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Create a piecewise-linear function with random knot points."""
        n_knots = rng.randint(2, 6)

        def func(x: np.ndarray) -> np.ndarray:
            x_min, x_max = x.min(), x.max()
            knots_x = np.sort(rng.uniform(x_min, x_max, n_knots))
            knots_y = rng.uniform(-2.0, 2.0, n_knots)

            result = np.zeros_like(x, dtype=float)
            for i in range(n_knots - 1):
                mask = (x >= knots_x[i]) & (x <= knots_x[i + 1])
                frac = (x[mask] - knots_x[i]) / (
                    knots_x[i + 1] - knots_x[i] + 1e-8
                )
                result[mask] = (
                    knots_y[i] + frac * (knots_y[i + 1] - knots_y[i])
                )
            # Extrapolate beyond edges
            result[x < knots_x[0]] = knots_y[0]
            result[x > knots_x[-1]] = knots_y[-1]
            return result

        return func

    @staticmethod
    def _make_polynomial(
        rng: np.random.RandomState,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Create a polynomial function with random coefficients.

        f(x) = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3
        """
        degree = rng.randint(1, 5)
        coeffs = rng.uniform(-1.0, 1.0, degree + 1)

        def func(x: np.ndarray) -> np.ndarray:
            result = np.zeros_like(x, dtype=float)
            for d, c in enumerate(coeffs):
                result += c * (x**d)
            return result

        return func
