# -*- coding: utf-8 -*-
"""
Linear Structural Causal Model (SCM) coupling mechanism.

This mechanism introduces directed lagged dependencies via a random directed
acyclic graph (DAG):

    x_j(t) = sum_{i in pa(j)} alpha_{ij} * z_i(t - tau_{ij}) + epsilon_j(t)

where pa(j) are the parents of variate j in the DAG, tau_{ij} are sampled lags,
and alpha_{ij} are causal coefficients.

Reference:
    Podest, P., et al. (2026). TiRex-2: Generalizing TiRex to Multivariate Data
    and Streaming. arXiv:2607.01204v1, Section 3.4, mechanism 5 & Appendix F.

Created on 2026/08/10 00:00:00
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/S2Generator
"""

from typing import List, Optional, Tuple

import numpy as np

from .base_coupling import BaseCoupling


class LinearSCM(BaseCoupling):
    """Linear Structural Causal Model with lagged directed dependencies.

    Variates are connected via a random DAG with lagged edges, introducing
    directed lead-lag causal structure.
    """

    min_variates = 2
    min_length = 2

    def __init__(
        self,
        dtype: np.dtype = np.float64,
        max_lag: int = 5,
        edge_probability: float = 0.4,
        coefficient_range: Tuple[float, float] = (-1.0, 1.0),
        noise_std: float = 0.1,
    ) -> None:
        """Initialize the linear SCM coupling mechanism.

        :param dtype: The numpy data type for generated data.
        :param max_lag: Maximum lag for causal edges.
        :param edge_probability: Probability of an edge between any pair (i,j).
        :param coefficient_range: Range (min, max) for edge coefficients.
        :param noise_std: Standard deviation of additive noise epsilon_j.
        """
        super().__init__(dtype=dtype)
        self._max_lag = max_lag
        self._edge_probability = edge_probability
        self._coefficient_range = coefficient_range
        self._noise_std = noise_std

    def __str__(self) -> str:
        return "LinearSCM"

    @property
    def max_lag(self) -> int:
        """Get the maximum lag."""
        return self._max_lag

    @property
    def edge_probability(self) -> float:
        """Get the edge probability."""
        return self._edge_probability

    def couple(
        self,
        rng: np.random.RandomState,
        series: np.ndarray,
        adjacency: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Apply linear SCM coupling to the input series.

        :param rng: The random number generator with fixed seed.
        :param series: Input univariate series of shape (T, Q).
        :param adjacency: Optional (Q, Q) binary graph where ``adjacency[i, j]``
                          is True if variate i is a parent of variate j. If None,
                          a random DAG is sampled.
        :param kwargs: Additional parameters.
        :return: Causally coupled multivariate series of shape (T, Q).
        """
        self._validate_series(
            series, min_variates=self.min_variates, min_length=self.min_length
        )

        T, Q = series.shape
        # Ensure max_lag >= 1 so that randint(1, max_lag + 1) stays valid even for
        # very short sequences (where T // 4 would otherwise be 0).
        max_lag = max(1, min(self._max_lag, T // 4))

        # Build the DAG over Q variates (user-specified or random)
        if adjacency is not None:
            adjacency, lags, coefficients = self._dag_from_adjacency(
                rng, adjacency, Q, max_lag
            )
        else:
            adjacency, lags, coefficients = self._generate_random_dag(rng, Q, max_lag)

        # Initialize output with the input series scattered
        result = np.zeros((T, Q), dtype=self._data_type)

        # Apply causal structure
        for j in range(Q):
            # Start with noise
            result[:, j] = rng.normal(0, self._noise_std, T)

            # Add contributions from parents
            for i in range(Q):
                if adjacency[i, j]:
                    tau = lags[i, j]
                    alpha = coefficients[i, j]
                    # Shift series i by tau steps
                    shifted = np.zeros(T, dtype=self._data_type)
                    if tau < T:
                        shifted[tau:] = series[: T - tau, i]
                    result[:, j] += alpha * shifted

        return result

    def _generate_random_dag(
        self,
        rng: np.random.RandomState,
        Q: int,
        max_lag: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate a random directed acyclic graph over Q nodes.

        Ensures acyclicity by only allowing edges from lower-index to
        higher-index nodes after a random topological ordering.

        :param rng: The random number generator.
        :param Q: Number of variates (nodes).
        :param max_lag: Maximum lag for causal edges.
        :return: A tuple of (adjacency, lags, coefficients), each of shape (Q, Q).
        """
        adjacency = np.zeros((Q, Q), dtype=bool)
        lags = np.zeros((Q, Q), dtype=int)
        coefficients = np.zeros((Q, Q), dtype=float)

        # Generate a random topological ordering
        order = rng.permutation(Q)

        # Add edges respecting the topological order (a -> b where a < b in order)
        for a_idx in range(Q):
            for b_idx in range(a_idx + 1, Q):
                i = order[a_idx]  # potential parent
                j = order[b_idx]  # potential child
                if rng.random() < self._edge_probability:
                    adjacency[i, j] = True
                    lags[i, j] = rng.randint(1, max_lag + 1)
                    coefficients[i, j] = rng.uniform(*self._coefficient_range)

        return adjacency, lags, coefficients

    def _dag_from_adjacency(
        self,
        rng: np.random.RandomState,
        adjacency: np.ndarray,
        Q: int,
        max_lag: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build a lagged DAG from a user-supplied adjacency matrix.

        The supplied graph only fixes the parent structure; lags and edge
        coefficients are still sampled per present edge.

        :param rng: The random number generator.
        :param adjacency: Binary (Q, Q) matrix, ``adjacency[i, j]`` = edge i -> j.
        :param Q: Number of variates (nodes).
        :param max_lag: Maximum lag for causal edges.
        :return: A tuple of (adjacency, lags, coefficients), each (Q, Q).
        """
        adjacency = np.asarray(adjacency)
        if adjacency.shape != (Q, Q):
            raise ValueError(
                f"adjacency must be (Q, Q) = ({Q}, {Q}), got {adjacency.shape}"
            )
        adjacency = adjacency.astype(bool)
        lags = np.zeros((Q, Q), dtype=int)
        coefficients = np.zeros((Q, Q), dtype=float)
        for i in range(Q):
            for j in range(Q):
                if adjacency[i, j]:
                    lags[i, j] = rng.randint(1, max_lag + 1)
                    coefficients[i, j] = rng.uniform(*self._coefficient_range)
        return adjacency, lags, coefficients
