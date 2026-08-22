# -*- coding: utf-8 -*-
"""
Nonlinear Structural Causal Model (SCM) coupling mechanism.

This mechanism extends linear SCM with nonlinear edge functions g_ij and an
optional multiplicative modulation gate h:

    x_j(t) = h(z_k(t - tau_k)) * sum_{i in pa(j)} g_ij(z_i(t - tau_{ij}))

This introduces state-dependent coupling as a proxy for threshold-driven and
regime-switching dynamics, characteristic of gated systems and many physical
and economic processes.

Reference:
    Podest, P., et al. (2026). TiRex-2: Generalizing TiRex to Multivariate Data
    and Streaming. arXiv:2607.01204v1, Section 3.4, mechanism 6 & Appendix F.

Created on 2026/08/10 00:00:00
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/S2Generator
"""

from typing import Callable, Optional, Tuple

import numpy as np

from .base_coupling import BaseCoupling
from .linear_scm import LinearSCM


class NonlinearSCM(BaseCoupling):
    """Nonlinear SCM with multiplicative modulation gate.

    x_j(t) = h(z_k(t - tau_k)) * sum_{i in pa(j)} g_ij(z_i(t - tau_{ij}))

    where g_ij are nonlinear edge functions and h is a multiplicative gate
    that enables state-dependent (regime-switching) coupling.
    """

    # Available nonlinearity types for edge functions and gates
    _NONLINEARITY_TYPES = [
        "tanh",
        "sigmoid",
        "relu",
        "sin",
        "softplus",
        "gelu",
    ]

    min_variates = 2
    min_length = 2

    def __init__(
        self,
        dtype: np.dtype = np.float64,
        max_lag: int = 5,
        edge_probability: float = 0.4,
        noise_std: float = 0.05,
        use_modulation_gate: Optional[bool] = None,
    ) -> None:
        """Initialize the nonlinear SCM coupling mechanism.

        :param dtype: The numpy data type for generated data.
        :param max_lag: Maximum lag for causal edges.
        :param edge_probability: Probability of an edge between any pair (i,j).
        :param noise_std: Standard deviation of additive noise.
        :param use_modulation_gate: Whether to use a multiplicative modulation gate.
                                   If None, randomly sampled per call.
        """
        super().__init__(dtype=dtype)
        self._max_lag = max_lag
        self._edge_probability = edge_probability
        self._noise_std = noise_std
        self._use_modulation_gate = use_modulation_gate

    def __str__(self) -> str:
        return "NonlinearSCM"

    @property
    def max_lag(self) -> int:
        """Get the maximum lag."""
        return self._max_lag

    @property
    def edge_probability(self) -> float:
        """Get the edge probability."""
        return self._edge_probability

    @property
    def nonlinearity_types(self) -> list:
        """Get the available nonlinearity types."""
        return self._NONLINEARITY_TYPES

    def couple(
        self,
        rng: np.random.RandomState,
        series: np.ndarray,
        adjacency: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Apply nonlinear SCM coupling to the input series.

        :param rng: The random number generator with fixed seed.
        :param series: Input univariate series of shape (T, Q).
        :param adjacency: Optional (Q, Q) binary graph where ``adjacency[i, j]``
                          is True if variate i is a parent of variate j. If None,
                          a random DAG is sampled.
        :param kwargs: Additional parameters.
        :return: Nonlinearly coupled multivariate series of shape (T, Q).
        """
        self._validate_series(
            series, min_variates=self.min_variates, min_length=self.min_length
        )

        T, Q = series.shape
        # Ensure max_lag >= 1 so that randint(1, max_lag + 1) stays valid even for
        # very short sequences (where T // 4 would otherwise be 0).
        max_lag = max(1, min(self._max_lag, T // 4))

        use_gate = (
            self._use_modulation_gate
            if self._use_modulation_gate is not None
            else rng.choice([True, False])
        )

        # Build the DAG over Q variates (user-specified or random)
        if adjacency is not None:
            adjacency, lags = self._dag_from_adjacency(rng, adjacency, Q, max_lag)
        else:
            adjacency, lags = self._generate_random_dag(rng, Q, max_lag)

        # Sample nonlinear edge functions
        edge_functions = {}
        for i in range(Q):
            for j in range(Q):
                if adjacency[i, j]:
                    edge_functions[(i, j)] = self._sample_nonlinearity(rng)

        # Sample modulation gate if used
        gate_function = None
        gate_source = None
        gate_lag = 0
        if use_gate and Q > 1:
            gate_source = rng.randint(0, Q)
            gate_function = self._sample_nonlinearity(rng)
            gate_lag = rng.randint(0, max_lag + 1)

        # Apply nonlinear SCM
        result = np.zeros((T, Q), dtype=self._data_type)

        for j in range(Q):
            # Start with noise
            result[:, j] = rng.normal(0, self._noise_std, T)

            # Sum contributions from parents through nonlinear edge functions
            parent_sum = np.zeros(T, dtype=self._data_type)
            for i in range(Q):
                if adjacency[i, j]:
                    tau = lags[i, j]
                    shifted = np.zeros(T, dtype=self._data_type)
                    if tau < T:
                        shifted[tau:] = series[: T - tau, i]
                    parent_sum += edge_functions[(i, j)](shifted)

            # Apply modulation gate
            if gate_function is not None and gate_source is not None:
                gate_input = np.zeros(T, dtype=self._data_type)
                if gate_lag < T:
                    gate_input[gate_lag:] = series[: T - gate_lag, gate_source]
                gate_values = gate_function(gate_input)
                # Scale gate to [0.1, 2.0] range for multiplicative modulation
                gate_values = 0.1 + 1.9 * (
                    (gate_values - gate_values.min())
                    / (gate_values.max() - gate_values.min() + 1e-8)
                )
                result[:, j] += gate_values * parent_sum
            else:
                result[:, j] += parent_sum

        return result

    def _generate_random_dag(
        self,
        rng: np.random.RandomState,
        Q: int,
        max_lag: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a random DAG with lag structure.

        :param rng: The random number generator.
        :param Q: Number of variates.
        :param max_lag: Maximum lag.
        :return: Tuple of (adjacency, lags), each shape (Q, Q).
        """
        adjacency = np.zeros((Q, Q), dtype=bool)
        lags = np.zeros((Q, Q), dtype=int)

        order = rng.permutation(Q)
        for a_idx in range(Q):
            for b_idx in range(a_idx + 1, Q):
                i = order[a_idx]
                j = order[b_idx]
                if rng.random() < self._edge_probability:
                    adjacency[i, j] = True
                    lags[i, j] = rng.randint(1, max_lag + 1)

        return adjacency, lags

    def _dag_from_adjacency(
        self,
        rng: np.random.RandomState,
        adjacency: np.ndarray,
        Q: int,
        max_lag: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build a lagged DAG from a user-supplied adjacency matrix.

        The supplied graph only fixes the parent structure; lags are still
        sampled per present edge.

        :param rng: The random number generator.
        :param adjacency: Binary (Q, Q) matrix, ``adjacency[i, j]`` = edge i -> j.
        :param Q: Number of variates (nodes).
        :param max_lag: Maximum lag for causal edges.
        :return: A tuple of (adjacency, lags), each (Q, Q).
        """
        adjacency = np.asarray(adjacency)
        if adjacency.shape != (Q, Q):
            raise ValueError(
                f"adjacency must be (Q, Q) = ({Q}, {Q}), got {adjacency.shape}"
            )
        adjacency = adjacency.astype(bool)
        lags = np.zeros((Q, Q), dtype=int)
        for i in range(Q):
            for j in range(Q):
                if adjacency[i, j]:
                    lags[i, j] = rng.randint(1, max_lag + 1)
        return adjacency, lags

    def _sample_nonlinearity(
        self, rng: np.random.RandomState
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Sample a random nonlinear activation function.

        :param rng: The random number generator.
        :return: A callable nonlinear function.
        """
        func_type = rng.choice(self._NONLINEARITY_TYPES)

        if func_type == "tanh":
            scale = rng.uniform(0.5, 3.0)
            return lambda x: np.tanh(scale * x)
        elif func_type == "sigmoid":
            scale = rng.uniform(0.5, 5.0)
            return lambda x: 1.0 / (1.0 + np.exp(-scale * x))
        elif func_type == "relu":
            return lambda x: np.maximum(0, x)
        elif func_type == "sin":
            scale = rng.uniform(0.5, 3.0)
            return lambda x: np.sin(scale * x)
        elif func_type == "softplus":
            return lambda x: np.log(1.0 + np.exp(x))
        elif func_type == "gelu":

            def gelu(x: np.ndarray) -> np.ndarray:
                return (
                    0.5
                    * x
                    * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
                )

            return gelu
        else:
            return lambda x: x
