# -*- coding: utf-8 -*-
"""
TiRex-2: Structural Causal Model (SCM) Synthetic Prior for time series.

This module implements the TiRex-2 SCM prior (Podest et al., 2026, Section 2.5)
for generating causally structured multivariate synthetic time series:

1. **Sample hyperparameters**: number of nodes, graph type, etc.
2. **Sample DAG**: generate a directed acyclic graph using various algorithms
3. **Compute Dynamic SCM**: propagate values through the DAG with temporal
   noise processes as exogenous inputs to root nodes
4. **Extract dataset**: choose observed variables from SCM nodes
5. **Post-processing**: apply observational transforms

Components:
- DAG sampling algorithms (chain, fork, collider, random, scale-free, bipartite)
- Combiner mechanisms (linear, MLP, polynomial, multiplicative, periodic, maxmin)
- Temporal noise processes (iid, random walk, AR(1), periodic, OU)
- Activation bank (ReLU, GELU, softplus, high-frequency sin, ...)
- Post-processing (outliers, missing values, scale-shift)

Reference:
    Podest, P., et al. (2026). TiRex-2: Generalizing TiRex to Multivariate
    Data and Streaming. arXiv:2607.01204v1.

Created on 2026/08/11
@author: Ruizhe Wang
@email: changewam6@gmail.com
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ===========================================================================
# DAG Generation Algorithms (TiRex-2 Section 2.5, item 1)
# ===========================================================================
# Multiple algorithms for sampling directed acyclic graphs with diverse
# structural properties: chains, forks, colliders, random, and scale-free-like.
# ===========================================================================


def _dag_chain(rng: np.random.RandomState, V: int) -> Tuple[List[List[int]], List[int]]:
    """Generate a chain DAG: 1 → 2 → 3 → ... → V.

    Each node has exactly one parent (the previous node), except node 0
    which is the root.

    :param rng: Random number generator (unused; deterministic structure).
    :param V: Number of nodes.
    :return: Tuple of (parents_list, roots_list).
    """
    parents = [[] for _ in range(V)]
    for i in range(1, V):
        parents[i] = [i - 1]
    roots = [0]
    return parents, roots


def _dag_fork(rng: np.random.RandomState, V: int) -> Tuple[List[List[int]], List[int]]:
    """Generate a fork DAG: root → all other nodes.

    One root node is parent to all other nodes.

    :param rng: Random number generator.
    :param V: Number of nodes.
    :return: Tuple of (parents_list, roots_list).
    """
    parents = [[] for _ in range(V)]
    root = rng.randint(0, V)
    for i in range(V):
        if i != root:
            parents[i] = [root]
    roots = [root]
    return parents, roots


def _dag_collider(
    rng: np.random.RandomState, V: int
) -> Tuple[List[List[int]], List[int]]:
    """Generate a collider DAG: all nodes → target.

    Multiple root nodes all point to a single target node.

    :param rng: Random number generator.
    :param V: Number of nodes.
    :return: Tuple of (parents_list, roots_list).
    """
    parents = [[] for _ in range(V)]
    target = rng.randint(0, V)
    roots = [i for i in range(V) if i != target]
    parents[target] = roots.copy()
    return parents, roots


def _dag_random(
    rng: np.random.RandomState, V: int, Pmax: int
) -> Tuple[List[List[int]], List[int]]:
    """Generate a random DAG using topological ordering.

    Each node can have up to Pmax parents, chosen from nodes earlier in
    a random topological order. This is the default DAG sampler.

    :param rng: Random number generator.
    :param V: Number of nodes.
    :param Pmax: Maximum parents per node.
    :return: Tuple of (parents_list, roots_list).
    """
    order = rng.permutation(V)
    parents = [[] for _ in range(V)]

    for pos in range(1, V):
        node = order[pos]
        possible = order[:pos]
        if len(possible) == 0:
            continue
        n_p = rng.randint(0, min(Pmax, len(possible)) + 1)
        if n_p > 0:
            parents[node] = sorted(
                rng.choice(possible, size=n_p, replace=False).tolist()
            )

    roots = [i for i in range(V) if len(parents[i]) == 0]
    return parents, roots


def _dag_scale_free(
    rng: np.random.RandomState, V: int, Pmax: int
) -> Tuple[List[List[int]], List[int]]:
    """Generate a scale-free-like DAG using preferential attachment.

    Nodes are more likely to have parents that already have many children
    (high out-degree), creating hub nodes.

    :param rng: Random number generator.
    :param V: Number of nodes.
    :param Pmax: Maximum parents per node.
    :return: Tuple of (parents_list, roots_list).
    """
    order = rng.permutation(V)
    parents = [[] for _ in range(V)]
    out_degree = np.zeros(V, dtype=int)

    for pos in range(1, V):
        node = order[pos]
        possible = order[:pos]
        if len(possible) == 0:
            continue

        # Preferential attachment: probability proportional to (out_degree + 1)
        weights = out_degree[possible] + 1
        weights = weights / weights.sum()

        n_p = rng.randint(1, min(Pmax, len(possible)) + 1)
        selected = rng.choice(possible, size=n_p, replace=False, p=weights)
        parents[node] = sorted(selected.tolist())
        for p in selected:
            out_degree[p] += 1

    roots = [i for i in range(V) if len(parents[i]) == 0]
    return parents, roots


def _dag_bipartite(
    rng: np.random.RandomState, V: int
) -> Tuple[List[List[int]], List[int]]:
    """Generate a bipartite-like DAG with two layers.

    First half of nodes are roots; second half are children with random
    parents from the first half.

    :param rng: Random number generator.
    :param V: Number of nodes.
    :return: Tuple of (parents_list, roots_list).
    """
    parents = [[] for _ in range(V)]
    n_roots = max(1, V // 2)
    order = rng.permutation(V)
    roots = order[:n_roots].tolist()
    children = order[n_roots:].tolist()

    for child in children:
        n_p = rng.randint(1, min(3, n_roots) + 1)
        selected = rng.choice(roots, size=n_p, replace=False)
        parents[child] = sorted(selected.tolist())

    return parents, roots


# Map of DAG generation algorithm names to functions
DAG_GENERATORS: Dict[str, Callable] = {
    "chain": _dag_chain,
    "fork": _dag_fork,
    "collider": _dag_collider,
    "random": _dag_random,
    "scale_free": _dag_scale_free,
    "bipartite": _dag_bipartite,
}


# ===========================================================================
# Noise Processes for Root Nodes (TiRex-2 Section 2.5)
# ===========================================================================
# Generate temporal noise sequences for exogenous (root) variables.
# ===========================================================================


def _noise_iid(rng: np.random.RandomState, L: int, **kwargs) -> np.ndarray:
    """IID Gaussian noise: ε_t ~ N(0, σ²).

    :param rng: Random number generator.
    :param L: Sequence length.
    :param kwargs: Optional 'scale' (default: sampled from U(0.1, 2.0)).
    :return: Noise sequence of shape (L,).
    """
    scale = kwargs.get("scale", rng.uniform(0.1, 2.0))
    return rng.normal(0, scale, L).astype(np.float64)


def _noise_random_walk(rng: np.random.RandomState, L: int, **kwargs) -> np.ndarray:
    """Random walk noise: x_t = x_{t-1} + ε_t, ε_t ~ N(0, σ²).

    :param rng: Random number generator.
    :param L: Sequence length.
    :param kwargs: Optional 'scale' (default: sampled from U(0.01, 0.2)).
    :return: Noise sequence of shape (L,).
    """
    scale = kwargs.get("scale", rng.uniform(0.01, 0.2))
    innovations = rng.normal(0, scale, L)
    return np.cumsum(innovations).astype(np.float64)


def _noise_ar1(rng: np.random.RandomState, L: int, **kwargs) -> np.ndarray:
    """AR(1) process: x_t = φ·x_{t-1} + ε_t, ε_t ~ N(0, σ²).

    :param rng: Random number generator.
    :param L: Sequence length.
    :param kwargs: Optional 'phi' (default: U(-0.9, 0.9)),
                   'scale' (default: U(0.05, 0.5)).
    :return: Noise sequence of shape (L,).
    """
    phi = kwargs.get("phi", rng.uniform(-0.9, 0.9))
    scale = kwargs.get("scale", rng.uniform(0.05, 0.5))
    x = np.zeros(L, dtype=np.float64)
    noise = rng.normal(0, scale, L)
    for t in range(1, L):
        x[t] = phi * x[t - 1] + noise[t]
    return x


def _noise_periodic(rng: np.random.RandomState, L: int, **kwargs) -> np.ndarray:
    """Periodic/sinusoidal noise: x_t = A·sin(2π·f·t/L + φ) + noise.

    :param rng: Random number generator.
    :param L: Sequence length.
    :param kwargs: Optional 'amplitude' (default: U(0.5, 3.0)),
                   'freq' (default: U(1, 10)),
                   'phase' (default: U(0, 2π)),
                   'noise_scale' (default: U(0.01, 0.2)).
    :return: Noise sequence of shape (L,).
    """
    amp = kwargs.get("amplitude", rng.uniform(0.5, 3.0))
    freq = kwargs.get("freq", rng.uniform(1.0, 10.0))
    phase = kwargs.get("phase", rng.uniform(0, 2 * np.pi))
    noise_scale = kwargs.get("noise_scale", rng.uniform(0.01, 0.2))

    t = np.arange(L, dtype=np.float64)
    signal = amp * np.sin(2 * np.pi * freq * t / L + phase)
    signal += rng.normal(0, noise_scale, L)
    return signal.astype(np.float64)


def _noise_ou(rng: np.random.RandomState, L: int, **kwargs) -> np.ndarray:
    """Ornstein-Uhlenbeck process (mean-reverting): dx = θ·(μ - x)·dt + σ·dW.

    Discrete approximation: x_t = x_{t-1} + θ·(μ - x_{t-1})·Δt + σ·√Δt·ε_t

    :param rng: Random number generator.
    :param L: Sequence length.
    :param kwargs: Optional 'theta' (default: U(0.1, 2.0)),
                   'mu' (default: U(-1, 1)),
                   'sigma' (default: U(0.05, 0.5)).
    :return: Noise sequence of shape (L,).
    """
    theta = kwargs.get("theta", rng.uniform(0.1, 2.0))
    mu = kwargs.get("mu", rng.uniform(-1.0, 1.0))
    sigma = kwargs.get("sigma", rng.uniform(0.05, 0.5))
    dt = 1.0 / L

    x = np.zeros(L, dtype=np.float64)
    x[0] = mu + rng.normal(0, sigma)
    for t in range(1, L):
        dx = theta * (mu - x[t - 1]) * dt + sigma * np.sqrt(dt) * rng.normal(0, 1)
        x[t] = x[t - 1] + dx
    return x


# Map of noise process names to functions
NOISE_PROCESSES: Dict[str, Callable] = {
    "iid": _noise_iid,
    "random_walk": _noise_random_walk,
    "ar1": _noise_ar1,
    "periodic": _noise_periodic,
    "ou": _noise_ou,
}


# ===========================================================================
# Combiner Mechanisms (TiRex-2 Section 2.5, item 2)
# ===========================================================================
# Functions that aggregate parent node values into a scalar for the child node.
# Each combiner maps (parent_values, rng) -> scalar output.
# ===========================================================================


def _combiner_linear(
    rng: np.random.RandomState,
    parent_values: np.ndarray,  # shape (n_parents,)
) -> np.ndarray:
    """Linear combination: Σ w_i·x_i + b.

    w_i ~ U(-2, 2), b ~ U(-1, 1).

    :param rng: Random number generator.
    :param parent_values: Array of parent values, shape (n_parents,).
    :return: Scalar combined value.
    """
    n = len(parent_values)
    w = rng.uniform(-2.0, 2.0, n)
    b = rng.uniform(-1.0, 1.0)
    return np.dot(w, parent_values) + b


def _combiner_mlp(
    rng: np.random.RandomState,
    parent_values: np.ndarray,
) -> np.ndarray:
    """Neural-network-style combiner: W₂·σ(W₁·x + b₁) + b₂.

    Single hidden layer with tanh activation.

    :param rng: Random number generator.
    :param parent_values: Array of parent values, shape (n_parents,).
    :return: Scalar combined value.
    """
    n = len(parent_values)
    if n == 0:
        return np.float64(0.0)
    hidden_dim = rng.randint(2, max(3, min(8, n * 2)) + 1)

    W1 = rng.normal(0, 1, (hidden_dim, n)) / np.sqrt(n)
    b1 = rng.normal(0, 0.5, hidden_dim)
    W2 = rng.normal(0, 1, (1, hidden_dim)) / np.sqrt(hidden_dim)
    b2 = rng.normal(0, 0.5)

    h = np.tanh(W1 @ parent_values + b1)
    return (W2 @ h + b2).item()


def _combiner_polynomial(
    rng: np.random.RandomState,
    parent_values: np.ndarray,
) -> np.ndarray:
    """Polynomial combination with random 2nd-order interactions.

    Σ w_i·x_i + Σ v_ij·x_i·x_j + b

    :param rng: Random number generator.
    :param parent_values: Array of parent values, shape (n_parents,).
    :return: Scalar combined value.
    """
    n = len(parent_values)
    # Linear terms
    w = rng.uniform(-1.0, 1.0, n)
    result = np.dot(w, parent_values)

    # Pairwise interaction terms (random subset to avoid explosion)
    if n >= 2:
        n_interactions = min(n * (n - 1) // 2, 5)
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((i, j))
        rng.shuffle(pairs)
        for i, j in pairs[:n_interactions]:
            v = rng.uniform(-0.5, 0.5)
            result += v * parent_values[i] * parent_values[j]

    result += rng.uniform(-0.5, 0.5)  # bias
    return result


def _combiner_multiplicative(
    rng: np.random.RandomState,
    parent_values: np.ndarray,
) -> np.ndarray:
    """Multiplicative combination: Π |x_i|^α_i · sign(Σ x_i).

    :param rng: Random number generator.
    :param parent_values: Array of parent values, shape (n_parents,).
    :return: Scalar combined value.
    """
    n = len(parent_values)
    if n == 0:
        return np.float64(1.0)

    # Add small epsilon to avoid sign issues
    eps = 1e-8
    alphas = rng.uniform(0.5, 1.5, n)
    abs_vals = np.abs(parent_values) + eps
    product = np.prod(abs_vals**alphas)

    # Use sign of majority
    sign = np.sign(np.sum(parent_values))
    if sign == 0:
        sign = 1.0
    return sign * min(product, 100.0)  # Clip to avoid overflow


def _combiner_periodic(
    rng: np.random.RandomState,
    parent_values: np.ndarray,
) -> np.ndarray:
    """Periodic/sinusoidal combination: sin(Σ w_i·x_i + φ).

    :param rng: Random number generator.
    :param parent_values: Array of parent values, shape (n_parents,).
    :return: Scalar combined value.
    """
    n = len(parent_values)
    if n == 0:
        return np.float64(0.0)
    w = rng.uniform(0.5, 2.0, n)
    phase = rng.uniform(0, 2 * np.pi)
    return np.sin(np.dot(w, parent_values) + phase)


def _combiner_maxmin(
    rng: np.random.RandomState,
    parent_values: np.ndarray,
) -> np.ndarray:
    """Max-min combiner: max(x) - min(x) or max(x) * min(x).

    :param rng: Random number generator.
    :param parent_values: Array of parent values, shape (n_parents,).
    :return: Scalar combined value.
    """
    n = len(parent_values)
    if n == 0:
        return np.float64(0.0)
    if n == 1:
        return parent_values[0]

    mode = rng.choice(["diff", "prod", "avg"])
    if mode == "diff":
        return np.max(parent_values) - np.min(parent_values)
    elif mode == "prod":
        return np.max(parent_values) * np.min(parent_values)
    else:
        return (np.max(parent_values) + np.min(parent_values)) / 2.0


# Map of combiner names to functions
COMBINERS: Dict[str, Callable] = {
    "linear": _combiner_linear,
    "mlp": _combiner_mlp,
    "polynomial": _combiner_polynomial,
    "multiplicative": _combiner_multiplicative,
    "periodic": _combiner_periodic,
    "maxmin": _combiner_maxmin,
}


# ===========================================================================
# Activation Functions
# ===========================================================================


def _activation_relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation."""
    return np.maximum(0, x)


def _activation_sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _activation_tanh(x: np.ndarray) -> np.ndarray:
    """Tanh activation."""
    return np.tanh(x)


def _activation_sin(x: np.ndarray) -> np.ndarray:
    """Sinusoidal activation."""
    return np.sin(x)


def _activation_gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation (approximation)."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def _activation_softplus(x: np.ndarray) -> np.ndarray:
    """Softplus activation."""
    return np.log(1.0 + np.exp(np.clip(x, -50, 50)))


def _activation_identity(x: np.ndarray) -> np.ndarray:
    """Identity/passthrough."""
    return x


def _activation_high_freq_sin(
    x: np.ndarray, rng: Optional[np.random.RandomState] = None
) -> np.ndarray:
    """High-frequency sinusoidal activation (TiRex-2, item 4).

    sin(ω·x) with ω sampled for high-frequency behavior.

    :param x: Input array.
    :param rng: Random number generator for sampling frequency.
    :return: Transformed array.
    """
    if rng is None:
        omega = 5.0
    else:
        omega = rng.uniform(3.0, 20.0)
    return np.sin(omega * x)


# Map of activation names to functions
ACTIVATIONS: Dict[str, Callable] = {
    "relu": _activation_relu,
    "sigmoid": _activation_sigmoid,
    "tanh": _activation_tanh,
    "sin": _activation_sin,
    "gelu": _activation_gelu,
    "softplus": _activation_softplus,
    "identity": _activation_identity,
    "high_freq_sin": _activation_high_freq_sin,
}


# ===========================================================================
# Post-Processing for TiRex-2
# ===========================================================================


def _postprocess_add_outliers(
    rng: np.random.RandomState,
    series: np.ndarray,
    outlier_prob: float = 0.02,
    outlier_scale: float = 5.0,
) -> np.ndarray:
    """Add random outliers to the series.

    :param rng: Random number generator.
    :param series: Input series of shape (T, Q) or (Q, T).
    :param outlier_prob: Probability of a value being an outlier.
    :param outlier_scale: Scale of outlier magnitude.
    :return: Series with outliers.
    """
    result = series.copy()
    mask = rng.random(result.shape) < outlier_prob
    outliers = rng.normal(0, outlier_scale, result.shape)
    result[mask] = result[mask] + outliers[mask]
    return result


def _postprocess_add_missing(
    rng: np.random.RandomState,
    series: np.ndarray,
    missing_prob: float = 0.05,
) -> np.ndarray:
    """Add missing values (NaN) to the series.

    :param rng: Random number generator.
    :param series: Input series.
    :param missing_prob: Probability of a value being missing.
    :return: Series with NaN values.
    """
    result = series.copy()
    mask = rng.random(result.shape) < missing_prob
    result[mask] = np.nan
    return result


def _postprocess_scale_shift(
    rng: np.random.RandomState,
    series: np.ndarray,
) -> np.ndarray:
    """Apply random scaling and shifting per variate.

    :param rng: Random number generator.
    :param series: Input series of shape (Q, T).
    :return: Transformed series.
    """
    Q = series.shape[0]
    result = series.copy()
    for q in range(Q):
        scale = rng.uniform(0.5, 2.0)
        shift = rng.uniform(-2.0, 2.0)
        valid = ~np.isnan(result[q])
        result[q, valid] = result[q, valid] * scale + shift
    return result


# ===========================================================================
# TiRex-2 SCM Pipeline
# ===========================================================================


class TiRex2Pipeline:
    """TiRex-2 Synthetic Prior pipeline for time series generation.

    Implements the SCM prior described in TiRex-2 (Section 2.5) extended
    with temporal dynamics (Dynamic SCM, item 7) for generating causally
    structured multivariate time series.

    The pipeline proceeds in 5 steps (Figure 9):

    1. **Sample hyperparameters**: number of nodes, graph type, etc.
    2. **Sample DAG**: generate a DAG using one of several algorithms
       (chain, fork, collider, random, scale-free, bipartite).
    3. **Compute Dynamic SCM**: for each time step, generate root node
       values from noise processes, then propagate through the DAG using
       combiner mechanisms and activation functions.
    4. **Extract dataset**: select observed variables from SCM nodes.
    5. **Post-processing**: add outliers, missing values, scaling.
    """

    def __init__(
        self,
        Vmin: int = 3,
        Vmax: int = 20,
        Pmax: int = 4,
        dag_weights: Optional[Dict[str, float]] = None,
        combiner_weights: Optional[Dict[str, float]] = None,
        noise_weights: Optional[Dict[str, float]] = None,
        activation_weights: Optional[Dict[str, float]] = None,
        apply_postprocessing: bool = True,
        dtype: np.dtype = np.float64,
    ) -> None:
        """Initialize the TiRex-2 SCM pipeline.

        :param Vmin: Minimum number of nodes in the DAG.
        :param Vmax: Maximum number of nodes.
        :param Pmax: Maximum parents per node.
        :param dag_weights: Sampling weights for DAG algorithms.
                           Default: uniform over all algorithms.
        :param combiner_weights: Sampling weights for combiner mechanisms.
                                Default: uniform.
        :param noise_weights: Sampling weights for noise processes.
                             Default: uniform.
        :param activation_weights: Sampling weights for activations.
                                  Default: uniform.
        :param apply_postprocessing: Whether to apply post-processing.
        :param dtype: The numpy data type.
        """
        self._Vmin = Vmin
        self._Vmax = Vmax
        self._Pmax = Pmax
        self._apply_postprocessing = apply_postprocessing
        self._dtype = dtype

        # Default weights: uniform over all options
        self._dag_names = list(DAG_GENERATORS.keys())
        self._combiner_names = list(COMBINERS.keys())
        self._noise_names = list(NOISE_PROCESSES.keys())
        self._activation_names = list(ACTIVATIONS.keys())

        n_dags = len(self._dag_names)
        n_combiners = len(self._combiner_names)
        n_noises = len(self._noise_names)
        n_activations = len(self._activation_names)

        self._dag_probs = dag_weights or {
            name: 1.0 / n_dags for name in self._dag_names
        }
        self._combiner_probs = combiner_weights or {
            name: 1.0 / n_combiners for name in self._combiner_names
        }
        self._noise_probs = noise_weights or {
            name: 1.0 / n_noises for name in self._noise_names
        }
        self._activation_probs = activation_weights or {
            name: 1.0 / n_activations for name in self._activation_names
        }

    def __str__(self) -> str:
        return "TiRex2Pipeline"

    def _sample_dag(
        self, rng: np.random.RandomState, V: int
    ) -> Tuple[List[List[int]], List[int], List[Tuple[int, int]]]:
        """Sample a DAG using a randomly chosen algorithm.

        :param rng: Random number generator.
        :param V: Number of nodes.
        :return: Tuple of (parents_list, roots_list, edge_list).
        """
        dag_name = rng.choice(self._dag_names, p=self._get_probs(self._dag_probs))
        gen_fn = DAG_GENERATORS[dag_name]

        if dag_name in ("chain", "fork", "collider", "bipartite"):
            parents, roots = gen_fn(rng, V)
        else:
            parents, roots = gen_fn(rng, V, self._Pmax)

        # Build edge list
        edges = []
        for child in range(V):
            for parent in parents[child]:
                edges.append((parent, child))

        return parents, roots, edges

    def _get_probs(self, prob_dict: Dict[str, float]) -> np.ndarray:
        """Get normalized probability array from dict.

        :param prob_dict: Dict mapping name -> probability weight.
        :return: Normalized probability array (same order as dict keys).
        """
        names = list(prob_dict.keys())
        probs = np.array([prob_dict[n] for n in names], dtype=float)
        return probs / probs.sum()

    def _generate_noise(
        self,
        rng: np.random.RandomState,
        L: int,
    ) -> np.ndarray:
        """Generate a noise sequence using a randomly chosen process.

        :param rng: Random number generator.
        :param L: Sequence length.
        :return: Noise sequence of shape (L,).
        """
        name = rng.choice(self._noise_names, p=self._get_probs(self._noise_probs))
        return NOISE_PROCESSES[name](rng, L)

    def _sample_combiner(self, rng: np.random.RandomState) -> Callable:
        """Sample a combiner mechanism.

        :param rng: Random number generator.
        :return: Combiner function (parent_values, rng) -> scalar.
        """
        name = rng.choice(self._combiner_names, p=self._get_probs(self._combiner_probs))
        return COMBINERS[name]

    def _sample_activation(self, rng: np.random.RandomState) -> Callable:
        """Sample an activation function.

        :param rng: Random number generator.
        :return: Activation function.
        """
        name = rng.choice(
            self._activation_names,
            p=self._get_probs(self._activation_probs),
        )
        act_fn = ACTIVATIONS[name]
        if name == "high_freq_sin":
            # Wrap to pass rng
            return lambda x, r=rng: act_fn(x, r)
        return act_fn

    def generate(
        self,
        rng: np.random.RandomState,
        n_inputs_points: int,
        input_dimension: Optional[int] = None,
        return_metadata: bool = False,
    ) -> Any:
        """Generate a causally structured multivariate time series.

        :param rng: The random number generator with fixed seed.
        :param n_inputs_points: Length T of the time series.
        :param input_dimension: Number of observed variates d.
                               If None, randomly sampled.
        :param return_metadata: If True, also return metadata.
        :return: Generated time series of shape (d, T), or (d, T) plus
                 metadata dict if return_metadata is True.
        """
        T = n_inputs_points

        # Step 1: Sample hyperparameters
        V = rng.randint(self._Vmin, self._Vmax + 1)
        if input_dimension is None:
            d = rng.randint(1, min(13, V))
        else:
            d = input_dimension

        # Step 2: Sample DAG
        parents, roots, edges = self._sample_dag(rng, V)
        E = len(edges)

        # Ensure at least one root
        if len(roots) == 0:
            # Force node 0 as root
            roots = [0]
            for child in range(V):
                parents[child] = [p for p in parents[child] if p != 0]

        # Step 3: Compute Dynamic SCM
        # Each node gets a time series of length T
        node_series: Dict[int, np.ndarray] = {}

        # Assign noise processes to root nodes
        for r in roots:
            node_series[r] = self._generate_noise(rng, T)

        # Assign combiner + activation to each non-root node
        node_combiners: Dict[int, Callable] = {}
        node_activations: Dict[int, Callable] = {}
        for v in range(V):
            if v not in roots:
                node_combiners[v] = self._sample_combiner(rng)
                node_activations[v] = self._sample_activation(rng)

        # Topological order for propagation
        in_degree = [len(parents[i]) for i in range(V)]
        topo_order = []
        queue = [i for i in range(V) if in_degree[i] == 0]

        while queue:
            node = queue.pop(0)
            topo_order.append(node)
            for child in range(V):
                if node in parents[child]:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)

        # Propagate through each time step
        for v in topo_order:
            if v in node_series:
                continue  # Root node already generated

            Pv = parents[v]
            node_series[v] = np.zeros(T, dtype=np.float64)
            combiner_fn = node_combiners[v]
            act_fn = node_activations[v]

            for t in range(T):
                # Gather parent values at time t
                p_vals = np.array([node_series[u][t] for u in Pv])
                # Combine
                combined = combiner_fn(rng, p_vals)
                # Activate
                try:
                    node_series[v][t] = act_fn(combined)
                except TypeError:
                    # high_freq_sin or other activations that don't accept rng
                    node_series[v][t] = act_fn(combined)

        # Step 4: Extract dataset - select d observed nodes
        all_nodes = list(range(V))
        observed = rng.choice(all_nodes, size=d, replace=False)
        observed = sorted(
            observed.tolist() if hasattr(observed, "tolist") else list(observed)
        )

        # Stack into output: shape (d, T)
        x = np.stack([node_series[v] for v in observed], axis=0)

        # Step 5: Post-processing
        if self._apply_postprocessing:
            # Randomly apply post-processing transforms
            if rng.random() < 0.3:
                x = _postprocess_add_outliers(
                    rng,
                    x,
                    outlier_prob=rng.uniform(0.005, 0.05),
                    outlier_scale=rng.uniform(2.0, 10.0),
                )
            if rng.random() < 0.3:
                x = _postprocess_add_missing(
                    rng,
                    x,
                    missing_prob=rng.uniform(0.01, 0.1),
                )
            if rng.random() < 0.5:
                x = _postprocess_scale_shift(rng, x)

        metadata: Dict[str, Any] = {}
        if return_metadata:
            metadata = {
                "n_nodes": V,
                "n_observed": d,
                "n_roots": len(roots),
                "n_edges": E,
                "sequence_length": T,
                "observed_nodes": observed,
                "root_nodes": roots,
                "edge_list": [(p, c) for p, c in edges],
                "dag_type": "sampled",  # could track which DAG was used
            }

        if return_metadata:
            return x.astype(self._dtype), metadata
        return x.astype(self._dtype)

    def generate_batch(
        self,
        rng: np.random.RandomState,
        n_samples: int,
        n_inputs_points: int,
        input_dimension: Optional[int] = None,
    ) -> List[np.ndarray]:
        """Generate a batch of N synthetic time series.

        :param rng: The random number generator.
        :param n_samples: Number of samples to generate.
        :param n_inputs_points: Length of each time series.
        :param input_dimension: Number of observed variates per sample.
        :return: List of generated time series.
        """
        dataset = []
        for _ in range(n_samples):
            x = self.generate(
                rng=rng,
                n_inputs_points=n_inputs_points,
                input_dimension=input_dimension,
                return_metadata=False,
            )
            dataset.append(x)
        return dataset

    @property
    def dag_algorithms(self) -> List[str]:
        """Get the list of DAG generation algorithms."""
        return self._dag_names

    @property
    def combiner_mechanisms(self) -> List[str]:
        """Get the list of combiner mechanism names."""
        return self._combiner_names

    @property
    def noise_processes(self) -> List[str]:
        """Get the list of noise process names."""
        return self._noise_names

    @property
    def activations(self) -> List[str]:
        """Get the list of activation function names."""
        return self._activation_names
