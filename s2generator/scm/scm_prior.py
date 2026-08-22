# -*- coding: utf-8 -*-
"""
Structural Causal Model (SCM) Synthetic Prior for tabular data.

This module implements the TabPFN-3 SCM prior (Prior Labs Team, 2026, Section 2.5)
for generating causally structured synthetic **tabular** datasets
(``N`` rows x ``P`` features + optional categorical target ``y``):

1. **Sample hyperparameters**: number of rows/features/classes, graph type, etc.
2. **Sample DAG**: generate a directed acyclic graph using various algorithms
3. **Compute SCM**: propagate values through the DAG in topological order with
   i.i.d. noise samples per node as exogenous inputs to root nodes
4. **Extract dataset**: choose observed features (X) and a target (Y) from SCM nodes
5. **Post-processing**: apply observational transforms

Components:
- DAG sampling algorithms (chain, fork, collider, random, scale-free, bipartite)
- Combiner mechanisms (linear, MLP, polynomial, multiplicative, periodic, maxmin)
- IID Gaussian noise for exogenous (root) nodes
- Activation bank (ReLU, GELU, softplus, high-frequency sin, ...)
- Categorical variables (quantile binning of feature columns)
- Many-class target (quantile binning of a target node)
- Post-processing (outliers, missing values, scale-shift)

Reference:
    Prior Labs Team (2026). TabPFN-3: Technical Report.
    arXiv:2605.13986v2.

Created on 2026/08/11
@author: Ruizhe Wang
@email: changewam6@gmail.com
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from ..utils._dag import adjacency_to_dag


# ===========================================================================
# DAG Generation Algorithms (TabPFN-3 Section 2.5, item 1)
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
# Noise for Root Nodes (TabPFN-3 Section 2.5, Figure 9)
# ===========================================================================
# An i.i.d. noise sample ε_i is drawn per node; root (exogenous) nodes are
# filled directly with such i.i.d. Gaussian samples over the N rows.
# ===========================================================================


def _noise_iid(rng: np.random.RandomState, L: int, **kwargs) -> np.ndarray:
    """IID Gaussian noise: ε_n ~ N(0, σ²), one draw per row.

    :param rng: Random number generator.
    :param L: Number of samples (rows) to generate.
    :param kwargs: Optional 'scale' (default: sampled from U(0.1, 2.0)).
    :return: Noise array of shape (L,).
    """
    scale = kwargs.get("scale", rng.uniform(0.1, 2.0))
    return rng.normal(0, scale, L).astype(np.float64)


# ===========================================================================
# Combiner Mechanisms (TabPFN-3 Section 2.5, item 2)
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
    x: np.ndarray,
    rng: Optional[np.random.RandomState] = None,
    omega: Optional[float] = None,
) -> np.ndarray:
    """High-frequency sinusoidal activation (TabPFN-3, item 4).

    sin(ω·x) with ω sampled for high-frequency behavior.

    :param x: Input array.
    :param rng: Random number generator for sampling frequency (used only when
                ``omega`` is not given).
    :param omega: Fixed angular frequency. If None, one is sampled (5.0 when
                  ``rng`` is also None, otherwise U(3, 20)).
    :return: Transformed array.
    """
    if omega is None:
        omega = rng.uniform(3.0, 20.0) if rng is not None else 5.0
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
# Post-Processing
# ===========================================================================


def _postprocess_add_outliers(
    rng: np.random.RandomState,
    matrix: np.ndarray,
    outlier_prob: float = 0.02,
    outlier_scale: float = 5.0,
) -> np.ndarray:
    """Add random outliers to the feature matrix.

    :param rng: Random number generator.
    :param matrix: Input matrix of shape (N, P).
    :param outlier_prob: Probability of a value being an outlier.
    :param outlier_scale: Scale of outlier magnitude.
    :return: Matrix with outliers.
    """
    result = matrix.copy()
    mask = rng.random(result.shape) < outlier_prob
    outliers = rng.normal(0, outlier_scale, result.shape)
    result[mask] = result[mask] + outliers[mask]
    return result


def _postprocess_add_missing(
    rng: np.random.RandomState,
    matrix: np.ndarray,
    missing_prob: float = 0.05,
) -> np.ndarray:
    """Add missing values (NaN) to the feature matrix.

    :param rng: Random number generator.
    :param matrix: Input matrix of shape (N, P).
    :param missing_prob: Probability of a value being missing.
    :return: Matrix with NaN values.
    """
    result = matrix.copy()
    mask = rng.random(result.shape) < missing_prob
    result[mask] = np.nan
    return result


def _postprocess_scale_shift(
    rng: np.random.RandomState,
    matrix: np.ndarray,
) -> np.ndarray:
    """Apply random scaling and shifting per feature (column).

    :param rng: Random number generator.
    :param matrix: Input matrix of shape (N, P).
    :return: Transformed matrix.
    """
    P = matrix.shape[1]
    result = matrix.copy()
    for p in range(P):
        scale = rng.uniform(0.5, 2.0)
        shift = rng.uniform(-2.0, 2.0)
        valid = ~np.isnan(result[:, p])
        result[valid, p] = result[valid, p] * scale + shift
    return result


# ===========================================================================
# Target Discretization and Categorical Binning
# ===========================================================================


def _discretize_target(
    z: np.ndarray,
    C: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Discretize a continuous target node into C classes via quantile binning.

    This is the many-class target mechanism (TabPFN-3, item 6): a continuous
    SCM node is cut at its ``1/C, ..., (C-1)/C`` quantiles, yielding naturally
    balanced classes whose decision boundaries are inherited from the causal
    structure of the node (cf. Figure 25).

    :param z: Continuous target values of shape (N,).
    :param C: Number of classes (>= 2).
    :param rng: Random number generator (unused; kept for a uniform signature).
    :return: Integer class labels of shape (N,), values in [0, C-1].
    """
    z = np.asarray(z, dtype=np.float64)
    qs = np.quantile(z, np.arange(1, C) / C)
    y = np.digitize(z, qs)
    return np.clip(y, 0, C - 1).astype(np.int64)


def _bin_categorical(
    values: np.ndarray,
    k: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Bin a continuous feature column into k ordered categorical levels.

    NaN values are preserved as NaN. This implements the base version of the
    categorical-variable treatment (TabPFN-3, item 3).

    :param values: Continuous feature values of shape (N,).
    :param k: Number of categorical levels (>= 2).
    :param rng: Random number generator (unused; kept for a uniform signature).
    :return: Integer-valued column of shape (N,) with NaN preserved.
    """
    values = np.asarray(values, dtype=np.float64)
    result = np.full(values.shape, np.nan, dtype=np.float64)
    valid = ~np.isnan(values)
    if valid.sum() == 0:
        return result
    v = values[valid]
    qs = np.quantile(v, np.arange(1, k) / k)
    bins = np.digitize(v, qs)
    result[valid] = np.clip(bins, 0, k - 1)
    return result


# ===========================================================================
# SCM Prior Pipeline
# ===========================================================================


class SCMPriorPipeline(object):
    """TabPFN-3 SCM prior pipeline for tabular data generation.

    Implements the SCM prior described in TabPFN-3 (Section 2.5) for
    generating causally structured synthetic tabular datasets: a feature
    matrix ``X`` of shape ``(N, P)`` and, optionally, a categorical target
    ``y`` of shape ``(N,)``.

    The pipeline proceeds in 5 steps (Figure 9):

    1. **Sample hyperparameters**: number of rows/features/classes, graph type.
    2. **Sample DAG**: generate a DAG using one of several algorithms
       (chain, fork, collider, random, scale-free, bipartite).
    3. **Compute SCM**: fill root nodes with i.i.d. Gaussian noise samples,
       then propagate through the DAG in topological order using combiner
       mechanisms, activations, and a per-node additive noise ε_v per the
       structural equation X_v = f_v(pa(X_v)) + ε_v.
    4. **Extract dataset**: choose observed features (X) and a target (Y)
       from the SCM nodes; the target is discretized into C classes.
    5. **Post-processing**: add outliers, missing values, scaling.
    """

    def __init__(
        self,
        Vmin: int = 3,
        Vmax: int = 20,
        Pmax: int = 4,
        Nmin: int = 32,
        Nmax: int = 512,
        dag_weights: Optional[Dict[str, float]] = None,
        combiner_weights: Optional[Dict[str, float]] = None,
        activation_weights: Optional[Dict[str, float]] = None,
        apply_postprocessing: bool = True,
        dtype: np.dtype = np.float64,
        noise_std: float = 0.1,
        categorical_prob: float = 0.3,
    ) -> None:
        """Initialize the SCM prior pipeline.

        :param Vmin: Minimum number of nodes in the DAG.
        :param Vmax: Maximum number of nodes.
        :param Pmax: Maximum parents per node.
        :param Nmin: Minimum number of rows when n_samples is not specified.
        :param Nmax: Maximum number of rows when n_samples is not specified.
        :param dag_weights: Sampling weights for DAG algorithms.
                           Default: uniform over all algorithms.
        :param combiner_weights: Sampling weights for combiner mechanisms.
                                Default: uniform.
        :param activation_weights: Sampling weights for activations.
                                  Default: uniform.
        :param apply_postprocessing: Whether to apply post-processing.
        :param dtype: The numpy data type for the feature matrix.
        :param noise_std: Standard deviation of the per-node additive noise ε_v
                          in the structural equation X_v = f_v(pa(X_v)) + ε_v
                          (TabPFN-3 Section 2.5, Figure 9). Set to 0.0 to
                          disable and recover deterministic non-root nodes.
        :param categorical_prob: Probability that an extracted feature column
                                 is binned into categorical levels.
        """
        self._Vmin = Vmin
        self._Vmax = Vmax
        self._Pmax = Pmax
        self._Nmin = Nmin
        self._Nmax = Nmax
        self._apply_postprocessing = apply_postprocessing
        self._dtype = dtype
        self._noise_std = noise_std
        self._categorical_prob = categorical_prob

        # Default weights: uniform over all options
        self._dag_names = list(DAG_GENERATORS.keys())
        self._combiner_names = list(COMBINERS.keys())
        self._activation_names = list(ACTIVATIONS.keys())

        n_dags = len(self._dag_names)
        n_combiners = len(self._combiner_names)
        n_activations = len(self._activation_names)

        self._dag_probs = dag_weights or {
            name: 1.0 / n_dags for name in self._dag_names
        }
        self._combiner_probs = combiner_weights or {
            name: 1.0 / n_combiners for name in self._combiner_names
        }
        self._activation_probs = activation_weights or {
            name: 1.0 / n_activations for name in self._activation_names
        }

    def __str__(self) -> str:
        return "SCMPriorPipeline"

    def _sample_dag(
        self, rng: np.random.RandomState, V: int
    ) -> Tuple[List[List[int]], List[int], List[Tuple[int, int]]]:
        """Sample a DAG using a randomly chosen algorithm.

        :param rng: Random number generator.
        :param V: Number of nodes.
        :return: Tuple of (parents_list, roots_list, edge_list).
        """
        dag_name = rng.choice(
            self._dag_names, p=self._get_probs(self._dag_probs, self._dag_names)
        )
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

    def _get_probs(self, prob_dict: Dict[str, float], names: List[str]) -> np.ndarray:
        """Get normalized probability array from a weight dict.

        :param prob_dict: Dict mapping name -> probability weight. Names absent
                          from the dict are assigned weight 0.
        :param names: Canonical ordering of names (the full option list).
        :return: Normalized probability array aligned with ``names``.
        :raises ValueError: If the total weight is not positive.
        """
        probs = np.array([prob_dict.get(n, 0.0) for n in names], dtype=float)
        total = probs.sum()
        if total <= 0.0:
            raise ValueError("sampling weights must sum to a positive value")
        return probs / total

    def _sample_combiner(self, rng: np.random.RandomState) -> Callable:
        """Sample a combiner mechanism.

        :param rng: Random number generator.
        :return: Combiner function (parent_values, rng) -> scalar.
        """
        name = rng.choice(
            self._combiner_names,
            p=self._get_probs(self._combiner_probs, self._combiner_names),
        )
        return COMBINERS[name]

    def _sample_activation(self, rng: np.random.RandomState) -> Callable:
        """Sample an activation function.

        :param rng: Random number generator.
        :return: Activation function.
        """
        name = rng.choice(
            self._activation_names,
            p=self._get_probs(self._activation_probs, self._activation_names),
        )
        act_fn = ACTIVATIONS[name]
        if name == "high_freq_sin":
            # Sample omega once per node so the activation is a deterministic
            # function of its input. Resampling per row would break the causal
            # semantics and silently consume extra RNG draws.
            omega = rng.uniform(3.0, 20.0)
            return lambda x, o=omega: act_fn(x, omega=o)
        return act_fn

    def generate(
        self,
        rng: np.random.RandomState,
        n_samples: Optional[int] = None,
        n_features: Optional[int] = None,
        n_classes: Optional[int] = None,
        adjacency: Optional[np.ndarray] = None,
        return_metadata: bool = False,
    ) -> Union[
        np.ndarray,
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, Dict[str, Any]],
        Tuple[np.ndarray, np.ndarray, Dict[str, Any]],
    ]:
        """Generate a causally structured synthetic tabular dataset.

        :param rng: The random number generator with fixed seed.
        :param n_samples: Number of rows N. If None, sampled from
                          [Nmin, Nmax].
        :param n_features: Number of features P. If None, randomly sampled.
        :param n_classes: Number of target classes C. If None, no target is
                          generated and only the feature matrix X is returned.
        :param adjacency: Optional binary adjacency matrix of shape (V, V)
                          describing a DAG. If provided, it is used instead of
                          sampling a DAG, and V is its number of nodes.
        :param return_metadata: If True, also return a metadata dictionary.
        :return: If n_classes is None, X of shape (N, P); otherwise (X, y).
                 Appends a metadata dict when return_metadata is True.
        """
        # Step 1: Sample hyperparameters and DAG
        if adjacency is not None:
            V = adjacency.shape[0]
            parents, roots, edges = adjacency_to_dag(adjacency)
        else:
            V = rng.randint(self._Vmin, self._Vmax + 1)
            parents, roots, edges = self._sample_dag(rng, V)
        E = len(edges)

        N = (
            n_samples
            if n_samples is not None
            else rng.randint(self._Nmin, self._Nmax + 1)
        )
        C = n_classes

        # Feature count is bounded by the number of available nodes; when a
        # target is generated, one node is reserved for it.
        max_P = (V - 1) if C is not None else V
        if max_P < 1:
            raise ValueError(
                "cannot extract any feature: the graph has only V=1 node and a "
                "target is requested, leaving no feature nodes"
            )
        if n_features is None:
            P = rng.randint(1, min(13, max_P) + 1)
        else:
            P = n_features
        if P > max_P:
            raise ValueError(f"n_features ({P}) exceeds available nodes ({max_P})")

        # Ensure at least one root
        if len(roots) == 0:
            roots = [0]
            for child in range(V):
                parents[child] = [p for p in parents[child] if p != 0]

        # Step 3: Compute SCM (per-row propagation)
        node_values: Dict[int, np.ndarray] = {}

        # Fill root nodes with i.i.d. Gaussian noise samples (Figure 9)
        for r in roots:
            node_values[r] = _noise_iid(rng, N)

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

        # Propagate through each row
        for v in topo_order:
            if v in node_values:
                continue  # Root node already generated

            Pv = parents[v]
            node_values[v] = np.zeros(N, dtype=np.float64)
            combiner_fn = node_combiners[v]
            act_fn = node_activations[v]

            # Per-node additive noise (TabPFN-3 Section 2.5, Figure 9):
            # X_v = f_v(pa(X_v)) + ε_v, with an independent ε_v per node.
            epsilon = rng.normal(0.0, self._noise_std, N)

            for n in range(N):
                # Gather parent values at row n
                p_vals = np.array([node_values[u][n] for u in Pv])
                # Combine
                combined = combiner_fn(rng, p_vals)
                # Activate and add the node's own noise term
                node_values[v][n] = act_fn(combined) + epsilon[n]

        # Step 4: Extract dataset - choose features and target
        all_nodes = list(range(V))

        if C is not None:
            target_node = int(rng.choice(all_nodes))
            remaining = [v for v in all_nodes if v != target_node]
        else:
            target_node = None
            remaining = all_nodes

        feature_nodes = sorted(rng.choice(remaining, size=P, replace=False).tolist())

        # Stack into feature matrix: shape (N, P)
        X = np.stack([node_values[v] for v in feature_nodes], axis=1)

        y = None
        if C is not None:
            y = _discretize_target(node_values[target_node], C, rng)

        # Step 5: Post-processing
        if self._apply_postprocessing:
            if rng.random() < 0.3:
                X = _postprocess_add_outliers(
                    rng,
                    X,
                    outlier_prob=rng.uniform(0.005, 0.05),
                    outlier_scale=rng.uniform(2.0, 10.0),
                )
            if rng.random() < 0.3:
                X = _postprocess_add_missing(
                    rng,
                    X,
                    missing_prob=rng.uniform(0.01, 0.1),
                )
            if rng.random() < 0.5:
                X = _postprocess_scale_shift(rng, X)

        # Categorical variables (TabPFN-3, item 3)
        categorical_features: List[int] = []
        if self._categorical_prob > 0.0:
            for j in range(P):
                if rng.random() < self._categorical_prob:
                    k = rng.randint(2, 11)
                    X[:, j] = _bin_categorical(X[:, j], k, rng)
                    categorical_features.append(int(feature_nodes[j]))

        X = X.astype(self._dtype)

        metadata: Dict[str, Any] = {}
        if return_metadata:
            metadata = {
                "n_rows": N,
                "n_features": P,
                "n_classes": C,
                "n_nodes": V,
                "n_edges": E,
                "n_roots": len(roots),
                "feature_nodes": feature_nodes,
                "target_node": target_node,
                "root_nodes": roots,
                "edge_list": [(p, c) for p, c in edges],
                "categorical_features": categorical_features,
                "graph_source": "custom" if adjacency is not None else "random",
            }

        if C is not None:
            if return_metadata:
                return X, y, metadata
            return X, y

        if return_metadata:
            return X, metadata
        return X

    def generate_batch(
        self,
        rng: np.random.RandomState,
        n_batches: int,
        n_samples: Optional[int] = None,
        n_features: Optional[int] = None,
        n_classes: Optional[int] = None,
        adjacency: Optional[np.ndarray] = None,
    ) -> List[Any]:
        """Generate a batch of N synthetic tabular datasets.

        :param rng: The random number generator.
        :param n_batches: Number of datasets to generate.
        :param n_samples: Number of rows per dataset (see ``generate``).
        :param n_features: Number of features per dataset (see ``generate``).
        :param n_classes: Number of target classes per dataset (see ``generate``).
        :param adjacency: Optional binary adjacency matrix describing a DAG.
                          If provided, it is reused for every sample.
        :return: List of generated datasets (each an X array or an (X, y) tuple).
        """
        dataset = []
        for _ in range(n_batches):
            out = self.generate(
                rng=rng,
                n_samples=n_samples,
                n_features=n_features,
                n_classes=n_classes,
                adjacency=adjacency,
                return_metadata=False,
            )
            dataset.append(out)
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
    def activations(self) -> List[str]:
        """Get the list of activation function names."""
        return self._activation_names
