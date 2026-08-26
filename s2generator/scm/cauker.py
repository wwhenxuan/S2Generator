# -*- coding: utf-8 -*-
"""
CAUKER: Causal-Kernel Generation for Time Series Classification.

This module implements the synthetic data generation pipeline from:

    Xie, S., Feofanov, V., Odonnat, A., et al. (2025).
    CAUKER: Classification Time Series Foundation Models Can Be Pretrained
    on Synthetic Data Only. arXiv:2508.02879v3.

The CAUKER pipeline combines Gaussian Process (GP) kernel composition with
Structural Causal Models (SCM) in a 5-step process (Algorithm 1):

1. **Kernel bank sampling**: sample K ~ U(1, Kmax) kernels from kernel bank
2. **Kernel composition**: combine kernels with random +/× operations
3. **Root node GP generation**: draw M mean functions, form GP priors, sample
4. **Activation bank sampling**: sample E ~ U(1, nA) activations for SCM edges
5. **Causal graph propagation**: generate DAG, propagate root signals through
   edges with activation functions and random linear layers

Key design choices (Section 3.2):
- Non-zero GP means preserve discriminative mean-level cues for classification
- SCM edges inject nonlinear causal semantics via activation functions
- Each SCM node provides one univariate series for pre-training

Reference:
    Xie, S., et al. (2025). CAUKER. arXiv:2508.02879v3, Algorithm 1 & Appendix C.

Created on 2026/08/10
@author: Ruizhe Wang
@email: changewam6@gmail.com
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..utils._dag import adjacency_to_dag
from ..utils._label import discretize_labels, label_single, summarize_series


# ===========================================================================
# Kernel Bank K: 36 kernel variants
# ===========================================================================
# Six base kernel types with varied hyperparameters, following Ansari et al. (2024)
# and extended as described in CAUKER Appendix C.2.
#
# Kernel types:
#   1. ExpSineSquared  – periodic patterns (periodicity, length_scale)
#   2. DotProduct       – linear trends (sigma_0)
#   3. RBF              – smooth local fluctuations (length_scale)
#   4. RationalQuadratic – multi-scale smoothness (length_scale, alpha)
#   5. WhiteKernel      – uncorrelated noise (noise_level)
#   6. ConstantKernel   – constant offset (constant_value)
# ===========================================================================


def _build_kernel_bank() -> List[Dict[str, Any]]:
    """Build the full kernel bank of 36 kernel variants.

    Each entry is a dict with:
        - 'name': str, kernel type name
        - 'params': dict, hyperparameters for kernel evaluation
        - 'covariance_fn': callable(t1, t2, params) -> covariance matrix

    :return: List of 36 kernel specifications.
    """
    kernels: List[Dict[str, Any]] = []

    # --- ExpSineSquared: 8 variants ---
    periodicities = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    for i, period in enumerate(periodicities):
        length_scale = period * np.random.RandomState(i).uniform(0.5, 2.0)
        kernels.append(
            {
                "name": f"ExpSineSquared_p{period:.0f}",
                "params": {"periodicity": period, "length_scale": length_scale},
                "covariance_fn": _cov_exp_sine_squared,
            }
        )

    # --- DotProduct: 4 variants ---
    for i, sigma_0 in enumerate([0.1, 1.0, 5.0, 10.0]):
        kernels.append(
            {
                "name": f"DotProduct_s{sigma_0}",
                "params": {"sigma_0": sigma_0},
                "covariance_fn": _cov_dot_product,
            }
        )

    # --- RBF: 8 variants ---
    length_scales_rbf = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]
    for i, ls in enumerate(length_scales_rbf):
        kernels.append(
            {
                "name": f"RBF_ls{ls}",
                "params": {"length_scale": ls},
                "covariance_fn": _cov_rbf,
            }
        )

    # --- RationalQuadratic: 6 variants ---
    ls_rq = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0]
    alphas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    for i, (ls, alpha) in enumerate(zip(ls_rq, alphas)):
        kernels.append(
            {
                "name": f"RationalQuadratic_ls{ls}_a{alpha}",
                "params": {"length_scale": ls, "alpha": alpha},
                "covariance_fn": _cov_rational_quadratic,
            }
        )

    # --- WhiteKernel: 5 variants ---
    noise_levels = [0.001, 0.01, 0.05, 0.1, 0.5]
    for i, nl in enumerate(noise_levels):
        kernels.append(
            {
                "name": f"WhiteKernel_n{nl}",
                "params": {"noise_level": nl},
                "covariance_fn": _cov_white,
            }
        )

    # --- ConstantKernel: 5 variants ---
    constant_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    for i, cv in enumerate(constant_values):
        kernels.append(
            {
                "name": f"ConstantKernel_v{cv}",
                "params": {"constant_value": cv},
                "covariance_fn": _cov_constant,
            }
        )

    return kernels


# --- Covariance functions ---


def _cov_exp_sine_squared(
    t1: np.ndarray, t2: np.ndarray, params: Dict[str, float]
) -> np.ndarray:
    """ExpSineSquared (periodic) kernel.

    k(t, t') = exp(-2 * sin^2(pi * |t - t'| / p) / l^2)

    :param t1: First time index array, shape (n1,).
    :param t2: Second time index array, shape (n2,).
    :param params: Dict with 'periodicity' (p) and 'length_scale' (l).
    :return: Covariance matrix of shape (n1, n2).
    """
    p = params["periodicity"]
    l = params["length_scale"]
    dt = np.abs(t1[:, None] - t2[None, :])
    return np.exp(-2.0 * np.sin(np.pi * dt / p) ** 2 / l**2)


def _cov_dot_product(
    t1: np.ndarray, t2: np.ndarray, params: Dict[str, float]
) -> np.ndarray:
    """DotProduct (linear) kernel.

    k(t, t') = sigma_0^2 + t * t'

    :param t1: First time index array, shape (n1,).
    :param t2: Second time index array, shape (n2,).
    :param params: Dict with 'sigma_0'.
    :return: Covariance matrix of shape (n1, n2).
    """
    s0 = params["sigma_0"]
    return s0**2 + t1[:, None] * t2[None, :]


def _cov_rbf(t1: np.ndarray, t2: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    """Radial Basis Function (squared-exponential) kernel.

    k(t, t') = exp(-0.5 * |t - t'|^2 / l^2)

    :param t1: First time index array, shape (n1,).
    :param t2: Second time index array, shape (n2,).
    :param params: Dict with 'length_scale' (l).
    :return: Covariance matrix of shape (n1, n2).
    """
    l = params["length_scale"]
    dt = t1[:, None] - t2[None, :]
    return np.exp(-0.5 * dt**2 / l**2)


def _cov_rational_quadratic(
    t1: np.ndarray, t2: np.ndarray, params: Dict[str, float]
) -> np.ndarray:
    """Rational Quadratic kernel (scale mixture of RBF kernels).

    k(t, t') = (1 + |t - t'|^2 / (2 * alpha * l^2))^(-alpha)

    :param t1: First time index array, shape (n1,).
    :param t2: Second time index array, shape (n2,).
    :param params: Dict with 'length_scale' (l) and 'alpha'.
    :return: Covariance matrix of shape (n1, n2).
    """
    l = params["length_scale"]
    alpha = params["alpha"]
    dt = t1[:, None] - t2[None, :]
    return (1.0 + dt**2 / (2.0 * alpha * l**2)) ** (-alpha)


def _cov_white(t1: np.ndarray, t2: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    """White noise kernel.

    k(t, t') = noise_level * delta(t, t')

    :param t1: First time index array, shape (n1,).
    :param t2: Second time index array, shape (n2,).
    :param params: Dict with 'noise_level'.
    :return: Covariance matrix of shape (n1, n2).
    """
    nl = params["noise_level"]
    return nl * np.eye(len(t1), len(t2))


def _cov_constant(
    t1: np.ndarray, t2: np.ndarray, params: Dict[str, float]
) -> np.ndarray:
    """Constant kernel.

    k(t, t') = constant_value for all t, t'

    :param t1: First time index array, shape (n1,).
    :param t2: Second time index array, shape (n2,).
    :param params: Dict with 'constant_value'.
    :return: Covariance matrix of shape (n1, n2).
    """
    cv = params["constant_value"]
    return np.full((len(t1), len(t2)), cv)


# ===========================================================================
# Mean Function Bank M: 4 types
# ===========================================================================
# Following CAUKER Section 3.2 & Appendix C.2:
#   1. Zero: μ(t) = 0
#   2. Linear: μ(t) = a·t + b
#   3. Exponential: μ(t) = a·exp(b·t)
#   4. Sparse anomalies: μ(t) with random spikes
# ===========================================================================


def _build_mean_bank() -> List[Dict[str, Any]]:
    """Build the mean function bank.

    Each entry is a dict with:
        - 'name': str, mean function type name
        - 'generate': callable(rng, t_grid) -> mean vector of shape (L,)

    :return: List of 4 mean function specifications.
    """
    return [
        {"name": "zero", "generate": _mean_zero},
        {"name": "linear", "generate": _mean_linear},
        {"name": "exponential", "generate": _mean_exponential},
        {"name": "sparse_anomalies", "generate": _mean_sparse_anomalies},
    ]


def _mean_zero(rng: np.random.RandomState, t_grid: np.ndarray) -> np.ndarray:
    """Zero mean function: μ(t) = 0.

    :param rng: Random number generator (unused, kept for interface consistency).
    :param t_grid: Time index array of shape (L,).
    :return: Zero vector of shape (L,).
    """
    return np.zeros(len(t_grid), dtype=np.float64)


def _mean_linear(rng: np.random.RandomState, t_grid: np.ndarray) -> np.ndarray:
    """Linear mean function: μ(t) = a·t + b.

    a ~ U(-2, 2), b ~ U(-5, 5).

    :param rng: Random number generator.
    :param t_grid: Time index array of shape (L,).
    :return: Linear mean vector of shape (L,).
    """
    a = rng.uniform(-2.0, 2.0)
    b = rng.uniform(-5.0, 5.0)
    return (a * t_grid + b).astype(np.float64)


def _mean_exponential(rng: np.random.RandomState, t_grid: np.ndarray) -> np.ndarray:
    """Exponential mean function: μ(t) = a·exp(b·t).

    a ~ U(-3, 3), b ~ U(-2, 2).

    :param rng: Random number generator.
    :param t_grid: Time index array of shape (L,), expected in [0, 1].
    :return: Exponential mean vector of shape (L,).
    """
    a = rng.uniform(-3.0, 3.0)
    b = rng.uniform(-2.0, 2.0)
    return (a * np.exp(b * t_grid)).astype(np.float64)


def _mean_sparse_anomalies(
    rng: np.random.RandomState, t_grid: np.ndarray
) -> np.ndarray:
    """Sparse anomaly mean function: piecewise-constant with random spikes.

    A small number of positions receive values from U(-5, 5); elsewhere zero.

    :param rng: Random number generator.
    :param t_grid: Time index array of shape (L,).
    :return: Sparse anomaly mean vector of shape (L,).
    """
    L = len(t_grid)
    mean = np.zeros(L, dtype=np.float64)
    n_anomalies = rng.randint(0, max(2, L // 50) + 1)
    if n_anomalies > 0:
        positions = rng.choice(L, size=n_anomalies, replace=False)
        mean[positions] = rng.uniform(-5.0, 5.0, size=n_anomalies)
    return mean


# ===========================================================================
# Activation Function Bank A: 6 types
# ===========================================================================
# Following CAUKER Appendix C.2:
#   1. Linear: f(x) = a·x + b, a ~ U(0.5, 2), b ~ U(-1, 1)
#   2. ReLU: f(x) = max(0, x)
#   3. Sigmoid: f(x) = 1/(1 + exp(-x))
#   4. Sin: f(x) = sin(x)
#   5. Modulo: f(x) = x mod c, c ~ U(1, 5)
#   6. Leaky ReLU: f(x) = x if x > 0 else alpha·x, alpha ~ U(0.01, 0.3)
# ===========================================================================


def _build_activation_bank(
    rng: np.random.RandomState,
) -> List[Dict[str, Any]]:
    """Build the activation function bank with randomized parameters.

    Each entry is a dict with:
        - 'name': str, activation type name
        - 'params': dict, randomly sampled parameters
        - 'apply': callable(x) -> transformed x

    :param rng: Random number generator for parameter sampling.
    :return: List of 6 activation function specifications.
    """
    # Each call returns a fresh set of activations with randomized parameters
    return [
        {
            "name": "linear",
            "params": {"a": rng.uniform(0.5, 2.0), "b": rng.uniform(-1.0, 1.0)},
            "apply": lambda x, a=None, b=None: (
                (a if a is not None else rng.uniform(0.5, 2.0)) * x
                + (b if b is not None else rng.uniform(-1.0, 1.0))
            ),
        },
        {
            "name": "relu",
            "params": {},
            "apply": lambda x: np.maximum(0, x),
        },
        {
            "name": "sigmoid",
            "params": {},
            "apply": lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50))),
        },
        {
            "name": "sin",
            "params": {},
            "apply": lambda x: np.sin(x),
        },
        {
            "name": "modulo",
            "params": {"c": rng.uniform(1.0, 5.0)},
            "apply": lambda x, c=None: np.mod(
                x, c if c is not None else rng.uniform(1.0, 5.0)
            ),
        },
        {
            "name": "leaky_relu",
            "params": {"alpha": rng.uniform(0.01, 0.3)},
            "apply": lambda x, alpha=None: np.where(
                x > 0,
                x,
                (alpha if alpha is not None else rng.uniform(0.01, 0.3)) * x,
            ),
        },
    ]


# ===========================================================================
# GP Sampling Utilities
# ===========================================================================


def _sample_composite_kernel(
    rng: np.random.RandomState,
    kernel_bank: List[Dict[str, Any]],
    Kmax: int = 5,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Sample and compose kernels (CAUKER Algorithm 1, lines 7-13).

    Samples K ~ U(1, Kmax) kernels from the bank and composes them
    with random +/× operations.

    :param rng: Random number generator.
    :param kernel_bank: List of kernel specifications.
    :param Kmax: Maximum number of kernels to sample.
    :return: A callable (t1, t2) -> covariance matrix.
    """
    K = rng.randint(1, Kmax + 1)
    selected_indices = rng.choice(len(kernel_bank), size=K, replace=True)
    selected = [kernel_bank[i] for i in selected_indices]

    if K == 1:
        k = selected[0]
        return lambda t1, t2, k=k: k["covariance_fn"](t1, t2, k["params"])

    # Compose: start with first kernel, then apply random +/× with subsequent
    ops = [rng.choice(["add", "mul"]) for _ in range(K - 1)]

    def composite_kernel(
        t1: np.ndarray,
        t2: np.ndarray,
        sel=selected,
        ops=ops,
    ) -> np.ndarray:
        K_mat = sel[0]["covariance_fn"](t1, t2, sel[0]["params"])
        for i, op in enumerate(ops):
            K_next = sel[i + 1]["covariance_fn"](t1, t2, sel[i + 1]["params"])
            if op == "add":
                K_mat = K_mat + K_next
            else:  # mul
                K_mat = K_mat * K_next
        return K_mat

    return composite_kernel


def _sample_mean(
    rng: np.random.RandomState,
    mean_bank: List[Dict[str, Any]],
    t_grid: np.ndarray,
) -> np.ndarray:
    """Sample and compose mean functions (CAUKER Algorithm 1, lines 15-17).

    Samples 2 mean functions and combines them with a random +/× operation.

    :param rng: Random number generator.
    :param mean_bank: List of mean function specifications.
    :param t_grid: Time index array of shape (L,).
    :return: Mean vector of shape (L,).
    """
    indices = rng.choice(len(mean_bank), size=2, replace=True)
    m1 = mean_bank[indices[0]]["generate"](rng, t_grid)
    m2 = mean_bank[indices[1]]["generate"](rng, t_grid)

    op = rng.choice(["add", "mul"])
    if op == "add":
        return (m1 + m2).astype(np.float64)
    else:
        return (m1 * m2).astype(np.float64)


def _sample_gp(
    rng: np.random.RandomState,
    mean: np.ndarray,
    cov_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    t_grid: np.ndarray,
    jitter: float = 1e-6,
) -> np.ndarray:
    """Sample a univariate time series from a GP prior.

    Draws from GP(mean, kernel) on the specified time grid using
    Cholesky decomposition: x = mean + L·z, where K = L·L^T and z ~ N(0, I).

    :param rng: Random number generator.
    :param mean: Mean vector of shape (L,).
    :param cov_fn: Covariance function (t1, t2) -> K of shape (L, L).
    :param t_grid: Time index array of shape (L,).
    :param jitter: Small diagonal jitter for numerical stability.
    :return: Sampled time series of shape (L,).
    """
    L = len(t_grid)
    K = cov_fn(t_grid, t_grid)
    # Add jitter for numerical stability
    K = K + jitter * np.eye(L, dtype=np.float64)

    try:
        L_chol = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        # Fallback: use eigenvalue decomposition for PSD repair
        eigvals, eigvecs = np.linalg.eigh(K)
        eigvals = np.maximum(eigvals, jitter)
        L_chol = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

    z = rng.normal(0, 1, size=L)
    sample = mean + L_chol @ z
    return sample.astype(np.float64)


# ===========================================================================
# DAG Generation
# ===========================================================================


def _generate_random_dag(
    rng: np.random.RandomState,
    V: int,
    Pmax: int,
) -> Tuple[List[List[int]], List[int], List[List[int]]]:
    """Generate a random Directed Acyclic Graph (DAG).

    Creates a DAG by assigning each node a random topological order and
    only allowing edges from lower-order to higher-order nodes. This
    guarantees acyclicity.

    :param rng: Random number generator.
    :param V: Number of nodes.
    :param Pmax: Maximum number of parents per node.
    :return: Tuple of (adjacency_list, root_nodes, edge_list).
             adjacency_list[i] = list of parent indices for node i.
             root_nodes = list of node indices with in-degree 0.
             edge_list = list of (parent, child) tuples.
    """
    # Random topological ordering
    order = rng.permutation(V)
    # Map: position in order -> original node index
    # We'll build graph where edges go from earlier to later positions
    parents: List[List[int]] = [[] for _ in range(V)]

    for pos in range(1, V):
        node = order[pos]
        possible_parents = order[:pos]
        if len(possible_parents) == 0:
            continue
        n_parents = rng.randint(0, min(Pmax, len(possible_parents)) + 1)
        if n_parents > 0:
            selected = rng.choice(possible_parents, size=n_parents, replace=False)
            parents[node] = sorted(selected.tolist())

    # Identify root nodes (in-degree 0)
    roots = [i for i in range(V) if len(parents[i]) == 0]

    # Build edge list
    edges = []
    for child in range(V):
        for parent in parents[child]:
            edges.append((parent, child))

    return parents, roots, edges


# ===========================================================================
# CAUKER Pipeline
# ===========================================================================


class CaukerPipeline(object):
    """CAUKER synthetic time-series generator for classification.

    Implements Algorithm 1 from Xie et al. (2025). The pipeline generates
    univariate time series by:

    1. Sampling and composing GP kernels from a bank of 36 variants
    2. Sampling and composing mean functions from a bank of 4 types
    3. Drawing root node signals from GP(mean, composite_kernel) priors
    4. Sampling activation functions for SCM edges
    5. Propagating signals through a random DAG with edge activations

    The result is a set of univariate time series with causally structured
    dependencies, suitable for pre-training classification TSFMs.
    """

    def __init__(
        self,
        Kmax: int = 5,
        Vmax: int = 20,
        Pmax: int = 4,
        target_length: int = 512,
        dtype: np.dtype = np.float64,
        standardize_activation_input: bool = True,
        min_node_std: float = 0.05,
    ) -> None:
        """Initialize the CAUKER pipeline.

        :param Kmax: Maximum number of kernels to sample per composite GP
                     (Algorithm 1, line 7: K ~ U(1, Kmax)).
        :param Vmax: Maximum number of nodes in the DAG
                     (Algorithm 1, line 20: V ~ U(d, Vmax)).
        :param Pmax: Maximum number of parents per node
                     (Algorithm 1, line 21).
        :param target_length: Target length L of each generated time series.
        :param dtype: The numpy data type for generated data.
        :param standardize_activation_input: If True, center/scale the linear
                     aggregate z before applying the edge activation (and
                     rescale afterwards) so that non-linear activations (ReLU,
                     sigmoid, sin, modulo) operate on a well-scaled signal. This
                     prevents ReLU from collapsing an all-negative aggregate to
                     an exactly-zero channel and keeps propagated channels on
                     their parents' scale. Default True. Set False for the
                     literal paper behaviour (activation applied directly to
                     W·z + b, Algorithm 1 line 35), which can emit degenerate
                     channels.
        :param min_node_std: Detection threshold for collapsed nodes. Any node
                     whose standard deviation falls below this value is treated
                     as degenerate (e.g. a flat GP root, or a collapsed
                     activation) and jittered with Gaussian noise of meaningful
                     amplitude so it keeps usable variation while retaining its
                     mean level. Set to 0.0 to disable. Default 0.05.
        """
        self._Kmax = Kmax
        self._Vmax = Vmax
        self._Pmax = Pmax
        self._target_length = target_length
        self._dtype = dtype
        self._standardize_activation_input = standardize_activation_input
        self._min_node_std = min_node_std

        # Build kernel and mean banks once
        self._kernel_bank = _build_kernel_bank()
        self._mean_bank = _build_mean_bank()

    def _ensure_variation(
        self, rng: np.random.RandomState, values: np.ndarray
    ) -> np.ndarray:
        """Guard against a collapsed (near-constant) node signal.

        A node can collapse to a flat/zero channel when, e.g., ReLU is applied
        to an all-negative linear aggregate ``z = W·parent + b``, or a GP draw
        lands on a constant kernel. Such channels carry no information and show
        up as dead or duplicate variates in the output. We detect collapse by
        the signal's standard deviation and inject a small Gaussian jitter so
        the channel retains meaningful variation.

        Well-behaved signals are returned unchanged and the RNG stream is left
        untouched, so this only alters degenerate cases and keeps generation
        deterministic for a fixed seed.

        :param rng: Random number generator.
        :param values: 1-D node signal of length L.
        :return: The (possibly jittered) signal.
        """
        if self._min_node_std <= 0.0:
            return values
        std = float(np.std(values))
        if std >= self._min_node_std:
            return values
        # Collapsed to a (near-)constant channel. Inject Gaussian noise of a
        # meaningful amplitude so the channel keeps usable variation, adding
        # around the existing mean to preserve its level (the paper's "mean
        # level as a discriminative cue").
        noise_std = max(self._min_node_std, 0.5)
        noise = rng.normal(0.0, noise_std, size=values.shape)
        return (values + noise).astype(np.float64)

    def __str__(self) -> str:
        return "CaukerPipeline"

    def generate(
        self,
        rng: np.random.RandomState,
        seq_length: int,
        num_channels: Optional[int] = None,
        adjacency: Optional[np.ndarray] = None,
        n_classes: Optional[int] = None,
        return_metadata: bool = False,
    ) -> Any:
        """Generate synthetic time series from the CAUKER pipeline.

        Implements Algorithm 1, lines 18-39.

        :param rng: The random number generator with fixed seed.
        :param seq_length: Target length L of each time series.
        :param num_channels: Number of observed variables d (output dimension).
                               If None, randomly sampled from {1, ..., min(12, V)}.
        :param adjacency: Optional binary adjacency matrix of shape (V, V)
                          describing a DAG. If provided, it is used instead of
                          generating a random DAG, and V is its number of nodes.
        :param n_classes: If given, also return a classification label for the
                          generated series (the CAUKER pretraining objective is
                          classification). The label is a single integer derived
                          deterministically from the series summary statistic;
                          for balanced labels use ``generate_batch``.
        :param return_metadata: If True, also return metadata about the
                               generation process.
        :return: Generated time series of shape (d, L). If ``n_classes`` is
                 given, returns ``(series, label)`` instead. A metadata dict is
                 appended when ``return_metadata`` is True.
        """
        L = seq_length
        t_grid = np.linspace(0, 1, L, dtype=np.float64)

        if adjacency is not None:
            # User-specified graph: V is the graph size, d is sampled within it.
            V = adjacency.shape[0]
            if num_channels is None:
                d = rng.randint(1, max(2, min(13, V)))
            else:
                d = num_channels
            if d > V:
                raise ValueError(f"num_channels ({d}) exceeds graph size ({V})")
            parents, roots, edges = adjacency_to_dag(adjacency)
            E = len(edges)
        else:
            # Sample the number of observed variables and total nodes
            if num_channels is None:
                d = rng.randint(1, max(2, min(13, self._Vmax)))
            else:
                d = num_channels

            V = rng.randint(max(d, 2), self._Vmax + 1)

            # Step 5: Generate DAG
            parents, roots, edges = _generate_random_dag(rng, V, self._Pmax)
            E = len(edges)

            # Ensure at least one root node (if DAG generation gave none, re-run)
            retries = 0
            while len(roots) == 0 and retries < 10:
                parents, roots, edges = _generate_random_dag(rng, V, self._Pmax)
                E = len(edges)
                retries += 1

            if len(roots) == 0:
                # Force at least one root
                roots = [0]
                # Remove incoming edges to node 0
                for child in range(V):
                    parents[child] = [p for p in parents[child] if p != 0]
                edges = [(p, c) for c in range(V) for p in parents[c]]

        # Step 4: Sample activation functions for each edge
        activation_bank = _build_activation_bank(rng)
        # Assign an activation to each edge
        edge_activations: Dict[Tuple[int, int], Callable] = {}
        edge_params: Dict[Tuple[int, int], Dict[str, Any]] = {}
        if E > 0:
            act_indices = rng.choice(len(activation_bank), size=E, replace=True)
            for idx, (parent, child) in enumerate(edges):
                act = activation_bank[act_indices[idx]]
                edge_activations[(parent, child)] = act["apply"]
                edge_params[(parent, child)] = act["params"]

        # Step 3: Generate root node signals from GP priors
        node_values: Dict[int, np.ndarray] = {}

        for r in roots:
            # Sample composite kernel
            cov_fn = _sample_composite_kernel(rng, self._kernel_bank, self._Kmax)
            # Sample mean function
            mean = _sample_mean(rng, self._mean_bank, t_grid)
            # Sample from GP
            node_values[r] = _sample_gp(rng, mean, cov_fn, t_grid)

            # Guard against collapsed (flat/zero) root signals, e.g. a GP draw
            # landing on a ConstantKernel. A flat channel would propagate through
            # the DAG and yield dead/duplicate variates downstream.
            node_values[r] = self._ensure_variation(rng, node_values[r])

        # Step 5 (continued): Propagate through DAG in topological order
        # Build topological order: nodes sorted by dependency
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

        # Process each non-root node in topological order
        for v in topo_order:
            if v in node_values:
                continue  # Already generated (root node)

            Pv = parents[v]
            # Concatenate parent signals along time axis
            # Each parent contributes one column; we concatenate horizontally
            parent_signals = np.column_stack([node_values[u] for u in Pv])  # (L, |Pv|)

            # Random linear layer: W ~ N(0, 1), b ~ N(0, 1)
            W = rng.normal(0, 1, size=(1, len(Pv)))
            b = rng.normal(0, 1)

            # Aggregate: z = W @ parent_signals^T + b -> shape (1, L)
            z = (W @ parent_signals.T + b).flatten()  # (L,)

            # Apply edge activation (use activation from first incoming edge)
            edge_key = (Pv[0], v)
            act_fn = edge_activations.get(edge_key)
            act_params = edge_params.get(edge_key, {})

            if self._standardize_activation_input:
                # Optional numerical-stability normalization (off by default):
                # center/scale z so ReLU/sigmoid/sin do not collapse, then
                # rescale the output back to the original signal magnitude.
                z_mean = float(np.mean(z))
                z_std = float(np.std(z)) + 1e-8
                z_act = (z - z_mean) / z_std
            else:
                # Paper behavior (Algorithm 1, line 35): t_v = φ(W·[e·j] + b).
                z_act = z

            if act_fn is not None:
                activated = act_fn(z_act, **act_params) if act_params else act_fn(z_act)
            else:
                activated = z_act

            if self._standardize_activation_input:
                activated = activated * z_std + z_mean

            node_values[v] = self._ensure_variation(rng, activated.astype(np.float64))

        # Select d observed nodes (Algorithm 1, line 36)
        all_nodes = list(range(V))
        observed = rng.choice(all_nodes, size=d, replace=False)
        observed = sorted(
            observed.tolist() if hasattr(observed, "tolist") else list(observed)
        )

        # Stack selected node values into output: shape (d, L)
        x = np.stack([node_values[v] for v in observed], axis=0)

        metadata: Dict[str, Any] = {}
        if return_metadata:
            metadata = {
                "target_length": L,
                "n_observed": d,
                "n_total_nodes": V,
                "n_roots": len(roots),
                "n_edges": E,
                "observed_nodes": observed,
                "root_nodes": roots,
                "edge_list": [(p, c) for p, c in edges],
                "graph_source": "custom" if adjacency is not None else "random",
            }

        if n_classes is not None:
            label = label_single(summarize_series(x), n_classes)
            if return_metadata:
                metadata["n_classes"] = n_classes
                return x.astype(self._dtype), label, metadata
            return x.astype(self._dtype), label

        if return_metadata:
            return x.astype(self._dtype), metadata
        return x.astype(self._dtype)

    def generate_batch(
        self,
        rng: np.random.RandomState,
        n_samples: int,
        seq_length: int,
        num_channels: Optional[int] = None,
        adjacency: Optional[np.ndarray] = None,
        n_classes: Optional[int] = None,
    ) -> List[Any]:
        """Generate a batch of N synthetic time series.

        Implements Algorithm 1, lines 18-39 loop over N samples.

        :param rng: The random number generator.
        :param n_samples: Number of samples to generate (N in Algorithm 1).
        :param seq_length: Target length of each time series.
        :param num_channels: Number of observed variables per sample.
        :param adjacency: Optional binary adjacency matrix describing a DAG.
                          If provided, it is reused for every sample.
        :param n_classes: If given, assign each generated series a balanced
                          class label by quantile-binning the per-series summary
                          statistic across the batch (RML2016-style labeled
                          dataset).
        :return: List of generated time series, each of shape (d, L). If
                 ``n_classes`` is given, returns a list of ``(series, label)``
                 tuples instead.
        """
        dataset = []
        for _ in range(n_samples):
            x = self.generate(
                rng=rng,
                seq_length=seq_length,
                num_channels=num_channels,
                adjacency=adjacency,
                return_metadata=False,
            )
            dataset.append(x)

        if n_classes is not None:
            stats = [summarize_series(x) for x in dataset]
            labels = discretize_labels(stats, n_classes)
            return [(x, int(y)) for x, y in zip(dataset, labels)]
        return dataset

    @property
    def kernel_bank(self) -> List[Dict[str, Any]]:
        """Get the kernel bank."""
        return self._kernel_bank

    @property
    def mean_bank(self) -> List[Dict[str, Any]]:
        """Get the mean function bank."""
        return self._mean_bank

    @property
    def n_kernels(self) -> int:
        """Get the number of kernels in the bank."""
        return len(self._kernel_bank)

    @property
    def n_mean_functions(self) -> int:
        """Get the number of mean functions in the bank."""
        return len(self._mean_bank)
