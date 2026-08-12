# -*- coding: utf-8 -*-
"""
Synthetic Multivariate Coupling (SCM) module for dataset construction.

This module implements the dataset construction methods from three papers:

1. **Synthetic multivariate coupling pipeline** (TiRex-2, Podest et al., 2026)
   - Identity / pass-through coupling
   - Functional coupling (deterministic covariate transformations)
   - Linear mixing (shared latent factors)
   - Cointegration (shared stochastic trends)
   - Linear structural causal models (lagged DAG-based dependencies)
   - Nonlinear structural causal models (with modulation gates)
   - Post-processing (time warping, masking, discretization, etc.)

2. **CAUKER: Causal-Kernel Generation** (Xie et al., 2025)
   - GP kernel bank with 36 composite kernel variants
   - Mean function bank (zero, linear, exponential, sparse anomalies)
   - Activation function bank (linear, ReLU, sigmoid, sin, modulo, Leaky ReLU)
   - Random DAG generation with topological SCM propagation
   - 5-step pipeline: kernel sampling → composition → GP generation →
     activation sampling → causal graph propagation

3. **TabPFN-3 Synthetic Prior** (Prior Labs Team, 2026)
   - Expanded DAG generation algorithms (chain, fork, collider, random,
     scale-free, bipartite)
   - Rich combiner mechanisms (linear, MLP, polynomial, multiplicative,
     periodic, maxmin)
   - Temporal noise processes (iid, random walk, AR(1), periodic, OU)
   - Dynamic SCM: time-evolving causal graph propagation
   - Post-processing (outliers, missing values, scale-shift)

The module provides both individual coupling mechanisms and unified pipeline
interfaces (`CouplingPipeline`, `CaukerPipeline`, `TabPFNSCMPipeline`)
for end-to-end dataset construction.

References:
    - Podest, P., et al. (2026). TiRex-2: Generalizing TiRex to Multivariate
      Data and Streaming. arXiv:2607.01204v1.
    - Xie, S., et al. (2025). CAUKER: Classification Time Series Foundation
      Models Can Be Pretrained on Synthetic Data Only. arXiv:2508.02879v3.
    - Prior Labs Team (2026). TabPFN-3: Technical Report.
      arXiv:2605.13986v2.

Created on 2026/08/12
@author: Ruizhe Wang
@email: changewam6@gmail.com
@url: https://github.com/wwhenxuan/S2Generator
"""

__all__ = [
    # Base class
    "BaseCoupling",
    # Coupling mechanisms (TiRex-2)
    "IdentityCoupling",
    "UnivariatePassThrough",
    "FunctionalCoupling",
    "LinearMixing",
    "Cointegration",
    "LinearSCM",
    "NonlinearSCM",
    # Post-processing (TiRex-2)
    "PostProcessor",
    "variate_permutation",
    "smooth_time_warping",
    "patch_masking",
    "partial_future_observability",
    "value_discretization",
    "time_discretization",
    # CAUKER pipeline
    "CaukerPipeline",
    # TabPFN-3 pipeline
    "TabPFNSCMPipeline",
    # Unified interfaces
    "CouplingPipeline",
]

# --- TiRex-2 coupling mechanisms (optional) ---
try:
    from s2generator.scm.tirex2 import (
        BaseCoupling,
        IdentityCoupling,
        UnivariatePassThrough,
        FunctionalCoupling,
        LinearMixing,
        Cointegration,
        LinearSCM,
        NonlinearSCM,
        PostProcessor,
        variate_permutation,
        smooth_time_warping,
        patch_masking,
        partial_future_observability,
        value_discretization,
        time_discretization,
        CouplingPipeline,
    )
except ImportError:
    BaseCoupling = None  # type: ignore[assignment]
    IdentityCoupling = None  # type: ignore[assignment]
    UnivariatePassThrough = None  # type: ignore[assignment]
    FunctionalCoupling = None  # type: ignore[assignment]
    LinearMixing = None  # type: ignore[assignment]
    Cointegration = None  # type: ignore[assignment]
    LinearSCM = None  # type: ignore[assignment]
    NonlinearSCM = None  # type: ignore[assignment]
    PostProcessor = None  # type: ignore[assignment]
    variate_permutation = None  # type: ignore[assignment]
    smooth_time_warping = None  # type: ignore[assignment]
    patch_masking = None  # type: ignore[assignment]
    partial_future_observability = None  # type: ignore[assignment]
    value_discretization = None  # type: ignore[assignment]
    time_discretization = None  # type: ignore[assignment]
    CouplingPipeline = None  # type: ignore[assignment]

# --- CAUKER pipeline (Xie et al., 2025) ---
from s2generator.scm.cauker import CaukerPipeline

# --- TabPFN-3 pipeline (Prior Labs Team, 2026) (optional) ---
try:
    from s2generator.scm.tabpfn_scm import TabPFNSCMPipeline
except ImportError:
    TabPFNSCMPipeline = None  # type: ignore[assignment]
