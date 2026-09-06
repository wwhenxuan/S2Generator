# -*- coding: utf-8 -*-
"""Heuristics for the first ForeCA weight vector."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ._entropy import omega
from ._spectrum import spectrum_of_linear_combination
from ._sfa import SlowFeatureAnalysis


def _unit(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float).reshape(-1)
    n = np.linalg.norm(w)
    if n < 1e-15:
        raise ValueError("Cannot normalize a zero weight vector.")
    w = w / n
    if w[0] < 0:
        w = -w
    return w


def initialize_weightvector(
    n_series: int,
    method: str = "rnorm",
    U: Optional[np.ndarray] = None,
    spectrum: Optional[np.ndarray] = None,
    random_state: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """
    Unit-norm starting vector for iterative ForeCA.

    :param n_series: Length ``K`` of the weight vector.
    :param method: One of ``rnorm``, ``runif``, ``rcauchy``, ``max``,
                   ``sfa``, ``sfa.slow``, ``sfa.fast``, ``pca``,
                   ``pca.large``, ``pca.small``.
    :param U: Whitened series ``(T, K)``, required for SFA / PCA / max.
    :param spectrum: Normalized multivariate spectrum, required for ``max``
                     and the auto ``sfa`` / ``pca`` variants.
    :param random_state: RNG for random methods.
    :return: Length-``K`` vector with unit Euclidean norm.
    """
    if n_series < 1:
        raise ValueError("n_series must be positive.")
    if n_series == 1:
        return np.ones(1)
    method = method.lower()
    rng = random_state if random_state is not None else np.random.RandomState()

    if method == "rnorm":
        return _unit(rng.randn(n_series))
    if method == "runif":
        return _unit(rng.uniform(-1.0, 1.0, size=n_series))
    if method == "rcauchy":
        return _unit(rng.standard_cauchy(size=n_series))

    if method == "max":
        if spectrum is None:
            raise ValueError("method='max' requires a multivariate spectrum.")
        k = spectrum.shape[1]
        omegas = []
        for i in range(k):
            omegas.append(float(omega(spectrum=np.real(spectrum[:, i, i]))))
        w = np.zeros(k)
        w[int(np.argmax(omegas))] = 1.0
        return _unit(w)

    if method.startswith("sfa"):
        if U is None:
            raise ValueError("SFA initializers require whitened data U.")
        sfa = SlowFeatureAnalysis().fit(U)
        slow = sfa.loadings[:, 0]
        fast = sfa.loadings[:, -1]
        if method == "sfa.slow":
            return _unit(slow)
        if method == "sfa.fast":
            return _unit(fast)
        if method == "sfa":
            if spectrum is None:
                raise ValueError("method='sfa' requires a multivariate spectrum.")
            omega_slow = float(
                omega(spectrum=spectrum_of_linear_combination(spectrum, slow))
            )
            omega_fast = float(
                omega(spectrum=spectrum_of_linear_combination(spectrum, fast))
            )
            return _unit(fast if omega_fast > omega_slow else slow)
        raise ValueError(f"Unknown SFA initializer '{method}'.")

    if method.startswith("pca"):
        if U is None:
            raise ValueError("PCA initializers require whitened data U.")
        _, _, vt = np.linalg.svd(U - U.mean(axis=0), full_matrices=False)
        large = vt[0]
        small = vt[-1]
        if method == "pca.large":
            return _unit(large)
        if method == "pca.small":
            return _unit(small)
        if method == "pca":
            if spectrum is None:
                raise ValueError("method='pca' requires a multivariate spectrum.")
            omega_large = float(
                omega(spectrum=spectrum_of_linear_combination(spectrum, large))
            )
            omega_small = float(
                omega(spectrum=spectrum_of_linear_combination(spectrum, small))
            )
            return _unit(small if omega_small > omega_large else large)
        raise ValueError(f"Unknown PCA initializer '{method}'.")

    raise ValueError(f"Unknown initializer '{method}'.")
