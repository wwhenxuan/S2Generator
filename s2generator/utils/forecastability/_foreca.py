# -*- coding: utf-8 -*-
"""Forecastable Component Analysis (Goerg 2013)."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from scipy.linalg import null_space

from ._entropy import omega
from ._init_weights import initialize_weightvector
from ._spectrum import (
    _as_2d,
    mvspectrum,
    spectrum_of_linear_combination,
)
from ._whiten import whiten


def _mvspectrum2wcov(
    spec: np.ndarray, kernel_weights: Optional[np.ndarray] = None
) -> np.ndarray:
    s = np.asarray(spec)
    if kernel_weights is not None:
        s = s * np.asarray(kernel_weights, dtype=float).reshape(-1, 1, 1)
    return 2.0 * np.real(s.sum(axis=0))


def _entropy_wcov(
    f_u: np.ndarray,
    f_current: np.ndarray,
    prior_weight: float,
    base: float,
) -> np.ndarray:
    """Weighted covariance with kernel :math:`-\\log f_y(\\lambda)` (ForeCA M-step)."""
    n_freq, k, _ = f_u.shape
    f_u = np.array(f_u, dtype=complex, copy=True)
    f_current = np.asarray(f_current, dtype=float).reshape(-1)
    if prior_weight > 0:
        mix = prior_weight / n_freq
        prior = np.zeros_like(f_u)
        p_pos = 1.0 / (2.0 * n_freq)
        idx = np.arange(k)
        prior[:, idx, idx] = p_pos
        f_u = (1.0 - mix) * f_u + mix * prior

    mask = f_current > 1e-15
    f_u = f_u[mask]
    f_current = f_current[mask]
    n_keep = f_current.size
    if n_keep == 0:
        raise ValueError("Spectral density is identically zero.")
    weights = -np.log(f_current) / np.log(base)
    return _mvspectrum2wcov(n_keep * f_u, weights) / n_keep


def _em_e_step(f_u: np.ndarray, weightvector: np.ndarray) -> np.ndarray:
    return spectrum_of_linear_combination(f_u, weightvector)


def _em_m_step(
    f_u: np.ndarray,
    f_current: np.ndarray,
    prior_weight: float,
    base: float,
) -> np.ndarray:
    matrix = _entropy_wcov(f_u, f_current, prior_weight=prior_weight, base=base)
    matrix = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(matrix)
    w = vectors[:, int(np.argmin(values))]
    if w[0] < 0:
        w = -w
    return w / np.linalg.norm(w)


def _em_one_weightvector(
    U: np.ndarray,
    f_u: np.ndarray,
    init: np.ndarray,
    max_iter: int,
    tol: float,
    prior_weight: float,
) -> np.ndarray:
    w = np.asarray(init, dtype=float).reshape(-1)
    nrm = np.linalg.norm(w)
    if nrm < 1e-15:
        raise ValueError("Initial weight vector is zero.")
    w = w / nrm
    n_freq = f_u.shape[0]
    base = float(2 * n_freq)
    prev_h = None
    for _ in range(max_iter):
        f_y = _em_e_step(f_u, w)
        matrix = _entropy_wcov(f_u, f_y, prior_weight=prior_weight, base=base)
        h = float(np.real(w @ matrix @ w))
        if prev_h is not None and abs(h - prev_h) < tol:
            break
        prev_h = h
        w = _em_m_step(f_u, f_y, prior_weight=prior_weight, base=base)
    return w


def _one_weightvector(
    U: np.ndarray,
    f_u: Optional[np.ndarray],
    n_starts: int,
    max_iter: int,
    tol: float,
    method: str,
    prior_weight: float,
    random_state: np.random.RandomState,
) -> np.ndarray:
    if f_u is None:
        f_u = mvspectrum(U, method=method, normalize=True)
    k = U.shape[1]
    starters: List[str] = ["rnorm", "sfa.slow", "max", "sfa.fast"]
    inits = []
    for i in range(n_starts):
        if i < len(starters):
            try:
                w0 = initialize_weightvector(
                    k,
                    method=starters[i],
                    U=U,
                    spectrum=f_u,
                    random_state=random_state,
                )
            except Exception:
                w0 = initialize_weightvector(
                    k, method="rnorm", random_state=random_state
                )
        else:
            w0 = initialize_weightvector(k, method="rnorm", random_state=random_state)
        inits.append(w0)

    best_w = inits[0]
    best_omega = -np.inf
    for w0 in inits:
        w = _em_one_weightvector(
            U, f_u, w0, max_iter=max_iter, tol=tol, prior_weight=prior_weight
        )
        fy = spectrum_of_linear_combination(f_u, w)
        om = float(np.asarray(omega(spectrum=fy)).reshape(-1)[0])
        if om > best_omega:
            best_omega = om
            best_w = w
    return best_w


class ForeCA:
    """
    Forecastable Component Analysis (Goerg, JMLR 2013).

    Finds linear combinations of a multivariate series that minimise spectral
    entropy (maximise :func:`omega`). Input ``X`` has shape ``(T, K)``.

    After :meth:`fit`, ``scores`` are mean-zero, unit-variance, uncorrelated
    ForeCs ordered from most to least forecastable.
    """

    def __init__(
        self,
        n_comp: int = 2,
        n_starts: int = 4,
        max_iter: int = 50,
        tol: float = 1e-6,
        method: str = "welch",
        prior_weight: float = 1e-3,
        random_state: Optional[int] = None,
    ) -> None:
        """
        :param n_comp: Number of forecastable components.
        :param n_starts: Random / heuristic EM restarts per component.
        :param max_iter: Maximum EM iterations per restart.
        :param tol: Convergence tolerance on the entropy objective.
        :param method: Spectrum estimator, ``"pgram"`` or ``"welch"``.
        :param prior_weight: Uniform prior mixed into the discrete spectral pmf.
        :param random_state: Seed for the weight-vector initializers.
        """
        if int(n_comp) < 1:
            raise ValueError("n_comp must be a positive integer.")
        self.n_comp = int(n_comp)
        self.n_starts = int(n_starts)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.method = method
        self.prior_weight = float(prior_weight)
        self.random_state = random_state

        self.center_: Optional[np.ndarray] = None
        self.whitening_: Optional[np.ndarray] = None
        self.weightvectors_: Optional[np.ndarray] = None
        self.loadings_: Optional[np.ndarray] = None
        self.scores_: Optional[np.ndarray] = None
        self.omega_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "ForeCA":
        """Fit ForeCA on a multivariate series of shape ``(T, K)``."""
        data = _as_2d(X)
        t, k = data.shape
        if k < 2:
            raise ValueError("ForeCA requires at least 2 series (columns).")
        if self.n_comp > k:
            raise ValueError(
                f"Cannot extract {self.n_comp} components from only {k} series."
            )
        rng = np.random.RandomState(self.random_state)
        pw = whiten(data)
        u = pw.U
        weights = np.zeros((k, self.n_comp))
        scores = np.zeros((t, self.n_comp))
        omegas = np.zeros(self.n_comp)

        basis = np.eye(k)
        for comp in range(self.n_comp):
            u_red = u @ basis
            remaining = u_red.shape[1]
            if remaining == 1:
                w_red = np.ones(1)
            else:
                w_red = _one_weightvector(
                    u_red,
                    f_u=None,
                    n_starts=self.n_starts,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    method=self.method,
                    prior_weight=self.prior_weight,
                    random_state=rng,
                )
            w_full = basis @ w_red.reshape(-1, 1)
            weights[:, comp] = w_full[:, 0]
            scores[:, comp] = (u @ w_full).ravel()
            omegas[comp] = float(
                np.asarray(omega(scores[:, comp], method=self.method)).reshape(-1)[0]
            )
            if comp + 1 < k:
                basis = null_space(weights[:, : comp + 1].T)
                if basis.size == 0:
                    break

        order = np.argsort(-omegas)
        self.center_ = pw.center
        self.whitening_ = pw.whitening
        self.weightvectors_ = weights[:, order]
        self.loadings_ = pw.whitening @ self.weightvectors_
        self.scores_ = scores[:, order]
        self.omega_ = omegas[order]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project new observations with the fitted loadings."""
        if self.loadings_ is None or self.center_ is None:
            raise RuntimeError("Call fit() before transform().")
        data = _as_2d(X)
        return (data - self.center_) @ self.loadings_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit ForeCA and return scores of shape ``(T, n_comp)``."""
        self.fit(X)
        assert self.scores_ is not None
        return self.scores_

    @property
    def scores(self) -> np.ndarray:
        if self.scores_ is None:
            raise RuntimeError("Call fit() before accessing scores.")
        return self.scores_

    @property
    def loadings(self) -> np.ndarray:
        if self.loadings_ is None:
            raise RuntimeError("Call fit() before accessing loadings.")
        return self.loadings_

    @property
    def weightvectors(self) -> np.ndarray:
        if self.weightvectors_ is None:
            raise RuntimeError("Call fit() before accessing weightvectors.")
        return self.weightvectors_

    @property
    def omega(self) -> np.ndarray:
        if self.omega_ is None:
            raise RuntimeError("Call fit() before accessing omega.")
        return self.omega_

    @property
    def whitening(self) -> np.ndarray:
        if self.whitening_ is None:
            raise RuntimeError("Call fit() before accessing whitening.")
        return self.whitening_

    @property
    def center(self) -> np.ndarray:
        if self.center_ is None:
            raise RuntimeError("Call fit() before accessing center.")
        return self.center_
