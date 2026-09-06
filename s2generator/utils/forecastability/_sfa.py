# -*- coding: utf-8 -*-
"""Slow Feature Analysis (linear sfa1) for temporally dependent signals."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ._spectrum import _as_2d
from ._whiten import whiten


class SlowFeatureAnalysis:
    """
    Linear Slow Feature Analysis (Wiskott & Sejnowski 2002; ForeCA ``sfa``).

    After ZCA whitening, the covariance of first differences is diagonalized.
    Components are ordered from **slowest** to **fastest**.

    Input ``X`` has shape ``(T, K)`` with time along axis 0.
    """

    def __init__(self, n_comp: Optional[int] = None) -> None:
        """
        :param n_comp: Number of components to keep. Default: all ``K``.
        """
        if n_comp is not None and int(n_comp) < 1:
            raise ValueError("n_comp must be a positive integer.")
        self.n_comp = None if n_comp is None else int(n_comp)
        self.center_: Optional[np.ndarray] = None
        self.whitening_: Optional[np.ndarray] = None
        self.loadings_: Optional[np.ndarray] = None
        self.eigenvalues_: Optional[np.ndarray] = None
        self.scores_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "SlowFeatureAnalysis":
        """Fit SFA on a multivariate series of shape ``(T, K)``."""
        data = _as_2d(X)
        t, k = data.shape
        if k < 2:
            raise ValueError("SFA requires at least 2 series (columns).")
        n_comp = k if self.n_comp is None else self.n_comp
        if n_comp > k:
            raise ValueError(f"Cannot extract {n_comp} components from {k} series.")

        pw = whiten(data)
        # Eigenvectors of cov(diff(U)): small eigenvalue = slow.
        # ForeCA uses eigen(solve(Sigma_delta)) so large lambda = slow, then
        # reverses the order. Equivalent: eigh(Sigma_delta) ascending.
        delta = np.diff(pw.U, axis=0)
        sigma_delta = np.cov(delta, rowvar=False)
        values, vectors = np.linalg.eigh(0.5 * (sigma_delta + sigma_delta.T))
        # ascending eigenvalues of Sigma_delta = slowest first
        order = np.argsort(values)
        vectors = vectors[:, order]
        values = values[order]

        loadings = pw.whitening @ vectors
        scores = pw.U @ vectors
        self.center_ = pw.center
        self.whitening_ = pw.whitening
        self.loadings_ = loadings[:, :n_comp]
        self.eigenvalues_ = values[:n_comp]
        self.scores_ = scores[:, :n_comp]
        self.n_comp = n_comp
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project new data with the fitted loadings."""
        if self.loadings_ is None or self.center_ is None:
            raise RuntimeError("Call fit() before transform().")
        data = _as_2d(X)
        return (data - self.center_) @ self.loadings_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit SFA and return the slow-feature scores of shape ``(T, n_comp)``."""
        self.fit(X)
        assert self.scores_ is not None
        return self.scores_

    @property
    def scores(self) -> np.ndarray:
        """Fitted scores of shape ``(T, n_comp)``."""
        if self.scores_ is None:
            raise RuntimeError("Call fit() before accessing scores.")
        return self.scores_

    @property
    def loadings(self) -> np.ndarray:
        """Loadings of shape ``(K, n_comp)`` mapping original series to scores."""
        if self.loadings_ is None:
            raise RuntimeError("Call fit() before accessing loadings.")
        return self.loadings_
