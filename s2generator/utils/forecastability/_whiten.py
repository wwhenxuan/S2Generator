# -*- coding: utf-8 -*-
"""ZCA whitening and symmetric matrix square roots."""

from __future__ import annotations

from typing import NamedTuple, Union

import numpy as np


class WhitenResult(NamedTuple):
    """Outputs of :func:`whiten`."""

    U: np.ndarray
    whitening: np.ndarray
    dewhitening: np.ndarray
    center: np.ndarray
    values: np.ndarray


def sqrt_matrix(
    mat: np.ndarray,
    return_sqrt_only: bool = True,
    symmetric: bool = True,
) -> Union[np.ndarray, tuple]:
    """
    Symmetric square root of a square matrix via the eigen-decomposition.

    :param mat: Square ``(K, K)`` array.
    :param return_sqrt_only: If True, return only :math:`A^{1/2}`.
    :param symmetric: Passed to :func:`numpy.linalg.eigh` when True.
    :return: Square-root matrix, or ``(values, vectors, sqrt, sqrt_inverse)``.
    """
    a = np.asarray(mat, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("mat must be a square 2-D array.")
    if symmetric:
        values, vectors = np.linalg.eigh(0.5 * (a + a.T))
    else:
        values, vectors = np.linalg.eig(a)
        values = np.real_if_close(values)
        vectors = np.real_if_close(vectors)

    if np.any(values < -1e-8):
        raise ValueError("Matrix is not positive semi-definite.")
    values = np.clip(values, 0.0, None)
    sqrt_vals = np.sqrt(values)
    sqrt_mat = vectors @ np.diag(sqrt_vals) @ vectors.T
    if return_sqrt_only:
        return sqrt_mat
    if np.any(values < 1e-12):
        raise ValueError("Exact inverse square root requires a full-rank matrix.")
    inv_sqrt = vectors @ np.diag(1.0 / sqrt_vals) @ vectors.T
    return values, vectors, sqrt_mat, inv_sqrt


def whiten(data: np.ndarray) -> WhitenResult:
    """
    Zero-phase (ZCA) whitening: center and map to covariance :math:`I`.

    :param data: ``(T,)`` or ``(T, K)``.
    :return: :class:`WhitenResult` with whitened ``U``, the whitening /
             dewhitening matrices, column means, and covariance eigenvalues.
    """
    from ._spectrum import _as_2d

    x = _as_2d(data)
    center = x.mean(axis=0)
    centered = x - center
    k = centered.shape[1]
    if k == 1:
        var = float(centered.var(axis=0, ddof=1))
        if var < 1e-12:
            raise ValueError("Cannot whiten a constant series.")
        scale = np.sqrt(var)
        whitening = np.array([[1.0 / scale]])
        dewhitening = np.array([[scale]])
        u = centered @ whitening
        return WhitenResult(u, whitening, dewhitening, center, np.array([var]))

    cov = np.cov(centered, rowvar=False)
    if np.allclose(cov, np.eye(k), atol=1e-8):
        u = centered.copy()
        eye = np.eye(k)
        return WhitenResult(u, eye, eye, center, np.ones(k))

    values, _, dewhitening, whitening = sqrt_matrix(
        cov, return_sqrt_only=False, symmetric=True
    )
    u = centered @ whitening
    return WhitenResult(u, whitening, dewhitening, center, values)
