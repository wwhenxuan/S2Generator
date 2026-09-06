# -*- coding: utf-8 -*-
"""Univariate and multivariate spectral density estimates."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy import signal


def _as_2d(series: np.ndarray) -> np.ndarray:
    """Cast a time series to shape ``(T, K)`` with time along axis 0."""
    data = np.asarray(series, dtype=float)
    if data.ndim == 1:
        return data.reshape(-1, 1)
    if data.ndim == 2:
        if data.shape[0] < 2:
            raise ValueError("Need at least 2 time points along axis 0.")
        return data
    raise ValueError(
        "series must be univariate (T,) or multivariate (T, K); "
        f"got shape {data.shape}."
    )


def univariate_spectrum(
    series: np.ndarray,
    method: str = "welch",
    nperseg: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Positive-frequency spectrum of a univariate series (DC omitted).

    :param series: 1-D array of length ``T``.
    :param method: ``"pgram"`` (raw periodogram) or ``"welch"`` (smoothed).
    :param nperseg: Welch segment length; default ``min(256, T)``.
    :return: ``(freqs, spec)`` with ``spec`` of length ``floor(T / 2)`` for
             ``pgram``, or the corresponding Welch grid for ``welch``.
    """
    x = np.asarray(series, dtype=float).reshape(-1)
    if x.size < 4:
        raise ValueError("Need at least 4 observations to estimate a spectrum.")
    method = method.lower()
    if method == "pgram":
        n = x.size
        fft = np.fft.fft(x)
        n_freq = n // 2
        spec = (np.abs(fft[1 : n_freq + 1]) ** 2) / n
        freqs = 2.0 * np.pi * np.arange(1, n_freq + 1) / n
        return freqs, spec.astype(float)
    if method in {"welch", "mvspec"}:
        n = x.size
        seg = nperseg if nperseg is not None else min(256, n)
        seg = max(8, min(seg, n))
        freqs_hz, spec = signal.welch(
            x,
            fs=1.0,
            nperseg=seg,
            noverlap=seg // 2,
            detrend="constant",
            scaling="spectrum",
            return_onesided=True,
        )
        # Drop the DC bin so the grid matches ForeCA (frequencies in (0, pi]).
        if freqs_hz.size > 1 and freqs_hz[0] == 0.0:
            freqs_hz = freqs_hz[1:]
            spec = spec[1:]
        freqs = 2.0 * np.pi * freqs_hz
        return freqs, np.maximum(spec.astype(float), 0.0)
    raise ValueError(f"Unknown spectrum method '{method}'. Use 'pgram' or 'welch'.")


def mvspectrum(
    series: np.ndarray,
    method: str = "welch",
    normalize: bool = False,
    nperseg: Optional[int] = None,
) -> np.ndarray:
    """
    Estimate the (multivariate) spectrum on positive frequencies.

    Univariate input returns a 1-D spectrum of length ``n_freq``. Multivariate
    input of shape ``(T, K)`` returns a Hermitian array of shape
    ``(n_freq, K, K)``.

    :param series: ``(T,)`` or ``(T, K)``.
    :param method: ``"pgram"`` or ``"welch"`` (alias ``"mvspec"``).
    :param normalize: If True, rescale so positive frequencies sum to 0.5
                      (univariate) or ``0.5 I_K`` on the diagonal (multivariate).
                      The series should already be whitened when this is True.
    :param nperseg: Welch segment length.
    :return: Spectrum array. Frequencies are stored as ``.freqs`` on the
             returned ndarray (a view with a custom attribute).
    """
    data = _as_2d(series)
    t, k = data.shape
    method = method.lower()

    if k == 1:
        freqs, spec = univariate_spectrum(data[:, 0], method=method, nperseg=nperseg)
        spec = spec.astype(float)
        if normalize:
            spec = normalize_mvspectrum(spec)
        return np.asarray(spec)

    if method == "pgram":
        freqs, spec = _mvpgram(data)
    elif method in {"welch", "mvspec"}:
        freqs, spec = _mvwelch(data, nperseg=nperseg)
    else:
        raise ValueError(f"Unknown spectrum method '{method}'. Use 'pgram' or 'welch'.")

    if normalize:
        spec = normalize_mvspectrum(spec)
    return np.asarray(spec)


def _mvpgram(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    t, k = data.shape
    n_freq = t // 2
    fft = np.fft.fft(data, axis=0)
    pos = fft[1 : n_freq + 1]
    spec = np.empty((n_freq, k, k), dtype=complex)
    for i in range(k):
        for j in range(k):
            spec[:, i, j] = pos[:, i] * np.conjugate(pos[:, j]) / t
    freqs = 2.0 * np.pi * np.arange(1, n_freq + 1) / t
    return freqs, spec


def _mvwelch(
    data: np.ndarray, nperseg: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    t, k = data.shape
    seg = nperseg if nperseg is not None else min(256, t)
    seg = max(8, min(seg, t))
    spec_blocks = []
    freqs_hz = None
    for i in range(k):
        row = []
        for j in range(k):
            f, pxy = signal.csd(
                data[:, i],
                data[:, j],
                fs=1.0,
                nperseg=seg,
                noverlap=seg // 2,
                detrend="constant",
                scaling="spectrum",
                return_onesided=True,
            )
            if freqs_hz is None:
                freqs_hz = f
            row.append(pxy)
        spec_blocks.append(row)
    spec = np.stack([np.stack(spec_blocks[i], axis=1) for i in range(k)], axis=1)
    # spec currently (n_freq, K, K) after stacking: wait
    # spec_blocks[i][j] is (n_freq,)
    # np.stack(row, axis=1) -> (n_freq, K) for each i
    # then stack axis=1 -> (n_freq, K, K)? stack of K arrays (n_freq, K) on axis 1
    # gives (n_freq, K, K). Yes.

    freqs_hz = np.asarray(freqs_hz)
    if freqs_hz.size > 1 and freqs_hz[0] == 0.0:
        freqs_hz = freqs_hz[1:]
        spec = spec[1:]
    freqs = 2.0 * np.pi * freqs_hz
    # Hermitian symmetrize numerically
    spec = 0.5 * (spec + np.conjugate(np.transpose(spec, (0, 2, 1))))
    return freqs, spec


def normalize_mvspectrum(spec: np.ndarray) -> np.ndarray:
    """
    Normalize a spectrum so positive frequencies add up to 0.5 (univariate)
    or a Hermitian matrix with 0.5 on the diagonal (multivariate).
    """
    spec = np.asarray(spec)
    if spec.ndim == 1:
        total = spec.real.sum()
        if total <= 0:
            raise ValueError("Cannot normalize a non-positive spectrum.")
        out = spec.real / total / 2.0
        return out

    if spec.ndim != 3 or spec.shape[1] != spec.shape[2]:
        raise ValueError("Multivariate spectrum must have shape (n_freq, K, K).")

    # Integrated covariance over positive frequencies, doubled for the full band.
    cov_hat = 2.0 * np.real(spec.sum(axis=0))
    cov_hat = 0.5 * (cov_hat + cov_hat.T)
    from ._whiten import sqrt_matrix

    _, _, _, cov_inv_sqrt = sqrt_matrix(cov_hat, return_sqrt_only=False)
    n_freq, k, _ = spec.shape
    out = np.empty_like(spec, dtype=complex)
    for f in range(n_freq):
        out[f] = 2.0 * cov_inv_sqrt.conj().T @ spec[f] @ cov_inv_sqrt
    return out / 2.0


def spectrum_of_linear_combination(spec: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Spectrum of :math:`y_t = X_t w` via the quadratic form
    :math:`f_y(\\lambda) = w^H f_X(\\lambda) w`.

    :param spec: Multivariate spectrum of shape ``(n_freq, K, K)``.
    :param weights: Length-``K`` vector (not necessarily unit-norm).
    :return: Real non-negative univariate spectrum of length ``n_freq``.
    """
    s = np.asarray(spec)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if s.ndim != 3 or s.shape[1] != s.shape[2]:
        raise ValueError("spec must have shape (n_freq, K, K).")
    if w.size != s.shape[1]:
        raise ValueError("weights length must match the spectrum dimension K.")
    # einsum: for each freq, w^H S w
    fy = np.einsum("i,fij,j->f", w, s, w)
    return np.real(fy).clip(min=0.0)
