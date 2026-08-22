# -*- coding: utf-8 -*-
"""
Pairwise correlation / similarity matrices for multivariate time series.

Input convention throughout: ``time_series`` has shape ``[num_samples, seq_length]``
with ``num_samples >= 2`` (each row is one channel / variate).

Created on 2026/03/22
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""

from __future__ import annotations

__all__ = [
    "AVAILABLE_CORRELATION_MEASURES",
    "parse_correlation_measures",
    "multivariate_correlation",
    "pearson_correlation_matrix",
    "spearman_correlation_matrix",
    "autocorrelation_similarity_matrix",
    "power_spectrum_similarity_matrix",
    "distribution_similarity_matrix",
    "wasserstein_distance_correlation_matrix",
]

from typing import List, Optional, Sequence, Union

import numpy as np
from scipy import signal, stats
from statsmodels.tsa.stattools import acf

from ._wasserstein_distance import wasserstein_distance

# Canonical measure names exposed to users
AVAILABLE_CORRELATION_MEASURES = (
    "pearson",
    "spearman",
    "autocorrelation",
    "power_spectrum",
    "distribution",
    "wasserstein",
)

# Aliases → canonical name
_MEASURE_ALIASES = {
    "pearson": "pearson",
    "corr": "pearson",
    "correlation": "pearson",
    "spearman": "spearman",
    "autocorrelation": "autocorrelation",
    "acf": "autocorrelation",
    "auto": "autocorrelation",
    "power_spectrum": "power_spectrum",
    "psd": "power_spectrum",
    "spectrum": "power_spectrum",
    "power": "power_spectrum",
    "distribution": "distribution",
    "dist": "distribution",
    "histogram": "distribution",
    "wasserstein": "wasserstein",
    "wdist": "wasserstein",
    "ws": "wasserstein",
}


def _validate_multivariate(time_series: np.ndarray) -> np.ndarray:
    """Validate and return float ndarray of shape ``[N, L]`` with ``N >= 2``."""
    data = np.asarray(time_series, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(
            "time_series must have shape [num_samples, seq_length], "
            f"got shape {data.shape}"
        )
    n_samples, seq_len = data.shape
    if n_samples < 2:
        raise ValueError(
            f"num_samples must be >= 2 for correlation matrices, got {n_samples}"
        )
    if seq_len < 2:
        raise ValueError(f"seq_length must be >= 2, got {seq_len}")
    return data


def parse_correlation_measures(
    measure: Union[str, Sequence[str]],
) -> List[str]:
    """
    Parse a measure specification into a list of canonical measure names.

    Accepts a single name, a space-separated string, or a sequence of names.
    """
    if isinstance(measure, str):
        tokens = [t for t in measure.replace(",", " ").split() if t]
    elif isinstance(measure, Sequence):
        tokens = []
        for item in measure:
            if not isinstance(item, str):
                raise TypeError(
                    f"measure list entries must be strings, got {type(item)}"
                )
            tokens.extend(t for t in item.replace(",", " ").split() if t)
    else:
        raise TypeError(
            "measure must be a string or a sequence of strings, " f"got {type(measure)}"
        )

    if not tokens:
        raise ValueError("at least one correlation measure must be provided")

    canonical: List[str] = []
    for token in tokens:
        key = token.strip().lower()
        if key not in _MEASURE_ALIASES:
            raise ValueError(
                f"unknown correlation measure {token!r}; "
                f"choose from {AVAILABLE_CORRELATION_MEASURES} "
                f"(aliases: {sorted(set(_MEASURE_ALIASES) - set(AVAILABLE_CORRELATION_MEASURES))})"
            )
        name = _MEASURE_ALIASES[key]
        if name not in canonical:
            canonical.append(name)
    return canonical


def pearson_correlation_matrix(time_series: np.ndarray) -> np.ndarray:
    """Pearson correlation matrix between rows (channels)."""
    data = _validate_multivariate(time_series)
    # np.corrcoef expects variables in rows when rowvar=True (default)
    corr = np.corrcoef(data)
    return np.asarray(corr, dtype=np.float64)


def spearman_correlation_matrix(time_series: np.ndarray) -> np.ndarray:
    """Spearman rank correlation matrix between rows (channels)."""
    data = _validate_multivariate(time_series)
    corr, _ = stats.spearmanr(data, axis=1)
    corr = np.asarray(corr, dtype=np.float64)
    if corr.ndim == 0:
        # Two series only → scalar
        n = data.shape[0]
        mat = np.eye(n, dtype=np.float64)
        mat[0, 1] = mat[1, 0] = float(corr)
        return mat
    return corr


def _safe_acf(series: np.ndarray, nlags: int) -> np.ndarray:
    """ACF vector with NaN/Inf sanitized."""
    values = acf(series, nlags=nlags, fft=True)
    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return values


def autocorrelation_similarity_matrix(
    time_series: np.ndarray,
    nlags: Optional[int] = None,
) -> np.ndarray:
    """
    Similarity of ACF shapes: Pearson correlation between ACF vectors of each pair.
    """
    data = _validate_multivariate(time_series)
    n_samples, seq_len = data.shape
    if nlags is None:
        nlags = min(40, max(1, seq_len // 4))
    nlags = int(min(nlags, seq_len - 1))

    acf_mat = np.stack([_safe_acf(data[i], nlags=nlags) for i in range(n_samples)])
    # Constant ACF rows → corrcoef may warn / yield NaN
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(acf_mat)
    corr = np.asarray(corr, dtype=np.float64)
    return np.nan_to_num(corr, nan=0.0)


def _power_spectrum(series: np.ndarray) -> np.ndarray:
    """Normalized Welch PSD magnitude vector."""
    nperseg = min(256, max(8, len(series) // 2))
    _, psd = signal.welch(series, nperseg=nperseg)
    psd = np.asarray(psd, dtype=np.float64)
    total = psd.sum()
    if total > 0:
        psd = psd / total
    return psd


def power_spectrum_similarity_matrix(time_series: np.ndarray) -> np.ndarray:
    """
    Similarity of power spectra: Pearson correlation between Welch PSD vectors.
    """
    data = _validate_multivariate(time_series)
    n_samples = data.shape[0]
    spectra = np.stack([_power_spectrum(data[i]) for i in range(n_samples)])
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(spectra)
    corr = np.asarray(corr, dtype=np.float64)
    return np.nan_to_num(corr, nan=0.0)


def distribution_similarity_matrix(
    time_series: np.ndarray,
    bins: int = 32,
) -> np.ndarray:
    """
    Similarity of empirical value distributions via Pearson correlation of
    shared-bin histograms.
    """
    data = _validate_multivariate(time_series)
    n_samples = data.shape[0]
    global_min = float(np.min(data))
    global_max = float(np.max(data))
    if not np.isfinite(global_min) or not np.isfinite(global_max):
        raise ValueError("time_series contains non-finite values")
    if global_max <= global_min:
        global_max = global_min + 1.0

    edges = np.linspace(global_min, global_max, bins + 1)
    hists = []
    for i in range(n_samples):
        hist, _ = np.histogram(data[i], bins=edges, density=False)
        hist = hist.astype(np.float64)
        total = hist.sum()
        hists.append(hist / total if total > 0 else hist)
    hist_mat = np.stack(hists)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(hist_mat)
    corr = np.asarray(corr, dtype=np.float64)
    return np.nan_to_num(corr, nan=0.0)


def wasserstein_distance_correlation_matrix(
    time_series: np.ndarray,
    mean_weight: float = 0.5,
    covar_weight: float = 0.5,
) -> np.ndarray:
    """
    Pairwise 1D / windowed Wasserstein distances between channels.

    Each channel ``i`` is treated as a length-``L`` series. For the dataset-level
    Wasserstein metric from ``_wasserstein_distance``, each channel is reshaped
    into a small windowed dataset ``[n_windows, window]`` so mean/covariance are
    well-defined; if the series is too short, falls back to
    ``scipy.stats.wasserstein_distance`` on the raw values.

    Returns a **distance** matrix (diagonal 0); smaller means more similar.
    """
    data = _validate_multivariate(time_series)
    n_samples, seq_len = data.shape
    dist = np.zeros((n_samples, n_samples), dtype=np.float64)

    window = min(32, max(4, seq_len // 8))
    use_dataset_metric = seq_len >= 2 * window

    def _as_dataset(series: np.ndarray) -> np.ndarray:
        # Non-overlapping windows → [n_windows, window]
        n_win = seq_len // window
        trimmed = series[: n_win * window].reshape(n_win, window)
        return trimmed

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            if use_dataset_metric:
                xi = _as_dataset(data[i])
                xj = _as_dataset(data[j])
                if xi.shape[0] >= 2 and xj.shape[0] >= 2:
                    d = wasserstein_distance(
                        xi.copy(),
                        xj.copy(),
                        mean_weight=mean_weight,
                        covar_weight=covar_weight,
                    )
                    if d is None or not np.isfinite(d):
                        d = float(stats.wasserstein_distance(data[i], data[j]))
                else:
                    d = float(stats.wasserstein_distance(data[i], data[j]))
            else:
                d = float(stats.wasserstein_distance(data[i], data[j]))
            dist[i, j] = dist[j, i] = float(d)
    return dist


def multivariate_correlation(
    time_series: np.ndarray,
    measure: Union[str, Sequence[str]] = "pearson",
    **kwargs,
) -> Union[np.ndarray, dict]:
    """
    Compute pairwise correlation / similarity / distance matrices.

    :param time_series: Array of shape ``[num_samples, seq_length]``, ``N >= 2``.
    :param measure: One measure name, a space-separated string, or a list of names.
                    Supported: ``pearson``, ``spearman``, ``autocorrelation``,
                    ``power_spectrum``, ``distribution``, ``wasserstein``.
    :param kwargs: Forwarded to specific estimators when relevant
                   (``nlags``, ``bins``, ``mean_weight``, ``covar_weight``).
    :return: A single ``(N, N)`` matrix if one measure is requested; otherwise a
             ``dict`` mapping measure name → matrix.
    """
    measures = parse_correlation_measures(measure)
    data = _validate_multivariate(time_series)

    results = {}
    for name in measures:
        if name == "pearson":
            results[name] = pearson_correlation_matrix(data)
        elif name == "spearman":
            results[name] = spearman_correlation_matrix(data)
        elif name == "autocorrelation":
            results[name] = autocorrelation_similarity_matrix(
                data, nlags=kwargs.get("nlags")
            )
        elif name == "power_spectrum":
            results[name] = power_spectrum_similarity_matrix(data)
        elif name == "distribution":
            results[name] = distribution_similarity_matrix(
                data, bins=int(kwargs.get("bins", 32))
            )
        elif name == "wasserstein":
            results[name] = wasserstein_distance_correlation_matrix(
                data,
                mean_weight=float(kwargs.get("mean_weight", 0.5)),
                covar_weight=float(kwargs.get("covar_weight", 0.5)),
            )
        else:  # pragma: no cover
            raise ValueError(f"unsupported measure: {name}")

    if len(measures) == 1:
        return results[measures[0]]
    return results
