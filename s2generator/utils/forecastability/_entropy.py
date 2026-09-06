# -*- coding: utf-8 -*-
"""Shannon entropy of a discrete pmf and spectral entropy / Omega scores."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from ._spectrum import _as_2d, univariate_spectrum


def discrete_entropy(
    probs: np.ndarray,
    base: float = 2.0,
    threshold: float = 0.0,
    prior_probs: Optional[np.ndarray] = None,
    prior_weight: float = 0.0,
) -> float:
    """
    Plug-in Shannon entropy of a discrete probability mass function.

    :param probs: Non-negative probabilities that sum to 1.
    :param base: Logarithm base (``2`` = bits).
    :param threshold: Masses below this value are set to 0 then renormalized.
    :param prior_probs: Optional prior mixed in when ``prior_weight > 0``.
                        Defaults to the uniform distribution.
    :param prior_weight: Mixture weight ``lambda`` in ``[0, 1]``.
    :return: Non-negative entropy value.
    """
    p = np.asarray(probs, dtype=float).reshape(-1)
    if p.size == 0:
        raise ValueError("probs must be a non-empty 1-D array.")
    if np.any(np.isnan(p)):
        raise ValueError("probs must not contain NaN.")
    if np.any(p < -1e-6):
        raise ValueError("Not all probabilities are non-negative.")
    p = np.clip(p, 0.0, None)
    if abs(p.sum() - 1.0) > 1e-6:
        raise ValueError("probs must sum to 1.")
    if not (0.0 <= prior_weight <= 1.0):
        raise ValueError("prior_weight must lie in [0, 1].")
    if base <= 0:
        raise ValueError("base must be positive.")

    if threshold > 0:
        p = p.copy()
        p[p < threshold] = 0.0
        total = p.sum()
        if total <= 0:
            raise ValueError("All probability mass was removed by thresholding.")
        p = p / total

    if prior_weight > 0:
        if prior_probs is None:
            prior = np.full(p.size, 1.0 / p.size)
        else:
            prior = np.asarray(prior_probs, dtype=float).reshape(-1)
            if prior.size != p.size:
                raise ValueError("prior_probs must have the same length as probs.")
            if np.any(prior < -1e-6) or abs(prior.sum() - 1.0) > 1e-5:
                raise ValueError("prior_probs must be a valid probability vector.")
            prior = np.clip(prior, 0.0, None)
            prior = prior / prior.sum()
        p = (1.0 - prior_weight) * p + prior_weight * prior

    p = p[p > 1e-9]
    p = p / p.sum()
    return float(-np.sum(p * (np.log(p) / np.log(base))))


def _spectrum_to_density(
    spec: np.ndarray,
    threshold: float = 0.0,
    prior_weight: float = 0.0,
) -> np.ndarray:
    """Mirror a positive-frequency spectrum into a two-sided discrete density."""
    spec = np.asarray(spec, dtype=float).reshape(-1)
    spec = np.maximum(spec, 0.0)
    density = np.concatenate([spec[::-1], spec])
    total = density.sum()
    if total <= 0:
        density = np.full(density.size, 1.0 / density.size)
    else:
        density = density / total
    return density


def spectral_entropy(
    series: Optional[np.ndarray] = None,
    method: str = "welch",
    prior_weight: float = 1e-3,
    threshold: float = 0.0,
    spectrum: Optional[np.ndarray] = None,
    **spectrum_kwargs,
) -> Union[float, np.ndarray]:
    """
    Spectral entropy of a (column-wise) time series.

    The positive-frequency spectrum is mirrored and treated as a discrete pmf.
    The logarithm base is ``2 * n_freq`` so a flat spectrum has entropy 1
    (Goerg 2013; ForeCA ``complete_entropy_control`` with ``base = NULL``).

    :param series: Univariate ``(T,)`` or multivariate ``(T, K)`` array.
    :param method: Spectrum estimator, ``"pgram"`` or ``"welch"``.
    :param prior_weight: Uniform-prior mixture weight for the discrete entropy.
    :param threshold: Drop spectral mass below this probability after mirroring.
    :param spectrum: Optional precomputed positive-frequency spectrum
                     (length ``n_freq`` or shape ``(n_freq, K)``).
    :param spectrum_kwargs: Forwarded to :func:`mvspectrum` / univariate Welch.
    :return: Scalar entropy in ``[0, 1]`` for a univariate series, or a length-``K``
             array when ``series`` / ``spectrum`` is multivariate.
    """
    if (series is None) == (spectrum is None):
        raise ValueError("Provide exactly one of series or spectrum.")

    if spectrum is None:
        data = _as_2d(series)
        n_series = data.shape[1]
        specs = []
        for k in range(n_series):
            _, spec_k = univariate_spectrum(
                data[:, k], method=method, **spectrum_kwargs
            )
            specs.append(spec_k)
        specs = np.column_stack(specs) if n_series > 1 else specs[0]
    else:
        specs = np.asarray(spectrum, dtype=float)
        if specs.ndim == 1:
            n_series = 1
        elif specs.ndim == 2:
            n_series = specs.shape[1]
        else:
            raise ValueError("spectrum must be 1-D or 2-D (n_freq, K).")

    def _one(spec_1d: np.ndarray) -> float:
        density = _spectrum_to_density(spec_1d, threshold=threshold)
        n_outcomes = density.size
        return discrete_entropy(
            density,
            base=float(n_outcomes),
            threshold=threshold,
            prior_weight=prior_weight,
        )

    if n_series == 1:
        spec_1d = specs.reshape(-1) if np.ndim(specs) > 1 else np.asarray(specs)
        return _one(spec_1d)

    return np.array([_one(specs[:, k]) for k in range(n_series)], dtype=float)


def omega(
    series: Optional[np.ndarray] = None,
    method: str = "welch",
    prior_weight: float = 1e-3,
    threshold: float = 0.0,
    spectrum: Optional[np.ndarray] = None,
    **spectrum_kwargs,
) -> Union[float, np.ndarray]:
    """
    Forecastability :math:`\\Omega` of a time series, in percent.

    Defined as :math:`\\Omega = (1 - H_s) \\times 100` where :math:`H_s` is
    the spectral entropy (normalized so white noise is near 0 and a pure
    sinusoid near 100). Multivariate input is scored column-wise.

    :param series: Univariate ``(T,)`` or multivariate ``(T, K)`` array.
                   Columns are standardized (center + scale) before the
                   spectrum is estimated, matching R ``scale()``.
    :param method: ``"pgram"`` or ``"welch"``.
    :param prior_weight: Passed to :func:`spectral_entropy`.
    :param threshold: Passed to :func:`spectral_entropy`.
    :param spectrum: Optional precomputed positive-frequency spectrum.
    :return: Float in ``[0, 100]``, or a length-``K`` array.
    """
    if series is not None:
        data = _as_2d(series)
        mean = data.mean(axis=0)
        std = data.std(axis=0, ddof=1)
        std = np.where(std < 1e-12, 1.0, std)
        series = (data - mean) / std
        if (
            np.asarray(series).shape[1] == 1
            and np.ndim(np.asarray(series).squeeze()) == 1
        ):
            # keep 2d internally; spectral_entropy accepts both
            pass

    h = spectral_entropy(
        series=series,
        method=method,
        prior_weight=prior_weight,
        threshold=threshold,
        spectrum=spectrum,
        **spectrum_kwargs,
    )
    out = (1.0 - np.asarray(h, dtype=float)) * 100.0
    if out.ndim == 0:
        return float(out)
    return out
