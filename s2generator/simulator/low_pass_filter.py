# -*- coding: utf-8 -*-
"""
Adaptive / manually configurable low-pass post-processing for simulator outputs.

White-noise excitation of an LTI system can introduce high-frequency roughness
(especially for Wiener-filter style generators). This module estimates a cutoff
from the reference series' cumulative spectral energy so that dominant periodic
content is retained while higher-frequency glitches are attenuated. A manual
``cutoff`` fully overrides the adaptive estimate.

Algorithm sketch
----------------
1. Periodogram ``P[k] = |RFFT(x - mean)|^2`` (DC zeroed).
2. Adaptive ``fc = k*/(K-1)`` where ``k*`` is the first bin whose cumulative
   energy reaches ``energy_ratio``; then clamp to ``[min_cutoff, max_cutoff]``.
3. Butterworth SOS at ``Wn = fc`` (Nyquist = 1) applied with ``sosfiltfilt``
   (zero phase); optional mean/std restore via ``revin``.

See ``LowPassFilter`` for the full description and the example notebook
``examples/simulator/low_pass_fliter.ipynb`` for formulas and visualizations.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
from scipy import signal


ArrayLike = Union[np.ndarray]


class LowPassFilter(object):
    """
    Zero-phase Butterworth low-pass filter with adaptive or manual cutoff.

    Motivation
    ----------
    Simulator pipelines excite a fitted dynamical system with white noise.
    The resulting samples often inherit broadband high-frequency roughness
    (especially Wiener-style AR filters). A fixed, overly low cutoff can erase
    useful mid-band periodicity; this class therefore estimates a data-driven
    cutoff from a **reference** series (typically the ``fit`` target) while
    still allowing a manual override.

    Frequency convention
    --------------------
    Cutoffs are fractions of the Nyquist frequency: ``fc ∈ (0, 1)``. With unit
    sampling rate, the physical cutoff in cycles/sample is ``f = fc / 2``.

    Adaptive cutoff (when ``cutoff`` is ``None``)
    ---------------------------------------------
    1. Mean-center the reference ``x[t]`` and form the one-sided periodogram

       ``P[k] = |RFFT(x - mean(x))[k]|^2``, with the DC bin ``P[0]`` zeroed.

    2. Build the cumulative energy ratio

       ``C[k] = sum_{i=0..k} P[i] / sum_j P[j]``.

    3. Take the smallest bin index ``k*`` with ``C[k*] >= energy_ratio`` (default
       ``0.98``) and map it to a Nyquist fraction

       ``fc = k* / (K - 1)``, where ``K = len(P)``.

    4. Clamp ``fc`` into ``[min_cutoff, max_cutoff]`` (defaults ``0.05``, ``0.95``).

    If a batch of references is provided, their periodograms are averaged before
    step 2. A provided manual ``cutoff`` **fully replaces** steps 1--3.

    Filtering
    ---------
    An order-``order`` digital Butterworth low-pass SOS is built at ``Wn = fc``
    (SciPy convention: Nyquist = 1) and applied with ``sosfiltfilt`` for
    approximately zero phase. Optionally (``revin=True``), each filtered trace
    is affine-rescaled to restore the original mean and standard deviation so
    amplitude scale is preserved after band-limiting.
    """

    def __init__(
        self,
        energy_ratio: float = 0.98,
        cutoff: Optional[float] = None,
        order: int = 4,
        min_cutoff: float = 0.05,
        max_cutoff: float = 0.95,
        revin: bool = True,
    ) -> None:
        """
        :param energy_ratio: Cumulative PSD energy fraction used for adaptive cutoff.
        :param cutoff: Manual cutoff relative to Nyquist. Overrides adaptation when set.
        :param order: Butterworth filter order.
        :param min_cutoff: Lower clamp for the adaptive / manual cutoff.
        :param max_cutoff: Upper clamp for the adaptive / manual cutoff.
        :param revin: If True, restore each filtered series' mean and std.
        """
        if not 0.0 < energy_ratio <= 1.0:
            raise ValueError("energy_ratio must be in (0, 1].")
        if cutoff is not None and not 0.0 < float(cutoff) < 1.0:
            raise ValueError("cutoff must be in (0, 1) when provided.")
        if order < 1:
            raise ValueError("order must be a positive integer.")
        if not 0.0 < min_cutoff < max_cutoff < 1.0:
            raise ValueError("Require 0 < min_cutoff < max_cutoff < 1.")

        self.energy_ratio = float(energy_ratio)
        self.cutoff = None if cutoff is None else float(cutoff)
        self.order = int(order)
        self.min_cutoff = float(min_cutoff)
        self.max_cutoff = float(max_cutoff)
        self.revin = bool(revin)

        self._cutoff: Optional[float] = None
        self._sos: Optional[np.ndarray] = None

    @property
    def cutoff_(self) -> float:
        """Actual cutoff relative to Nyquist after ``fit``."""
        if self._cutoff is None:
            raise ValueError(
                "The low-pass filter has not been fitted yet; please call `fit` first."
            )
        return self._cutoff

    def fit(self, reference: ArrayLike) -> "LowPassFilter":
        """
        Estimate or lock the cutoff from a reference series and build the filter.

        :param reference: Reference / target series, shape ``[T]`` or ``[N, T]``.
        :return: ``self``
        """
        reference = self._as_2d(reference)
        if reference.shape[1] < 8:
            raise ValueError(
                "reference series length must be at least 8 samples for spectral estimation."
            )

        if self.cutoff is not None:
            fc = self.cutoff
        else:
            # Average PSD across rows when a batch of references is provided.
            psd_stack = [self._power_spectrum(row) for row in reference]
            psd = np.mean(np.vstack(psd_stack), axis=0)
            fc = self._cutoff_from_psd(psd)

        self._cutoff = self._clamp_cutoff(fc)
        # butter requires Wn strictly inside (0, 1) for analog=False / fs=None (Nyquist=1)
        wn = float(np.clip(self._cutoff, 1e-4, 1.0 - 1e-4))
        self._sos = signal.butter(self.order, wn, btype="low", output="sos")
        return self

    def transform(self, series: ArrayLike) -> np.ndarray:
        """
        Apply the fitted low-pass filter.

        :param series: Simulated series, shape ``[T]``, ``[N, T]``, or ``[N, T, C]``.
        :return: Filtered series with the same shape as the input.
        """
        if self._sos is None:
            raise ValueError(
                "The low-pass filter has not been fitted yet; please call `fit` first."
            )

        arr = np.asarray(series, dtype=np.float64)
        original_shape = arr.shape

        if arr.ndim == 1:
            return self._filter_1d(arr)

        if arr.ndim == 2:
            out = np.empty_like(arr)
            for i in range(arr.shape[0]):
                out[i] = self._filter_1d(arr[i])
            return out

        if arr.ndim == 3:
            out = np.empty_like(arr)
            for i in range(arr.shape[0]):
                for c in range(arr.shape[2]):
                    out[i, :, c] = self._filter_1d(arr[i, :, c])
            return out

        raise ValueError(
            f"Unsupported series shape {original_shape}; expected 1D, 2D, or 3D."
        )

    def fit_transform(self, reference: ArrayLike, series: ArrayLike) -> np.ndarray:
        """Fit on ``reference`` and filter ``series``."""
        return self.fit(reference).transform(series)

    def _filter_1d(self, series: np.ndarray) -> np.ndarray:
        mean = float(np.mean(series))
        std = float(np.std(series))
        filtered = signal.sosfiltfilt(self._sos, series)
        if self.revin and std > 1e-12:
            f_mean = float(np.mean(filtered))
            f_std = float(np.std(filtered))
            if f_std > 1e-12:
                filtered = (filtered - f_mean) / f_std * std + mean
            else:
                filtered = filtered - f_mean + mean
        return filtered

    def _clamp_cutoff(self, fc: float) -> float:
        return float(np.clip(fc, self.min_cutoff, self.max_cutoff))

    @staticmethod
    def _as_2d(series: ArrayLike) -> np.ndarray:
        arr = np.asarray(series, dtype=np.float64)
        if arr.ndim == 1:
            return arr[None, :]
        if arr.ndim == 2:
            return arr
        if arr.ndim == 3:
            # Flatten batch and channels into rows of length T
            n, t, c = arr.shape
            return np.transpose(arr, (0, 2, 1)).reshape(n * c, t)
        raise ValueError(
            f"Unsupported reference shape {arr.shape}; expected 1D, 2D, or 3D."
        )

    @staticmethod
    def _power_spectrum(series: np.ndarray) -> np.ndarray:
        """One-sided power spectrum via rFFT (robust for short series)."""
        x = np.asarray(series, dtype=np.float64)
        x = x - np.mean(x)
        spectrum = np.fft.rfft(x)
        psd = np.abs(spectrum) ** 2
        # Ignore the DC bin for cumulative-energy cutoff estimation.
        if psd.size > 1:
            psd = psd.copy()
            psd[0] = 0.0
        return psd

    def _cutoff_from_psd(self, psd: np.ndarray) -> float:
        total = float(np.sum(psd))
        if total <= 0.0 or psd.size <= 1:
            return self.max_cutoff
        cumulative = np.cumsum(psd) / total
        idx = int(np.searchsorted(cumulative, self.energy_ratio, side="left"))
        idx = min(max(idx, 1), psd.size - 1)
        # rFFT bins map to [0, Nyquist]; normalize by last bin index.
        return float(idx / (psd.size - 1))


def apply_lowpass(
    series: ArrayLike,
    reference: ArrayLike,
    energy_ratio: float = 0.98,
    cutoff: Optional[float] = None,
    order: int = 4,
    min_cutoff: float = 0.05,
    max_cutoff: float = 0.95,
    revin: bool = True,
) -> np.ndarray:
    """Convenience one-shot adaptive / manual low-pass filtering."""
    return LowPassFilter(
        energy_ratio=energy_ratio,
        cutoff=cutoff,
        order=order,
        min_cutoff=min_cutoff,
        max_cutoff=max_cutoff,
        revin=revin,
    ).fit_transform(reference=reference, series=series)


def maybe_attach_lowpass(
    owner: Any,
    enabled: bool,
    kwargs: Optional[dict],
    reference: ArrayLike,
) -> None:
    """
    Attach a fitted ``LowPassFilter`` to ``owner._lowpass_filter`` when enabled.

    :param owner: Simulator instance.
    :param enabled: Whether low-pass post-processing is requested.
    :param kwargs: Optional constructor kwargs for ``LowPassFilter``.
    :param reference: Reference series used for adaptive cutoff estimation.
    """
    if not enabled:
        owner._lowpass_filter = None
        return
    filter_kwargs = {} if kwargs is None else dict(kwargs)
    lowpass_filter = LowPassFilter(**filter_kwargs)
    owner._lowpass_filter = lowpass_filter.fit(reference)


def maybe_apply_lowpass(owner: Any, series: ArrayLike) -> np.ndarray:
    """Apply ``owner._lowpass_filter`` when present; otherwise return ``series`` unchanged."""
    lowpass_filter = getattr(owner, "_lowpass_filter", None)
    if lowpass_filter is None:
        return np.asarray(series)
    return lowpass_filter.transform(series)
