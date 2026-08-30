# -*- coding: utf-8 -*-
"""
Hammerstein–Wiener block-oriented simulator for white-noise-to-target generation.

Created for S2Generator simulator suite.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import signal
from scipy.interpolate import PchipInterpolator
from scipy.linalg import toeplitz
from statsmodels.tsa.stattools import acf

from s2generator.utils._tools import yule_walker
from s2generator.simulator.low_pass_filter import (
    maybe_attach_lowpass,
    maybe_apply_lowpass,
)


def _vandermonde(x: np.ndarray, degree: int) -> np.ndarray:
    """Columns ``[1, x, x^2, ..., x^degree]``."""
    x = np.asarray(x, dtype=np.float64).ravel()
    return np.column_stack([x**k for k in range(degree + 1)])


def apply_polynomial(x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """Evaluate ``sum_k c[k] * x**k`` elementwise (Horner)."""
    x = np.asarray(x, dtype=np.float64)
    coeffs = np.asarray(coeffs, dtype=np.float64).ravel()
    out = np.zeros_like(x, dtype=np.float64)
    for c in coeffs[::-1]:
        out = out * x + c
    return out


def fit_polynomial_ridge(
    x: np.ndarray,
    y: np.ndarray,
    degree: int,
    ridge: float = 1e-6,
) -> np.ndarray:
    """Ridge least-squares polynomial fit of given degree."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size != y.size:
        raise ValueError("x and y must have the same length.")
    if degree < 0:
        raise ValueError("degree must be >= 0.")
    if x.size < degree + 1:
        raise ValueError("Need at least degree+1 samples to fit the polynomial.")

    phi = _vandermonde(x, degree)
    scales = np.ones(degree + 1, dtype=np.float64)
    for k in range(1, degree + 1):
        s = float(np.std(phi[:, k]))
        if s > 1e-12:
            phi[:, k] = phi[:, k] / s
            scales[k] = s

    reg = ridge * np.eye(degree + 1)
    reg[0, 0] = 0.0
    beta = np.linalg.solve(phi.T @ phi + reg, phi.T @ y)
    return beta / scales


def _quantile_knots(
    source_samples: np.ndarray,
    target_samples: np.ndarray,
    n_quantiles: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build sorted unique quantile knots mapping source → target."""
    source_samples = np.asarray(source_samples, dtype=np.float64).ravel()
    target_samples = np.asarray(target_samples, dtype=np.float64).ravel()
    n_quantiles = int(
        max(8, min(n_quantiles, source_samples.size, target_samples.size))
    )
    probs = np.linspace(0.001, 0.999, n_quantiles)
    x_q = np.quantile(source_samples, probs)
    y_q = np.quantile(target_samples, probs)
    # Enforce strictly increasing x for PCHIP
    order = np.argsort(x_q)
    x_q, y_q = x_q[order], y_q[order]
    uniq_x, idx = np.unique(x_q, return_index=True)
    return uniq_x, y_q[idx]


def apply_static_map(
    x: np.ndarray,
    x_knots: np.ndarray,
    y_knots: np.ndarray,
) -> np.ndarray:
    """Monotone static nonlinearity via clipped PCHIP on quantile knots."""
    x = np.asarray(x, dtype=np.float64)
    x_knots = np.asarray(x_knots, dtype=np.float64).ravel()
    y_knots = np.asarray(y_knots, dtype=np.float64).ravel()
    if x_knots.size < 2:
        return np.full_like(x, float(y_knots[0]) if y_knots.size else 0.0)
    x_clip = np.clip(x, x_knots[0], x_knots[-1])
    return PchipInterpolator(x_knots, y_knots, extrapolate=False)(x_clip)


def fit_quantile_polynomial(
    source_samples: np.ndarray,
    target_samples: np.ndarray,
    degree: int,
    n_quantiles: int = 64,
    ridge: float = 1e-6,
) -> np.ndarray:
    """
    Fit a polynomial on source→target quantile pairs (diagnostic / API helper).
    """
    x_q, y_q = _quantile_knots(source_samples, target_samples, n_quantiles=n_quantiles)
    return fit_polynomial_ridge(x_q, y_q, degree=degree, ridge=ridge)


class HammersteinWienerSimulator(object):
    """
    Simulate new time series with a Hammerstein–Wiener (HW) block structure.

    Classical **Wiener filter** theory seeks an *optimal linear* operator under
    a mean-square criterion and therefore matches targets primarily through
    **second-order** statistics (autocorrelation / power spectrum).  In contrast,
    the **Hammerstein–Wiener model** is a *block-oriented nonlinear system*:

    .. code-block:: text

        w  --►  f(·)  --►  G(z)  --►  g(·)  --►  y
              Hammerstein   linear     Wiener
              (static)      dynamics   (static)

    Generation uses

    .. math::

        u = f(w), \\qquad v = G \\ast u, \\qquad y = g(v),

    where ``w`` is white noise, ``f`` / ``g`` are **static nonlinearities**
    (monotone quantile maps, with optional polynomial summaries), and ``G`` is
    the same Yule–Walker AR / causal linear core used by
    :class:`~s2generator.simulator.wiener_filter.WienerFilterSimulator`.

    **Why this helps.**  A purely linear Wiener-style generator excited by
    Gaussian noise cannot reproduce skewed or heavy-tailed amplitude laws even
    when the ACF is correct.  The static maps ``f`` and ``g`` absorb those
    higher-order effects while ``G`` still shapes the spectrum — so HW typically
    matches histograms / skewness / kurtosis of nonlinear targets much better
    than linear Wiener alone, without giving up second-order fidelity.

    Fitting (staged):

    1. Estimate linear ``G`` by Yule–Walker on the (optionally ReVIN-normalized)
       target — same ACF matching as the linear Wiener simulator.
    2. Estimate Hammerstein ``f`` as a quantile map
       ``N(0,1) →`` AR-residual law (monotone PCHIP), scaled to innovation power.
    3. Estimate Wiener ``g`` as a quantile map from the law of ``G(f(w))`` to
       the law of the target.

    Polynomial coefficients of the same quantile pairs are also stored for
    inspection (``input_coeffs`` / ``output_coeffs``).

    The public API mirrors other simulators: ``fit`` / ``transform`` / ``invoke``,
    with optional ``revin`` and low-pass post-processing.
    """

    def __init__(
        self,
        filter_order: int = 6,
        input_degree: int = 3,
        output_degree: int = 3,
        n_quantiles: int = 64,
        ridge: float = 1e-4,
        revin: Optional[bool] = True,
        random_state: Optional[int] = 42,
        lowpass: bool = False,
        lowpass_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        :param filter_order: Length of the AR coefficient vector (includes ``a0=1``).
        :param input_degree: Degree of the polynomial summary of Hammerstein ``f``.
        :param output_degree: Degree of the polynomial summary of Wiener ``g``.
        :param n_quantiles: Number of quantile knots for the static maps.
        :param ridge: Ridge strength for polynomial summaries.
        :param revin: If True, fit in z-scored space and restore mean/std on output.
        :param random_state: Seed for the internal RNG.
        :param lowpass: Optional low-pass post-process after ``transform``.
        :param lowpass_kwargs: Forwarded to ``LowPassFilter``.
        """
        if filter_order < 2:
            raise ValueError("filter_order must be >= 2.")
        if input_degree < 0 or output_degree < 0:
            raise ValueError("Polynomial degrees must be non-negative.")

        self.filter_order = int(filter_order)
        self.lag_max = self.filter_order * 2
        self.input_degree = int(input_degree)
        self.output_degree = int(output_degree)
        self.n_quantiles = int(max(8, n_quantiles))
        self.ridge = float(ridge)

        self.acf_vals = None
        self.R = None
        self._coeffs = None
        self._sigma_sq = None
        self._input_coeffs = None
        self._output_coeffs = None
        self._input_xq = None
        self._input_yq = None
        self._output_xq = None
        self._output_yq = None
        self._v_scale = 1.0

        self.revin = bool(revin)
        self.mean, self.std = None, None

        self.random_state = random_state
        self.rng = np.random.RandomState(seed=random_state)

        self.lowpass = lowpass
        self.lowpass_kwargs = lowpass_kwargs
        self._lowpass_filter = None

        self.time_series = None
        self.residuals = None
        self.simulated_series = None

    def __str__(self) -> str:
        return self.__class__.__name__

    @property
    def coeffs(self) -> np.ndarray:
        if self._coeffs is None:
            raise ValueError("Model is not fitted; call `fit` first.")
        return self._coeffs

    @property
    def sigma_sq(self) -> float:
        if self._sigma_sq is None:
            raise ValueError("Model is not fitted; call `fit` first.")
        return float(self._sigma_sq)

    @property
    def input_coeffs(self) -> np.ndarray:
        if self._input_coeffs is None:
            raise ValueError("Model is not fitted; call `fit` first.")
        return self._input_coeffs

    @property
    def output_coeffs(self) -> np.ndarray:
        if self._output_coeffs is None:
            raise ValueError("Model is not fitted; call `fit` first.")
        return self._output_coeffs

    def acf(
        self,
        time_series: np.ndarray,
        lag_max: Optional[int] = None,
        fft: Optional[bool] = True,
    ) -> np.ndarray:
        """Autocorrelation of a 1-D series up to ``lag_max``."""
        return acf(
            time_series,
            nlags=self.lag_max if lag_max is None else lag_max,
            fft=fft,
        )

    def check_inputs(self, time_series: np.ndarray) -> np.ndarray:
        """Validate and flatten the target series to 1-D float64."""
        if not isinstance(time_series, np.ndarray):
            raise ValueError("The input time series must be a NumPy array.")
        if time_series.ndim > 2:
            raise ValueError(
                "The input time series must be 1D [seq_len,] or 2D "
                f"[num_samples, seq_len], got shape {time_series.shape}."
            )
        if time_series.ndim == 2:
            time_series = time_series.reshape(-1)
        time_series = np.asarray(time_series, dtype=np.float64)
        if len(time_series) < 2 * self.filter_order:
            raise ValueError(
                f"Input length must be at least {2 * self.filter_order} "
                "for stable ACF / Yule–Walker estimation."
            )
        if not np.all(np.isfinite(time_series)):
            raise ValueError("Input time series must be finite.")
        if float(np.std(time_series)) < 1e-12:
            raise ValueError("Input time series has (near) zero variance.")
        return time_series

    def _ar_residuals(self, series: np.ndarray) -> np.ndarray:
        """Apply the AR polynomial ``A(z)`` to obtain approximate innovations."""
        # lfilter(b=A, a=[1], x) = A(z) x
        e = signal.lfilter(b=self._coeffs, a=np.array([1.0]), x=series)
        return e[self.filter_order :]

    def _linear_filter(self, drive: np.ndarray) -> np.ndarray:
        """Apply ``G = 1/A`` and drop the warm-up prefix."""
        filtered = signal.lfilter(a=self._coeffs, b=np.array([1.0]), x=drive)
        return filtered[self.filter_order :]

    def fit(self, time_series: np.ndarray) -> "HammersteinWienerSimulator":
        """
        Fit Hammerstein ``f``, linear ``G``, and Wiener ``g`` to the target series.
        """
        time_series = self.check_inputs(time_series)
        self.time_series = time_series.copy()

        y = time_series
        if self.revin:
            self.mean = float(np.mean(y))
            self.std = float(np.std(y))
            if self.std < 1e-12:
                self.std = 1.0
            y = (y - self.mean) / self.std
        else:
            self.mean, self.std = 0.0, 1.0

        # ---- 1) Linear block G via Yule–Walker ----
        self.acf_vals = self.acf(time_series=y)
        self.R = toeplitz(self.acf_vals[: self.filter_order])
        self._coeffs, self._sigma_sq = yule_walker(A=self.R)
        self._sigma_sq = float(max(float(self._sigma_sq), 1e-12))

        # ---- 2) Hammerstein f: N(0,1) → residual law (monotone quantile map) ----
        residuals = self._ar_residuals(y)
        res_std = float(np.std(residuals))
        if res_std < 1e-12:
            res_std = 1.0
        residuals_z = (residuals - float(np.mean(residuals))) / res_std

        gauss = self.rng.normal(0.0, 1.0, size=max(len(residuals_z), 512))
        self._input_xq, self._input_yq = _quantile_knots(
            gauss, residuals_z, n_quantiles=self.n_quantiles
        )
        # Scale map outputs so Var(f(N(0,1))) ≈ sigma_sq
        probe = apply_static_map(
            self.rng.normal(0.0, 1.0, size=4096), self._input_xq, self._input_yq
        )
        probe_std = float(np.std(probe))
        if probe_std > 1e-12:
            scale = np.sqrt(self._sigma_sq) / probe_std
            self._input_yq = self._input_yq * scale
        self._input_coeffs = fit_polynomial_ridge(
            self._input_xq, self._input_yq, degree=self.input_degree, ridge=self.ridge
        )

        # ---- 3) Wiener g: law(G(f(w))) → law(y) ----
        w_ref = self.rng.normal(
            0.0, 1.0, size=max(len(y) * 2, 512) + self.filter_order
        )
        u_ref = apply_static_map(w_ref, self._input_xq, self._input_yq)
        v_ref = self._linear_filter(u_ref)
        v_std = float(np.std(v_ref))
        self._v_scale = v_std if v_std > 1e-12 else 1.0
        v_unit = v_ref / self._v_scale

        self._output_xq, self._output_yq = _quantile_knots(
            v_unit, y, n_quantiles=self.n_quantiles
        )
        self._output_coeffs = fit_polynomial_ridge(
            self._output_xq,
            self._output_yq,
            degree=self.output_degree,
            ridge=self.ridge,
        )

        # Diagnostics on a fresh draw through the full chain
        w_diag = self.rng.normal(0.0, 1.0, size=len(y) + self.filter_order)
        y_hat = self.invoke(w_diag)
        n = min(len(y), len(y_hat))
        self.residuals = y[-n:] - y_hat[-n:]

        maybe_attach_lowpass(
            self,
            enabled=self.lowpass,
            kwargs=self.lowpass_kwargs,
            reference=self.time_series,
        )
        return self

    def invoke(self, white_noise: np.ndarray) -> np.ndarray:
        """
        Push a prepared white-noise path through ``f → G → g``.

        :param white_noise: 1-D array of length ``seq_len + filter_order``
            (treated as approximately unit Gaussian; only lightly re-scaled).
        :return: Simulated series of length ``seq_len`` (normalized space if ``revin``).
        """
        if (
            self._coeffs is None
            or self._input_xq is None
            or self._output_xq is None
        ):
            raise ValueError("Model is not fitted; call `fit` first.")

        white_noise = np.asarray(white_noise, dtype=np.float64).ravel()
        if white_noise.size <= self.filter_order:
            raise ValueError(
                "white_noise length must exceed filter_order "
                f"(got {white_noise.size}, need > {self.filter_order})."
            )

        w_std = float(np.std(white_noise))
        w = white_noise / w_std if w_std > 1e-8 else white_noise

        u = apply_static_map(w, self._input_xq, self._input_yq)
        v = self._linear_filter(u)
        v_unit = v / max(self._v_scale, 1e-12)
        return apply_static_map(v_unit, self._output_xq, self._output_yq)

    def transform(
        self,
        num_samples: int,
        seq_length: int,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate new samples by exciting the fitted HW system with white noise.

        :return: Array of shape ``[num_samples, seq_length]``.
        """
        if self._coeffs is None:
            raise ValueError("Model is not fitted; call `fit` first.")

        rng = (
            np.random.RandomState(seed=random_state)
            if random_state is not None
            else self.rng
        )

        white_noise = rng.normal(
            0.0,
            1.0,
            size=(num_samples, seq_length + self.filter_order),
        )
        simulated = np.zeros((num_samples, seq_length), dtype=np.float64)
        for i in range(num_samples):
            simulated[i] = self.invoke(white_noise[i])

        if self.revin:
            simulated = simulated * self.std + self.mean

        self.simulated_series = maybe_apply_lowpass(self, simulated)
        return self.simulated_series
