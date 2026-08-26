# -*- coding: utf-8 -*-
"""
ARMA excitation: random-order autoregressive–moving-average stimulus series.

The generator draws AR / MA coefficients and then filters a Gaussian innovation
sequence.  Two design goals matter here:

1. **Speed.**  A naive Python loop that rebuilds the AR lag vector and redraws
   MA shocks at every time index is O(n · (p + q)) interpreter work.  The
   canonical linear recurrence is exactly what ``scipy.signal.lfilter``
   implements in compiled code, so we pre-draw the whole innovation path and
   apply one IIR filter.
2. **Shape.**  White-looking ARMA draws usually come from real roots that sit
   well inside the unit circle.  Quasi-periodic trajectories instead come from
   a complex-conjugate pair of AR roots with modulus close to one and a
   non-trivial angular frequency.  Stationarity of that pair is optional.

Created on 2025/08/13 21:48:34
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.signal import lfilter

from s2generator.excitation.base_excitation import BaseExcitation


def arma_series(
    rng: np.random.RandomState,
    time_series: np.ndarray,
    p_params: np.ndarray,
    q_params: np.ndarray,
    clip_value: Optional[float] = 1024.0,
) -> np.ndarray:
    """
    Simulate one ARMA path and write it into ``time_series`` in place.

    The process follows the standard linear recurrence

    .. math::

        x_t = \\sum_{i=1}^{p} \\varphi_i x_{t-i}
              + \\varepsilon_t
              + \\sum_{j=1}^{q} \\theta_j \\varepsilon_{t-j},

    with :math:`\\varepsilon_t \\sim \\mathcal{N}(0, 1)`.  Missing pre-sample
    lags are treated as zero (the usual causal start-up).

    Implementation notes:
    - All innovations :math:`\\varepsilon_1,\\ldots,\\varepsilon_n` are drawn
      **once**.  The previous Python loop sampled a fresh length-``q`` Gaussian
      vector at every ``t``, which is not a shared MA shock process and also
      dominated the runtime.
    - ``scipy.signal.lfilter`` realises the same recurrence as a single IIR
      pass (numerator = MA polynomial, denominator = AR polynomial).

    :param rng: NumPy random number generator with a fixed seed.
    :param time_series: Pre-allocated 1-D buffer (typically zeros).  The
                        simulated path is written back into this array.
    :param p_params: AR coefficients :math:`(\\varphi_1,\\ldots,\\varphi_p)`.
    :param q_params: MA coefficients :math:`(\\theta_1,\\ldots,\\theta_q)`.
    :param clip_value: Symmetric numerical guard.  ``None`` disables clipping.
    :return: The same object as ``time_series``, filled with the ARMA path.
    """
    n = int(len(time_series))
    phi = np.asarray(p_params, dtype=np.float64).reshape(-1)
    theta = np.asarray(q_params, dtype=np.float64).reshape(-1)

    # Shared innovation path: one Gaussian shock per time index.
    eps = rng.standard_normal(n)

    # lfilter convention:
    #   a[0] y[n] = b[0] x[n] + b[1] x[n-1] + ... - a[1] y[n-1] - ...
    # Matching the ARMA recurrence therefore takes
    #   b = (1, θ_1, ..., θ_q),   a = (1, -φ_1, ..., -φ_p).
    b = np.concatenate(([1.0], theta)) if theta.size else np.array([1.0])
    a = np.concatenate(([1.0], -phi)) if phi.size else np.array([1.0])

    y = np.asarray(lfilter(b, a, eps), dtype=np.float64)

    # Keep a hard ceiling so a mildly explosive draw cannot overflow float64
    # before the caller rejects / retries the path.
    if clip_value is not None:
        y = np.clip(y, -float(clip_value), float(clip_value))

    # NaN / Inf (possible when the AR polynomial is strictly unstable) → 0
    # so the in-place write stays finite.
    if not np.all(np.isfinite(y)):
        y = np.nan_to_num(
            y,
            nan=0.0,
            posinf=float(clip_value or 0.0),
            neginf=-float(clip_value or 0.0),
        )

    time_series[:] = y
    return time_series


class AutoregressiveMovingAverage(BaseExcitation):
    """Generate stimulus series from a random-order ARMA(p, q) model.

    AR coefficients are built from the roots of the characteristic polynomial
    rather than by rejection-sampling unbounded uniforms.  Placing at least
    one complex-conjugate pair close to the unit circle yields a visible
    oscillation instead of a near-white residual.

    Set ``stationary=False`` to allow roots on or outside the unit circle
    (unit-root / mildly explosive paths).  ``generate`` still retries and
    clips against ``upper_bound`` so the returned array stays usable.
    """

    def __init__(
        self,
        p_min: Optional[int] = 1,
        p_max: Optional[int] = 3,
        q_min: Optional[int] = 1,
        q_max: Optional[int] = 5,
        upper_bound: float = 512,
        stationary: bool = True,
        dtype: np.dtype = np.float64,
    ) -> None:
        """
        :param p_min: Minimum AR order (inclusive).  Orders ``>= 2`` are
                      required for a complex-conjugate pair, i.e. for a
                      genuine oscillatory mode.
        :param p_max: Maximum AR order (exclusive, as ``RandomState.randint``).
        :param q_min: Minimum MA order (inclusive).
        :param q_max: Maximum MA order (exclusive).
        :param upper_bound: Reject (and retry) a draw whose peak absolute
                            value exceeds this threshold.
        :param stationary: If True, every AR root is forced inside the unit
                           circle so the process is (weakly) stationary.  If
                           False, moduli may reach or exceed 1 and the series
                           can wander or grow.
        :param dtype: NumPy dtype of the generated array.
        """
        super().__init__(dtype=dtype)

        # Order sampling box.  ``randint`` uses a half-open interval [min, max).
        self.p_min = p_min
        self.p_max = p_max
        self.q_min = q_min
        self.q_max = q_max

        # Stationarity switch for the AR polynomial.
        self.stationary = bool(stationary)

        # Last realised order / coefficients (exposed via ``order`` / ``params``).
        self.p_order, self.q_order = None, None
        self.p_params, self.q_params = None, None

        # Peak-amplitude guard used inside ``generate``.
        self.upper_bound = upper_bound

    def __call__(
        self,
        rng: np.random.RandomState,
        seq_length: int = 512,
        num_channels: int = 1,
    ) -> np.ndarray:
        """Call the `generate` method to stimulate time series generation"""
        return self.generate(rng=rng, seq_length=seq_length, num_channels=num_channels)

    def __str__(self) -> str:
        """Get the name of the time series generator"""
        return "ARMA"

    @staticmethod
    def create_autoregressive_params(
        rng: np.random.RandomState,
        p_order: int,
        stationary: bool = True,
    ) -> np.ndarray:
        """
        Draw AR coefficients from the roots of the characteristic polynomial.

        A direct uniform draw on :math:`(\\varphi_1,\\ldots,\\varphi_p)` almost
        never produces a sustained oscillation: the roots sit too far inside
        the unit circle and the spectrum is nearly flat.  Sampling the roots
        first, then converting them to coefficients, makes the spectrum
        controllable.

        **Periodicity.**  Whenever :math:`p \\ge 2` we place at least one
        complex-conjugate pair

        .. math::

            r e^{\\pm i \\omega}, \\qquad
            \\varphi_1 = 2 r \\cos\\omega,\\;
            \\varphi_2 = -r^2,

        with angular frequency :math:`\\omega = 2\\pi / T` for a period ``T``
        of roughly 8–48 samples (several visible cycles on a length-256/512
        window).  The modulus ``r`` is close to 1 so the oscillation decays
        slowly.

        **Stationarity.**  The causal AR process is (weakly) stationary iff
        every root satisfies :math:`|r| < 1`.  When ``stationary=True`` we
        draw moduli in ``[0.85, 0.98]``.  When ``stationary=False`` moduli
        may sit in ``[0.95, 1.12]``, which covers near-unit-root and mildly
        explosive oscillations.

        :param rng: NumPy random number generator with a fixed seed.
        :param p_order: AR order :math:`p \\ge 1`.
        :param stationary: Whether all roots must lie inside the unit circle.
        :return: Coefficient vector :math:`(\\varphi_1,\\ldots,\\varphi_p)`.
        """
        if p_order < 1:
            raise ValueError("p_order must be >= 1")

        # Modulus box: persist but decay (stationary) vs persist / grow.
        if stationary:
            rho_lo, rho_hi = 0.85, 0.98
            real_hi = 0.92
        else:
            rho_lo, rho_hi = 0.95, 1.12
            real_hi = 1.08

        roots = []
        remaining = int(p_order)

        def _oscillatory_pair() -> Tuple[complex, complex]:
            # Period in samples.  Avoid T≈2 (Nyquist flicker) and T→∞ (a
            # near-constant drift that looks like a real unit root).
            period = float(rng.uniform(8.0, 48.0))
            omega = 2.0 * np.pi / period
            rho = float(rng.uniform(rho_lo, rho_hi))
            return rho * np.exp(1j * omega), rho * np.exp(-1j * omega)

        # First pair: this is the dominant visible cycle when p >= 2.
        if remaining >= 2:
            roots.extend(_oscillatory_pair())
            remaining -= 2

        # Extra pairs: more spectral peaks, or two real roots.
        while remaining >= 2:
            if rng.rand() < 0.75:
                roots.extend(_oscillatory_pair())
            else:
                roots.append(float(rng.uniform(-real_hi, real_hi)))
                roots.append(float(rng.uniform(-real_hi, real_hi)))
            remaining -= 2

        # Odd leftover order → one real root (a persistent low-frequency mode).
        if remaining == 1:
            # Bias away from 0 so AR(1) is not just weakly correlated noise.
            sign = -1.0 if rng.rand() < 0.5 else 1.0
            mag = float(rng.uniform(max(0.5, rho_lo * 0.7), real_hi))
            roots.append(sign * mag)

        # (x - r_1)...(x - r_p) = x^p - φ_1 x^{p-1} - ...  so φ = -poly[1:].
        char = np.poly(roots)
        phi = -np.real(np.asarray(char, dtype=np.complex128)[1:])
        return np.asarray(phi, dtype=np.float64)

    @staticmethod
    def create_moving_average_params(
        rng: np.random.RandomState, q_order: int
    ) -> np.ndarray:
        """
        Draw MA coefficients as a signed geometric kernel.

        A uniform cloud on :math:`[-1, 1]^q` leaves the innovations almost
        white.  A decaying kernel :math:`\\theta_j \\propto s_j \\, \\alpha^j`
        is a low-pass colouring of :math:`\\varepsilon_t`, so the AR cycle
        is not buried under high-frequency jitter.  Coefficients stay inside
        :math:`[-1, 1]` (invertibility is not strictly enforced).

        :param rng: NumPy random number generator with a fixed seed.
        :param q_order: MA order :math:`q \\ge 1`.
        :return: Coefficient vector :math:`(\\theta_1,\\ldots,\\theta_q)`.
        """
        if q_order < 1:
            raise ValueError("q_order must be >= 1")

        lags = np.arange(1, q_order + 1, dtype=np.float64)
        # Decay in (0.4, 0.85): fast enough to stay in [-1, 1], slow enough
        # that several lags still carry mass.
        alpha = float(rng.uniform(0.4, 0.85))
        signs = rng.choice(np.array([-1.0, 1.0]), size=q_order)
        scale = float(rng.uniform(0.35, 0.90))
        theta = signs * scale * (alpha**lags)
        return np.asarray(theta, dtype=np.float64)

    @property
    def order(self) -> Dict[str, int]:
        """Get the order of the autoregressive process and the moving average process in the ARMA model."""
        return {"AR(p)": self.p_order, "MA(q)": self.q_order}

    @property
    def params(self) -> Dict[str, np.ndarray]:
        """Get the parameters of the autoregressive process and the moving average process in the ARMA model."""
        return {"AR(p)": self.p_params, "MA(q)": self.q_params}

    def arma_series(
        self,
        rng: np.random.RandomState,
        time_series: np.ndarray,
        p_params: Optional[np.ndarray] = None,
        q_params: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate an ARMA process based on the specified parameters.

        :param rng: Random number generator of NumPy with fixed seed.
        :param time_series: The zeros time series.
        :param p_params: The parameters of the AR(p) process.
        :param q_params: The parameters of the MA(q) process.
        """
        return arma_series(
            rng=rng,
            time_series=time_series,
            p_params=self.p_params if p_params is None else p_params,
            q_params=self.q_params if q_params is None else q_params,
        )

    def create_params(self, rng: np.random.RandomState) -> None:
        """
        Sample a random ARMA order and the matching coefficient vectors.

        :param rng: The random number generator of NumPy with fixed seed.
        :return: None.
        """
        # Half-open integer interval [min, max), identical to the previous API.
        self.p_order = rng.randint(low=self.p_min, high=self.p_max)
        self.q_order = rng.randint(low=self.q_min, high=self.q_max)

        self.p_params = self.create_autoregressive_params(
            rng=rng, p_order=self.p_order, stationary=self.stationary
        )
        self.q_params = self.create_moving_average_params(rng=rng, q_order=self.q_order)

    def generate(
        self,
        rng: np.random.RandomState,
        seq_length: int = 512,
        num_channels: int = 1,
    ) -> np.ndarray:
        """
        Generate ARMA time series of the requested length and dimension.

        Each column is an independent ARMA draw.  Paths that explode past
        ``upper_bound`` (typical when ``stationary=False``) are discarded and
        resampled; after enough failures the last path is clipped so the
        loop cannot run forever.

        :param rng: The random number generator of NumPy with fixed seed.
        :param seq_length: The number of input points.
        :param num_channels: The dimension of the time series.
        :return: Array of shape ``(seq_length, num_channels)``.
        """
        time_series = self.create_zeros(
            seq_length=seq_length, num_channels=num_channels
        )

        # Independent retry budget per column.  Non-stationary roots can
        # overflow; we would rather clip than spin indefinitely.
        max_attempts = 48

        for col in range(num_channels):
            accepted = False
            last = None
            for _ in range(max_attempts):
                self.create_params(rng=rng)
                # Fresh buffer: arma_series writes in place.
                buffer = np.zeros(seq_length, dtype=np.float64)
                last = self.arma_series(
                    rng=rng,
                    time_series=buffer,
                    p_params=self.p_params,
                    q_params=self.q_params,
                )
                peak = float(np.max(np.abs(last))) if last.size else 0.0
                if np.all(np.isfinite(last)) and peak <= self.upper_bound:
                    time_series[:, col] = last
                    accepted = True
                    break
            if not accepted:
                # Last-resort finite path: clip the most recent draw.
                fallback = (
                    np.zeros(seq_length, dtype=np.float64) if last is None else last
                )
                fallback = np.nan_to_num(
                    fallback, nan=0.0, posinf=self.upper_bound, neginf=-self.upper_bound
                )
                time_series[:, col] = np.clip(
                    fallback, -self.upper_bound, self.upper_bound
                )

        return time_series


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    arma = AutoregressiveMovingAverage(p_min=2, p_max=5, stationary=True)

    for i in range(10):
        rng = np.random.RandomState(i)

        time = arma.generate(rng=rng, seq_length=256)
        plt.plot(time)
        plt.show()
