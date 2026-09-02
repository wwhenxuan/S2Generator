# -*- coding: utf-8 -*-
"""
Map a real time series to complex IQ whose phase comes from the series
and whose spectrum shape comes from a reference wireless channel.

Created on 2026/09/02
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/S2Generator
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Sequence, Union

import numpy as np
from numpy.fft import fft, fftfreq, ifft
from scipy.signal import butter, filtfilt, get_window, hilbert

from s2generator.excitation.autoregressive_moving_average import (
    AutoregressiveMovingAverage,
)


ArrayLike = Union[np.ndarray, Sequence[float]]
Mode = Literal["baseband", "analytic"]


class IQSimulator(object):
    """
    Convert a real time series into a complex IQ pair with a prescribed spectrum.

    Motivation
    ----------
    A wireless fading coefficient is a *complex baseband* process: its I and Q
    channels are not a Hilbert pair, and its Doppler spectrum is two-sided.
    An analytic signal (one-sided spectrum) is the other common target.  The
    two cannot be produced by stacking Hilbert, a one-sided PSD match, and a
    hard low-pass filter — those three steps fight each other.  This simulator
    therefore separates the two legitimate goals.

    What is transferred
    -------------------
    The **phase** of the output comes from a real stimulus ``x[t]`` (a user
    series or an ARMA draw).  The **magnitude spectrum** comes from a reference
    channel PSD ``P[k]``, typically the mean windowed periodogram of DeepMIMO
    CSI snapshots.

    Let ``X[k] = FFT(x)`` after a padded Hilbert seed ``z[t]`` with
    ``z = I + j Q``.  Weak FFT bins (the whole negative-frequency half after
    Hilbert) receive interpolated unwrapped phase.  Magnitudes are then blended

        ``|Z'[k]| = (1 - mix) |Z[k]| + mix sqrt(P[k])``

    while the (infilled) phase is kept.  An inverse FFT yields the IQ pair.

    Modes
    -----
    ``mode="baseband"`` (default)
        Reference is a complex fading coefficient.  Negative frequencies are
        *kept*, so the two-sided Doppler shape of ``P[k]`` is reproduced.

    ``mode="analytic"``
        The reference is first projected onto positive frequencies; after the
        magnitude match the same analytic projection is applied and the lost
        energy is restored.  ``corr(H(I), Q)`` is then close to one.

    Pipeline
    --------
    1. Crop a padded window of ``x``, z-score it, optionally low-pass it.
    2. Hilbert-transform the *padded* window and crop back to ``out_len``.
    3. If a target PSD is fitted, replace magnitudes (``_spectral_shape``).
    4. Optionally normalise to unit mean power.

    When ``transform`` is called without a series, two independent ARMA
    channels are drawn via :class:`AutoregressiveMovingAverage` and each
    column is converted to IQ, returning an array of shape ``[2, 2, L]``.
    """

    def __init__(
        self,
        mode: Mode = "baseband",
        match_mix: float = 1.0,
        context_factor: float = 4.0,
        unit_power: bool = True,
        keep_dc: bool = False,
        apply_lpf: Optional[bool] = None,
        f_cut: Optional[float] = None,
        f_cut_scale: float = 1.0,
        lpf_order: int = 4,
        psd_window: str = "hann",
        random_state: Optional[int] = 42,
        arma_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        :param mode: ``\"baseband\"`` keeps a two-sided Doppler spectrum;
                     ``\"analytic\"`` projects onto positive frequencies.
        :param match_mix: Blend in ``[0, 1]`` from the Hilbert seed magnitude
                          (0) toward ``sqrt(target_psd)`` (1).
        :param context_factor: Extra samples on each side of the Hilbert window,
                               as a multiple of the output length.
        :param unit_power: If True, scale each IQ pair to unit mean power.
        :param keep_dc: Keep the DC FFT bin (Rician LOS).  Default drops it.
        :param apply_lpf: If None, low-pass only when there is *no* PSD match.
                          Set True/False to force the smoother on or off.
        :param f_cut: Manual low-pass cutoff in cycles/sample.  If None, it is
                      estimated from the fitted PSD (or 0.05 without a PSD).
        :param f_cut_scale: Multiplier applied to ``f_cut`` before filtering.
        :param lpf_order: Butterworth order of the optional low-pass stage.
        :param psd_window: Window name for :meth:`estimate_mean_psd_from_channels`.
        :param random_state: Seed used when ARMA stimulus series are drawn.
        :param arma_kwargs: Extra kwargs forwarded to
                            :class:`AutoregressiveMovingAverage`.
        """
        if mode not in ("baseband", "analytic"):
            raise ValueError(f"mode must be 'baseband' or 'analytic', got {mode!r}")
        if not 0.0 <= float(match_mix) <= 1.0:
            raise ValueError("match_mix must lie in [0, 1].")
        if lpf_order < 1:
            raise ValueError("lpf_order must be a positive integer.")

        self.mode: Mode = mode
        self.match_mix = float(match_mix)
        self.context_factor = float(context_factor)
        self.unit_power = bool(unit_power)
        self.keep_dc = bool(keep_dc)
        self.apply_lpf = apply_lpf
        self.f_cut = None if f_cut is None else float(f_cut)
        self.f_cut_scale = float(f_cut_scale)
        self.lpf_order = int(lpf_order)
        self.psd_window = str(psd_window)
        self.random_state = random_state

        arma_kwargs = {} if arma_kwargs is None else dict(arma_kwargs)
        self.arma = AutoregressiveMovingAverage(**arma_kwargs)

        self.target_psd: Optional[np.ndarray] = None
        self.channel_ri: Optional[np.ndarray] = None

    def fit(
        self,
        channel_ri: Optional[ArrayLike] = None,
        target_psd: Optional[ArrayLike] = None,
        n_fft: Optional[int] = None,
    ) -> "IQSimulator":
        """
        Store a two-sided reference PSD from CSI snapshots or a ready periodogram.

        Provide **either** ``channel_ri`` **or** ``target_psd``.  Channel stacks
        are accepted as ``(N, T, 2)`` or ``(N, 2, T)``.

        :param channel_ri: Real/imaginary CSI traces used to estimate the PSD.
        :param target_psd: Two-sided periodogram, already averaged.
        :param n_fft: FFT length for the periodogram; defaults to the snapshot
                      length ``T``.
        :return: ``self``, so that ``fit(...).transform(...)`` can be chained.
        """
        if channel_ri is None and target_psd is None:
            raise ValueError("fit requires channel_ri or target_psd")
        if channel_ri is not None and target_psd is not None:
            raise ValueError("provide only one of channel_ri or target_psd")

        if target_psd is not None:
            psd = np.asarray(target_psd, dtype=np.float64).ravel()
            if psd.size < 8:
                raise ValueError(f"target_psd is too short: {psd.size}")
            if np.any(~np.isfinite(psd)):
                raise ValueError("target_psd must be finite")
            self.target_psd = np.maximum(psd, 0.0)
            self.channel_ri = None
            return self

        traces = self.check_channel_inputs(channel_ri)
        self.channel_ri = traces
        self.target_psd = self.estimate_mean_psd_from_channels(
            traces,
            n_fft=n_fft,
            window=self.psd_window,
            analytic=self.mode == "analytic",
            keep_dc=self.keep_dc,
        )
        return self

    def transform(
        self,
        time_series: Optional[ArrayLike] = None,
        seq_length: Optional[int] = None,
        num_channels: int = 2,
        start: Optional[int] = None,
        random_state: Optional[int] = None,
        return_complex: bool = False,
    ) -> np.ndarray:
        """
        Map one or more real series onto IQ pairs.

        If ``time_series`` is omitted, two (or ``num_channels``) independent
        ARMA paths are drawn and converted.  The default return layout is
        real/imag with I in row 0 and Q in row 1, matching
        :func:`s2generator.utils.visualization.plot_iq_series`.

        :param time_series: Real stimulus.  ``None`` draws ARMA channels;
                            1-D arrays yield one IQ pair; 2-D arrays yield one
                            pair per channel.
        :param seq_length: Output length.  Defaults to ``len(target_psd)``
                           or 128.  For ARMA draws the generated path is longer
                           than this so Hilbert can use context padding.
        :param num_channels: Number of ARMA columns when ``time_series`` is None.
        :param start: Crop offset inside the stimulus; ``None`` centres the crop.
        :param random_state: Override of the constructor seed for ARMA draws.
        :param return_complex: If True, return complex arrays instead of
                               ``[2, L]`` / ``[C, 2, L]`` real stacks.
        :return: ``[2, L]`` for one series, ``[C, 2, L]`` for several channels.
        """
        out_len = self._resolve_out_len(seq_length)
        if time_series is None:
            series = self._default_arma_series(
                seq_length=out_len,
                num_channels=num_channels,
                random_state=random_state,
            )
        else:
            series = self.check_inputs(time_series)

        if series.ndim == 1:
            iq = self.time_series_to_iq(
                series,
                out_len=out_len,
                start=start,
                return_real_imag=not return_complex,
            )
            if return_complex:
                return np.asarray(iq, dtype=np.complex128)
            return np.asarray(iq, dtype=np.float64).T

        batch = []
        for col in range(series.shape[1]):
            converted = self.time_series_to_iq(
                series[:, col],
                out_len=out_len,
                start=start,
                return_real_imag=not return_complex,
            )
            batch.append(converted)
        if return_complex:
            return np.stack(
                [np.asarray(item, dtype=np.complex128) for item in batch], axis=0
            )
        stacked = np.stack(
            [np.asarray(item, dtype=np.float64).T for item in batch], axis=0
        )
        return stacked

    def check_inputs(self, time_series: ArrayLike) -> np.ndarray:
        """
        Validate a stimulus series and return a 1-D or ``(L, C)`` float array.

        :param time_series: Candidate real array (1-D or 2-D).
        :return: Float64 array, shape ``(L,)`` or ``(L, C)``.
        """
        if time_series is None:
            raise ValueError("time_series must not be None")
        data = np.asarray(time_series, dtype=np.float64)
        if data.ndim == 0:
            raise ValueError("time_series must be 1-D or 2-D")
        if data.ndim > 2:
            raise ValueError(
                "time_series must be 1-D [seq_length] or 2-D "
                f"[seq_length, channels]; got shape {data.shape}"
            )
        if np.any(~np.isfinite(data)):
            raise ValueError("time_series must be finite")
        if data.ndim == 2:
            data = self._as_channels_last(data)
            if data.shape[0] < 8:
                raise ValueError(f"time_series too short: {data.shape[0]}")
            return data
        if data.size < 8:
            raise ValueError(f"time_series too short: {data.size}")
        return data.reshape(-1)

    def check_channel_inputs(self, channel_ri: ArrayLike) -> np.ndarray:
        """
        Validate CSI real/imag stacks used by :meth:`fit`.

        :param channel_ri: Array of shape ``(N, T, 2)`` or ``(N, 2, T)``.
        :return: Array of shape ``(N, T, 2)``.
        """
        data = np.asarray(channel_ri, dtype=np.float64)
        z = self._as_complex_channels(data)
        if z.shape[0] < 1 or z.shape[1] < 8:
            raise ValueError(f"channel snapshots are too short: {z.shape}")
        if np.any(~np.isfinite(z)):
            raise ValueError("channel_ri must be finite")
        return np.stack([np.real(z), np.imag(z)], axis=-1)

    def estimate_mean_psd_from_channels(
        self,
        channel_ri: ArrayLike,
        n_fft: Optional[int] = None,
        window: Optional[str] = None,
        analytic: bool = False,
        keep_dc: bool = False,
    ) -> np.ndarray:
        """
        Average windowed periodogram of complex channel snapshots.

        :param channel_ri: ``(N, T, 2)`` or ``(N, 2, T)`` real/imag stacks.
        :param n_fft: FFT length, default ``T``.
        :param window: ``scipy.signal.get_window`` name.  Hann reduces Doppler
                       leakage on short snapshots (T=128).
        :param analytic: If True, project each snapshot to an analytic spectrum
                         *before* averaging.
        :param keep_dc: Keep the DC bin (Rician LOS).  Default drops it.
        :return: Two-sided periodogram of length ``n_fft``.
        """
        z = self._as_complex_channels(np.asarray(channel_ri)).astype(
            np.complex128, copy=False
        )
        t_len = z.shape[1]
        n_fft = int(n_fft or t_len)
        z = z - z.mean(axis=1, keepdims=True)
        name = self.psd_window if window is None else window
        if name and str(name).lower() not in {"boxcar", "ones", "none"}:
            tap = get_window(name, t_len, fftbins=True).astype(np.float64)
            tap = tap / np.sqrt(np.mean(tap**2) + 1e-18)
            z = z * tap
        spectrum = fft(z, n=n_fft, axis=1)
        if analytic:
            spectrum = self.project_analytic_spectrum(spectrum, keep_dc=True)
        psd = np.mean(np.abs(spectrum) ** 2, axis=0)
        if not keep_dc:
            psd = psd.copy()
            psd[0] = 0.0
        return psd

    def project_analytic_spectrum(
        self, spectrum: ArrayLike, keep_dc: bool = False
    ) -> np.ndarray:
        """
        Zero negative frequencies (and Nyquist if the length is even).

        Operates on the last axis of ``spectrum``.

        :param spectrum: Complex FFT array.
        :param keep_dc: If False, the DC bin is also zeroed.
        :return: Copy of ``spectrum`` with the negative-frequency half removed.
        """
        result = np.asarray(spectrum, dtype=np.complex128).copy()
        n = result.shape[-1]
        if n % 2 == 0:
            result[..., n // 2 :] = 0.0
        else:
            result[..., n // 2 + 1 :] = 0.0
        if not keep_dc:
            result[..., 0] = 0.0
        return result

    def f_cut_from_psd(self, psd: ArrayLike, energy_frac: float = 0.9) -> float:
        """
        Two-sided bandwidth: smallest ``|f|`` capturing ``energy_frac`` of
        the energy outside DC, in cycles/sample.

        :param psd: Two-sided periodogram.
        :param energy_frac: Cumulative energy fraction in ``(0, 1]``.
        :return: Cutoff in cycles/sample, at least ``1 / n``.
        """
        values = np.asarray(psd, dtype=np.float64).ravel()
        n = len(values)
        freqs = fftfreq(n)
        energy = values.copy()
        energy[0] = 0.0
        total = float(energy.sum())
        if total <= 0:
            return 0.05
        order = np.argsort(np.abs(freqs))
        cumulative = np.cumsum(energy[order]) / total
        k = int(np.searchsorted(cumulative, energy_frac))
        k = min(k, n - 1)
        return float(max(np.abs(freqs[order][k]), 1.0 / n))

    def time_series_to_iq(
        self,
        time_series: ArrayLike,
        target_psd: Optional[ArrayLike] = None,
        out_len: Optional[int] = None,
        mode: Optional[Mode] = None,
        f_cut: Optional[float] = None,
        f_cut_scale: Optional[float] = None,
        lpf_order: Optional[int] = None,
        apply_lpf: Optional[bool] = None,
        context_factor: Optional[float] = None,
        start: Optional[int] = None,
        match_mix: Optional[float] = None,
        unit_power: Optional[bool] = None,
        keep_dc: Optional[bool] = None,
        return_real_imag: bool = False,
    ) -> np.ndarray:
        """
        Convert a real 1-D series into complex IQ.

        Keyword arguments default to the values stored on the instance
        (and to the fitted PSD when ``target_psd`` is omitted).

        :param time_series: Real 1-D array (length >= 8).  Longer than
                            ``out_len`` is recommended so Hilbert / filtfilt
                            can use context padding.
        :param target_psd: Two-sided reference periodogram.  ``None`` uses
                           the fitted PSD, or Hilbert-only if none is fitted.
        :param out_len: Output length.  Defaults to ``len(target_psd)`` or
                        ``min(128, len(time_series))``.
        :param mode: Override of the constructor mode.
        :param f_cut: Override of the low-pass cutoff (cycles/sample).
        :param f_cut_scale: Override of the cutoff multiplier.
        :param lpf_order: Override of the Butterworth order.
        :param apply_lpf: Override of the LPF switch.  ``None`` means
                          “LPF only when there is no PSD match”.
        :param context_factor: Override of the Hilbert context multiple.
        :param start: Crop offset; ``None`` centres the window.
        :param match_mix: Override of the spectral-shape blend.
        :param unit_power: Override of unit-power normalisation.
        :param keep_dc: Override of the DC-bin policy.
        :param return_real_imag: If True, return ``(L, 2)`` instead of complex.
        :return: Complex 1-D array of length ``out_len``, or ``(L, 2)``.
        """
        use_mode: Mode = self.mode if mode is None else mode
        if use_mode not in ("baseband", "analytic"):
            raise ValueError(f"mode must be 'baseband' or 'analytic', got {use_mode!r}")

        x_full = np.asarray(self.check_inputs(time_series), dtype=np.float64).ravel()
        psd_src = self.target_psd if target_psd is None else target_psd
        psd = None if psd_src is None else np.asarray(psd_src, dtype=np.float64).ravel()

        if out_len is None:
            out_len = len(psd) if psd is not None else int(min(128, x_full.size))
        out_len = int(out_len)
        if out_len < 8:
            raise ValueError(f"out_len too small: {out_len}")
        if out_len > x_full.size:
            raise ValueError(
                f"out_len ({out_len}) cannot exceed time_series length ({x_full.size})"
            )

        cutoff = self.f_cut if f_cut is None else f_cut
        if cutoff is None:
            cutoff = (
                self.f_cut_from_psd(psd, energy_frac=0.9) if psd is not None else 0.05
            )
        scale = self.f_cut_scale if f_cut_scale is None else float(f_cut_scale)
        effective_f_cut = float(np.clip(float(cutoff) * scale, 1e-4, 0.49))

        mix = self.match_mix if match_mix is None else float(match_mix)
        matching = psd is not None and mix > 0.0
        lpf_flag = self.apply_lpf if apply_lpf is None else apply_lpf
        if lpf_flag is None:
            lpf_flag = not matching

        if start is None:
            start = max(0, (x_full.size - out_len) // 2)
        start = int(np.clip(start, 0, x_full.size - out_len))

        ctx = self.context_factor if context_factor is None else float(context_factor)
        max_pad = int(max(0.0, ctx) * out_len)
        pad = min(max_pad, start, x_full.size - (start + out_len))
        lo, hi = start - pad, start + out_len + pad
        segment = x_full[lo:hi]
        segment = (segment - np.mean(segment)) / (np.std(segment) + 1e-12)

        order = self.lpf_order if lpf_order is None else int(lpf_order)
        if lpf_flag:
            segment = self._butter_lowpass(segment, f_cut=effective_f_cut, order=order)

        # Hilbert on the padded real window, then crop.  This is the complex
        # seed: Im ≈ H{Re} on the support of the window, with weaker edges
        # than Hilbert(core) alone.
        z_pad = hilbert(segment - np.mean(segment))
        z = np.asarray(z_pad[pad : pad + out_len], dtype=np.complex128)

        dc_flag = self.keep_dc if keep_dc is None else bool(keep_dc)
        if matching:
            z = self._spectral_shape(
                z,
                psd,
                mix=mix,
                mode=use_mode,
                keep_dc=dc_flag,
            )

        do_unit = self.unit_power if unit_power is None else bool(unit_power)
        if do_unit:
            power = float(np.mean(np.abs(z) ** 2))
            if power > 0.0:
                z = z / np.sqrt(power)

        if return_real_imag:
            return np.column_stack([np.real(z), np.imag(z)])
        return z

    def _default_arma_series(
        self,
        seq_length: int,
        num_channels: int = 2,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        """
        Draw independent ARMA stimulus channels, longer than ``seq_length``.

        Extra samples give Hilbert / filtfilt a context pad.  Each column is
        one real series that :meth:`time_series_to_iq` will map to IQ.

        :param seq_length: Desired IQ length.
        :param num_channels: Number of independent ARMA columns.
        :param random_state: Optional seed override.
        :return: Array of shape ``(L_gen, num_channels)``.
        """
        if num_channels < 1:
            raise ValueError("num_channels must be a positive integer")
        seed = self.random_state if random_state is None else random_state
        rng = np.random.RandomState(seed)
        extra = max(int(max(self.context_factor, 0.0) * seq_length), 8)
        gen_len = int(seq_length + extra)
        return self.arma.generate(
            rng=rng, seq_length=gen_len, num_channels=int(num_channels)
        )

    def _resolve_out_len(self, seq_length: Optional[int]) -> int:
        """Pick the IQ length from the user, the fitted PSD, or 128."""
        if seq_length is not None:
            out_len = int(seq_length)
        elif self.target_psd is not None:
            out_len = int(len(self.target_psd))
        else:
            out_len = 128
        if out_len < 8:
            raise ValueError(f"seq_length too small: {out_len}")
        return out_len

    @staticmethod
    def _as_channels_last(data: np.ndarray) -> np.ndarray:
        """Interpret a 2-D array as ``(seq_length, channels)``."""
        if data.shape[1] == 2 and data.shape[0] != 2:
            return data
        if data.shape[0] == 2 and data.shape[1] != 2:
            return data.T
        return data

    @staticmethod
    def _as_complex_channels(channel_ri: np.ndarray) -> np.ndarray:
        """Interpret ``(N, T, 2)`` or ``(N, 2, T)`` as complex ``(N, T)``."""
        x = np.asarray(channel_ri)
        if x.ndim != 3:
            raise ValueError(f"expected (N, T, 2) or (N, 2, T), got {x.shape}")
        if x.shape[-1] == 2:
            return x[..., 0] + 1j * x[..., 1]
        if x.shape[1] == 2:
            return x[:, 0, :] + 1j * x[:, 1, :]
        raise ValueError(f"cannot interpret channel layout with shape {x.shape}")

    @staticmethod
    def _resample_psd_to_length(psd: np.ndarray, out_len: int) -> np.ndarray:
        """Resample a two-sided PSD onto another FFT length in log-power."""
        values = np.maximum(np.asarray(psd, dtype=np.float64).ravel(), 0.0)
        src_n = len(values)
        if src_n == out_len:
            return values.copy()

        src_f = fftfreq(src_n)
        dst_f = fftfreq(out_len)
        order = np.argsort(src_f)
        src_f_sorted = src_f[order]
        log_psd = np.log(np.maximum(values[order], 1e-30))

        src_f_ext = np.concatenate(
            [src_f_sorted - 1.0, src_f_sorted, src_f_sorted + 1.0]
        )
        log_ext = np.concatenate([log_psd, log_psd, log_psd])
        return np.exp(np.interp(dst_f, src_f_ext, log_ext))

    @staticmethod
    def _butter_lowpass(x: np.ndarray, f_cut: float, order: int = 4) -> np.ndarray:
        """Zero-phase Butterworth low-pass. ``f_cut`` is in cycles/sample."""
        wn = float(np.clip(f_cut / 0.5, 1e-3, 0.99))
        b, a = butter(order, wn, btype="low")
        return filtfilt(b, a, np.asarray(x, dtype=np.float64))

    @staticmethod
    def _infill_phase(spectrum: np.ndarray) -> np.ndarray:
        """
        Keep ``angle(Z)`` on reliable bins; fill the rest by interpolating
        unwrapped phase along the sorted (wrapped) frequency axis.

        After a Hilbert transform the entire negative-frequency half is ~0, so
        this step is what lets two-sided spectral shaping put energy back on
        −f without using conjugate-symmetric (i.e. real-valued) phase.

        :param spectrum: Complex 1-D FFT vector.
        :return: Phase vector of the same length, in radians.
        """
        z = np.asarray(spectrum, dtype=np.complex128)
        n = z.size
        mag = np.abs(z)
        ang = np.angle(z)
        if n < 2:
            return ang

        thresh = 1e-8 * (float(mag.max()) + 1e-15)
        strong = mag > thresh
        if int(np.count_nonzero(strong)) < 2:
            return np.unwrap(np.linspace(0.0, np.pi, n, endpoint=False))

        freqs = fftfreq(n)
        order = np.argsort(freqs)
        f_s = freqs[order]
        ang_s = ang[order]
        strong_s = strong[order]

        f_v = f_s[strong_s]
        ang_rep = np.concatenate([ang_s[strong_s], ang_s[strong_s], ang_s[strong_s]])
        ph_ext = np.unwrap(ang_rep)
        f_ext = np.concatenate([f_v - 1.0, f_v, f_v + 1.0])
        ph_s = np.interp(f_s, f_ext, ph_ext)

        filled = np.empty(n, dtype=np.float64)
        filled[order] = ph_s
        out = ang.copy()
        out[~strong] = filled[~strong]
        return out

    def _spectral_shape(
        self,
        z: np.ndarray,
        target_psd: np.ndarray,
        mix: float,
        mode: Mode,
        keep_dc: bool,
    ) -> np.ndarray:
        """
        Replace ``|FFT(z)|`` by a blend toward ``sqrt(target_psd)``.

        The (infilled) phase is kept.  In analytic mode the spectrum is then
        projected and energy-aligned.

        :param z: Complex seed of length ``n``.
        :param target_psd: Two-sided reference periodogram.
        :param mix: Blend in ``[0, 1]``.
        :param mode: ``\"baseband\"`` or ``\"analytic\"``.
        :param keep_dc: Whether to retain the DC bin.
        :return: Inverse-FFT of the shaped spectrum.
        """
        z = np.asarray(z, dtype=np.complex128)
        n = len(z)
        spectrum = fft(z)
        phase = self._infill_phase(spectrum)
        mag = np.abs(spectrum)

        tgt = self._resample_psd_to_length(target_psd, n)
        tgt = np.maximum(tgt, 0.0)
        if mode == "analytic":
            tgt = tgt.copy()
            if n % 2 == 0:
                tgt[n // 2 :] = 0.0
            else:
                tgt[n // 2 + 1 :] = 0.0
        if not keep_dc:
            tgt[0] = 0.0

        tgt_mag = np.sqrt(tgt)
        mix = float(np.clip(mix, 0.0, 1.0))
        new_mag = (1.0 - mix) * mag + mix * tgt_mag
        if not keep_dc:
            new_mag[0] = 0.0

        shaped = new_mag * np.exp(1j * phase)

        if mode == "analytic":
            energy_before = float(np.sum(np.abs(shaped) ** 2))
            shaped = self.project_analytic_spectrum(shaped, keep_dc=keep_dc)
            energy_after = float(np.sum(np.abs(shaped) ** 2))
            if energy_after > 0.0 and energy_before > 0.0:
                shaped = shaped * np.sqrt(energy_before / energy_after)
        elif not keep_dc:
            shaped[0] = 0.0

        return ifft(shaped)
