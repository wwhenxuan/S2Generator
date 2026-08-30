# -*- coding: utf-8 -*-
"""
Created on 2025/08/12 13:40:16
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com

Intrinsic Mode Function (IMF) style excitation: additive synthesis of oscillatory
modes with optional localized envelopes, trends, chirps, and EMD-like amplitude
hierarchies.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from pysdkit.data import (
    add_noise,
    generate_sin_signal,
    generate_cos_signal,
    generate_am_signal,
    generate_sawtooth_wave,
)

from s2generator.augmentation import (
    add_linear_trend,
    add_piecewise_linear_trend,
    add_nonlinear_trend,
)
from s2generator.excitation.base_excitation import BaseExcitation

# A dictionary of all available Eigenmodel functions (PySDKit backends)
ALL_IMF_DICT = {
    "generate_sin_signal": generate_sin_signal,
    "generate_cos_signal": generate_cos_signal,
    "generate_am_signal": generate_am_signal,
    "generate_sawtooth_wave": generate_sawtooth_wave,
}

_DEFAULT_ENVELOPE_FAMILIES = ("gaussian", "sech", "tukey", "asymmetric")
_DEFAULT_TREND_KINDS = ("linear", "piecewise", "nonlinear")


def _normalize_to_simplex(weights: np.ndarray) -> np.ndarray:
    """Clip negatives to zero and normalize so weights sum to 1."""
    arr = np.asarray(weights, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Probability weights must be non-empty.")
    arr = np.clip(arr, 0.0, None)
    total = float(np.sum(arr))
    if total <= 0.0:
        raise ValueError(
            "Probability weights must contain at least one positive value."
        )
    return arr / total


def _check_probability_dict(prob_dict: Dict[str, float]) -> Dict[str, float]:
    """
    Validates and normalizes a probability dictionary for intrinsic mode functions.

    Keys must exist in ``ALL_IMF_DICT``. Values are clipped to be non-negative and
    renormalized to a probability simplex (sum to 1).
    """
    if not prob_dict:
        raise ValueError("`prob_dict` must be non-empty.")

    keys: List[str] = []
    values: List[float] = []
    for key, value in prob_dict.items():
        if key not in ALL_IMF_DICT:
            raise ValueError(f"Illegal key: {key} in `prob_dict`!")
        keys.append(key)
        values.append(float(value))

    probs = _normalize_to_simplex(np.asarray(values, dtype=np.float64))
    return {key: float(prob) for key, prob in zip(keys, probs)}


def _check_probability_list(prob_list: List[float]) -> Dict[str, float]:
    """
    Validates and normalizes a probability list for intrinsic mode functions.

    Length must be in ``[1, len(ALL_IMF_DICT)]``. Weights map to the first
    ``len(prob_list)`` keys of ``ALL_IMF_DICT`` and are simplex-normalized.
    """
    length = len(prob_list)
    total_imfs = len(ALL_IMF_DICT)

    if length > total_imfs or length <= 0:
        raise ValueError(
            f"Invalid `prob_list` length: {length}. Must be 1-{total_imfs}"
        )

    probs = _normalize_to_simplex(np.asarray(prob_list, dtype=np.float64))
    keys = list(ALL_IMF_DICT.keys())[:length]
    return {imf: float(prob) for imf, prob in zip(keys, probs)}


def _get_energy(signal: np.ndarray) -> float:
    """RMS energy ``sqrt(mean(x^2))`` (falls back to 0 for empty / all-zero)."""
    x = np.asarray(signal, dtype=np.float64)
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x**2)))


def get_adaptive_sampling_rate(duration: float, length: int) -> float:
    """
    Computes the minimum sampling rate required to achieve a target signal length.

    Formula: sampling_rate = ceil(signal_length / time_duration)
    """
    return float(np.ceil(length / duration))


def _unit_time(seq_length: int) -> np.ndarray:
    """Normalized time axis ``t ∈ [0, 1]``."""
    if seq_length <= 1:
        return np.zeros(seq_length, dtype=np.float64)
    return np.linspace(0.0, 1.0, seq_length, dtype=np.float64)


def _make_envelope(
    t: np.ndarray,
    center: float,
    width: float,
    family: str = "gaussian",
    asymmetry: float = 1.0,
    inverted: bool = False,
) -> np.ndarray:
    """
    Build a localized amplitude envelope on ``t ∈ [0, 1]``.

    ``width`` is a fraction of the unit interval (clamped away from 0).
    ``asymmetry`` stretches the right side relative to the left for
    ``asymmetric`` / gaussian-like families.
    """
    t = np.asarray(t, dtype=np.float64)
    center = float(np.clip(center, 0.0, 1.0))
    width = float(max(width, 1e-3))
    family = str(family).strip().lower()
    asymmetry = float(max(asymmetry, 1e-3))

    left = t < center
    # Effective sigma on each side
    w_left = width
    w_right = width * asymmetry

    if family == "sech":
        # sech((t-c)/w); use side-dependent width
        z = np.where(left, (t - center) / w_left, (t - center) / w_right)
        # sech(x) = 1/cosh(x)
        env = 1.0 / np.cosh(np.clip(z, -50.0, 50.0))
    elif family == "tukey":
        # Cosine-tapered window of total width ~ 2*width around center
        half = max(width, 1e-3)
        dist = np.abs(t - center)
        env = np.zeros_like(t)
        inner = dist <= 0.5 * half
        taper = (dist > 0.5 * half) & (dist <= half)
        env[inner] = 1.0
        x = (dist[taper] - 0.5 * half) / max(0.5 * half, 1e-8)
        env[taper] = 0.5 * (1.0 + np.cos(np.pi * x))
    elif family == "asymmetric":
        z = np.where(left, (t - center) / w_left, (t - center) / w_right)
        env = np.exp(-0.5 * z**2)
    else:
        # gaussian (default)
        z = np.where(left, (t - center) / w_left, (t - center) / w_right)
        env = np.exp(-0.5 * z**2)

    env = np.asarray(env, dtype=np.float64)
    peak = float(np.max(env)) if env.size else 0.0
    if peak > 0:
        env = env / peak

    if inverted:
        env = 1.0 - env
        # Keep non-negative and re-normalize peak
        env = np.clip(env, 0.0, None)
        peak = float(np.max(env)) if env.size else 0.0
        if peak > 0:
            env = env / peak

    return env


def _chirp_carrier(
    t: np.ndarray,
    f0: float,
    beta: float,
    phase0: float = 0.0,
    use_cos: bool = False,
) -> np.ndarray:
    """
    Linear chirp on unit time: ``φ(t) = 2π (f0 t + ½ β t²) + φ0``.

    ``f0`` and ``β`` are in cycles over the unit interval (i.e. number of
    oscillations across ``t∈[0,1]`` when ``β=0``).
    """
    phase = 2.0 * np.pi * (f0 * t + 0.5 * beta * t**2) + phase0
    return np.cos(phase) if use_cos else np.sin(phase)


def _tone_carrier(
    t: np.ndarray,
    frequency: float,
    phase0: float = 0.0,
    use_cos: bool = False,
) -> np.ndarray:
    """Constant-frequency carrier on unit time (``frequency`` cycles over [0, 1])."""
    phase = 2.0 * np.pi * frequency * t + phase0
    return np.cos(phase) if use_cos else np.sin(phase)


def _spectral_spread(x: np.ndarray) -> float:
    """Second-moment bandwidth proxy on the one-sided power spectrum."""
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)
    if x.size < 4:
        return 0.0
    psd = np.abs(np.fft.rfft(x)) ** 2
    if psd.size <= 1:
        return 0.0
    freqs = np.fft.rfftfreq(len(x))
    psd = psd.copy()
    psd[0] = 0.0
    total = float(np.sum(psd))
    if total <= 0:
        return 0.0
    mu = float(np.sum(freqs * psd) / total)
    var = float(np.sum(((freqs - mu) ** 2) * psd) / total)
    return float(np.sqrt(max(var, 0.0)))


class IntrinsicModeFunction(BaseExcitation):
    """
    Generates excitation time series via Intrinsic Mode Function (IMF) synthesis.

    A channel is an additive mixture of oscillatory modes (base tones / chirps,
    PySDKit choice waveforms, optional wavelet-like bursts), with optional
    localized envelopes and trend overlays, plus energy-adaptive noise and an
    energy cap.

    See also: PySDKit — https://github.com/wwhenxuan/PySDKit
    """

    def __init__(
        self,
        min_base_imfs: int = 2,
        max_base_imfs: int = 4,
        min_choice_imfs: int = 1,
        max_choice_imfs: int = 5,
        probability_dict: Optional[Dict[str, float]] = None,
        probability_list: Optional[List[float]] = None,
        min_duration: float = 0.5,
        max_duration: float = 10.0,
        min_amplitude: float = 0.01,
        max_amplitude: float = 10.0,
        min_frequency: float = 0.01,
        max_frequency: float = 8.0,
        noise_level: float = 0.1,
        upper_energy: Optional[float] = 32,
        # Envelope / wavelet-like bursts
        envelope_prob: float = 0.40,
        envelope_families: Optional[Sequence[str]] = None,
        envelope_center_range: Tuple[float, float] = (0.05, 0.95),
        envelope_width_range: Tuple[float, float] = (0.05, 0.35),
        envelope_invert_prob: float = 0.15,
        min_wavelets: int = 0,
        max_wavelets: int = 3,
        wavelet_amp_range: Tuple[float, float] = (0.2, 2.0),
        # Trends
        trend_prob: float = 0.35,
        trend_kinds: Optional[Sequence[str]] = None,
        trend_strength_range: Tuple[float, float] = (0.3, 1.5),
        trend_apply_on: str = "component",
        max_trend_segments: int = 4,
        # Chirp + amplitude hierarchy
        chirp_prob: float = 0.25,
        chirp_rate_range: Tuple[float, float] = (-6.0, 6.0),
        amplitude_decay_with_freq: bool = True,
        amplitude_decay_gamma: float = 0.5,
        dtype: np.dtype = np.float64,
    ) -> None:
        """
        :param envelope_prob: Probability of applying a localized envelope to a
            base/choice oscillatory component.
        :param envelope_families: Envelope families to sample from.
        :param envelope_center_range: Peak location on the unit interval.
        :param envelope_width_range: Envelope width fraction of the unit interval.
        :param envelope_invert_prob: Probability of an inverted envelope (mid dip).
        :param min_wavelets / max_wavelets: Extra localized burst count (inclusive).
        :param wavelet_amp_range: Amplitude range for wavelet-like bursts.
        :param trend_prob: Probability of injecting a trend (per component or sum).
        :param trend_kinds: Subset of ``{linear, piecewise, nonlinear}``.
        :param trend_strength_range: Strength range forwarded to trend helpers.
        :param trend_apply_on: ``\"component\"`` or ``\"sum\"``.
        :param max_trend_segments: Max pieces for piecewise trends.
        :param chirp_prob: Probability that a synthetic carrier is a linear chirp.
        :param chirp_rate_range: Chirp rate ``β`` over the unit interval.
        :param amplitude_decay_with_freq: If True, quieter high-frequency modes.
        :param amplitude_decay_gamma: Exponent for freq–amplitude coupling.
        """
        super().__init__(dtype=dtype)

        self.min_base_imfs = min_base_imfs
        self.max_base_imfs = max_base_imfs
        self.base_imfs = [generate_sin_signal, generate_cos_signal]

        self.min_choice_imfs = min_choice_imfs
        self.max_choice_imfs = max_choice_imfs

        (
            self.available_dict,
            self.available_list,
            self.available_probability,
        ) = self._processing_probability(
            probability_dict=probability_dict, probability_list=probability_list
        )

        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_amplitude = min_amplitude
        self.max_amplitude = max_amplitude
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self.upper_energy = upper_energy
        self.noise_level = noise_level

        self.envelope_prob = float(np.clip(envelope_prob, 0.0, 1.0))
        self.envelope_families = tuple(
            envelope_families
            if envelope_families is not None
            else _DEFAULT_ENVELOPE_FAMILIES
        )
        self.envelope_center_range = envelope_center_range
        self.envelope_width_range = envelope_width_range
        self.envelope_invert_prob = float(np.clip(envelope_invert_prob, 0.0, 1.0))
        self.min_wavelets = int(min_wavelets)
        self.max_wavelets = int(max_wavelets)
        self.wavelet_amp_range = wavelet_amp_range

        self.trend_prob = float(np.clip(trend_prob, 0.0, 1.0))
        self.trend_kinds = tuple(
            trend_kinds if trend_kinds is not None else _DEFAULT_TREND_KINDS
        )
        self.trend_strength_range = trend_strength_range
        trend_apply_on = str(trend_apply_on).strip().lower()
        if trend_apply_on not in {"component", "sum"}:
            raise ValueError("trend_apply_on must be 'component' or 'sum'")
        self.trend_apply_on = trend_apply_on
        self.max_trend_segments = int(max(1, max_trend_segments))

        self.chirp_prob = float(np.clip(chirp_prob, 0.0, 1.0))
        self.chirp_rate_range = chirp_rate_range
        self.amplitude_decay_with_freq = bool(amplitude_decay_with_freq)
        self.amplitude_decay_gamma = float(max(0.0, amplitude_decay_gamma))

        if self.min_wavelets < 0 or self.max_wavelets < self.min_wavelets:
            raise ValueError("Require 0 <= min_wavelets <= max_wavelets")
        if self.min_base_imfs < 0 or self.max_base_imfs < self.min_base_imfs:
            raise ValueError("Require 0 <= min_base_imfs <= max_base_imfs")
        if self.min_choice_imfs < 0 or self.max_choice_imfs < self.min_choice_imfs:
            raise ValueError("Require 0 <= min_choice_imfs <= max_choice_imfs")

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
        return self.__class__.__name__

    @property
    def all_imfs_dict(self) -> Dict[str, Callable]:
        """Get a dictionary of all available Eigen model functions"""
        return ALL_IMF_DICT

    @property
    def all_imfs_list(self) -> List[Callable]:
        """Get a list of all available eigenmode functions"""
        return list(self.all_imfs_dict.values())

    @property
    def default_probability_dict(self) -> Dict[str, float]:
        """Get the default probability dictionary when the user specifies parameters for the input"""
        return {
            "generate_sin_signal": 0.30,
            "generate_cos_signal": 0.30,
            "generate_am_signal": 0.20,
            "generate_sawtooth_wave": 0.20,
        }

    def _processing_probability(
        self,
        probability_dict: Optional[Dict[str, float]] = None,
        probability_list: Optional[List[float]] = None,
    ) -> Tuple[Dict[str, float], List[str], List[float]]:
        """
        Processes and validates input probability distributions for IMF selection.

        Handles four configuration scenarios:
        1. Both None: Uses default probability distribution
        2. Only dict provided: Validates and normalizes dictionary
        3. Only list provided: Validates and normalizes list
        4. Both provided: Prioritizes dictionary input
        """
        if probability_dict is None and probability_list is None:
            available_dict = _check_probability_dict(self.default_probability_dict)

        elif probability_dict is not None and probability_list is None:
            available_dict = _check_probability_dict(prob_dict=probability_dict)

        elif probability_dict is None and probability_list is not None:
            available_dict = _check_probability_list(prob_list=probability_list)

        elif probability_dict is not None and probability_list is not None:
            available_dict = _check_probability_dict(prob_dict=probability_dict)

        else:
            raise ValueError("Must provide either probability_dict or probability_list")

        available_list = list(available_dict.keys())
        available_probability = list(available_dict.values())
        return available_dict, available_list, available_probability

    def _add_noise(
        self,
        imfs: np.ndarray,
        seq_length: int,
        rng: Optional[np.random.RandomState] = None,
    ) -> np.ndarray:
        """
        Generates adaptive Gaussian noise proportional to signal RMS energy.

        ``STD = noise_level × RMS(imfs)``. Uses ``rng`` when provided so that
        full ``generate`` calls remain reproducible under a fixed seed.
        """
        std = self.noise_level * _get_energy(signal=imfs)
        if std <= 0.0:
            return np.zeros(seq_length, dtype=np.float64)
        if rng is not None:
            return rng.normal(loc=0.0, scale=std, size=seq_length)
        return add_noise(N=seq_length, Mean=0, STD=std)

    def get_random_duration(
        self, rng: np.random.RandomState, number: int
    ) -> np.ndarray:
        """Uniform durations in ``[min_duration, max_duration]``."""
        return rng.uniform(low=self.min_duration, high=self.max_duration, size=number)

    def get_random_amplitude(
        self, rng: np.random.RandomState, number: int
    ) -> np.ndarray:
        """Uniform amplitudes in ``[min_amplitude, max_amplitude]``."""
        return rng.uniform(low=self.min_amplitude, high=self.max_amplitude, size=number)

    def get_random_frequency(
        self, rng: np.random.RandomState, number: int
    ) -> np.ndarray:
        """Uniform frequencies in ``[min_frequency, max_frequency]``."""
        return rng.uniform(low=self.min_frequency, high=self.max_frequency, size=number)

    def _scale_amplitude_for_frequency(
        self, amplitude: float, frequency: float
    ) -> float:
        """EMD-like hierarchy: quieter high-frequency modes when enabled."""
        if not self.amplitude_decay_with_freq:
            return float(amplitude)
        f = max(float(frequency), 1e-8)
        f_ref = max(float(self.max_frequency), f)
        scale = (f_ref / f) ** self.amplitude_decay_gamma
        # Keep scale in a sane band so low-freq modes do not explode
        scale = float(np.clip(scale, 0.25, 4.0))
        return float(amplitude) * scale

    def _sample_envelope(
        self, t: np.ndarray, rng: np.random.RandomState
    ) -> np.ndarray:
        """Sample a random localized (or inverted) envelope on ``t``."""
        c_lo, c_hi = self.envelope_center_range
        w_lo, w_hi = self.envelope_width_range
        center = float(rng.uniform(c_lo, c_hi))
        width = float(rng.uniform(w_lo, w_hi))
        family = str(rng.choice(self.envelope_families))
        asymmetry = float(rng.uniform(0.5, 2.0))
        inverted = bool(rng.rand() < self.envelope_invert_prob)
        return _make_envelope(
            t,
            center=center,
            width=width,
            family=family,
            asymmetry=asymmetry,
            inverted=inverted,
        )

    def _maybe_apply_envelope(
        self, component: np.ndarray, t: np.ndarray, rng: np.random.RandomState
    ) -> np.ndarray:
        if rng.rand() >= self.envelope_prob:
            return component
        return component * self._sample_envelope(t, rng)

    def _maybe_apply_trend(
        self, series: np.ndarray, rng: np.random.RandomState
    ) -> np.ndarray:
        if self.trend_prob <= 0.0 or rng.rand() >= self.trend_prob:
            return series
        if not self.trend_kinds:
            return series

        kind = str(rng.choice(self.trend_kinds)).lower()
        s_lo, s_hi = self.trend_strength_range
        strength = float(rng.uniform(s_lo, s_hi))
        direction = "upward" if rng.rand() < 0.5 else "downward"

        if kind == "linear":
            return add_linear_trend(
                series,
                trend_strength=strength,
                direction=direction,
                normalize=False,
            )
        if kind == "piecewise":
            n_seg = int(rng.randint(2, self.max_trend_segments + 1))
            n_seg = min(n_seg, max(1, len(series)))
            return add_piecewise_linear_trend(
                series,
                num_segments=n_seg,
                strength_range=self.trend_strength_range,
                normalize=False,
                rng=rng,
            )
        # nonlinear (default fallback)
        return add_nonlinear_trend(
            series,
            kind=None,
            trend_strength=strength,
            direction=direction,
            normalize=False,
            rng=rng,
        )

    def _synthesize_carrier(
        self,
        t: np.ndarray,
        frequency: float,
        rng: np.random.RandomState,
        use_cos: bool = False,
    ) -> np.ndarray:
        """Tone or linear chirp on the unit interval."""
        phase0 = float(rng.uniform(0.0, 2.0 * np.pi))
        if rng.rand() < self.chirp_prob:
            b_lo, b_hi = self.chirp_rate_range
            beta = float(rng.uniform(b_lo, b_hi))
            return _chirp_carrier(
                t, f0=frequency, beta=beta, phase0=phase0, use_cos=use_cos
            )
        return _tone_carrier(t, frequency=frequency, phase0=phase0, use_cos=use_cos)

    def _finalize_component(
        self,
        component: np.ndarray,
        amplitude: float,
        frequency: float,
        t: np.ndarray,
        rng: np.random.RandomState,
        apply_envelope: bool = True,
    ) -> np.ndarray:
        """Scale, optional envelope, optional per-component trend."""
        amp = self._scale_amplitude_for_frequency(amplitude, frequency)
        out = amp * np.asarray(component, dtype=np.float64)
        if apply_envelope:
            out = self._maybe_apply_envelope(out, t, rng)
        if self.trend_apply_on == "component":
            out = self._maybe_apply_trend(out, rng)
        return out

    def get_base_imfs(
        self, imfs: np.ndarray, rng: np.random.RandomState, seq_length: int
    ) -> np.ndarray:
        """
        Generates fundamental IMF components (sine/cosine/chirp) and adds them.

        Carriers are synthesized on ``t∈[0,1]`` for consistent frequency meaning.
        """
        if self.max_base_imfs <= 0:
            return imfs

        base_number = rng.randint(
            low=self.min_base_imfs, high=self.max_base_imfs + 1
        )
        t = _unit_time(seq_length)

        for use_cos, amplitude, frequency in zip(
            rng.rand(base_number) < 0.5,
            self.get_random_amplitude(rng=rng, number=base_number),
            self.get_random_frequency(rng=rng, number=base_number),
        ):
            carrier = self._synthesize_carrier(
                t, frequency=float(frequency), rng=rng, use_cos=bool(use_cos)
            )
            imfs = imfs + self._finalize_component(
                carrier,
                amplitude=float(amplitude),
                frequency=float(frequency),
                t=t,
                rng=rng,
            )

        return imfs

    def get_choice_imfs(
        self, imfs: np.ndarray, rng: np.random.RandomState, seq_length: int
    ) -> np.ndarray:
        """Adds randomly selected IMF components from available PySDKit types."""
        if self.max_choice_imfs <= 0 or not self.available_list:
            return imfs

        choice_number = rng.randint(
            low=self.min_choice_imfs, high=self.max_choice_imfs + 1
        )
        t = _unit_time(seq_length)

        for choice_function, amplitude, frequency, duration in zip(
            rng.choice(
                self.available_list, size=choice_number, p=self.available_probability
            ),
            self.get_random_amplitude(rng=rng, number=choice_number),
            self.get_random_frequency(rng=rng, number=choice_number),
            self.get_random_duration(rng=rng, number=choice_number),
        ):
            func = ALL_IMF_DICT[choice_function]
            sampling_rate = get_adaptive_sampling_rate(
                duration=float(duration), length=seq_length
            )
            # Nyquist for the PySDKit sampling grid
            nyquist = 0.5 * float(sampling_rate)

            if func == generate_am_signal:
                carrier_hi = max(2, int(min(150, max(2, 0.4 * nyquist))))
                carrier_lo = max(1, min(50, carrier_hi - 1))
                mod_hi = max(2, int(min(16, max(2, 0.1 * nyquist))))
                component = generate_am_signal(
                    duration=float(duration),
                    sampling_rate=sampling_rate,
                    mod_index=int(rng.randint(1, 4)),
                    carrier_freq=int(rng.randint(carrier_lo, carrier_hi + 1)),
                    modulating_freq=int(rng.randint(1, mod_hi)),
                    noise_level=0.0,
                )[1][:seq_length]
            else:
                # Prefer unit-time synthetic carriers for sin/cos when chirp/envelope
                # diversity is desired; still allow PySDKit path for sawtooth etc.
                if func in (generate_sin_signal, generate_cos_signal):
                    component = self._synthesize_carrier(
                        t,
                        frequency=float(frequency),
                        rng=rng,
                        use_cos=(func is generate_cos_signal),
                    )
                else:
                    component = func(
                        duration=float(duration),
                        sampling_rate=sampling_rate,
                        frequency=float(frequency),
                        noise_level=0.0,
                    )[1][:seq_length]

            if len(component) < seq_length:
                pad = np.zeros(seq_length - len(component), dtype=np.float64)
                component = np.concatenate([component, pad])
            component = np.asarray(component[:seq_length], dtype=np.float64)

            imfs = imfs + self._finalize_component(
                component,
                amplitude=float(amplitude),
                frequency=float(frequency),
                t=t,
                rng=rng,
            )

        return imfs

    def get_wavelet_imfs(
        self, imfs: np.ndarray, rng: np.random.RandomState, seq_length: int
    ) -> np.ndarray:
        """Add localized wavelet-like oscillatory bursts."""
        if self.max_wavelets <= 0:
            return imfs

        n_burst = rng.randint(low=self.min_wavelets, high=self.max_wavelets + 1)
        if n_burst <= 0:
            return imfs

        t = _unit_time(seq_length)
        a_lo, a_hi = self.wavelet_amp_range

        for _ in range(n_burst):
            frequency = float(self.get_random_frequency(rng=rng, number=1)[0])
            amplitude = float(rng.uniform(a_lo, a_hi))
            carrier = self._synthesize_carrier(
                t, frequency=frequency, rng=rng, use_cos=bool(rng.rand() < 0.5)
            )
            # Always localize bursts
            env = self._sample_envelope(t, rng)
            burst = carrier * env
            imfs = imfs + self._finalize_component(
                burst,
                amplitude=amplitude,
                frequency=frequency,
                t=t,
                rng=rng,
                apply_envelope=False,  # already enveloped
            )

        return imfs

    def adjust_upper_energy(
        self, signal: np.ndarray, rng: np.random.RandomState
    ) -> np.ndarray:
        """
        Rescale the signal so its mean-square energy matches a random target
        in ``(0.05, 1.05] * upper_energy``.
        """
        if self.upper_energy is None:
            return signal

        energy = float(np.mean(np.asarray(signal, dtype=np.float64) ** 2))
        if energy <= 0.0:
            return signal

        target = (float(rng.rand()) + 0.05) * float(self.upper_energy)
        return signal * (target / energy)

    def generate(
        self,
        rng: np.random.RandomState,
        seq_length: int = 512,
        num_channels: int = 1,
    ) -> np.ndarray:
        """
        Generates multi-dimensional time series through IMF composition.

        Per channel:
        1. Base oscillatory / chirp components (with optional envelopes & trends)
        2. Choice IMF components (AM / sawtooth / ...)
        3. Wavelet-like localized bursts
        4. Optional sum-level trend
        5. Energy-adaptive Gaussian noise
        6. Mean-square energy cap
        """
        imfs = np.zeros(shape=(seq_length, num_channels), dtype=self.dtype)

        for i in range(num_channels):
            channel = np.zeros(seq_length, dtype=np.float64)
            channel = self.get_base_imfs(imfs=channel, rng=rng, seq_length=seq_length)
            channel = self.get_choice_imfs(
                imfs=channel, rng=rng, seq_length=seq_length
            )
            channel = self.get_wavelet_imfs(
                imfs=channel, rng=rng, seq_length=seq_length
            )

            if self.trend_apply_on == "sum":
                channel = self._maybe_apply_trend(channel, rng)

            channel = channel + self._add_noise(
                imfs=channel, seq_length=seq_length, rng=rng
            )
            channel = self.adjust_upper_energy(channel, rng=rng)
            imfs[:, i] = channel.astype(self.dtype, copy=False)

        return imfs
