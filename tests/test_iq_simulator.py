# -*- coding: utf-8 -*-
"""
Created on 2026/09/02
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/S2Generator
"""

import unittest

import matplotlib

matplotlib.use("Agg")

import numpy as np
from numpy.fft import fft, fftfreq

from s2generator.simulator import IQSimulator
from s2generator.utils.data import load_deepmimo_iq
from s2generator.utils.visualization import plot_iq_analysis, plot_iq_series


class TestIQSimulator(unittest.TestCase):
    """The Unittest for IQSimulator class and its conversion helpers."""

    @staticmethod
    def _make_series(length: int = 256, seed: int = 0) -> np.ndarray:
        """Build a reproducible damped cosine used as a real stimulus."""
        rng = np.random.RandomState(seed)
        t = np.linspace(0.0, 1.0, length, endpoint=False)
        return np.cos(2.0 * np.pi * 4.0 * t) * np.exp(-1.5 * t) + 0.02 * rng.randn(
            length
        )

    @staticmethod
    def _make_channel(n: int = 8, length: int = 128, seed: int = 1) -> np.ndarray:
        """Synthetic complex baseband traces with a two-sided Doppler lobe."""
        rng = np.random.RandomState(seed)
        t = np.arange(length)
        traces = []
        for _ in range(n):
            f_d = rng.uniform(0.02, 0.08)
            i_s = rng.normal(0.0, 1.0) * np.cos(2.0 * np.pi * f_d * t)
            q_s = rng.normal(0.0, 1.0) * np.sin(2.0 * np.pi * f_d * t)
            traces.append(np.stack([i_s, q_s], axis=-1))
        return np.stack(traces, axis=0)

    @staticmethod
    def _make_simulator(**kwargs) -> IQSimulator:
        """Create an IQSimulator with conservative defaults for unit testing."""
        defaults = {"random_state": 0, "match_mix": 1.0, "unit_power": True}
        defaults.update(kwargs)
        return IQSimulator(**defaults)

    def test_create_instance(self) -> None:
        """Construct IQSimulator instances across modes and mix values."""
        for mode in ("baseband", "analytic"):
            for mix in (0.0, 0.5, 1.0):
                for unit_power in (True, False):
                    with self.subTest(mode=mode, mix=mix, unit_power=unit_power):
                        sim = IQSimulator(
                            mode=mode,
                            match_mix=mix,
                            unit_power=unit_power,
                            random_state=0,
                        )
                        self.assertIsInstance(sim, IQSimulator)
                        self.assertEqual(sim.mode, mode)

    def test_invalid_constructor(self) -> None:
        """Reject constructor arguments that violate model constraints."""
        with self.assertRaises(ValueError):
            IQSimulator(mode="invalid")
        with self.assertRaises(ValueError):
            IQSimulator(match_mix=1.5)
        with self.assertRaises(ValueError):
            IQSimulator(lpf_order=0)

    def test_fit_channel_layouts(self) -> None:
        """fit should accept both (N, T, 2) and (N, 2, T) CSI stacks."""
        channel = self._make_channel()
        sim_a = self._make_simulator().fit(channel_ri=channel)
        sim_b = self._make_simulator().fit(channel_ri=np.transpose(channel, (0, 2, 1)))
        self.assertEqual(sim_a.target_psd.shape, (channel.shape[1],))
        np.testing.assert_allclose(sim_a.target_psd, sim_b.target_psd, rtol=1e-10)

    def test_fit_target_psd(self) -> None:
        """fit should store a ready-made periodogram without CSI snapshots."""
        psd = np.linspace(1.0, 0.1, 64)
        sim = self._make_simulator().fit(target_psd=psd)
        np.testing.assert_allclose(sim.target_psd, psd)
        self.assertIsNone(sim.channel_ri)

    def test_fit_requires_one_source(self) -> None:
        """fit should reject missing or duplicated PSD sources."""
        sim = self._make_simulator()
        with self.assertRaises(ValueError):
            sim.fit()
        with self.assertRaises(ValueError):
            sim.fit(channel_ri=self._make_channel(), target_psd=np.ones(32))

    def test_check_inputs_and_errors(self) -> None:
        """check_inputs should accept 1-D / 2-D series and reject invalid arrays."""
        sim = self._make_simulator()
        series = self._make_series(64)
        np.testing.assert_array_equal(sim.check_inputs(series), series)
        dual = np.stack([series, series], axis=1)
        self.assertEqual(sim.check_inputs(dual).shape, (64, 2))
        self.assertEqual(sim.check_inputs(dual.T).shape, (64, 2))
        with self.assertRaises(ValueError):
            sim.check_inputs(np.ones(4))
        with self.assertRaises(ValueError):
            sim.check_inputs(np.array([np.nan, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]))
        with self.assertRaises(ValueError):
            sim.check_inputs(np.ones((2, 2, 2)))
        with self.assertRaises(ValueError):
            sim.check_channel_inputs(np.ones((3, 16)))

    def test_transform_default_arma(self) -> None:
        """Omitting the stimulus should draw two ARMA channels and return [2, 2, L]."""
        sim = self._make_simulator().fit(channel_ri=self._make_channel())
        iq = sim.transform(seq_length=64, num_channels=2)
        self.assertEqual(iq.shape, (2, 2, 64))
        self.assertTrue(np.isfinite(iq).all())

    def test_transform_one_series(self) -> None:
        """A 1-D stimulus should map to a single [2, L] IQ pair."""
        sim = self._make_simulator().fit(channel_ri=self._make_channel())
        series = self._make_series(200)
        iq = sim.transform(time_series=series, seq_length=128)
        self.assertEqual(iq.shape, (2, 128))
        self.assertTrue(np.isfinite(iq).all())
        complex_iq = sim.transform(
            time_series=series, seq_length=128, return_complex=True
        )
        self.assertEqual(complex_iq.shape, (128,))
        self.assertTrue(np.iscomplexobj(complex_iq))

    def test_transform_two_dimensional(self) -> None:
        """A 2-D stimulus should map each column to its own IQ pair."""
        sim = self._make_simulator().fit(channel_ri=self._make_channel())
        series = np.stack(
            [self._make_series(180, 0), self._make_series(180, 1)], axis=1
        )
        iq = sim.transform(time_series=series, seq_length=128)
        self.assertEqual(iq.shape, (2, 2, 128))

    def test_transform_without_fit(self) -> None:
        """Hilbert-only conversion should work when no PSD has been fitted."""
        sim = self._make_simulator()
        iq = sim.transform(time_series=self._make_series(96), seq_length=64)
        self.assertEqual(iq.shape, (2, 64))
        analytic = hilbert_q_correlation(iq)
        self.assertGreater(analytic, 0.9)

    def test_time_series_to_iq_unit_power(self) -> None:
        """unit_power should force the complex series onto unit mean power."""
        sim = self._make_simulator(unit_power=True).fit(channel_ri=self._make_channel())
        z = sim.time_series_to_iq(self._make_series(200), out_len=128)
        power = float(np.mean(np.abs(z) ** 2))
        self.assertAlmostEqual(power, 1.0, places=6)

    def test_time_series_to_iq_too_short(self) -> None:
        """time_series_to_iq should reject short series and oversized out_len."""
        sim = self._make_simulator()
        with self.assertRaises(ValueError):
            sim.time_series_to_iq(np.ones(4))
        with self.assertRaises(ValueError):
            sim.time_series_to_iq(self._make_series(32), out_len=64)

    def test_match_mix_extremes(self) -> None:
        """match_mix=0 should keep the Hilbert seed away from a full PSD match."""
        channel = self._make_channel()
        series = self._make_series(256)
        mixed = (
            self._make_simulator(match_mix=1.0)
            .fit(channel_ri=channel)
            .time_series_to_iq(series, out_len=128)
        )
        seed = (
            self._make_simulator(match_mix=0.0)
            .fit(channel_ri=channel)
            .time_series_to_iq(series, out_len=128)
        )
        self.assertGreater(float(np.mean(np.abs(mixed - seed))), 1e-3)

    def test_baseband_keeps_negative_frequency(self) -> None:
        """Baseband mode should retain energy on negative Doppler bins."""
        channel = self._make_channel()
        series = self._make_series(256)
        z = (
            self._make_simulator(mode="baseband")
            .fit(channel_ri=channel)
            .time_series_to_iq(series, out_len=128)
        )
        spectrum = fft(z - np.mean(z))
        n = spectrum.size
        neg = float(np.sum(np.abs(spectrum[n // 2 :]) ** 2))
        pos = float(np.sum(np.abs(spectrum[1 : n // 2]) ** 2))
        self.assertGreater(neg / (pos + 1e-18), 0.05)

    def test_analytic_zeros_negative_frequency(self) -> None:
        """Analytic mode should put essentially all energy on positive frequencies."""
        channel = self._make_channel()
        series = self._make_series(256)
        z = (
            self._make_simulator(mode="analytic")
            .fit(channel_ri=channel)
            .time_series_to_iq(series, out_len=128)
        )
        spectrum = fft(z - np.mean(z))
        n = spectrum.size
        neg = float(np.sum(np.abs(spectrum[n // 2 :]) ** 2))
        total = float(np.sum(np.abs(spectrum) ** 2)) + 1e-18
        self.assertLess(neg / total, 1e-6)

    def test_project_analytic_spectrum(self) -> None:
        """project_analytic_spectrum should zero DC and the negative-frequency half."""
        sim = self._make_simulator()
        spectrum = np.ones(16, dtype=np.complex128)
        projected = sim.project_analytic_spectrum(spectrum, keep_dc=False)
        self.assertEqual(projected[0], 0.0)
        self.assertTrue(np.all(projected[8:] == 0.0))
        self.assertNotEqual(projected[1], 0.0)

    def test_f_cut_from_psd(self) -> None:
        """f_cut_from_psd should grow when energy is placed farther from DC."""
        sim = self._make_simulator()
        n = 64
        freqs = fftfreq(n)
        narrow = np.exp(-((np.abs(freqs) / 0.02) ** 2))
        wide = np.exp(-((np.abs(freqs) / 0.15) ** 2))
        self.assertLess(sim.f_cut_from_psd(narrow), sim.f_cut_from_psd(wide))
        self.assertEqual(sim.f_cut_from_psd(np.zeros(n)), 0.05)

    def test_infill_phase_strong_and_weak(self) -> None:
        """_infill_phase should keep strong bins and invent a ramp when all bins are weak."""
        sim = self._make_simulator()
        strong = np.exp(1j * np.linspace(0.0, np.pi, 32, endpoint=False))
        phase = sim._infill_phase(strong)
        np.testing.assert_allclose(phase, np.angle(strong), atol=1e-10)
        weak = 1e-20 * np.ones(16, dtype=np.complex128)
        filled = sim._infill_phase(weak)
        self.assertEqual(filled.shape, (16,))
        self.assertTrue(np.isfinite(filled).all())
        tiny = sim._infill_phase(np.array([1.0 + 0j]))
        self.assertEqual(tiny.shape, (1,))

    def test_resample_psd_to_length(self) -> None:
        """_resample_psd_to_length should copy equal lengths and interpolate others."""
        sim = self._make_simulator()
        psd = np.linspace(1.0, 0.2, 32)
        same = sim._resample_psd_to_length(psd, 32)
        np.testing.assert_array_equal(same, psd)
        longer = sim._resample_psd_to_length(psd, 64)
        self.assertEqual(longer.shape, (64,))
        self.assertTrue(np.all(longer > 0.0))

    def test_butter_lowpass(self) -> None:
        """_butter_lowpass should attenuate a high-frequency tone."""
        sim = self._make_simulator()
        t = np.linspace(0.0, 1.0, 256, endpoint=False)
        x = np.sin(2.0 * np.pi * 40.0 * t)
        y = sim._butter_lowpass(x, f_cut=0.05, order=4)
        self.assertLess(float(np.std(y)), float(np.std(x)))

    def test_spectral_shape_runs(self) -> None:
        """_spectral_shape should return a finite complex series of the same length."""
        sim = self._make_simulator()
        z = self._make_series(64).astype(np.complex128)
        psd = np.ones(64)
        shaped = sim._spectral_shape(z, psd, mix=1.0, mode="baseband", keep_dc=False)
        self.assertEqual(shaped.shape, (64,))
        self.assertTrue(np.isfinite(shaped).all())

    def test_as_complex_channels_errors(self) -> None:
        """_as_complex_channels should reject layouts that are not I/Q stacks."""
        sim = self._make_simulator()
        with self.assertRaises(ValueError):
            sim._as_complex_channels(np.ones((4, 8, 3)))
        with self.assertRaises(ValueError):
            sim._as_complex_channels(np.ones((4, 8)))

    def test_default_arma_series(self) -> None:
        """_default_arma_series should return a longer-than-requested 2-D path."""
        sim = self._make_simulator()
        drawn = sim._default_arma_series(seq_length=32, num_channels=2, random_state=0)
        self.assertEqual(drawn.ndim, 2)
        self.assertEqual(drawn.shape[1], 2)
        self.assertGreaterEqual(drawn.shape[0], 32)
        with self.assertRaises(ValueError):
            sim._default_arma_series(seq_length=32, num_channels=0)

    def test_resolve_out_len(self) -> None:
        """_resolve_out_len should use seq_length, then the PSD length, then 128."""
        sim = self._make_simulator()
        self.assertEqual(sim._resolve_out_len(32), 32)
        self.assertEqual(sim._resolve_out_len(None), 128)
        sim.fit(target_psd=np.ones(48))
        self.assertEqual(sim._resolve_out_len(None), 48)
        with self.assertRaises(ValueError):
            sim._resolve_out_len(4)

    def test_load_deepmimo_and_generate(self) -> None:
        """Packaged DeepMIMO CSI should fit the simulator and yield a finite IQ pair."""
        traces = load_deepmimo_iq(speed_kmh=30, subcarrier=0)
        self.assertEqual(traces.ndim, 3)
        self.assertEqual(traces.shape[-1], 2)
        iq = (
            self._make_simulator()
            .fit(channel_ri=traces)
            .transform(time_series=self._make_series(200), seq_length=128)
        )
        self.assertEqual(iq.shape, (2, 128))
        self.assertTrue(np.isfinite(iq).all())

    def test_plot_helpers_accept_generated_iq(self) -> None:
        """Generated IQ pairs should be accepted by the IQ visualisation helpers."""
        sim = self._make_simulator().fit(channel_ri=self._make_channel())
        iq = sim.transform(time_series=self._make_series(160), seq_length=96)
        fig_a = plot_iq_series(iq, overlay=True)
        fig_b = plot_iq_analysis(iq, overlay=True)
        self.assertGreaterEqual(len(fig_a.axes), 1)
        self.assertGreaterEqual(len(fig_b.axes), 6)


def hilbert_q_correlation(iq: np.ndarray) -> float:
    """Correlation between Q and the Hilbert transform of I."""
    from scipy.signal import hilbert as analytic

    i_s, q_s = np.asarray(iq[0], dtype=np.float64), np.asarray(iq[1], dtype=np.float64)
    h_q = np.imag(analytic(i_s - np.mean(i_s)))
    return float(np.corrcoef(h_q, q_s)[0, 1])


if __name__ == "__main__":
    unittest.main()
