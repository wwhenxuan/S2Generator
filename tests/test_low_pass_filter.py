# -*- coding: utf-8 -*-
"""Unit tests for adaptive / manual low-pass post-processing."""

import unittest

import numpy as np

from s2generator.simulator import LowPassFilter, WienerFilterSimulator, apply_lowpass
from s2generator.simulator.low_pass_filter import (
    maybe_attach_lowpass,
    maybe_apply_lowpass,
)


class TestLowPassFilter(unittest.TestCase):
    """Tests for LowPassFilter core behavior."""

    def setUp(self) -> None:
        rng = np.random.RandomState(0)
        t = np.arange(256, dtype=np.float64)
        # Dominant low-frequency sinusoid plus weak high-frequency noise
        self.reference = np.sin(2 * np.pi * 0.03 * t) + 0.05 * rng.randn(256)
        self.noisy = self.reference + 0.8 * np.sin(2 * np.pi * 0.35 * t)

    def test_adaptive_cutoff_in_range(self) -> None:
        lpf = LowPassFilter(energy_ratio=0.98).fit(self.reference)
        self.assertGreaterEqual(lpf.cutoff_, 0.05)
        self.assertLessEqual(lpf.cutoff_, 0.95)

    def test_manual_cutoff_overrides_adaptive(self) -> None:
        lpf = LowPassFilter(cutoff=0.2, energy_ratio=0.5).fit(self.reference)
        self.assertAlmostEqual(lpf.cutoff_, 0.2, places=6)

    def test_high_frequency_power_decreases(self) -> None:
        filtered = LowPassFilter(cutoff=0.15).fit_transform(self.reference, self.noisy)
        freqs = np.fft.rfftfreq(len(self.noisy))
        high_band = freqs > 0.25
        raw_power = np.mean(np.abs(np.fft.rfft(self.noisy))[high_band] ** 2)
        filt_power = np.mean(np.abs(np.fft.rfft(filtered))[high_band] ** 2)
        self.assertLess(filt_power, raw_power)

    def test_zero_phase_keeps_waveform_aligned(self) -> None:
        t = np.arange(512, dtype=np.float64)
        clean = np.sin(2 * np.pi * 0.02 * t)
        filtered = LowPassFilter(cutoff=0.2, revin=False).fit_transform(clean, clean)
        # Zero-phase filtfilt should preserve alignment of in-band content.
        mid = slice(100, 400)
        corr = float(np.corrcoef(clean[mid], filtered[mid])[0, 1])
        self.assertGreater(corr, 0.99)

    def test_shape_1d_and_2d(self) -> None:
        lpf = LowPassFilter(cutoff=0.3).fit(self.reference)
        out_1d = lpf.transform(self.noisy)
        self.assertEqual(out_1d.shape, self.noisy.shape)
        batch = np.vstack([self.noisy, self.noisy])
        out_2d = lpf.transform(batch)
        self.assertEqual(out_2d.shape, batch.shape)

    def test_apply_lowpass_helper(self) -> None:
        out = apply_lowpass(self.noisy, self.reference, cutoff=0.25)
        self.assertEqual(out.shape, self.noisy.shape)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_maybe_attach_and_apply(self) -> None:
        owner = type("Owner", (), {})()
        maybe_attach_lowpass(
            owner, enabled=False, kwargs=None, reference=self.reference
        )
        self.assertIsNone(owner._lowpass_filter)
        unchanged = maybe_apply_lowpass(owner, self.noisy)
        np.testing.assert_array_equal(unchanged, self.noisy)

        maybe_attach_lowpass(
            owner, enabled=True, kwargs={"cutoff": 0.2}, reference=self.reference
        )
        self.assertIsNotNone(owner._lowpass_filter)
        filtered = maybe_apply_lowpass(owner, self.noisy)
        self.assertEqual(filtered.shape, self.noisy.shape)
        self.assertFalse(np.allclose(filtered, self.noisy))

    def test_invalid_params(self) -> None:
        with self.assertRaises(ValueError):
            LowPassFilter(energy_ratio=0.0)
        with self.assertRaises(ValueError):
            LowPassFilter(cutoff=1.5)
        with self.assertRaises(ValueError):
            LowPassFilter(min_cutoff=0.8, max_cutoff=0.2)


class TestWienerLowpassIntegration(unittest.TestCase):
    """End-to-end low-pass wiring for WienerFilterSimulator."""

    def test_wiener_lowpass_transform(self) -> None:
        rng = np.random.RandomState(1)
        t = np.arange(200, dtype=np.float64)
        x = np.sin(2 * np.pi * 0.04 * t) + 0.1 * rng.randn(200)

        sim = WienerFilterSimulator(
            filter_order=4,
            lowpass=True,
            lowpass_kwargs={"energy_ratio": 0.95},
            random_state=0,
        )
        sim.fit(x)
        y = sim.transform(num_samples=3, seq_len=len(x), random_state=0)

        self.assertEqual(y.shape, (3, len(x)))
        self.assertTrue(np.all(np.isfinite(y)))
        self.assertIsNotNone(sim._lowpass_filter)
        self.assertGreater(sim._lowpass_filter.cutoff_, 0.0)

    def test_wiener_default_no_lowpass(self) -> None:
        rng = np.random.RandomState(2)
        x = rng.randn(128)
        sim = WienerFilterSimulator(filter_order=4, random_state=0)
        sim.fit(x)
        self.assertIsNone(sim._lowpass_filter)
        y = sim.transform(num_samples=2, seq_len=64, random_state=0)
        self.assertEqual(y.shape, (2, 64))


if __name__ == "__main__":
    unittest.main()
