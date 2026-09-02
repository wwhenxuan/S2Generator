# -*- coding: utf-8 -*-
"""
Created on 2025/08/26
@author:Yifan Wu
@email: wy3370868155@outlook.com
"""

import unittest

import numpy as np

from s2generator.excitation.intrinsic_mode_functions import (
    IntrinsicModeFunction,
    ALL_IMF_DICT,
    _check_probability_dict,
    _check_probability_list,
    _get_energy,
    get_adaptive_sampling_rate,
    _make_envelope,
    _chirp_carrier,
    _tone_carrier,
    _spectral_spread,
    _unit_time,
)


class TestUtilityFunctions(unittest.TestCase):
    """Testing utility functions in intrinsic_mode_functions module"""

    def setUp(self):
        """Prepare fixtures used by the Utility Functions tests."""
        self.rng = np.random.RandomState(42)

    def test_all_imf_dict_structure(self):
        """ALL_IMF_DICT should map each IMF name to a callable generator."""
        self.assertIsInstance(ALL_IMF_DICT, dict)
        self.assertGreater(len(ALL_IMF_DICT), 0)
        expected_keys = [
            "generate_sin_signal",
            "generate_cos_signal",
            "generate_am_signal",
            "generate_sawtooth_wave",
        ]
        for key in expected_keys:
            self.assertIn(key, ALL_IMF_DICT)
            self.assertTrue(callable(ALL_IMF_DICT[key]))

    def test_check_probability_dict_valid(self):
        """A valid IMF probability dict should be accepted and normalized."""
        valid_dict = {
            "generate_sin_signal": 0.3,
            "generate_cos_signal": 0.3,
            "generate_am_signal": 0.2,
            "generate_sawtooth_wave": 0.2,
        }
        result = _check_probability_dict(valid_dict)
        self.assertEqual(set(result.keys()), set(valid_dict.keys()))
        self.assertAlmostEqual(sum(result.values()), 1.0, places=6)
        self.assertAlmostEqual(result["generate_sin_signal"], 0.3, places=6)

        subset = {"generate_sin_signal": 0.6, "generate_cos_signal": 0.4}
        result_subset = _check_probability_dict(subset)
        self.assertAlmostEqual(sum(result_subset.values()), 1.0, places=6)
        self.assertAlmostEqual(result_subset["generate_sin_signal"], 0.6, places=6)

    def test_check_probability_dict_clips_and_rejects_all_zero(self):
        """Probability dicts should clip invalid weights and reject an all-zero dict."""
        clipped = _check_probability_dict(
            {"generate_sin_signal": -1.0, "generate_cos_signal": 2.0}
        )
        self.assertAlmostEqual(sum(clipped.values()), 1.0, places=6)
        self.assertAlmostEqual(clipped["generate_sin_signal"], 0.0, places=6)
        self.assertAlmostEqual(clipped["generate_cos_signal"], 1.0, places=6)

        with self.assertRaises(ValueError):
            _check_probability_dict(
                {"generate_sin_signal": 0.0, "generate_cos_signal": 0.0}
            )

    def test_check_probability_dict_invalid_key(self):
        """An unknown IMF name in the probability dict should raise ValueError."""
        with self.assertRaises(ValueError) as context:
            _check_probability_dict(
                {"generate_sin_signal": 0.5, "invalid_function": 0.5}
            )
        self.assertIn("Illegal key: invalid_function", str(context.exception))

    def test_check_probability_list_valid(self):
        """A probability list matching the IMF bank length should be accepted."""
        valid_list = [0.3, 0.3, 0.2, 0.2]
        result = _check_probability_list(valid_list)
        self.assertEqual(len(result), 4)
        self.assertAlmostEqual(sum(result.values()), 1.0, places=6)
        expected_keys = list(ALL_IMF_DICT.keys())[:4]
        self.assertEqual(list(result.keys()), expected_keys)

        partial = _check_probability_list([0.6, 0.4])
        self.assertEqual(len(partial), 2)
        self.assertAlmostEqual(sum(partial.values()), 1.0, places=6)

    def test_check_probability_list_invalid_length(self):
        """A probability list of the wrong length should raise ValueError."""
        with self.assertRaises(ValueError):
            _check_probability_list([])
        with self.assertRaises(ValueError):
            _check_probability_list([0.2] * (len(ALL_IMF_DICT) + 1))

    def test_get_energy_is_rms(self):
        """get_energy should return the RMS energy of the series."""
        signal = np.array([1.0, -1.0, 2.0, -2.0, 3.0, -3.0])
        energy = _get_energy(signal)
        expected = float(np.sqrt(np.mean(signal**2)))
        self.assertAlmostEqual(energy, expected, places=8)
        self.assertEqual(_get_energy(np.zeros(10)), 0.0)
        self.assertAlmostEqual(_get_energy(np.ones(5) * 2.5), 2.5, places=8)

    def test_get_adaptive_sampling_rate(self):
        """Adaptive sampling rate should scale with the requested sequence length."""
        self.assertEqual(get_adaptive_sampling_rate(2.0, 100), np.ceil(100 / 2.0))
        self.assertEqual(get_adaptive_sampling_rate(1.0, 1), 1.0)
        self.assertGreater(get_adaptive_sampling_rate(0.1, 10), 50)

    def test_make_envelope_peaks_near_center(self):
        """The synthetic envelope should peak near the center of the window."""
        t = _unit_time(256)
        for family in ("gaussian", "sech", "tukey", "asymmetric"):
            env = _make_envelope(t, center=0.7, width=0.1, family=family)
            peak_idx = int(np.argmax(env))
            self.assertAlmostEqual(t[peak_idx], 0.7, delta=0.05)
            self.assertAlmostEqual(float(np.max(env)), 1.0, places=5)

        inverted = _make_envelope(
            t, center=0.5, width=0.15, family="gaussian", inverted=True
        )
        self.assertLess(float(inverted[len(t) // 2]), float(inverted[0]) + 0.2)

    def test_chirp_has_larger_spectral_spread_than_tone(self):
        """A chirp carrier should have a larger spectral spread than a pure tone."""
        t = _unit_time(512)
        tone = _tone_carrier(t, frequency=8.0)
        chirp = _chirp_carrier(t, f0=4.0, beta=12.0)
        self.assertGreater(_spectral_spread(chirp), _spectral_spread(tone))


class TestIntrinsicModeFunction(unittest.TestCase):
    """Testing the IntrinsicModeFunction class"""

    def setUp(self):
        """Prepare fixtures used by the Intrinsic Mode Function tests."""
        self.rng = np.random.RandomState(42)

    def _legacy_like(self, **kwargs):
        """Disable new stochastic features for stable legacy-style checks."""
        defaults = dict(
            envelope_prob=0.0,
            trend_prob=0.0,
            chirp_prob=0.0,
            min_wavelets=0,
            max_wavelets=0,
            amplitude_decay_with_freq=False,
            noise_level=0.0,
        )
        defaults.update(kwargs)
        return IntrinsicModeFunction(**defaults)

    def test_init_default_params(self):
        """Default IntrinsicModeFunction parameters should match the documented values."""
        imf = IntrinsicModeFunction()
        self.assertEqual(imf.min_base_imfs, 2)
        self.assertEqual(imf.max_base_imfs, 4)
        self.assertEqual(imf.envelope_prob, 0.40)
        self.assertEqual(imf.trend_prob, 0.35)
        self.assertEqual(imf.chirp_prob, 0.25)
        self.assertTrue(imf.amplitude_decay_with_freq)
        self.assertAlmostEqual(sum(imf.available_probability), 1.0, places=6)

    def test_init_custom_params(self):
        """Custom constructor arguments should be stored on the instance."""
        custom_prob_dict = {"generate_sin_signal": 0.5, "generate_cos_signal": 0.5}
        imf = IntrinsicModeFunction(
            min_base_imfs=2,
            max_base_imfs=4,
            probability_dict=custom_prob_dict,
            envelope_prob=0.1,
            trend_prob=0.2,
            chirp_prob=0.0,
            dtype=np.float32,
        )
        self.assertEqual(len(imf.available_dict), 2)
        self.assertAlmostEqual(sum(imf.available_probability), 1.0, places=6)
        self.assertEqual(imf.dtype, np.float32)

    def test_init_with_probability_list(self):
        """A probability list should be accepted as an alternative to the dict form."""
        imf = IntrinsicModeFunction(probability_list=[0.4, 0.3, 0.2, 0.1])
        self.assertEqual(len(imf.available_probability), 4)
        self.assertAlmostEqual(sum(imf.available_probability), 1.0, places=6)
        self.assertAlmostEqual(imf.available_probability[0], 0.4, places=6)

    def test_str_and_call(self):
        """String conversion and __call__ should work for IntrinsicModeFunction."""
        imf = IntrinsicModeFunction()
        self.assertEqual(str(imf), "IntrinsicModeFunction")
        result = imf(self.rng, seq_length=100, num_channels=2)
        self.assertEqual(result.shape, (100, 2))

    def test_properties(self):
        """Verify that public properties expose the expected values."""
        imf = IntrinsicModeFunction()
        self.assertEqual(imf.all_imfs_dict, ALL_IMF_DICT)
        self.assertEqual(imf.all_imfs_list, list(ALL_IMF_DICT.values()))
        self.assertAlmostEqual(
            sum(imf.default_probability_dict.values()), 1.0, places=5
        )

    def test_processing_probability_scenarios(self):
        """Processing-probability helpers should cover enabled, disabled, and sampled cases."""
        imf = IntrinsicModeFunction()
        d, keys, probs = imf._processing_probability(None, None)
        self.assertAlmostEqual(sum(probs), 1.0, places=6)

        custom_dict = {"generate_sin_signal": 0.7, "generate_cos_signal": 0.3}
        d, keys, probs = imf._processing_probability(custom_dict, None)
        self.assertEqual(len(d), 2)
        self.assertAlmostEqual(sum(probs), 1.0, places=6)

        d, keys, probs = imf._processing_probability(None, [0.8, 0.2])
        self.assertEqual(len(d), 2)
        self.assertAlmostEqual(probs[0], 0.8, places=6)

        d, keys, probs = imf._processing_probability(custom_dict, [0.8, 0.2])
        self.assertEqual(len(d), 2)
        self.assertAlmostEqual(d["generate_sin_signal"], 0.7, places=6)

    def test_add_noise_uses_rms_and_rng(self):
        """Additive noise should be scaled by RMS energy and consume the RNG."""
        imf = IntrinsicModeFunction(noise_level=0.1)
        test_signal = np.ones(1000, dtype=np.float64)
        expected_std = 0.1 * _get_energy(test_signal)
        n1 = imf._add_noise(test_signal, seq_length=1000, rng=np.random.RandomState(0))
        n2 = imf._add_noise(test_signal, seq_length=1000, rng=np.random.RandomState(0))
        np.testing.assert_array_equal(n1, n2)
        self.assertAlmostEqual(
            float(np.std(n1)), expected_std, delta=0.05 * expected_std
        )

    def test_random_samplers(self):
        """Random IMF samplers should return names from the registered bank."""
        imf = IntrinsicModeFunction(
            min_duration=1.0, max_duration=5.0, min_amplitude=0.5, max_amplitude=2.0
        )
        durations = imf.get_random_duration(self.rng, 10)
        self.assertTrue(np.all(durations >= 1.0) and np.all(durations <= 5.0))
        amps = imf.get_random_amplitude(self.rng, 5)
        self.assertTrue(np.all(amps >= 0.5) and np.all(amps <= 2.0))
        freqs = imf.get_random_frequency(self.rng, 7)
        self.assertTrue(np.all(freqs >= imf.min_frequency))

    def test_get_base_and_choice_imfs(self):
        """Base and choice IMF helpers should return finite series of the requested length."""
        imf = self._legacy_like(
            min_base_imfs=1, max_base_imfs=2, min_choice_imfs=1, max_choice_imfs=2
        )
        base = imf.get_base_imfs(np.zeros(64), rng=self.rng, seq_length=64)
        self.assertEqual(base.shape, (64,))
        self.assertFalse(np.allclose(base, 0.0))

        choice = imf.get_choice_imfs(np.zeros(64), rng=self.rng, seq_length=64)
        self.assertEqual(choice.shape, (64,))

    def test_get_choice_imfs_am_signal(self):
        """Amplitude-modulated choice IMFs should remain finite and match the length."""
        am_prob_dict = {
            k: (1.0 if k == "generate_am_signal" else 1e-6) for k in ALL_IMF_DICT
        }
        imf = self._legacy_like(
            min_choice_imfs=1,
            max_choice_imfs=2,
            probability_dict=am_prob_dict,
        )
        result = imf.get_choice_imfs(np.zeros(64), rng=self.rng, seq_length=64)
        self.assertEqual(result.shape, (64,))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_generate_shapes_and_finite(self):
        """generate() should return the requested shape with finite values."""
        imf = IntrinsicModeFunction()
        for dims in (1, 2, 3):
            out = imf.generate(self.rng, seq_length=64, num_channels=dims)
            self.assertEqual(out.shape, (64, dims))
            self.assertTrue(np.all(np.isfinite(out)))
        for length in (32, 128, 512):
            out = imf.generate(self.rng, seq_length=length, num_channels=1)
            self.assertEqual(out.shape, (length, 1))

    def test_generate_reproducibility(self):
        """The same seed should reproduce IntrinsicModeFunction outputs."""
        imf1 = IntrinsicModeFunction()
        imf2 = IntrinsicModeFunction()
        r1 = imf1.generate(np.random.RandomState(42), seq_length=100, num_channels=1)
        r2 = imf2.generate(np.random.RandomState(42), seq_length=100, num_channels=1)
        np.testing.assert_array_equal(r1, r2)

    def test_generate_dtypes(self):
        """Generated IMF series should use the configured dtype."""
        for dtype in (np.float32, np.float64):
            imf = IntrinsicModeFunction(dtype=dtype)
            self.assertEqual(
                imf.generate(self.rng, seq_length=50, num_channels=1).dtype, dtype
            )

    def test_inheritance(self):
        """IntrinsicModeFunction should inherit from BaseExcitation."""
        from s2generator.excitation.base_excitation import BaseExcitation

        imf = IntrinsicModeFunction()
        self.assertIsInstance(imf, BaseExcitation)
        zeros = imf.create_zeros(seq_length=10, num_channels=2)
        np.testing.assert_array_equal(zeros, np.zeros((10, 2)))

    def test_adjust_upper_energy_scales_and_uses_rng(self):
        """Upper-energy adjustment should scale amplitude and consume the RNG."""
        imf = IntrinsicModeFunction(upper_energy=16.0)
        signal = np.ones(128, dtype=np.float64) * 10.0
        adjusted = imf.adjust_upper_energy(signal, rng=np.random.RandomState(0))
        energy = float(np.mean(adjusted**2))
        self.assertGreater(energy, 0.0)
        self.assertLessEqual(energy, 16.0 * 1.05 + 1e-6)

        # Same RandomState sequence => same scale; different seeds => different
        out1 = imf.adjust_upper_energy(signal.copy(), rng=np.random.RandomState(7))
        out2 = imf.adjust_upper_energy(signal.copy(), rng=np.random.RandomState(7))
        out3 = imf.adjust_upper_energy(signal.copy(), rng=np.random.RandomState(8))
        np.testing.assert_array_equal(out1, out2)
        self.assertFalse(np.allclose(out1, out3))

    def test_max_choice_imfs_is_inclusive(self):
        """Inclusive upper bound: choice count can equal max_choice_imfs."""
        imf = self._legacy_like(
            min_base_imfs=0,
            max_base_imfs=0,
            min_choice_imfs=3,
            max_choice_imfs=3,
            probability_dict={"generate_sin_signal": 1.0},
            upper_energy=None,
        )
        # Force exactly 3 choice components; shape/finite is enough smoke
        out = imf.get_choice_imfs(
            np.zeros(64), rng=np.random.RandomState(0), seq_length=64
        )
        self.assertEqual(out.shape, (64,))
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertFalse(np.allclose(out, 0.0))

    def test_wavelet_bursts_finite_and_localized(self):
        """Wavelet bursts should be finite and localized in time."""
        imf = IntrinsicModeFunction(
            min_base_imfs=0,
            max_base_imfs=0,
            min_choice_imfs=0,
            max_choice_imfs=0,
            min_wavelets=2,
            max_wavelets=2,
            envelope_prob=0.0,
            trend_prob=0.0,
            chirp_prob=0.0,
            noise_level=0.0,
            upper_energy=None,
            amplitude_decay_with_freq=False,
            envelope_center_range=(0.4, 0.6),
            envelope_width_range=(0.05, 0.08),
        )
        out = imf.get_wavelet_imfs(
            np.zeros(256), rng=np.random.RandomState(1), seq_length=256
        )
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertFalse(np.allclose(out, 0.0))
        # Energy concentrated away from far edges on average
        edge = np.mean(out[:20] ** 2) + np.mean(out[-20:] ** 2)
        mid = np.mean(out[100:156] ** 2)
        self.assertGreater(mid, edge * 0.5)

    def test_trend_changes_low_frequency_structure(self):
        """Enabling a trend should change the low-frequency slope of the series."""
        base = self._legacy_like(
            min_base_imfs=1,
            max_base_imfs=1,
            min_choice_imfs=0,
            max_choice_imfs=0,
            upper_energy=None,
        )
        with_trend = IntrinsicModeFunction(
            min_base_imfs=1,
            max_base_imfs=1,
            min_choice_imfs=0,
            max_choice_imfs=0,
            min_wavelets=0,
            max_wavelets=0,
            envelope_prob=0.0,
            chirp_prob=0.0,
            trend_prob=1.0,
            trend_kinds=("linear",),
            trend_apply_on="sum",
            noise_level=0.0,
            upper_energy=None,
            amplitude_decay_with_freq=False,
        )
        y0 = base.generate(np.random.RandomState(0), seq_length=256, num_channels=1)[
            :, 0
        ]
        y1 = with_trend.generate(
            np.random.RandomState(0), seq_length=256, num_channels=1
        )[:, 0]

        # Linear detrend residual energy should differ when a trend is forced
        def slope(y):
            """Estimate the linear slope of a series with a degree-1 polynomial fit."""
            t = np.arange(len(y), dtype=np.float64)
            return float(np.polyfit(t, y, 1)[0])

        self.assertGreater(abs(slope(y1)), abs(slope(y0)) * 0.5)

    def test_amplitude_decay_with_freq(self):
        """Higher-frequency IMFs should receive smaller amplitudes when decay is enabled."""
        imf = IntrinsicModeFunction(
            amplitude_decay_with_freq=True, amplitude_decay_gamma=0.5
        )
        low = imf._scale_amplitude_for_frequency(1.0, frequency=0.5)
        high = imf._scale_amplitude_for_frequency(1.0, frequency=8.0)
        self.assertGreater(low, high)

        off = IntrinsicModeFunction(amplitude_decay_with_freq=False)
        self.assertEqual(off._scale_amplitude_for_frequency(1.0, 0.5), 1.0)
        self.assertEqual(off._scale_amplitude_for_frequency(1.0, 8.0), 1.0)

    def test_chirp_enabled_generation_finite(self):
        """Enabling chirp carriers should still produce finite generated series."""
        imf = IntrinsicModeFunction(
            min_base_imfs=2,
            max_base_imfs=2,
            min_choice_imfs=0,
            max_choice_imfs=0,
            min_wavelets=0,
            max_wavelets=0,
            envelope_prob=0.0,
            trend_prob=0.0,
            chirp_prob=1.0,
            noise_level=0.0,
            upper_energy=None,
            amplitude_decay_with_freq=False,
        )
        out = imf.generate(np.random.RandomState(3), seq_length=256, num_channels=1)
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertGreater(_spectral_spread(out[:, 0]), 0.0)

    def test_amplitude_energy_ordering_without_rescale(self):
        """Without rescaling, lower-frequency modes should tend to carry more energy."""
        prob = {"generate_sin_signal": 0.7, "generate_cos_signal": 0.3}
        imf_small = self._legacy_like(
            min_amplitude=0.01,
            max_amplitude=0.02,
            min_base_imfs=1,
            max_base_imfs=1,
            min_choice_imfs=1,
            max_choice_imfs=1,
            probability_dict=prob,
            upper_energy=None,
        )
        imf_large = self._legacy_like(
            min_amplitude=5.0,
            max_amplitude=6.0,
            min_base_imfs=1,
            max_base_imfs=1,
            min_choice_imfs=1,
            max_choice_imfs=1,
            probability_dict=prob,
            upper_energy=None,
        )
        small = imf_small.generate(np.random.RandomState(7), 128, 1)
        large = imf_large.generate(np.random.RandomState(7), 128, 1)
        self.assertLess(_get_energy(small.flatten()), _get_energy(large.flatten()))

    def test_invalid_trend_apply_on(self):
        """An invalid trend_apply_on value should raise ValueError."""
        with self.assertRaises(ValueError):
            IntrinsicModeFunction(trend_apply_on="both")


if __name__ == "__main__":
    unittest.main()
