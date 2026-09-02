# -*- coding: utf-8 -*-
"""
Created on 2026/06/21
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/S2Generator
"""

import unittest

import numpy as np

from s2generator.utils.data import (
    generate_arma_samples,
    generate_chirp_signal,
    generate_damped_oscillation,
    generate_electrocardiogram,
    generate_electroencephalogram,
    generate_exponential_signal,
    generate_impulse_signal,
    generate_logarithmic_signal,
    generate_nonstationary_sine,
    generate_ramp_signal,
    generate_sawtooth_wave,
    generate_sine_with_local_frequency_changes,
    generate_square_wave,
    generate_step_signal,
    generate_stock_price,
    generate_triangle_wave,
    generate_variable_frequency_sine,
)


class TestUtilsData(unittest.TestCase):
    """Unit tests for synthetic time-series generation helpers in ``utils.data``."""

    SEQ_LENGTH = 128

    def _assert_valid_series(self, series: np.ndarray) -> None:
        """Helper to verify a generated one-dimensional series is usable."""
        self.assertIsInstance(series, np.ndarray)
        self.assertEqual(series.shape, (self.SEQ_LENGTH,))
        self.assertFalse(np.isnan(series).any())
        self.assertFalse(np.isinf(series).any())

    def test_generate_arma_samples(self) -> None:
        """
        Test ARMA(1,1) sample generation.

        Verify that the function returns a series with the requested length and
        that ``return_params=True`` also returns the generation parameters.
        """
        np.random.seed(0)

        series = generate_arma_samples(seq_length=self.SEQ_LENGTH)
        self._assert_valid_series(series)

        series, params = generate_arma_samples(
            seq_length=self.SEQ_LENGTH, return_params=True
        )
        self._assert_valid_series(series)
        self.assertEqual(len(params), 3)

        with self.assertRaises(ValueError):
            generate_arma_samples(seq_length=0)

    def test_generate_nonstationary_sine(self) -> None:
        """
        Test non-stationary sine generation with linear trend.

        Verify output shape and optional parameter return values.
        """
        np.random.seed(1)

        series = generate_nonstationary_sine(seq_length=self.SEQ_LENGTH)
        self._assert_valid_series(series)

        series, freq, sample_rate = generate_nonstationary_sine(
            seq_length=self.SEQ_LENGTH, return_params=True
        )
        self._assert_valid_series(series)
        self.assertGreater(freq, 0)
        self.assertGreater(sample_rate, 0)

    def test_generate_variable_frequency_sine(self) -> None:
        """
        Test variable-frequency sine generation.

        Verify that the generated sequence has the expected length and that
        start/end frequency metadata can be returned.
        """
        np.random.seed(2)

        series = generate_variable_frequency_sine(seq_length=self.SEQ_LENGTH)
        self._assert_valid_series(series)

        series, start_freq, end_freq, sample_rate = generate_variable_frequency_sine(
            seq_length=self.SEQ_LENGTH, return_params=True
        )
        self._assert_valid_series(series)
        self.assertGreaterEqual(start_freq, 0)
        self.assertGreaterEqual(end_freq, 0)
        self.assertGreater(sample_rate, 0)

    def test_generate_sine_with_local_frequency_changes(self) -> None:
        """
        Test sine generation with local frequency changes.

        Verify basic generation and optional instantaneous-frequency output.
        """
        np.random.seed(3)

        series = generate_sine_with_local_frequency_changes(
            seq_length=self.SEQ_LENGTH,
            change_positions=[0.3, 0.7],
            directions=["increase", "decrease"],
        )
        self._assert_valid_series(series)

        series, inst_freq, base_freq, sample_rate = (
            generate_sine_with_local_frequency_changes(
                seq_length=self.SEQ_LENGTH,
                change_positions=[0.5],
                return_params=True,
            )
        )
        self._assert_valid_series(series)
        self.assertEqual(len(inst_freq), self.SEQ_LENGTH)
        self.assertGreater(base_freq, 0)
        self.assertGreater(sample_rate, 0)

    def test_generate_triangle_wave(self) -> None:
        """
        Test triangle wave generation.

        Verify output length and bounded amplitude behavior.
        """
        np.random.seed(4)

        series = generate_triangle_wave(seq_length=self.SEQ_LENGTH, noise_std=0.0)
        self._assert_valid_series(series)
        self.assertLessEqual(np.max(np.abs(series)), 1.5 + 1e-6)

    def test_generate_square_wave(self) -> None:
        """
        Test square wave generation.

        Verify output length and rejection of invalid duty cycles.
        """
        np.random.seed(5)

        series = generate_square_wave(seq_length=self.SEQ_LENGTH, noise_std=0.0)
        self._assert_valid_series(series)

        with self.assertRaises(ValueError):
            generate_square_wave(seq_length=self.SEQ_LENGTH, duty_cycle=1.5)

    def test_generate_sawtooth_wave(self) -> None:
        """
        Test sawtooth wave generation.

        Verify output length for both rising and falling sawtooth settings.
        """
        np.random.seed(6)

        rising = generate_sawtooth_wave(
            seq_length=self.SEQ_LENGTH, width=1.0, noise_std=0.0
        )
        falling = generate_sawtooth_wave(
            seq_length=self.SEQ_LENGTH, width=0.0, noise_std=0.0
        )
        self._assert_valid_series(rising)
        self._assert_valid_series(falling)

    def test_generate_damped_oscillation(self) -> None:
        """
        Test damped oscillation generation.

        Verify normal and flipped outputs both have the requested length.
        """
        np.random.seed(7)

        series = generate_damped_oscillation(seq_length=self.SEQ_LENGTH)
        flipped = generate_damped_oscillation(seq_length=self.SEQ_LENGTH, flip=True)
        self._assert_valid_series(series)
        self._assert_valid_series(flipped)

    def test_generate_chirp_signal(self) -> None:
        """
        Test chirp signal generation.

        Verify output length and optional parameter return values.
        """
        np.random.seed(8)

        series = generate_chirp_signal(seq_length=self.SEQ_LENGTH)
        self._assert_valid_series(series)

        series, start_freq, end_freq, sample_rate = generate_chirp_signal(
            seq_length=self.SEQ_LENGTH, return_params=True
        )
        self._assert_valid_series(series)
        self.assertGreaterEqual(start_freq, 0)
        self.assertGreaterEqual(end_freq, 0)
        self.assertGreater(sample_rate, 0)

    def test_generate_impulse_signal(self) -> None:
        """
        Test impulse signal generation.

        Verify that impulses increase the signal amplitude above the background.
        """
        np.random.seed(9)

        series = generate_impulse_signal(
            seq_length=self.SEQ_LENGTH,
            impulse_position=[0.2, 0.8],
            noise_std=0.0,
        )
        self._assert_valid_series(series)
        self.assertGreater(np.max(series), 0.0)

    def test_generate_step_signal(self) -> None:
        """
        Test step signal generation.

        Verify multi-step generation and optional parameter return values.
        """
        np.random.seed(10)

        series = generate_step_signal(
            seq_length=self.SEQ_LENGTH,
            step_position=[0.3, 0.7],
            step_height=[1.0, -0.5],
            noise_std=0.0,
        )
        self._assert_valid_series(series)

        series, positions, heights = generate_step_signal(
            seq_length=self.SEQ_LENGTH,
            step_position=0.5,
            return_params=True,
        )
        self._assert_valid_series(series)
        self.assertEqual(len(positions), len(heights))

    def test_generate_ramp_signal(self) -> None:
        """
        Test ramp signal generation.

        Verify single and multiple ramp segments can be generated successfully.
        """
        np.random.seed(11)

        series = generate_ramp_signal(
            seq_length=self.SEQ_LENGTH,
            start_position=[0.1, 0.5],
            end_position=[0.3, 0.8],
            ramp_height=[1.0, -0.5],
            noise_std=0.0,
        )
        self._assert_valid_series(series)

    def test_generate_exponential_signal(self) -> None:
        """
        Test exponential growth and decay signal generation.

        Verify both growth/decay modes and optional parameter return values.
        """
        np.random.seed(12)

        growth = generate_exponential_signal(
            seq_length=self.SEQ_LENGTH, decay=False, normalize=True
        )
        decay = generate_exponential_signal(
            seq_length=self.SEQ_LENGTH, decay=True, normalize=True
        )
        self._assert_valid_series(growth)
        self._assert_valid_series(decay)

        series, growth_rate, sample_rate, amp, offset, is_decay = (
            generate_exponential_signal(seq_length=self.SEQ_LENGTH, return_params=True)
        )
        self._assert_valid_series(series)
        self.assertIsInstance(is_decay, bool)
        self.assertGreater(sample_rate, 0)

    def test_generate_logarithmic_signal(self) -> None:
        """
        Test logarithmic signal generation.

        Verify output length and rejection of invalid logarithm bases.
        """
        np.random.seed(13)

        series = generate_logarithmic_signal(seq_length=self.SEQ_LENGTH)
        self._assert_valid_series(series)

        with self.assertRaises(ValueError):
            generate_logarithmic_signal(seq_length=self.SEQ_LENGTH, log_base=1.0)

    def test_generate_stock_price(self) -> None:
        """
        Test simulated stock price generation.

        Verify that all generated prices remain positive and parameters can be returned.
        """
        np.random.seed(14)

        prices = generate_stock_price(seq_length=self.SEQ_LENGTH)
        self._assert_valid_series(prices)
        self.assertTrue(np.all(prices > 0))

        prices, initial_price, drift, volatility, jump_probability = (
            generate_stock_price(seq_length=self.SEQ_LENGTH, return_params=True)
        )
        self._assert_valid_series(prices)
        self.assertGreater(initial_price, 0)
        self.assertGreaterEqual(jump_probability, 0.0)
        self.assertLessEqual(jump_probability, 1.0)

    def test_generate_electrocardiogram(self) -> None:
        """
        Test simulated ECG signal generation.

        Verify output length and optional heart-rate metadata return.
        """
        np.random.seed(15)

        ecg = generate_electrocardiogram(seq_length=self.SEQ_LENGTH)
        self._assert_valid_series(ecg)

        ecg, heart_rate, sample_rate, amp = generate_electrocardiogram(
            seq_length=self.SEQ_LENGTH, return_params=True
        )
        self._assert_valid_series(ecg)
        self.assertGreater(heart_rate, 0)
        self.assertGreater(sample_rate, 0)
        self.assertGreaterEqual(amp, 0)

    def test_generate_electroencephalogram(self) -> None:
        """
        Test simulated EEG signal generation.

        Verify output length and optional rhythm-weight metadata return.
        """
        np.random.seed(16)

        eeg = generate_electroencephalogram(seq_length=self.SEQ_LENGTH)
        self._assert_valid_series(eeg)

        (
            eeg,
            sample_rate,
            amp,
            alpha_weight,
            beta_weight,
            theta_weight,
            delta_weight,
        ) = generate_electroencephalogram(
            seq_length=self.SEQ_LENGTH, return_params=True
        )
        self._assert_valid_series(eeg)
        self.assertGreater(sample_rate, 0)
        self.assertGreaterEqual(alpha_weight, 0)
        self.assertGreaterEqual(beta_weight, 0)
        self.assertGreaterEqual(theta_weight, 0)
        self.assertGreaterEqual(delta_weight, 0)


class TestUtilsDataValidation(unittest.TestCase):
    """Validation tests shared by multiple data-generation helpers."""

    def test_invalid_seq_length(self) -> None:
        """
        Test rejection of non-positive sequence lengths.

        All generators should raise ``ValueError`` when ``seq_length <= 0``.
        """
        generators = [
            generate_arma_samples,
            generate_nonstationary_sine,
            generate_variable_frequency_sine,
            generate_sine_with_local_frequency_changes,
            generate_triangle_wave,
            generate_square_wave,
            generate_sawtooth_wave,
            generate_damped_oscillation,
            generate_chirp_signal,
            generate_impulse_signal,
            generate_step_signal,
            generate_ramp_signal,
            generate_exponential_signal,
            generate_logarithmic_signal,
            generate_stock_price,
            generate_electrocardiogram,
            generate_electroencephalogram,
        ]

        for generator in generators:
            with self.subTest(generator=generator.__name__):
                with self.assertRaises(ValueError):
                    generator(seq_length=0)

    def test_sine_local_frequency_invalid_direction(self) -> None:
        """
        Test validation of local frequency change directions.

        Invalid direction strings should be rejected by the generator.
        """
        with self.assertRaises(ValueError):
            generate_sine_with_local_frequency_changes(
                seq_length=128,
                change_positions=[0.5],
                directions=["invalid"],
            )


if __name__ == "__main__":
    unittest.main()
