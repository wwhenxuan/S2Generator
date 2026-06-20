# -*- coding: utf-8 -*-
"""
Created on 2026/03/02 16:02:37
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/S2Generator
"""

import unittest

import numpy as np

from s2generator.augmentation import (
    amplitude_modulation,
    censor_augmentation,
    empirical_mode_modulation,
    frequency_perturbation,
    wiener_filter,
    add_linear_trend,
    time_series_mixup,
)

from s2generator.augmentation._frequency_perturbation import sample_random_perturbation


class TestDataAugmentation(unittest.TestCase):
    """Testing the data augmentation module for time series data"""

    # Random number generator for testing
    rng = np.random.RandomState(42)

    def test_sample_random_perturbation(self) -> None:
        """Test the function for sampling random perturbations in the frequency domain"""
        K = 10
        min_alpha = 0.1
        max_alpha = 0.5

        random_perturbations = sample_random_perturbation(
            K=K, min_alpha=min_alpha, max_alpha=max_alpha, rng=self.rng
        )

        # Check the length of the output
        self.assertEqual(
            len(random_perturbations),
            K,
            msg="Wrong length of random perturbations in `test_sample_random_perturbation` method",
        )

        # Check the value range of the output
        for alpha in random_perturbations:
            self.assertTrue(
                (alpha >= min_alpha and alpha <= max_alpha)
                or (alpha <= -min_alpha and alpha >= -max_alpha),
                msg="Random perturbation value out of range in `test_sample_random_perturbation` method",
            )

    def test_frequency_perturbation(self) -> None:
        """Test the function for performing frequency domain perturbation on time series data"""
        # Generate a simple time series for testing
        t = np.linspace(0, 1, 100)
        series = np.sin(2 * np.pi * 5 * t) + 0.5 * np.random.normal(size=100)

        min_alpha = 0.1
        max_alpha = 0.5
        r = 0.3

        perturbed_series = frequency_perturbation(
            time_series=series,
            min_alpha=min_alpha,
            max_alpha=max_alpha,
            r=r,
            rng=self.rng,
        )

        # Check that the output has the same length as the input
        self.assertEqual(
            len(perturbed_series),
            len(series),
            msg="Output length does not match input length in `test_frequency_perturbation` method",
        )

        # Check that the output is different from the input (since we added perturbations)
        self.assertFalse(
            np.array_equal(perturbed_series, series),
            msg="Perturbed series is identical to original series in `test_frequency_perturbation` method",
        )

    def test_censor_augmentation(self) -> None:
        """Test the function for performing censoring augmentation on time series data."""
        # Generate a simple time series for testing
        t = np.linspace(0, 1, 100)
        series = np.sin(2 * np.pi * 5 * t) + 0.5 * np.random.normal(size=100)

        upper_quantile = 0.65
        lower_quantile = 0.35
        bernoulli_p = 0.8

        censored_series = censor_augmentation(
            time_series=series.copy(),
            upper_quantile=upper_quantile,
            lower_quantile=lower_quantile,
            bernoulli_p=bernoulli_p,
            rng=self.rng,
        )

        # Check that the output has the same length as the input
        self.assertEqual(
            len(censored_series),
            len(series),
            msg="Output length does not match input length in `test_censor_augmentation` method",
        )

        # Check that the output is different from the input (since we applied censoring)
        self.assertFalse(
            np.array_equal(censored_series, series),
            msg="Censored series is identical to original series in `test_censor_augmentation` method",
        )

    def test_amplitude_modulation(self) -> None:
        """Test the function for performing amplitude modulation on time series data."""
        # Generate a simple time series for testing
        t = np.linspace(0, 1, 100)
        series = np.sin(2 * np.pi * 5 * t) + 0.5 * np.random.normal(size=100)

        amplitude_mean, amplitude_variation = 1.0, 1.0

        modulated_series = amplitude_modulation(
            time_series=series.copy(),
            amplitude_mean=amplitude_mean,
            amplitude_variation=amplitude_variation,
            rng=self.rng,
        )

        # Check that the output has the same length as the input
        self.assertEqual(
            len(modulated_series),
            len(series),
            msg="Output length does not match input length in `test_amplitude_modulation` method",
        )

        # Check that the output is different from the input (since we applied amplitude modulation)
        self.assertFalse(
            np.array_equal(modulated_series, series),
            msg="Modulated series is identical to original series in `test_amplitude_modulation` method",
        )

    def test_time_series_mixup(self) -> None:
        """Test the function for performing time series mixup augmentation."""
        # Generate two simple time series for testing
        t = np.linspace(0, 1, 100)
        series_a = np.sin(2 * np.pi * 5 * t) + 0.5 * np.random.normal(size=100)
        series_b = np.cos(2 * np.pi * 5 * t) + 0.5 * np.random.normal(size=100)

        alpha = 0.7

        # Apply time series mixup augmentation
        mixed_series = time_series_mixup(
            a=series_a.copy(), b=series_b.copy(), alpha=alpha
        )

        # Check that the output has the same length as the input
        self.assertEqual(
            len(mixed_series),
            len(series_a),
            msg="Output length does not match input length in `test_time_series_mixup` method",
        )

        # Check that the output is different from both inputs (since we applied mixup)
        self.assertFalse(
            np.array_equal(mixed_series, series_a),
            msg="Mixed series is identical to first input series in `test_time_series_mixup` method",
        )
        self.assertFalse(
            np.array_equal(mixed_series, series_b),
            msg="Mixed series is identical to second input series in `test_time_series_mixup` method",
        )

    def test_add_linear_trend(self) -> None:
        """Test the function for adding a linear trend to time series data."""
        # Generate a simple time series for testing
        t = np.linspace(0, 1, 100)
        series = np.sin(2 * np.pi * 5 * t) + 0.5 * np.random.normal(size=100)

        # Test the upward trend
        trended_series = add_linear_trend(time_series=series.copy(), direction="upward")
        # Check the outputs size
        self.assertEqual(
            len(trended_series),
            len(series),
            msg="Output length does not match input length in `test_add_linear_trend` method",
        )
        # Check the upward trend
        trend_upward = trended_series - series
        self.assertTrue(
            trend_upward[-1] > trend_upward[0],
            msg="Upward trend is not correctly applied in `test_add_linear_trend` method",
        )

        # Test the downward trend
        trended_series = add_linear_trend(
            time_series=series.copy(), direction="downward"
        )
        # Check the downward trend
        trend_downward = trended_series - series
        self.assertTrue(
            trend_downward[-1] < trend_downward[0],
            msg="Downward trend is not correctly applied in `test_add_linear_trend` method",
        )

        # Check that the output is different from the input (since we applied a linear trend)
        self.assertFalse(
            np.array_equal(trended_series, series),
            msg="Trended series is identical to original series in `test_add_linear_trend` method",
        )

    def test_empirical_mode_modulation(self) -> None:
        """Test the function for performing empirical mode modulation on time series data."""
        # Generate a simple time series for testing
        t = np.linspace(0, 1, 100)
        series = np.sin(2 * np.pi * 5 * t) + 0.5 * np.random.normal(size=100)

        # Apply empirical mode modulation augmentation
        modulated_series = empirical_mode_modulation(
            time_series=series.copy(), rng=self.rng
        )

        # Check that the output has the same length as the input
        self.assertEqual(
            len(modulated_series),
            len(series),
            msg="Output length does not match input length in `test_empirical_mode_modulation` method",
        )

        # Check that the output is different from the input (since we applied empirical mode modulation)
        self.assertFalse(
            np.array_equal(modulated_series, series),
            msg="Modulated series is identical to original series in `test_empirical_mode_modulation` method",
        )

    def test_wiener_filter(self) -> None:
        """Test the function for performing Wiener filtering on time series data."""
        # Generate a simple time series for testing
        t = np.linspace(0, 1, 100)
        series = np.sin(2 * np.pi * 5 * t) + 0.5 * np.random.normal(size=100)

        # Apply Wiener filter augmentation
        filtered_series = wiener_filter(time_series=series.copy())

        # Check that the output has the same length as the input
        self.assertEqual(
            len(filtered_series),
            len(series),
            msg="Output length does not match input length in `test_wiener_filter` method",
        )

        # Check that the output is different from the input (since we applied Wiener filtering)
        self.assertFalse(
            np.array_equal(filtered_series, series),
            msg="Filtered series is identical to original series in `test_wiener_filter` method",
        )


if __name__ == "__main__":
    unittest.main()
