# -*- coding: utf-8 -*-
"""
Created on 2026/03/02 16:02:37
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/S2Generator
"""
import unittest

import numpy as np

from s2generator.augmentation import frequency_perturbation
from s2generator.augmentation.frequency_perturbation import sample_random_perturbation


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
            series=series, min_alpha=min_alpha, max_alpha=max_alpha, r=r, rng=self.rng
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


if __name__ == "__main__":
    unittest.main()
