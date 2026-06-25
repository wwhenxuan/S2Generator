# -*- coding: utf-8 -*-
"""
Created on 2026/06/21 23:30:00
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/S2Generator
"""

import unittest

import numpy as np

from s2generator.simulator import (
    KalmanFilterSimulator,
    MultivariateSimulator,
    WienerFilterSimulator,
)


class TestMultivariateSimulator(unittest.TestCase):
    """The Unittest for MultivariateSimulator class."""

    @staticmethod
    def _make_multivariate_series(
        seq_len: int = 200, n_channels: int = 3, seed: int = 0
    ) -> np.ndarray:
        """
        Build a reproducible multivariate series by filtering shared white noise.

        Each channel uses a different linear filter applied to the same excitation,
        so the resulting channels are cross-correlated.
        """
        rng = np.random.RandomState(seed)
        white_noise = rng.normal(size=seq_len + 8)
        filters = [
            np.array([1.0, 0.6, -0.2, 0.1, 0.0, 0.0]),
            np.array([1.0, -0.4, 0.3, 0.2, 0.0, 0.0]),
            np.array([1.0, 0.2, 0.2, -0.3, 0.1, 0.0]),
            np.array([1.0, 0.5, -0.1, 0.0, 0.0, 0.0]),
            np.array([1.0, -0.2, 0.4, -0.1, 0.0, 0.0]),
            np.array([1.0, 0.3, 0.1, 0.1, -0.2, 0.0]),
        ]

        channels = []
        for coeff in filters[:n_channels]:
            filtered = np.convolve(white_noise, coeff, mode="valid")
            channels.append(filtered[:seq_len])

        return np.stack(channels, axis=1)

    def test_single_simulator_template(self) -> None:
        """
        Test fitting and generation when one simulator template is shared by all channels.

        Verify output shape and that shared-excitation channels remain cross-correlated.
        """
        time_series = self._make_multivariate_series(seq_len=200, n_channels=3)
        simulator = MultivariateSimulator(
            WienerFilterSimulator(filter_order=6, random_state=42),
            n_jobs=1,
        )

        simulator.fit(time_series)
        generated = simulator.transform(num_samples=4, seq_len=120, random_state=7)

        self.assertEqual(len(simulator.simulators), 3)
        self.assertEqual(generated.shape, (4, 120, 3))

        corr = np.corrcoef(generated[0].T)
        off_diag = corr[np.triu_indices(3, k=1)]
        self.assertTrue(np.any(np.abs(off_diag) > 0.05))

    def test_simulator_list_with_default_overflow(self) -> None:
        """
        Test channel-specific simulator assignment with Wiener fallback for extra channels.

        When the list is shorter than the number of channels, remaining channels should
        still be fitted and generated successfully.
        """
        time_series = self._make_multivariate_series(seq_len=200, n_channels=4)
        simulator = MultivariateSimulator(
            [
                WienerFilterSimulator(filter_order=5, random_state=0),
                KalmanFilterSimulator(state_order=5, random_state=1),
            ],
            n_jobs=1,
        )

        simulator.fit(time_series)
        generated = simulator.transform(num_samples=2, seq_len=100, random_state=3)

        self.assertEqual(len(simulator.simulators), 4)
        self.assertIsInstance(simulator.simulators[0], WienerFilterSimulator)
        self.assertIsInstance(simulator.simulators[1], KalmanFilterSimulator)
        self.assertIsInstance(simulator.simulators[2], WienerFilterSimulator)
        self.assertIsInstance(simulator.simulators[3], WienerFilterSimulator)
        self.assertEqual(generated.shape, (2, 100, 4))

    def test_parallel_fit(self) -> None:
        """
        Test parallel per-channel fitting with ``n_jobs=-1``.

        Parallel fitting should produce the same number of fitted channel simulators as
        the input channel count.
        """
        time_series = self._make_multivariate_series(seq_len=180, n_channels=6)
        simulator = MultivariateSimulator(
            WienerFilterSimulator(filter_order=5, random_state=42),
            n_jobs=-1,
        )

        simulator.fit(time_series)
        self.assertEqual(len(simulator.simulators), 6)

    def test_check_inputs(self) -> None:
        """
        Test validation of multivariate input shapes and invalid values.

        Only two-dimensional arrays with at least one channel should be accepted.
        """
        simulator = MultivariateSimulator(WienerFilterSimulator(filter_order=5))

        with self.assertRaises(ValueError):
            simulator.check_inputs(np.random.randn(100))

        with self.assertRaises(ValueError):
            simulator.check_inputs(np.ones((100, 2)))

        valid = simulator.check_inputs(self._make_multivariate_series())
        self.assertEqual(valid.ndim, 2)


if __name__ == "__main__":
    unittest.main()
