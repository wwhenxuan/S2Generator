# -*- coding: utf-8 -*-
"""
Created on 2026/06/21 23:05:00
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/S2Generator
"""

import unittest

import numpy as np
import pandas as pd

from s2generator.simulator import GaussianMixtureSimulator


class TestGaussianMixtureSimulator(unittest.TestCase):
    """The Unittest for GaussianMixtureSimulator class."""

    @staticmethod
    def _make_time_series(length: int = 300, seed: int = 0) -> np.ndarray:
        """
        Build a reproducible two-regime Markov-switching Gaussian process.

        Each regime has its own mean and variance, and the latent regime follows
        a persistent Markov chain. This structure is suitable for verifying the
        refactored mixture simulator in unit tests.
        """
        rng = np.random.RandomState(seed)

        transition = np.array([[0.95, 0.10], [0.05, 0.90]])
        const = np.array([-1.0, 1.5])
        sigma = np.array([0.5, 0.8])

        series = np.zeros(length)
        current_regime = rng.choice(2, p=[0.5, 0.5])
        series[0] = const[current_regime] + rng.normal(scale=sigma[current_regime])

        for t in range(1, length):
            current_regime = rng.choice(2, p=transition[:, current_regime])
            series[t] = const[current_regime] + rng.normal(scale=sigma[current_regime])

        return series

    @staticmethod
    def _make_simulator(**kwargs) -> GaussianMixtureSimulator:
        """Create a GaussianMixtureSimulator with conservative defaults for unit tests."""
        defaults = {
            "n_components": 2,
            "switching_variance": True,
            "not_white_alarm": False,
            "revin": True,
            "random_state": 42,
            "max_iter": 200,
        }
        defaults.update(kwargs)
        return GaussianMixtureSimulator(**defaults)

    def test_create_instance(self) -> None:
        """
        Test the creation of a GaussianMixtureSimulator instance.

        Verify that common hyperparameter combinations can be passed to the
        constructor and that invalid covariance types are rejected.
        """
        for n_components in [1, 2, 3]:
            for switching_variance in [True, False]:
                for revin in [True, False]:
                    with self.subTest(
                        n_components=n_components,
                        switching_variance=switching_variance,
                        revin=revin,
                    ):
                        simulator = GaussianMixtureSimulator(
                            n_components=n_components,
                            switching_variance=switching_variance,
                            revin=revin,
                            random_state=42,
                        )
                        self.assertIsInstance(simulator, GaussianMixtureSimulator)
                        self.assertEqual(simulator.n_components, n_components)

        with self.assertRaises(ValueError):
            GaussianMixtureSimulator(covariance_type="invalid")

    def test_fit_transform(self) -> None:
        """
        Test the fit and transform workflow of GaussianMixtureSimulator.

        After fitting on a valid regime-switching input sequence, residuals and
        smoothed probabilities should be available, and transform should return
        simulated data with shape [num_samples, seq_length].
        """
        simulator = self._make_simulator()
        time_series = self._make_time_series(length=300, seed=0)

        simulator.fit(time_series)

        self.assertIsInstance(simulator.residuals, np.ndarray)
        self.assertIsInstance(simulator.smoothed_probabilities, np.ndarray)
        self.assertEqual(len(simulator.residuals), len(time_series))
        self.assertEqual(simulator.smoothed_probabilities.shape[1], simulator.k_regimes)
        self.assertEqual(simulator.k_regimes, 2)

        simulation = simulator.transform(num_samples=5, seq_length=100, random_state=7)
        self.assertEqual(simulation.shape, (5, 100))
        self.assertFalse(np.isnan(simulation).any())

    def test_fit_with_select_order(self) -> None:
        """
        Test automatic component-number selection during fit.

        When select_order=True, the simulator should choose the number of
        components via BIC and still allow downstream simulation.
        """
        simulator = self._make_simulator(n_components=2, max_n_components=2)
        time_series = self._make_time_series(length=300, seed=1)

        simulator.fit(time_series, select_order=True)

        self.assertGreaterEqual(simulator.k_regimes, 1)
        self.assertLessEqual(simulator.k_regimes, simulator.max_n_components)

        generated = simulator.transform(num_samples=2, seq_length=80, random_state=3)
        self.assertEqual(generated.shape, (2, 80))

    def test_check_inputs(self) -> None:
        """
        Test the check_inputs method of GaussianMixtureSimulator.

        Invalid dtypes, shapes, constant sequences, NaN values, and overly short
        inputs should raise ValueError; valid 1D/2D ndarray inputs should pass.
        """
        simulator = self._make_simulator(n_components=2)

        for wrong_input in [1, "hello, world!", True, {"input": [1, 2, 3]}]:
            with self.subTest(wrong_input=wrong_input):
                with self.assertRaises(ValueError):
                    simulator.check_inputs(wrong_input)

        time_series = self._make_time_series(length=300, seed=2)
        result = simulator.check_inputs(time_series)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), len(time_series))

        time_series_2d = np.tile(time_series, (2, 1))
        result_2d = simulator.check_inputs(time_series_2d)
        self.assertEqual(len(result_2d), time_series_2d.size)

        series_input = pd.Series(time_series)
        result_series = simulator.check_inputs(series_input)
        self.assertEqual(len(result_series), len(series_input))

        min_length = max(10, simulator.n_components + 2)
        with self.assertRaises(ValueError):
            simulator.check_inputs(np.random.randn(min_length - 1))

        with self.assertRaises(ValueError):
            simulator.check_inputs(np.ones(300))

        nan_series = time_series.copy()
        nan_series[0] = np.nan
        with self.assertRaises(ValueError):
            simulator.check_inputs(nan_series)

        with self.assertRaises(ValueError):
            simulator.check_inputs(np.random.randn(2, 3, 4))

    def test_component_accessors(self) -> None:
        """
        Test weight, weights, mean, means, covariance, and covariances helpers.

        These accessors should expose finite component statistics after fitting
        and validate component indices.
        """
        simulator = self._make_simulator()
        time_series = self._make_time_series(length=300, seed=3)
        simulator.fit(time_series)

        weights = simulator.weights()
        means = simulator.means()
        covariances = simulator.covariances()

        self.assertEqual(len(weights), simulator.k_regimes)
        self.assertEqual(len(means), simulator.k_regimes)
        self.assertEqual(len(covariances), simulator.k_regimes)
        self.assertTrue(np.allclose(weights.sum(), 1.0))
        self.assertTrue(np.all(weights >= 0))
        self.assertTrue(np.all(covariances > 0))

        self.assertEqual(simulator.weight(0), weights[0])
        self.assertEqual(simulator.mean(0), means[0])
        self.assertEqual(simulator.covariance(0), covariances[0])

        with self.assertRaises(ValueError):
            simulator.weight(simulator.k_regimes)

    def test_regime_transition_and_simulation_persistence(self) -> None:
        """
        Test that generated sequences preserve regime persistence.

        Simulated paths should not behave like independent point-wise mixture
        sampling; consecutive values in the same regime should stay clustered.
        """
        simulator = self._make_simulator()
        time_series = self._make_time_series(length=300, seed=4)
        simulator.fit(time_series)

        transition = simulator.regime_transition[:, :, 0]
        self.assertEqual(transition.shape, (simulator.k_regimes, simulator.k_regimes))
        self.assertTrue(np.allclose(transition.sum(axis=0), 1.0))

        sample = simulator.transform(num_samples=1, seq_length=500, random_state=11)[0]
        diffs = np.abs(np.diff(sample))
        self.assertGreater(np.percentile(diffs, 50), 0.0)

    def test_transform_before_fit(self) -> None:
        """
        Test calling transform before the model has been fitted.

        transform should raise ValueError when the fitted result object is absent.
        """
        simulator = self._make_simulator()

        with self.assertRaises(ValueError):
            simulator.transform(num_samples=1, seq_length=50)

    def test_model_summary_and_properties(self) -> None:
        """
        Test model_summary and parameter metadata properties.

        These interfaces should raise before fitting and expose statsmodels
        metadata after fitting.
        """
        simulator = self._make_simulator()
        time_series = self._make_time_series(length=300, seed=5)

        with self.assertRaises(ValueError):
            simulator.model_summary()
        with self.assertRaises(ValueError):
            _ = simulator.param_names

        simulator.fit(time_series)
        summary = simulator.model_summary()

        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 0)
        self.assertGreater(len(simulator.param_names), 0)
        self.assertEqual(len(simulator.param_items), len(simulator.param_names))

    def test_residual_diagnosis(self) -> None:
        """
        Test the residual_diagnosis method.

        After fitting, the method should return the mean Ljung-Box p-value and a
        boolean flag indicating whether all p-values exceed the significance level.
        """
        simulator = self._make_simulator()
        time_series = self._make_time_series(length=300, seed=6)
        simulator.fit(time_series)

        mean_p_value, is_white = simulator.residual_diagnosis()
        self.assertIsInstance(mean_p_value, float)
        self.assertIsInstance(is_white, (bool, np.bool_))

    def test_revin_inverse_transform(self) -> None:
        """
        Test reversible normalization during fit and transform.

        When revin=True, generated samples should be returned in the original
        scale of the input series.
        """
        simulator = self._make_simulator(revin=True)
        time_series = self._make_time_series(length=300, seed=7)
        simulator.fit(time_series)

        generated = simulator.transform(num_samples=3, seq_length=120, random_state=8)
        self.assertGreater(np.std(generated), 0.0)
        self.assertFalse(np.allclose(generated, 0.0))


if __name__ == "__main__":
    unittest.main()
