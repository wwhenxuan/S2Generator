# -*- coding: utf-8 -*-
"""
Created on 2026/06/21 12:18:13
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/S2Generator
"""

import unittest

import numpy as np
import pandas as pd

from s2generator.simulator import MarkovSwitchingSimulator


class TestMarkovSwitchingSimulator(unittest.TestCase):
    """The Unittest for MarkovSwitchingSimulator class."""

    @staticmethod
    def _make_time_series(length: int = 300, seed: int = 0) -> np.ndarray:
        """
        Build a reproducible two-regime AR(1) process for fitting tests.

        Regime 0 uses low volatility and weak persistence, while regime 1 uses
        higher volatility and stronger persistence. This structure is suitable
        for verifying MSAR fitting and simulation in unit tests.
        """
        rng = np.random.RandomState(seed)

        # transition[j, i] = P(S_t = j | S_{t-1} = i)
        transition = np.array([[0.95, 0.10], [0.05, 0.90]])
        const = np.array([0.10, -0.10])
        ar1 = np.array([0.35, 0.85])
        sigma = np.array([0.35, 1.10])

        series = np.zeros(length)
        current_regime = rng.choice(2, p=[0.5, 0.5])
        series[0] = const[current_regime] + rng.normal(scale=sigma[current_regime])

        for t in range(1, length):
            current_regime = rng.choice(2, p=transition[:, current_regime])
            series[t] = (
                const[current_regime]
                + ar1[current_regime] * (series[t - 1] - const[current_regime])
                + rng.normal(scale=sigma[current_regime])
            )

        return series

    @staticmethod
    def _make_simulator(**kwargs) -> MarkovSwitchingSimulator:
        """Create a MarkovSwitchingSimulator with conservative defaults for unit tests."""
        defaults = {
            "max_k_regimes": 2,
            "max_order": 2,
            "switching_variance": True,
            "not_white_alarm": False,
            "revin": True,
            "random_state": 42,
            "maxiter": 200,
        }
        defaults.update(kwargs)
        return MarkovSwitchingSimulator(**defaults)

    def test_create_instance(self) -> None:
        """
        Test the creation of a MarkovSwitchingSimulator instance.

        Verify that different combinations of hyperparameters can be passed to
        the constructor and that a valid simulator object is returned.
        """

        # Traverse several common hyperparameter combinations
        for max_k_regimes in [1, 2, 3]:
            for max_order in [1, 2, 3]:
                for switching_variance in [True, False]:
                    for revin in [True, False]:
                        for random_state in [None, 0, 42]:
                            with self.subTest(
                                max_k_regimes=max_k_regimes,
                                max_order=max_order,
                                switching_variance=switching_variance,
                                revin=revin,
                                random_state=random_state,
                            ):
                                simulator = MarkovSwitchingSimulator(
                                    max_k_regimes=max_k_regimes,
                                    max_order=max_order,
                                    switching_variance=switching_variance,
                                    revin=revin,
                                    random_state=random_state,
                                )
                                self.assertIsInstance(
                                    simulator, MarkovSwitchingSimulator
                                )

    def test_fit_transform(self) -> None:
        """
        Test the fit and transform workflow of MarkovSwitchingSimulator.

        After fitting on a valid regime-switching input sequence, residuals and
        smoothed probabilities should be available, and transform should return
        simulated data with shape [num_samples, seq_len].
        """

        simulator = self._make_simulator()
        time_series = self._make_time_series(length=300, seed=0)

        # Fit the MSAR model on the input sequence
        simulator.fit(time_series)

        # Residuals and smoothed regime probabilities should be stored after fitting
        self.assertIsInstance(simulator.residuals, np.ndarray)
        self.assertIsInstance(simulator.smoothed_probabilities, np.ndarray)
        self.assertGreater(len(simulator.residuals), 0)
        # statsmodels returns smoothed probabilities with shape [nobs, k_regimes]
        self.assertEqual(simulator.smoothed_probabilities.shape[1], simulator.k_regimes)

        # Selected model orders should be assigned after fitting
        self.assertIsNotNone(simulator.k_regimes)
        self.assertIsNotNone(simulator.order)
        self.assertGreaterEqual(simulator.k_regimes, 1)
        self.assertGreaterEqual(simulator.order, 1)

        # Generate new sequences from the fitted model
        simulation = simulator.transform(num_samples=5, seq_len=100, random_state=7)

        # Output shape should be [num_samples, seq_len]
        self.assertEqual(simulation.shape, (5, 100))
        self.assertFalse(np.isnan(simulation).any())

    def test_fit_with_select_order(self) -> None:
        """
        Test automatic (k_regimes, order) selection during fit.

        When select_order=True, the simulator should choose the model order via
        BIC and still allow downstream simulation.
        """

        simulator = self._make_simulator(max_k_regimes=2, max_order=2)
        time_series = self._make_time_series(length=300, seed=1)

        # Enable automatic order selection
        simulator.fit(time_series, select_order=True)

        # Selected orders should lie within the configured search range
        self.assertGreaterEqual(simulator.k_regimes, 1)
        self.assertLessEqual(simulator.k_regimes, simulator.max_k_regimes)
        self.assertGreaterEqual(simulator.order, 1)
        self.assertLessEqual(simulator.order, simulator.max_order)

        generated = simulator.transform(num_samples=2, seq_len=80, random_state=3)
        self.assertEqual(generated.shape, (2, 80))

    def test_check_inputs(self) -> None:
        """
        Test the check_inputs method of MarkovSwitchingSimulator.

        Invalid dtypes, shapes, constant sequences, NaN values, and overly short
        inputs should raise ValueError; valid 1D/2D ndarray inputs should pass.
        """

        simulator = self._make_simulator(max_k_regimes=2, max_order=2)

        # Invalid input types should be rejected
        for wrong_input in [1, "hello, world!", True, {"input": [1, 2, 3]}]:
            with self.subTest(wrong_input=wrong_input):
                with self.assertRaises(ValueError):
                    simulator.check_inputs(wrong_input)

        # Valid 1D ndarray input should be returned as a numpy array
        time_series = self._make_time_series(length=300, seed=2)
        result = simulator.check_inputs(time_series)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), len(time_series))

        # Valid 2D ndarray input should be flattened
        time_series_2d = np.tile(time_series, (2, 1))
        result_2d = simulator.check_inputs(time_series_2d)
        self.assertEqual(len(result_2d), time_series_2d.size)

        # pandas Series input should also be accepted
        series_input = pd.Series(time_series)
        result_series = simulator.check_inputs(series_input)
        self.assertEqual(len(result_series), len(series_input))

        min_length = max(10, simulator.max_order + simulator.max_k_regimes + 2)
        with self.assertRaises(ValueError):
            simulator.check_inputs(np.random.randn(min_length - 1))

        # Constant input with zero variance should be rejected
        with self.assertRaises(ValueError):
            simulator.check_inputs(np.ones(300))

        # Input containing NaN values should be rejected
        nan_series = time_series.copy()
        nan_series[0] = np.nan
        with self.assertRaises(ValueError):
            simulator.check_inputs(nan_series)

        # Input with more than two dimensions should be rejected
        with self.assertRaises(ValueError):
            simulator.check_inputs(np.random.randn(2, 3, 4))

    def test_model_properties(self) -> None:
        """
        Test param_names, params, and param_items properties.

        These properties should raise ValueError before fitting and expose
        consistent parameter metadata from statsmodels after fitting.
        """

        simulator = self._make_simulator()
        time_series = self._make_time_series(length=300, seed=0)

        # Parameter metadata should not be available before fitting
        with self.assertRaises(ValueError):
            _ = simulator.param_names
        with self.assertRaises(ValueError):
            _ = simulator.params
        with self.assertRaises(ValueError):
            _ = simulator.param_items

        simulator.fit(time_series)

        # After fitting, parameter metadata should be available
        self.assertIsInstance(simulator.param_names, list)
        self.assertGreater(len(simulator.param_names), 0)
        self.assertGreater(len(simulator.params), 0)
        self.assertEqual(len(simulator.param_items), len(simulator.param_names))

        for name, value in simulator.param_items:
            self.assertIn(name, simulator.param_names)
            self.assertTrue(np.isfinite(value))

    def test_model_summary(self) -> None:
        """
        Test the model_summary method.

        After fitting, model_summary should return a non-empty textual description
        of the fitted MSAR model.
        """

        simulator = self._make_simulator()
        time_series = self._make_time_series(length=300, seed=4)

        # Summary should not be available before fitting
        with self.assertRaises(ValueError):
            simulator.model_summary()

        simulator.fit(time_series)
        summary = simulator.model_summary()

        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 0)

    def test_transform_before_fit(self) -> None:
        """
        Test calling transform before the model has been fitted.

        transform should raise ValueError when the fitted result object is absent.
        """

        simulator = self._make_simulator()

        with self.assertRaises(ValueError):
            simulator.transform(num_samples=1, seq_len=50)

    def test_select_msar_order(self) -> None:
        """
        Test the select_msar_order method.

        For a valid input series, the method should return integer regime count
        and AR order within the configured search bounds.
        """

        simulator = self._make_simulator(max_k_regimes=2, max_order=2)
        time_series = self._make_time_series(length=300, seed=5)

        k_regimes, order = simulator.select_msar_order(endog=time_series)

        self.assertIsInstance(k_regimes, int)
        self.assertIsInstance(order, int)
        self.assertGreaterEqual(k_regimes, 1)
        self.assertLessEqual(k_regimes, simulator.max_k_regimes)
        self.assertGreaterEqual(order, 1)
        self.assertLessEqual(order, simulator.max_order)

    def test_residual_diagnosis(self) -> None:
        """
        Test the residual_diagnosis method.

        After fitting, the method should return the mean Ljung-Box p-value and a
        boolean flag indicating whether all p-values exceed the significance level.
        """

        simulator = self._make_simulator()
        time_series = self._make_time_series(length=300, seed=6)
        simulator.fit(time_series)

        mean_p_value, is_white = simulator.residual_diagnosis(lags=10)

        self.assertIsInstance(mean_p_value, (float, np.floating))
        self.assertIsInstance(is_white, (bool, np.bool_))
        self.assertGreaterEqual(mean_p_value, 0.0)
        self.assertLessEqual(mean_p_value, 1.0)

        # Calling diagnosis before fit should fail
        unfitted = self._make_simulator()
        with self.assertRaises(ValueError):
            unfitted.residual_diagnosis()

    def test_regime_transition(self) -> None:
        """
        Test the regime_transition property.

        After fitting, the transition matrix should have shape
        [k_regimes, k_regimes, 1] and each column should sum to approximately one.
        """

        simulator = self._make_simulator()
        time_series = self._make_time_series(length=300, seed=7)

        with self.assertRaises(ValueError):
            _ = simulator.regime_transition

        simulator.fit(time_series)
        transition = simulator.regime_transition

        self.assertEqual(transition.shape[0], simulator.k_regimes)
        self.assertEqual(transition.shape[1], simulator.k_regimes)
        column_sums = transition[:, :, 0].sum(axis=0)
        np.testing.assert_allclose(column_sums, 1.0, atol=1e-6)

    def test_extract_regime_parameters(self) -> None:
        """
        Test the internal _extract_regime_parameters helper.

        Parsed intercepts, AR coefficients, and variances should have the expected
        shapes for the fitted number of regimes and AR order.
        """

        simulator = self._make_simulator()
        time_series = self._make_time_series(length=300, seed=8)
        simulator.fit(time_series)

        const, ar, variance = simulator._extract_regime_parameters()

        self.assertEqual(const.shape, (simulator.k_regimes,))
        self.assertEqual(ar.shape, (simulator.k_regimes, simulator.order))
        self.assertEqual(variance.shape, (simulator.k_regimes,))
        self.assertTrue(np.all(variance > 0))

    def test_simulate_internal(self) -> None:
        """
        Test the internal _simulate helper.

        A fitted simulator should be able to generate a single sample path with
        the requested length and without NaN values.
        """

        simulator = self._make_simulator()
        time_series = self._make_time_series(length=300, seed=9)
        simulator.fit(time_series)

        simulated = simulator._simulate(seq_len=120, random_state=11)

        self.assertEqual(simulated.shape, (120,))
        self.assertFalse(np.isnan(simulated).any())

    def test_revin_false(self) -> None:
        """
        Test transform behavior when reversible normalization is disabled.

        With revin=False, mean and std should remain None and generated samples
        should not be inverse-transformed.
        """

        simulator = self._make_simulator(revin=False)
        time_series = self._make_time_series(length=300, seed=10)
        simulator.fit(time_series)

        # Normalization statistics should not be recorded
        self.assertIsNone(simulator.mean)
        self.assertIsNone(simulator.std)

        generated = simulator.transform(num_samples=3, seq_len=60, random_state=15)
        self.assertEqual(generated.shape, (3, 60))

    def test_reproducibility(self) -> None:
        """
        Test reproducibility of transform with a fixed random seed.

        Two transform calls with the same random_state should produce identical
        simulated sequences after the same fit operation.
        """

        simulator = self._make_simulator(random_state=42)
        time_series = self._make_time_series(length=300, seed=11)
        simulator.fit(time_series)

        sample_a = simulator.transform(num_samples=2, seq_len=80, random_state=123)
        sample_b = simulator.transform(num_samples=2, seq_len=80, random_state=123)

        self.assertTrue(np.allclose(sample_a, sample_b))

    def test_not_white_alarm(self) -> None:
        """
        Test the not_white_alarm switch during fit.

        When residual whiteness is not satisfied and not_white_alarm=True, fit
        may raise ValueError; disabling the alarm should allow fit to finish.
        """

        time_series = self._make_time_series(length=300, seed=12)

        strict_simulator = MarkovSwitchingSimulator(
            max_k_regimes=2,
            max_order=2,
            switching_variance=True,
            not_white_alarm=True,
            revin=True,
            random_state=42,
        )
        tolerant_simulator = MarkovSwitchingSimulator(
            max_k_regimes=2,
            max_order=2,
            switching_variance=True,
            not_white_alarm=False,
            revin=True,
            random_state=42,
        )

        # High-order or misspecified models may fail the residual whiteness check
        try:
            strict_simulator.fit(time_series)
        except ValueError as exc:
            self.assertIn("white noise", str(exc))
            tolerant_simulator.fit(time_series)
            self.assertIsNotNone(tolerant_simulator.model)
        else:
            tolerant_simulator.fit(time_series)
            self.assertIsNotNone(tolerant_simulator.model)


if __name__ == "__main__":
    unittest.main()
