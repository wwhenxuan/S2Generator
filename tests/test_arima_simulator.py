# -*- coding: utf-8 -*-
"""
Created on 2026/06/21
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/S2Generator
"""

import unittest

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from s2generator.simulator import ARIMASimulator


class TestARIMASimulator(unittest.TestCase):
    """The Unittest for ARIMASimulator class."""

    @staticmethod
    def _make_time_series(length: int = 120, seed: int = 0) -> np.ndarray:
        """
        Build a reproducible input sequence for fitting tests.

        A simple AR(1)-like cumulative process is used so that differencing and
        low-order ARIMA fitting remain numerically stable in unit tests.
        """
        rng = np.random.RandomState(seed)
        noise = rng.randn(length)
        series = np.zeros(length)
        for t in range(1, length):
            series[t] = 0.6 * series[t - 1] + noise[t]
        return series

    @staticmethod
    def _make_simulator(**kwargs) -> ARIMASimulator:
        """Create an ARIMASimulator with conservative defaults for unit testing."""
        defaults = {
            "max_p": 2,
            "max_d": 1,
            "max_q": 2,
            "not_white_alarm": False,
            "revin": True,
            "random_state": 42,
        }
        defaults.update(kwargs)
        return ARIMASimulator(**defaults)

    def test_create_instance(self) -> None:
        """
        Test the creation of an ARIMASimulator instance.

        Verify that different combinations of hyperparameters can be passed to
        the constructor and that a valid simulator object is returned.
        """

        # Traverse several common hyperparameter combinations
        for max_p in [1, 2, 3]:
            for max_d in [0, 1, 2]:
                for max_q in [1, 2, 3]:
                    for revin in [True, False]:
                        for random_state in [None, 0, 42]:
                            with self.subTest(
                                max_p=max_p,
                                max_d=max_d,
                                max_q=max_q,
                                revin=revin,
                                random_state=random_state,
                            ):
                                simulator = ARIMASimulator(
                                    max_p=max_p,
                                    max_d=max_d,
                                    max_q=max_q,
                                    revin=revin,
                                    random_state=random_state,
                                )
                                self.assertIsInstance(simulator, ARIMASimulator)

    def test_fit_transform(self) -> None:
        """
        Test the fit and transform workflow of ARIMASimulator.

        After fitting on a valid input sequence, residuals should be available and
        transform should return simulated data with shape [num_samples, seq_length].
        """

        simulator = self._make_simulator()
        time_series = self._make_time_series(length=120, seed=0)

        # Fit the ARIMA model on the input sequence
        simulator.fit(time_series)

        # Residuals should be stored as a pandas Series or ndarray-like object
        self.assertIsNotNone(simulator.residuals)
        self.assertGreater(len(simulator.residuals), 0)

        # Model orders should be assigned after fitting
        self.assertIsNotNone(simulator.p_order)
        self.assertIsNotNone(simulator.q_order)
        self.assertIsNotNone(simulator.d_order)

        # Generate new sequences from the fitted model
        simulation = simulator.transform(num_samples=5, seq_length=100, random_state=7)

        # Output shape should be [num_samples, seq_length]
        self.assertEqual(simulation.shape, (5, 100))
        self.assertFalse(np.isnan(simulation).any())

    def test_fit_with_select_order(self) -> None:
        """
        Test automatic ARMA order selection during fit.

        When select_order=True, the simulator should choose (p, q) via BIC and
        still allow downstream simulation.
        """

        simulator = self._make_simulator(max_p=2, max_q=2, max_d=1)
        time_series = self._make_time_series(length=120, seed=1)

        # Enable automatic order selection
        simulator.fit(time_series, select_order=True)

        # Selected orders should lie within the configured search range
        self.assertGreaterEqual(simulator.p_order, 0)
        self.assertLessEqual(simulator.p_order, simulator.max_p)
        self.assertGreaterEqual(simulator.q_order, 0)
        self.assertLessEqual(simulator.q_order, simulator.max_q)

        generated = simulator.transform(num_samples=2, seq_length=80, random_state=3)
        self.assertEqual(generated.shape, (2, 80))

    def test_check_inputs(self) -> None:
        """
        Test the check_inputs method of ARIMASimulator.

        Invalid dtypes, shapes, constant sequences, NaN values, and overly short
        inputs should raise ValueError; valid 1D/2D ndarray inputs should pass.
        """

        simulator = self._make_simulator()

        # Invalid input types should be rejected
        for wrong_input in [1, "hello, world!", True, {"input": [1, 2, 3]}]:
            with self.subTest(wrong_input=wrong_input):
                with self.assertRaises(ValueError):
                    simulator.check_inputs(wrong_input)

        # Valid 1D ndarray input should be returned as a pandas Series
        time_series = self._make_time_series(length=50, seed=2)
        result = simulator.check_inputs(time_series)
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(time_series))

        # Valid 2D ndarray input should be flattened
        time_series_2d = np.tile(time_series, (2, 1))
        result_2d = simulator.check_inputs(time_series_2d)
        self.assertEqual(len(result_2d), time_series_2d.size)

        # pandas Series input should also be accepted
        series_input = pd.Series(time_series)
        result_series = simulator.check_inputs(series_input)
        self.assertEqual(len(result_series), len(series_input))

        # Input shorter than 10 points should be rejected
        with self.assertRaises(ValueError):
            simulator.check_inputs(np.random.randn(5))

        # Constant input with zero variance should be rejected
        with self.assertRaises(ValueError):
            simulator.check_inputs(np.ones(20))

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

        These properties should be accessible only after a successful fit and
        should expose consistent parameter metadata from statsmodels.
        """

        simulator = self._make_simulator()
        time_series = self._make_time_series(length=120, seed=3)

        # Before fitting, underlying model is None and attribute access should fail
        with self.assertRaises(AttributeError):
            _ = simulator.param_names
        with self.assertRaises(AttributeError):
            _ = simulator.params
        with self.assertRaises(AttributeError):
            _ = simulator.param_items

        simulator.fit(time_series)

        # After fitting, parameter metadata should be available
        self.assertIsInstance(simulator.param_names, list)
        self.assertGreater(len(simulator.param_names), 0)
        self.assertIsNotNone(simulator.params)
        self.assertEqual(len(simulator.param_items), len(simulator.param_names))

        for name, value in simulator.param_items:
            self.assertIn(name, simulator.param_names)
            self.assertTrue(np.isfinite(value))

    def test_model_summary(self) -> None:
        """
        Test the model_summary method.

        After fitting, model_summary should return a non-empty textual description
        of the fitted ARIMA model.
        """

        simulator = self._make_simulator()
        time_series = self._make_time_series(length=120, seed=4)

        # Summary should not be available before fitting
        with self.assertRaises(AttributeError):
            simulator.model_summary()

        simulator.fit(time_series)
        summary = simulator.model_summary()

        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 0)

    def test_transform_before_fit(self) -> None:
        """
        Test calling transform before the model has been fitted.

        When no statsmodels result object is attached, simulation should fail.
        """

        simulator = self._make_simulator()

        # transform must refuse to run before fit
        with self.assertRaises(ValueError):
            simulator.transform(num_samples=1, seq_length=50)

    def test_adf_test(self) -> None:
        """
        Test the adf_test stationarity helper.

        The method should return a boolean indicating whether the ADF test
        rejects the unit-root null hypothesis at the configured significance level.
        """

        simulator = self._make_simulator(signif=0.05)
        time_series = self._make_time_series(length=120, seed=5)

        # adf_test expects pandas-like input because it calls dropna() internally
        white_noise = pd.Series(np.random.RandomState(6).randn(120))
        self.assertIsInstance(simulator.adf_test(white_noise), bool)
        self.assertIsInstance(simulator.adf_test(pd.Series(time_series)), bool)

    def test_diff_stationary(self) -> None:
        """
        Test the diff_stationary method.

        The method should return a differenced series and a non-negative
        differencing order not exceeding max_d.
        """

        simulator = self._make_simulator(max_d=2)
        time_series = pd.Series(self._make_time_series(length=120, seed=7))

        stationary_series, diff_count = simulator.diff_stationary(time_series)

        # Differencing order should be within the configured upper bound
        self.assertGreaterEqual(diff_count, 0)
        self.assertLessEqual(diff_count, simulator.max_d)
        self.assertGreater(len(stationary_series), 0)
        self.assertIsInstance(stationary_series, pd.Series)

    def test_select_arma_order(self) -> None:
        """
        Test the select_arma_order method.

        For a stationary input series, the method should return a tuple of
        integer AR and MA orders selected by BIC.
        """

        simulator = self._make_simulator(max_p=2, max_q=2)
        time_series = pd.Series(self._make_time_series(length=120, seed=8))
        stationary_series, _ = simulator.diff_stationary(time_series)

        p_order, q_order = simulator.select_arma_order(stationary_series)

        self.assertIsInstance(p_order, int)
        self.assertIsInstance(q_order, int)
        self.assertGreaterEqual(p_order, 0)
        self.assertGreaterEqual(q_order, 0)
        self.assertLessEqual(p_order, simulator.max_p)
        self.assertLessEqual(q_order, simulator.max_q)

    def test_residual_diagnosis(self) -> None:
        """
        Test the residual_diagnosis method.

        After fitting, the method should return the mean Ljung-Box p-value and a
        boolean flag indicating whether all p-values exceed the significance level.
        """

        simulator = self._make_simulator()
        time_series = self._make_time_series(length=120, seed=9)
        simulator.fit(time_series)

        mean_p_value, is_white = simulator.residual_diagnosis(lags=10)

        self.assertIsInstance(mean_p_value, (float, np.floating))
        self.assertIsInstance(is_white, (bool, np.bool_))
        self.assertGreaterEqual(mean_p_value, 0.0)
        self.assertLessEqual(mean_p_value, 1.0)

    def test_acf_and_pacf(self) -> None:
        """
        Test the static acf and pacf helper methods.

        Both functions should return autocorrelation estimates with length
        nlags + 1 when nlags is explicitly provided.
        """

        time_series = self._make_time_series(length=120, seed=10)
        nlags = 10

        acf_vals = ARIMASimulator.acf(time_series, nlags=nlags, fft=True)
        # Use a statsmodels-supported PACF method for compatibility across versions
        pacf_vals = ARIMASimulator.pacf(time_series, nlags=nlags, method="ywmle")

        self.assertEqual(len(acf_vals), nlags + 1)
        self.assertEqual(len(pacf_vals), nlags + 1)
        self.assertAlmostEqual(acf_vals[0], 1.0)

    def test_eacf(self) -> None:
        """
        Test the static eacf helper method.

        The EACF routine should return a matrix, a significance threshold, and
        a formatted DataFrame summary.
        """

        time_series = self._make_time_series(length=120, seed=11)

        eacf_matrix, threshold, eacf_df = ARIMASimulator.eacf(
            time_series=time_series,
            symbolize=True,
            max_ar=3,
            max_ma=3,
        )

        self.assertEqual(eacf_matrix.shape, (4, 3))
        self.assertIsInstance(threshold, float)
        self.assertIsInstance(eacf_df, pd.DataFrame)

    def test_plot_helpers(self) -> None:
        """
        Test plotting helper methods for ACF, PACF, and EACF.

        Each plotting method should return a matplotlib Figure object without
        requiring an active GUI backend.
        """

        simulator = self._make_simulator()
        time_series = self._make_time_series(length=120, seed=12)

        acf_fig = ARIMASimulator.plot_acf(time_series, lags=10)
        pacf_fig = ARIMASimulator.plot_pacf(time_series, lags=10)
        eacf_fig = simulator.plot_eacf(time_series, max_ar=2, max_ma=2)

        self.assertIsInstance(acf_fig, plt.Figure)
        self.assertIsInstance(pacf_fig, plt.Figure)
        self.assertIsInstance(eacf_fig, plt.Figure)

        plt.close(acf_fig)
        plt.close(pacf_fig)
        plt.close(eacf_fig)

    def test_plot_shapiro_wilk(self) -> None:
        """
        Test the plot_shapiro_wilk residual diagnostic helper.

        After fitting, the method should return a figure together with the
        Shapiro-Wilk statistic and p-value.
        """

        simulator = self._make_simulator()
        time_series = self._make_time_series(length=120, seed=13)
        simulator.fit(time_series)

        fig, stat, p_value = simulator.plot_shapiro_wilk(bins=10, dpi=100)

        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(stat, float)
        self.assertIsInstance(p_value, float)
        plt.close(fig)

    def test_revin_false(self) -> None:
        """
        Test transform behavior when reversible normalization is disabled.

        With revin=False, generated samples should be returned directly from the
        fitted ARIMA simulator without inverse mean/std restoration.
        """

        simulator = self._make_simulator(revin=False)
        time_series = self._make_time_series(length=120, seed=14)
        simulator.fit(time_series)

        # mean/std should not be recorded when revin is disabled
        self.assertIsNone(simulator.mean)
        self.assertIsNone(simulator.std)

        generated = simulator.transform(num_samples=3, seq_length=60, random_state=15)
        self.assertEqual(generated.shape, (3, 60))

    def test_not_white_alarm(self) -> None:
        """
        Test the not_white_alarm switch during fit.

        When residual whiteness is not satisfied and not_white_alarm=True, fit
        should raise ValueError; disabling the alarm should allow fit to finish.
        """

        time_series = self._make_time_series(length=120, seed=16)

        strict_simulator = ARIMASimulator(
            max_p=5,
            max_q=5,
            max_d=2,
            not_white_alarm=True,
            revin=True,
            random_state=42,
        )
        tolerant_simulator = ARIMASimulator(
            max_p=5,
            max_q=5,
            max_d=2,
            not_white_alarm=False,
            revin=True,
            random_state=42,
        )

        # High-order models may fail the residual whiteness check
        try:
            strict_simulator.fit(time_series)
        except ValueError as exc:
            self.assertIn("white noise", str(exc))
            # The tolerant simulator should still complete fitting on the same data
            tolerant_simulator.fit(time_series)
            self.assertIsNotNone(tolerant_simulator.model)
        else:
            # If strict fitting succeeds, tolerant fitting should succeed as well
            tolerant_simulator.fit(time_series)
            self.assertIsNotNone(tolerant_simulator.model)


if __name__ == "__main__":
    unittest.main()
