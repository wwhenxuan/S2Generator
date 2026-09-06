# -*- coding: utf-8 -*-
"""
Created on 2026/06/21 02:16:57
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/S2Generator
"""

import unittest

import numpy as np
from scipy.linalg import toeplitz

from s2generator.simulator import KalmanFilterSimulator, WienerFilterSimulator
from s2generator.utils.tools import yule_walker


class TestKalmanFilterSimulator(unittest.TestCase):
    """The Unittest for KalmanFilterSimulator class."""

    @staticmethod
    def _make_time_series(length: int = 100, seed: int = 0) -> np.ndarray:
        """
        Build a reproducible random input sequence for fitting tests.

        The length defaults to 100 so that it satisfies the minimum requirement
        ``2 * state_order`` for common state orders used in unit tests.
        """
        rng = np.random.RandomState(seed)
        return rng.rand(length)

    @staticmethod
    def _make_simulator(**kwargs) -> KalmanFilterSimulator:
        """Create a KalmanFilterSimulator with conservative defaults for unit tests."""
        defaults = {
            "state_order": 5,
            "revin": True,
            "random_state": 42,
            "observation_noise": 1e-8,
        }
        defaults.update(kwargs)
        return KalmanFilterSimulator(**defaults)

    def test_create_instance(self) -> None:
        """
        Test the creation of a KalmanFilterSimulator instance.

        Verify that different combinations of state_order, revin, random_state,
        and observation_noise can be passed to the constructor successfully.
        """

        # Traverse several common hyperparameter combinations
        for state_order in [3, 5, 7, 9, 25]:
            for revin in [True, False]:
                for random_state in [None, 0, 42]:
                    for observation_noise in [1e-8, 1e-6]:
                        with self.subTest(
                            state_order=state_order,
                            revin=revin,
                            random_state=random_state,
                            observation_noise=observation_noise,
                        ):
                            simulator = KalmanFilterSimulator(
                                state_order=state_order,
                                revin=revin,
                                random_state=random_state,
                                observation_noise=observation_noise,
                            )
                            self.assertIsInstance(simulator, KalmanFilterSimulator)
                            self.assertEqual(simulator.state_order, state_order)
                            self.assertEqual(
                                simulator.observation_noise, observation_noise
                            )

    def test_fit_transform(self) -> None:
        """
        Test the fit and transform workflow of KalmanFilterSimulator.

        After fitting on a valid input sequence, residuals and innovations should
        be available, and transform should return data with shape [num_samples, seq_length].
        """

        simulator = self._make_simulator(state_order=5)
        time_series = self._make_time_series(length=100, seed=0)

        # Fit the state-space model on the input sequence
        simulator.fit(time_series)

        # Residuals and innovations should be stored after fitting
        self.assertIsInstance(simulator.residuals, np.ndarray)
        self.assertIsInstance(simulator.innovations, np.ndarray)
        self.assertEqual(simulator.residuals.shape, time_series.shape)
        self.assertEqual(simulator.innovations.shape, time_series.shape)

        # State-space matrices should be constructed during fit
        F, G, H, Q, R = simulator.state_space_matrices
        self.assertEqual(F.shape[0], simulator.state_order - 1)
        self.assertEqual(G.shape, (simulator.state_order - 1, 1))
        self.assertEqual(H.shape, (1, simulator.state_order - 1))

        # Generate new sequences from the fitted model
        simulation = simulator.transform(num_samples=5, seq_length=100, random_state=7)

        # Output shape should be [num_samples, seq_length]
        self.assertEqual(simulation.shape, (5, 100))
        self.assertFalse(np.isnan(simulation).any())

    def test_check_inputs(self) -> None:
        """
        Test the check_inputs method of KalmanFilterSimulator.

        Invalid dtypes, shapes, constant sequences, NaN values, and overly short
        inputs should raise ValueError; valid 1D/2D ndarray inputs should pass.
        """

        simulator = self._make_simulator(state_order=5)

        # Invalid input types should be rejected
        for wrong_input in [1, "hello, world!", True, [1, 2, 3], {"input": [1, 2, 3]}]:
            with self.subTest(wrong_input=wrong_input):
                with self.assertRaises(ValueError):
                    simulator.check_inputs(wrong_input)

        # Valid 1D ndarray input should be returned unchanged in shape
        time_series = self._make_time_series(length=100, seed=1)
        result = simulator.check_inputs(time_series)
        self.assertEqual(result.shape, time_series.shape)

        # Valid 2D ndarray input should be flattened
        time_series_2d = np.random.rand(2, 100)
        result_2d = simulator.check_inputs(time_series_2d)
        self.assertEqual(result_2d.shape, time_series_2d.flatten().shape)

        # Input shorter than 2 * state_order should be rejected
        with self.assertRaises(ValueError):
            simulator.check_inputs(np.random.rand(2 * simulator.state_order - 1))

        # Constant input with zero variance should be rejected
        with self.assertRaises(ValueError):
            simulator.check_inputs(np.ones(100))

        # Input containing NaN values should be rejected
        nan_series = time_series.copy()
        nan_series[0] = np.nan
        with self.assertRaises(ValueError):
            simulator.check_inputs(nan_series)

        # Input with more than two dimensions should be rejected
        with self.assertRaises(ValueError):
            simulator.check_inputs(np.random.rand(2, 3, 4))

    def test_coeffs(self) -> None:
        """
        Test the coeffs property of KalmanFilterSimulator.

        The property should raise ValueError before fitting and return an AR
        coefficient vector of length state_order after fitting.
        """

        for state_order in [3, 5, 7, 9]:
            simulator = KalmanFilterSimulator(state_order=state_order, random_state=0)

            # Accessing coeffs before fit should fail
            with self.assertRaises(ValueError):
                _ = simulator.coeffs

            time_series = self._make_time_series(
                length=max(100, 2 * state_order), seed=2
            )
            simulator.fit(time_series)

            # Coefficient vector length should match state_order
            self.assertEqual(len(simulator.coeffs), state_order)
            self.assertAlmostEqual(simulator.coeffs[0], 1.0)

    def test_sigma_sq(self) -> None:
        """
        Test the sigma_sq property of KalmanFilterSimulator.

        The property should raise ValueError before fitting and return a positive
        process-noise variance after fitting.
        """

        simulator = self._make_simulator(state_order=5)

        # Accessing sigma_sq before fit should fail
        with self.assertRaises(ValueError):
            _ = simulator.sigma_sq

        time_series = self._make_time_series(length=100, seed=3)
        simulator.fit(time_series)

        # Process-noise variance should be a positive scalar after fitting
        self.assertIsNotNone(simulator.sigma_sq)
        self.assertGreater(simulator.sigma_sq, 0.0)

    def test_set_coeffs(self) -> None:
        """
        Test the set_coeffs method of KalmanFilterSimulator.

        Valid coefficient arrays should be stored correctly, while invalid shapes
        or dtypes should trigger AssertionError.
        """

        for state_order in [3, 5, 7, 9]:
            simulator = KalmanFilterSimulator(state_order=state_order)

            # Generate coefficient vector with the expected length
            coeffs = np.random.rand(state_order)
            coeffs[0] = 1.0

            simulator.set_coeffs(coeffs=coeffs)

            # Stored coefficients should match the manually assigned values
            self.assertEqual(len(simulator.coeffs), state_order)
            self.assertTrue(np.allclose(simulator.coeffs, coeffs))

        simulator = KalmanFilterSimulator(state_order=5)

        # Invalid coefficient inputs should be rejected
        for wrong_coeffs in [np.random.rand(4), [1, 2, 3, 4, 5, 6], "hello, world!"]:
            with self.subTest(wrong_coeffs=wrong_coeffs):
                with self.assertRaises(AssertionError):
                    simulator.set_coeffs(wrong_coeffs)

    def test_set_sigma_sq(self) -> None:
        """
        Test the set_sigma_sq method of KalmanFilterSimulator.

        Valid positive numeric values should be accepted, while invalid inputs
        should trigger AssertionError.
        """

        simulator = self._make_simulator(state_order=5)

        # Assign a valid process-noise variance
        sigma_sq = np.random.rand()
        simulator.set_sigma_sq(sigma_sq=sigma_sq)
        self.assertEqual(simulator.sigma_sq, sigma_sq)

        # Invalid sigma_sq values should be rejected
        for wrong_sigma_sq in [
            "hello, world!",
            [1, 2, 3],
            {"sigma_sq": sigma_sq},
            -0.5,
            0,
        ]:
            with self.subTest(wrong_sigma_sq=wrong_sigma_sq):
                with self.assertRaises(AssertionError):
                    simulator.set_sigma_sq(wrong_sigma_sq)

    def test_state_space_matrices(self) -> None:
        """
        Test the state_space_matrices property.

        After fitting, the companion matrices should have consistent dimensions
        and the observation-noise variance R should match the constructor setting.
        """

        observation_noise = 1e-7
        simulator = self._make_simulator(
            state_order=6, observation_noise=observation_noise
        )
        time_series = self._make_time_series(length=120, seed=4)

        # Property should not be available before fitting
        with self.assertRaises(ValueError):
            _ = simulator.state_space_matrices

        simulator.fit(time_series)
        F, G, H, Q, R = simulator.state_space_matrices

        p = simulator.state_order - 1

        # Companion-form dimensions should follow AR order p = state_order - 1
        self.assertEqual(F.shape, (p, p))
        self.assertEqual(G.shape, (p, 1))
        self.assertEqual(H.shape, (1, p))
        self.assertEqual(Q.shape, (p, p))
        self.assertAlmostEqual(R, observation_noise)
        self.assertAlmostEqual(Q[0, 0], simulator.sigma_sq)

    def test_invoke(self) -> None:
        """
        Test the invoke method of KalmanFilterSimulator.

        invoke should raise ValueError before fitting and return a 1D sequence
        with length len(white_noise) - state_order after fitting.
        """

        simulator = self._make_simulator(state_order=5)
        white_noise = np.random.randn(100 + simulator.state_order)

        # invoke should fail before the model is fitted
        with self.assertRaises(ValueError):
            simulator.invoke(white_noise=white_noise)

        time_series = self._make_time_series(length=100, seed=5)
        simulator.fit(time_series)

        output = simulator.invoke(white_noise=white_noise)

        # Output length equals input white-noise length minus state_order
        self.assertEqual(output.shape, (len(white_noise) - simulator.state_order,))

    def test_transform_before_fit(self) -> None:
        """
        Test calling transform before the model has been fitted.

        transform should raise ValueError when coefficients have not been estimated.
        """

        simulator = self._make_simulator(state_order=5)

        with self.assertRaises(ValueError):
            simulator.transform(num_samples=1, seq_length=50)

    def test_acf(self) -> None:
        """
        Test the acf helper method of KalmanFilterSimulator.

        The method should return autocorrelation values with length lag_max + 1.
        """

        simulator = self._make_simulator(state_order=5)
        time_series = self._make_time_series(length=100, seed=6)

        acf_vals = simulator.acf(time_series=time_series)
        self.assertEqual(len(acf_vals), simulator.lag_max + 1)
        self.assertAlmostEqual(acf_vals[0], 1.0)

        # Custom lag_max should override the default lag setting
        custom_lag = 12
        acf_custom = simulator.acf(time_series=time_series, lag_max=custom_lag)
        self.assertEqual(len(acf_custom), custom_lag + 1)

    def test_revin_false(self) -> None:
        """
        Test transform behavior when reversible normalization is disabled.

        With revin=False, mean and std should remain None and generated samples
        should not be inverse-transformed.
        """

        simulator = self._make_simulator(state_order=5, revin=False)
        time_series = self._make_time_series(length=100, seed=7)
        simulator.fit(time_series)

        # Normalization statistics should not be recorded
        self.assertIsNone(simulator.mean)
        self.assertIsNone(simulator.std)

        generated = simulator.transform(num_samples=3, seq_length=60, random_state=8)
        self.assertEqual(generated.shape, (3, 60))

    def test_reproducibility(self) -> None:
        """
        Test reproducibility of transform with a fixed random seed.

        Two transform calls with the same random_state should produce identical
        simulated sequences after the same fit operation.
        """

        simulator = self._make_simulator(state_order=5, random_state=42)
        time_series = self._make_time_series(length=100, seed=9)
        simulator.fit(time_series)

        sample_a = simulator.transform(num_samples=2, seq_length=80, random_state=123)
        sample_b = simulator.transform(num_samples=2, seq_length=80, random_state=123)

        self.assertTrue(np.allclose(sample_a, sample_b))

    def test_consistency_with_wiener_filter(self) -> None:
        """
        Test numerical consistency with WienerFilterSimulator.

        Because both simulators estimate the same Yule-Walker parameters and implement
        the same white-noise excitation mapping, invoke outputs should closely match
        when given identical coefficients and white-noise realizations.
        """

        state_order = 6
        time_series = self._make_time_series(length=120, seed=10)
        white_noise = np.random.RandomState(11).randn(120 + state_order)

        kalman_sim = KalmanFilterSimulator(
            state_order=state_order, revin=False, random_state=0
        )
        wiener_sim = WienerFilterSimulator(
            filter_order=state_order, revin=False, random_state=0
        )

        kalman_sim.fit(time_series)
        wiener_sim.fit(time_series)

        # Estimated parameters should match between the two simulators
        self.assertTrue(np.allclose(kalman_sim.coeffs, wiener_sim.coeffs))
        self.assertAlmostEqual(kalman_sim.sigma_sq, wiener_sim.sigma_sq)

        kalman_output = kalman_sim.invoke(white_noise=white_noise)
        wiener_output = wiener_sim.invoke(white_noise=white_noise)

        # Generated sequences should be numerically close
        self.assertTrue(np.allclose(kalman_output, wiener_output, atol=1e-10))

    def test_yule_walker(self) -> None:
        """
        Test the yule_walker function used inside KalmanFilterSimulator.

        For different state orders, yule_walker should return coefficient vectors
        of length state_order and a valid process-noise variance estimate.
        """

        time_series = self._make_time_series(length=120, seed=12)

        for state_order in [5, 7, 9]:
            with self.subTest(state_order=state_order):
                simulator = KalmanFilterSimulator(state_order=state_order)
                A = toeplitz(simulator.acf(time_series)[:state_order])
                coeffs, sigma_sq = yule_walker(A=A)

                self.assertEqual(len(coeffs), state_order)
                self.assertIsInstance(sigma_sq, (float, np.ndarray))


if __name__ == "__main__":
    unittest.main()
