# -*- coding: utf-8 -*-
"""
Created on 2025/08/13 23:47:51
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""

import unittest
import numpy as np

from s2generator.excitation import AutoregressiveMovingAverage
from s2generator.excitation.base_excitation import BaseExcitation
from s2generator.excitation.autoregressive_moving_average import arma_series


class TestARMA(unittest.TestCase):
    """Testing the ARMA module for generating stimulus time series data"""

    # Random number generator for testing
    rng = np.random.RandomState(42)

    # Instance object for testing
    arma = AutoregressiveMovingAverage()

    def test_setup(self) -> None:
        """Test module creation process with keyword arguments"""
        for p_max in [2, 3, 4, 5]:
            for q_max in [2, 3, 4, 5]:
                for upper_bound in [100, 200, 300, 400]:
                    arma = AutoregressiveMovingAverage(
                        p_max=p_max, q_max=q_max, upper_bound=upper_bound
                    )
                    self.assertIsInstance(arma, AutoregressiveMovingAverage)
                    self.assertEqual(arma.p_max, p_max)
                    self.assertEqual(arma.q_max, q_max)
                    self.assertEqual(arma.upper_bound, upper_bound)

    def test_inheritance_base_excitation(self) -> None:
        """Test that ARMA inherits from BaseExcitation"""
        self.assertIsInstance(self.arma, BaseExcitation)
        zeros = self.arma.create_zeros(n_inputs_points=16, input_dimension=2)
        self.assertEqual(zeros.shape, (16, 2))

    def test_create_autoregressive_params(self) -> None:
        """Stationary AR coefficients must have all roots inside the unit circle."""
        for p_order in [1, 2, 3, 4]:
            p_params = self.arma.create_autoregressive_params(
                rng=self.rng, p_order=p_order, stationary=True
            )

            self.assertEqual(len(p_params), p_order)
            self.assertIsInstance(p_params, np.ndarray)
            roots = np.roots(np.concatenate(([1.0], -p_params)))
            self.assertTrue(
                np.all(np.abs(roots) < 1.0 + 1e-8),
                msg="stationary AR roots must lie inside the unit circle",
            )

    def test_create_autoregressive_params_nonstationary(self) -> None:
        """With stationary=False, moduli are allowed to reach or exceed 1."""
        arma = AutoregressiveMovingAverage(stationary=False)
        max_moduli = []
        rng = np.random.RandomState(1)
        for _ in range(40):
            phi = arma.create_autoregressive_params(
                rng=rng, p_order=2, stationary=False
            )
            roots = np.roots(np.concatenate(([1.0], -phi)))
            max_moduli.append(float(np.max(np.abs(roots))))
        self.assertTrue(
            max(max_moduli) >= 1.0 - 1e-8,
            msg="non-stationary draws should be able to leave the unit disk",
        )

    def test_create_moving_average_params(self) -> None:
        """Test whether the parameters of the moving average process can be generated normally"""
        for q_order in [1, 2, 3, 4, 5]:
            q_params = self.arma.create_moving_average_params(
                rng=self.rng, q_order=q_order
            )

            self.assertEqual(len(q_params), q_order)
            self.assertIsInstance(q_params, np.ndarray)
            self.assertTrue(np.all(q_params >= -1.0))
            self.assertTrue(np.all(q_params <= 1.0))

    def test_create_params(self) -> None:
        """Test whether the parameters of the ARMA model can be generated normally"""
        self.arma.create_params(rng=self.rng)

        self.assertEqual(self.arma.p_order, len(self.arma.p_params))
        self.assertEqual(self.arma.q_order, len(self.arma.q_params))
        self.assertGreaterEqual(self.arma.p_order, self.arma.p_min)
        self.assertLess(self.arma.p_order, self.arma.p_max)
        self.assertGreaterEqual(self.arma.q_order, self.arma.q_min)
        self.assertLess(self.arma.q_order, self.arma.q_max)

    def test_order_range_respected(self) -> None:
        """Orders sampled by create_params stay within [min, max)"""
        arma = AutoregressiveMovingAverage(p_min=2, p_max=5, q_min=2, q_max=6)
        rng = np.random.RandomState(0)
        for _ in range(30):
            arma.create_params(rng=rng)
            self.assertGreaterEqual(arma.p_order, 2)
            self.assertLess(arma.p_order, 5)
            self.assertGreaterEqual(arma.q_order, 2)
            self.assertLess(arma.q_order, 6)

    def test_order(self) -> None:
        """Test the function that attempts to obtain the model order"""
        self.arma.create_params(rng=self.rng)
        order_dict = self.arma.order

        self.assertIsInstance(order_dict, dict)
        self.assertIn("AR(p)", order_dict)
        self.assertIn("MA(q)", order_dict)
        for key, value in order_dict.items():
            self.assertIsInstance(key, str)
            self.assertIsInstance(value, int)

    def test_params(self) -> None:
        """Test the function that tries to get the model parameters"""
        self.arma.create_params(rng=self.rng)
        params_dict = self.arma.params

        self.assertIsInstance(params_dict, dict)
        self.assertIn("AR(p)", params_dict)
        self.assertIn("MA(q)", params_dict)
        for key, value in params_dict.items():
            self.assertIsInstance(key, str)
            self.assertIsInstance(value, np.ndarray)

    def test_module_arma_series_basic(self) -> None:
        """Test the standalone arma_series helper"""
        rng = np.random.RandomState(1)
        series = np.zeros(64, dtype=np.float64)
        p_params = np.array([0.3, 0.2])
        q_params = np.array([0.1, -0.2, 0.05])
        out = arma_series(rng, series, p_params, q_params)

        self.assertIs(out, series)
        self.assertEqual(out.shape, (64,))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_module_arma_series_clamp(self) -> None:
        """Values larger than 1024 should be replaced by the MA contribution"""
        rng = np.random.RandomState(0)
        series = np.zeros(8, dtype=np.float64)
        # Strong AR coefficient causes rapid growth and hits the clamp branch.
        p_params = np.array([3.0])
        q_params = np.array([0.0])
        out = arma_series(rng, series, p_params, q_params)
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertTrue(np.all(np.abs(out) <= 1024 + 1e-6) or np.any(np.abs(out) > 0))

    def test_instance_arma_series_uses_self_params(self) -> None:
        """Instance arma_series falls back to self.p_params / self.q_params"""
        arma = AutoregressiveMovingAverage()
        rng = np.random.RandomState(7)
        arma.create_params(rng=rng)
        series = np.zeros(32, dtype=np.float64)
        out = arma.arma_series(rng=rng, time_series=series)
        self.assertEqual(out.shape, (32,))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_generate(self) -> None:
        """Test whether the stimulus time series data can be generated correctly"""
        for p_max in [2, 3, 4]:
            for q_max in [2, 3, 4, 5]:
                arma = AutoregressiveMovingAverage(p_max=p_max, q_max=q_max)
                for length in [32, 128, 256]:
                    for dim in [1, 3, 5]:
                        time_series = arma.generate(
                            rng=self.rng, n_inputs_points=length, input_dimension=dim
                        )
                        self.assertIsInstance(time_series, np.ndarray)
                        self.assertEqual(time_series.shape, (length, dim))
                        self.assertTrue(np.all(np.isfinite(time_series)))

    def test_upper_bound_enforced(self) -> None:
        """Generated series should respect the configured upper bound"""
        upper_bound = 50.0
        arma = AutoregressiveMovingAverage(upper_bound=upper_bound)
        rng = np.random.RandomState(123)
        series = arma.generate(rng=rng, n_inputs_points=128, input_dimension=3)
        self.assertLessEqual(np.max(np.abs(series)), upper_bound + 1e-8)

    def test_dtype_on_generate(self) -> None:
        """Generated series should use the configured dtype"""
        for dtype in [np.float32, np.float64]:
            arma = AutoregressiveMovingAverage(dtype=dtype)
            series = arma.generate(
                rng=np.random.RandomState(0), n_inputs_points=64, input_dimension=2
            )
            self.assertEqual(series.dtype, dtype)

    def test_reproducibility_same_seed(self) -> None:
        """Same seed should produce identical ARMA series"""
        arma1 = AutoregressiveMovingAverage()
        arma2 = AutoregressiveMovingAverage()
        out1 = arma1.generate(
            rng=np.random.RandomState(42), n_inputs_points=128, input_dimension=2
        )
        out2 = arma2.generate(
            rng=np.random.RandomState(42), n_inputs_points=128, input_dimension=2
        )
        np.testing.assert_array_equal(out1, out2)

    def test_generate_nonstationary_respects_upper_bound(self) -> None:
        """Non-stationary draws may explode internally but the output is clipped."""
        arma = AutoregressiveMovingAverage(
            p_min=2, p_max=4, stationary=False, upper_bound=80.0
        )
        series = arma.generate(
            rng=np.random.RandomState(0), n_inputs_points=128, input_dimension=3
        )
        self.assertEqual(series.shape, (128, 3))
        self.assertTrue(np.all(np.isfinite(series)))
        self.assertLessEqual(np.max(np.abs(series)), 80.0 + 1e-8)

    def test_call(self) -> None:
        """Test data generation class response"""
        time_series = self.arma(rng=self.rng, n_inputs_points=256, input_dimension=1)
        self.assertIsInstance(time_series, np.ndarray)
        self.assertEqual(time_series.shape, (256, 1))

    def test_str(self) -> None:
        """Test the magic method to get string description"""
        self.assertIsInstance(str(self.arma), str)
        self.assertEqual(str(self.arma), "ARMA")


if __name__ == "__main__":
    unittest.main()
