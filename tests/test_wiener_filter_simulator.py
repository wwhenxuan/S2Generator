# -*- coding: utf-8 -*-
"""
Created on 2026/03/02 12:16:05
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/S2Generator
"""

import unittest

import numpy as np
from scipy.linalg import toeplitz

from s2generator.simulator import WienerFilterSimulator
from s2generator.utils._tools import yule_walker


class TestWienerFilterSimulator(unittest.TestCase):
    """The Unittest for WienerFilterSimulator class."""

    def test_create_instance(self) -> None:
        """Test the creation of a WienerFilterSimulator instance."""

        # Test with different filter orders
        for filter_order in [1, 3, 5, 7, 9, 25]:
            # Test with different combinations of revin and random_state parameters
            for revin in [True, False]:
                for random_state in [None, 0, 42]:
                    # Use subTest to test different combinations of parameters
                    with self.subTest(
                        filter_order=filter_order,
                        revin=revin,
                        random_state=random_state,
                    ):
                        # Create an instance of WienerFilterSimulator with the specified parameters
                        simulator = WienerFilterSimulator(
                            filter_order=filter_order,
                            revin=revin,
                            random_state=random_state,
                        )
                        self.assertIsInstance(simulator, WienerFilterSimulator)

    def test_fit_transform(self) -> None:
        """Test the fit_transform method of WienerFilterSimulator."""

        # Create a WienerFilterSimulator instance
        simulator = WienerFilterSimulator(filter_order=5)

        # Generate random input signal for fitting and transforming
        np.random.seed(0)  # For reproducibility
        time_series = np.random.rand(100)

        # Fit the model with the input signal
        simulator.fit(time_series)

        # Check the residual variance after fitting (should be a non-negative value)
        self.assertIsInstance(simulator.residuals, np.ndarray)

        # Check the shape of the residuals (should be the same as the input signal)
        self.assertEqual(simulator.residuals.shape, time_series.shape)

        # Transform the input signal using the fitted model
        simulation = simulator.transform(num_samples=5, seq_length=100)

        # Check the shape of the simulated data (should be (num_samples, seq_length))
        self.assertEqual(simulation.shape, (5, 100))

    def test_check_inputs(self) -> None:
        """Test the check_inputs method of WienerFilterSimulator."""

        # Create a WienerFilterSimulator instance
        simulator = WienerFilterSimulator()

        # The the wrong output signal with a different shape
        for wrong_input_signal in [
            1,
            "hello, world!",
            True,
            [1, 2, 3],
            {"input_signal": [1, 2, 3]},
        ]:
            with self.subTest(wrong_input_signal=wrong_input_signal):
                with self.assertRaises(ValueError):
                    simulator.check_inputs(wrong_input_signal)

        # Generate random input and output signals for fitting
        np.random.seed(0)  # For reproducibility
        time_series = np.random.rand(100)

        # Test the valid time series
        result = simulator.check_inputs(time_series)
        self.assertEqual(result.shape, time_series.shape)

        # The the 2D array input signal
        time_series_2d = np.random.rand(2, 100)
        result = simulator.check_inputs(time_series_2d)
        self.assertEqual(result.shape, time_series_2d.flatten().shape)

    def test_coeffs(self) -> None:
        """The test for the coeffs property of WienerFilterSimulator."""

        for filter_order in [3, 5, 7, 9]:
            # Create a WienerFilterSimulator instance with different filter orders
            simulator = WienerFilterSimulator(filter_order=filter_order)

            # Try to access the coeffs property before fitting the model (should raise an error)
            with self.assertRaises(ValueError):
                _ = simulator.coeffs

            time_series = np.random.rand(100)
            simulator.fit(time_series)

            # Check if the coefficients are initialized correctly
            self.assertEqual(len(simulator._coeffs), filter_order)

    def test_sigma_sq(self) -> None:
        """Test the sigma_sq property of WienerFilterSimulator."""

        # Create a WienerFilterSimulator instance
        simulator = WienerFilterSimulator(filter_order=5)

        # Try to access the sigma_sq property before fitting the model (should raise an error)
        with self.assertRaises(ValueError):
            _ = simulator.sigma_sq

        # Generate random input signal for fitting
        time_series = np.random.rand(100)
        simulator.fit(time_series)

        # Check if the sigma_sq is initialized correctly (should be None)
        self.assertIsNotNone(simulator.sigma_sq)

    def test_set_coeffs(self) -> None:
        """Test the set_coeffs method of WienerFilterSimulator."""

        for filter_order in [3, 5, 7, 9]:
            # Create a WienerFilterSimulator instance with different filter orders
            simulator = WienerFilterSimulator(filter_order=filter_order)

            # Generate random coefficients with the correct shape
            coeffs = np.random.rand(filter_order)

            # Set the coefficients using the set_coeffs method
            simulator.set_coeffs(coeffs=coeffs)

            # Check if the coefficients are initialized correctly
            self.assertEqual(len(simulator.coeffs), filter_order)

            # Check if the coefficients are set correctly
            self.assertTrue(np.allclose(simulator.coeffs, coeffs))

        # Create a WienerFilterSimulator instance
        simulator = WienerFilterSimulator(filter_order=5)

        # Test the wrong coefficients with a different shape
        for wrong_coeffs in [np.random.rand(4), [1, 2, 3, 4, 5, 6], "hello, world!"]:
            with self.subTest(wrong_coeffs=wrong_coeffs):
                with self.assertRaises(AssertionError):
                    simulator.set_coeffs(wrong_coeffs)

    def test_set_sigma_sq(self) -> None:
        """Test the set_sigma_sq method of WienerFilterSimulator."""

        # Create a WienerFilterSimulator instance
        simulator = WienerFilterSimulator(filter_order=5)

        # Generate a random sigma_sq value
        sigma_sq = np.random.rand()

        # Set the sigma_sq using the set_sigma_sq method
        simulator.set_sigma_sq(sigma_sq=sigma_sq)

        # Check if the sigma_sq is set correctly
        self.assertEqual(simulator.sigma_sq, sigma_sq)

        # Test the wrong sigma_sq with a non-numeric value
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

    def test_yule_walker(self) -> None:
        """Test the yule_walker function used in WienerFilterSimulator."""

        # Generate a random time series
        np.random.seed(0)  # For reproducibility
        time_series = np.random.rand(100)

        # Test the yule_walker function with different filter orders
        for filter_order in [5, 7, 9]:
            with self.subTest(filter_order=filter_order):
                simulator = WienerFilterSimulator(filter_order=filter_order)
                A = toeplitz(simulator.acf(time_series)[:filter_order])
                coeffs, sigma_sq = yule_walker(A=A)

                # Check if the coefficients and sigma_sq are returned correctly
                self.assertEqual(len(coeffs), filter_order)
                self.assertIsInstance(sigma_sq, (float, np.ndarray))
