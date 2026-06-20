# -*- coding: utf-8 -*-
"""
Created on 2025/08/27 10:44:34
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/S2Generator

Test module for Wasserstein distance calculation between datasets.
This module validates the implementation of Wasserstein distance metrics
for assessing similarity between multivariate time series datasets.
"""

import unittest
import numpy as np

from s2generator.utils import (
    wasserstein_distance,
    wasserstein_distance_matrix,
    plot_wasserstein_heatmap,
)
from s2generator.utils._wasserstein_distance import (
    dataset_max_min_normalization,
    time_series_to_distribution,
    check_inputs,
)


class TestWasserstein(unittest.TestCase):
    """
    Test suite for Wasserstein distance calculation between datasets.

    Validates the implementation of:
    1. Input data validation
    2. Data normalization techniques
    3. Distribution estimation from time series
    4. Wasserstein distance calculation
    5. Edge case handling
    """

    # Define test data parameters
    n_samples = 100  # Number of samples in test dataset
    n_length = 48  # Length of each time series
    # Generate random test data
    data = np.random.randn(n_samples, n_length)

    def test_check_inputs(self) -> None:
        """
        Tests the input validation function for data integrity checks.

        Validates:
        - Proper handling of valid input data
        - Type error raising for invalid data types
        - Value error raising for incorrect data dimensions
        """
        # Test valid data input
        self.assertTrue(
            check_inputs(self.data), msg="Input validation failed for valid data!"
        )

        # Test invalid data type (string instead of array)
        with self.assertRaises(TypeError):
            check_inputs(data="hello, world!")

        # Test invalid data dimensions (3D instead of 2D)
        with self.assertRaises(ValueError):
            check_inputs(data=np.zeros(shape=(100, 10, 48)))

    def test_dataset_normalization(self) -> None:
        """
        Tests the max-min normalization function for dataset scaling.

        Validates:
        - Proper scaling to [0, 1] range across different input scales
        - Correct handling of zero-input edge case
        - Numerical precision in normalization
        """
        for scale in [-10, -0.5, 0.5, 10]:
            # Create scaled input data
            inputs = self.data.copy() * scale

            # Apply max-min normalization
            outputs = dataset_max_min_normalization(data=inputs)

            # Validate minimum value normalization
            min_diff = np.allclose(a=np.min(outputs), b=0, atol=1e-3)
            self.assertTrue(min_diff, msg="Minimum value normalization error!")

            # Validate maximum value normalization
            max_diff = np.allclose(a=np.max(outputs), b=1, atol=1e-3)
            self.assertTrue(max_diff, msg="Maximum value normalization error!")

        # Test zero-input edge case
        zero_outputs = dataset_max_min_normalization(data=np.zeros_like(self.data))
        zero_diff = np.allclose(a=np.sum(zero_outputs), b=0, atol=1e-3)
        self.assertTrue(zero_diff, msg="Zero-input normalization error!")

    def test_time_series_to_distribution(self) -> None:
        """
        Tests the conversion of multivariate time series to distribution parameters.

        Validates:
        - Correct output types (numpy arrays)
        - Proper dimensionality of mean vector and covariance matrix
        - Consistency with input data dimensions
        """
        # Convert time series to distribution parameters
        mean_vector, cov_matrix = time_series_to_distribution(data=self.data)

        # Validate output types
        self.assertIsInstance(mean_vector, np.ndarray, msg="Mean vector type error!")
        self.assertIsInstance(
            cov_matrix, np.ndarray, msg="Covariance matrix type error!"
        )

        # Validate output dimensions
        self.assertEqual(
            first=len(mean_vector),
            second=self.n_length,
            msg="Mean vector dimension mismatch!",
        )
        self.assertEqual(
            first=cov_matrix.shape,
            second=(self.n_length, self.n_length),
            msg="Covariance matrix dimension mismatch!",
        )

    def test_wasserstein_distance(self) -> None:
        """
        Tests the core Wasserstein distance calculation function.

        Validates:
        - Correct output type (float)
        - Identity property (distance to self should be zero)
        - Numerical stability
        """
        # Create test datasets
        x = np.random.randn(self.n_samples, self.n_length)
        y = np.random.randn(self.n_samples, self.n_length)

        # Calculate distance between different datasets
        outputs = wasserstein_distance(x=x, y=y)
        self.assertIsInstance(outputs, float, msg="Distance output type error!")

        # Validate identity property (distance to self is zero)
        outputs = wasserstein_distance(x=x, y=x)
        zero_diff = np.allclose(a=outputs, b=0, atol=1e-3)
        self.assertTrue(zero_diff, msg="Non-zero distance for identical inputs!")

    def test_return_all(self) -> None:
        """
        Tests the extended return functionality of Wasserstein distance.

        Validates:
        - Multiple return values when return_all=True
        - Correct types for all return values
        - Consistency between return values
        """
        # Create test datasets
        x = np.random.randn(self.n_samples, self.n_length)
        y = np.random.randn(self.n_samples, self.n_length)

        # Calculate distance with extended return
        distance, mean_value, covar_value = wasserstein_distance(
            x=x, y=y, return_all=True
        )

        # Validate return types
        self.assertIsInstance(distance, float, msg="Distance return type error!")
        self.assertIsInstance(mean_value, float, msg="Mean value return type error!")
        self.assertIsInstance(covar_value, float, msg="Covariance return type error!")

    def test_wasserstein_distance_matrix(self) -> None:
        """Test the calculation of Wasserstein distance matrix between multiple time series data sets"""
        # Create the list of test datasets
        datasets = [np.random.randn(self.n_samples, self.n_length) for _ in range(10)]

        # Get the distance matrix
        distance_matrix = wasserstein_distance_matrix(datasets)

        # Test the data type of the matrix
        self.assertIsInstance(
            distance_matrix, np.ndarray, msg="Distance matrix type error!"
        )

        # Test the symmetric matrix
        for i in range(10):
            for j in range(10):
                self.assertEqual(
                    first=distance_matrix[i, j],
                    second=distance_matrix[j, i],
                    msg="Distance matrix mismatch!",
                )

    def test_plot_wasserstein_heatmap(self) -> None:
        """Functions for testing matrices for visualization"""
        # Creating a matrix for visualization
        matrix = np.random.randn(10, 10)

        # Create a list of dataset indices
        data_list = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]

        for i in range(len(data_list)):
            data_list[i] = data_list[i] * 3

        # Execute visualization algorithm
        plot_wasserstein_heatmap(
            matrix, data_list, figsize=(8, 6), dpi=200, fontsize=10, cmap="Blues"
        )


if __name__ == "__main__":
    unittest.main()
