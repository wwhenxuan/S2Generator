# -*- coding: utf-8 -*-
"""
Created on 2025/08/13 21:48:34
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""

import numpy as np
from typing import Optional, Dict
from s2generator.excitation.base_excitation import BaseExcitation


def arma_series(
    rng: np.random.RandomState,
    time_series: np.ndarray,
    p_params: np.ndarray,
    q_params: np.ndarray,
) -> np.ndarray:
    """
    Generate an ARMA process based on the specified parameters.

    :param rng: Random number generator of NumPy with fixed seed.
    :param time_series: The zeros time series.
    :param p_params: The parameters of the AR(p) process.
    :param q_params: The parameters of the MA(q) process.
    """
    # TODO: 这里的参数控制需要进一步的调整
    for index in range(len(time_series)):
        # Get the previous p AR values
        index_p = max(0, index - len(p_params))
        p_vector = np.flip(time_series[index_p:index])

        # Compute the dot product of p values and model parameters
        p_value = np.dot(p_vector, p_params[0 : len(p_vector)])

        # Generate q values through a white noise sequence
        q_value = np.dot(rng.randn(len(q_params)), q_params)

        sum_value = p_value + rng.randn(1) + q_value
        if sum_value > 1024:
            sum_value = q_value
        time_series[index] = sum_value
    return time_series


class AutoregressiveMovingAverage(BaseExcitation):
    """Generate motivating time series data by constructing random parameterized moving average and autoregressive."""

    def __init__(
        self,
        p_min: Optional[int] = 1,
        p_max: Optional[int] = 3,
        q_min: Optional[int] = 1,
        q_max: Optional[int] = 5,
        upper_bound: float = 512,
        dtype: np.dtype = np.float64,
    ) -> None:
        """
        :param p_min: Minimum value for the AR(p) process.
        :param p_max: Maximum value for the AR(p) process.
        :param q_min: Minimum value for the MA(q) process.
        :param q_max: Maximum value for the MA(q) process.
        :param upper_bound: The upper bound number of the ARMA process.
        :param dtype: The data type of NumPy in ARMA process.
        """
        super().__init__(dtype=dtype)

        # The min and max order of AR(p) and MA(q)
        self.p_min = p_min
        self.p_max = p_max
        self.q_min = q_min
        self.q_max = q_max

        # Save the order and params in AMRA
        self.p_order, self.q_order = None, None
        self.p_params, self.q_params = None, None

        # The value upper bound
        self.upper_bound = upper_bound

    def __call__(
        self,
        rng: np.random.RandomState,
        n_inputs_points: int = 512,
        input_dimension: int = 1,
    ) -> np.ndarray:
        """Call the `generate` method to stimulate time series generation"""
        return self.generate(
            rng=rng, n_inputs_points=n_inputs_points, input_dimension=input_dimension
        )

    def __str__(self) -> str:
        """Get the name of the time series generator"""
        return "ARMA"

    @staticmethod
    def create_autoregressive_params(
        rng: np.random.RandomState, p_order: int
    ) -> np.ndarray:
        """
        Constructing the model parameters for the autoregressive process.
        Because the autoregressive process utilizes historical information accumulated in the time series,
        it can easily lead to numerical explosion in the generated stimulus time series.
        To ensure that the autoregressive process can stably generate a stationary time series,
        we impose certain constraints on the parameters of the autoregressive process.

        1. The absolute value of the last parameter (i.e., :math:`\\varphi_p`) is less than 1: :math:`|\\varphi_p| < 1`
        2. The sum of all parameters is less than 1: :math:`\\varphi_1 + \\varphi_2 + \dots + \\varphi_p < 1`

        :param rng: The random number generator of NumPy with fixed seed.
        :param p_order: The order of the autoregressive process.
        :return: The autoregressive parameters.
        """
        # Generate the last params first
        p_last = rng.uniform(low=-0.99, high=0.99)

        # Generate other params
        p_params = np.append(rng.uniform(low=-1.0, high=1.0, size=p_order - 1), p_last)

        # Calculate the sum of parameters
        total = np.sum(p_params)

        # Scale the parameters so that the sum is < 1 (while keeping |φ_p| < 1)
        if total >= 1:
            scale_factor = 0.95 / (
                total + 0.1
            )  # Make sure the sum is < 1 after scaling
            p_params *= scale_factor

        return p_params

    @staticmethod
    def create_moving_average_params(
        rng: np.random.RandomState, q_order: int
    ) -> np.ndarray:
        """
        Constructing model parameters of the sliding average process.

        :param rng: The random number generator of NumPy with fixed seed.
        :param q_order: The order of the moving average process.
        :return: The moving average parameters.
        """
        return rng.uniform(low=-1.0, high=1.0, size=q_order)

    @property
    def order(self) -> Dict[str, int]:
        """Get the order of the autoregressive process and the moving average process in the ARMA model."""
        return {"AR(p)": self.p_order, "MA(q)": self.q_order}

    @property
    def params(self) -> Dict[str, np.ndarray]:
        """Get the parameters of the autoregressive process and the moving average process in the ARMA model."""
        return {"AR(p)": self.p_params, "MA(q)": self.q_params}

    def arma_series(
        self,
        rng: np.random.RandomState,
        time_series: np.ndarray,
        p_params: Optional[np.ndarray] = None,
        q_params: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate an ARMA process based on the specified parameters.

        :param rng: Random number generator of NumPy with fixed seed.
        :param time_series: The zeros time series.
        :param p_params: The parameters of the AR(p) process.
        :param q_params: The parameters of the MA(q) process.
        """
        return arma_series(
            rng=rng,
            time_series=time_series,
            p_params=self.p_params if p_params is None else p_params,
            q_params=self.q_params if q_params is None else q_params,
        )

    def create_params(self, rng: np.random.RandomState) -> None:
        """
        Constructing parameters for moving average autoregressive time series data.

        :param rng: The random number generator of NumPy with fixed seed.
        :return: None.
        """
        # First, randomly generate the order of the model
        self.p_order = rng.randint(low=self.p_min, high=self.p_max)
        self.q_order = rng.randint(low=self.q_min, high=self.q_max)

        # Generate the parameters of AR(p)
        self.p_params = self.create_autoregressive_params(rng=rng, p_order=self.p_order)

        # Generate the parameters of MA(q)
        self.q_params = self.create_moving_average_params(rng=rng, q_order=self.q_order)

    def generate(
        self,
        rng: np.random.RandomState,
        n_inputs_points: int = 512,
        input_dimension: int = 1,
    ) -> np.ndarray:
        """
        Generate ARMA stationary time series based on the specified input points and dimensions.

        :param rng: The random number generator of NumPy with fixed seed.
        :param n_inputs_points: The number of input points.
        :param input_dimension: The dimension of the time series.
        :return: The generated ARMA time series.
        """
        # Generate all zero time series data
        time_series = self.create_zeros(
            n_inputs_points=n_inputs_points, input_dimension=input_dimension
        )

        # Generate clusters with numerical explosion through a while loop
        index = 0
        while index < input_dimension:
            # Randomly generate model parameters
            self.create_params(rng=rng)

            # Generate the AMRA series
            arma = self.arma_series(
                rng=rng,
                time_series=time_series[:, index],
                p_params=self.p_params,
                q_params=self.q_params,
            )

            # Check the upper bound
            if np.max(np.abs(arma)) <= self.upper_bound:
                time_series[:, index] = arma
                # Remove the pointer
                index += 1

        return time_series


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    arma = AutoregressiveMovingAverage()

    for i in range(10):
        rng = np.random.RandomState(i)

        time = arma.generate(rng=rng, n_inputs_points=256)
        plt.plot(time)
        plt.show()
