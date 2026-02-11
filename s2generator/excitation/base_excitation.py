# -*- coding: utf-8 -*-
"""
This file is mainly used to build a unified stimulus time series generation interface module.
We specify the general parameters of data generation through the abstract class.
Then, we specify the `generate` method to generate specific data through the abstract method.

Created on 2025/08/11 09:34:54
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/S2Generator
"""
import numpy as np
from abc import ABC, abstractmethod


class BaseExcitation(ABC):
    """Base class for generating stimulus time series data"""

    def __init__(self, dtype: np.dtype = np.float64) -> None:
        self.data_type = dtype

    def __str__(self) -> str:
        return self.__class__.__name__

    def create_zeros(
        self, n_inputs_points: int = 512, input_dimension: int = 1
    ) -> np.ndarray:
        """
        Constructs an empty time series data of the specified length and dimension.

        :param n_inputs_points: The length of the generated time series data.
        :param input_dimension: The dimension of the generated time series data.
        :return: The zeros time series with the specified dimension and length.
        """
        return np.zeros(shape=(n_inputs_points, input_dimension), dtype=self.data_type)

    @property
    def dtype(self) -> np.dtype:
        """Get the current data type"""
        return self.data_type

    @abstractmethod
    def generate(
        self, rng: np.random.RandomState, n_inputs_points: int = 512, input_dimension=1
    ) -> np.ndarray:
        """Generate a unified interface for time series data"""
