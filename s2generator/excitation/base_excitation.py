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

    def create_zeros(self, seq_length: int = 512, num_channels: int = 1) -> np.ndarray:
        """
        Constructs an empty time series data of the specified length and dimension.

        :param seq_length: The length of the generated time series data.
        :param num_channels: The dimension of the generated time series data.
        :return: The zeros time series with the specified dimension and length.
        """
        return np.zeros(shape=(seq_length, num_channels), dtype=self.data_type)

    @property
    def dtype(self) -> np.dtype:
        """Get the current data type"""
        return self.data_type

    @abstractmethod
    def generate(
        self, rng: np.random.RandomState, seq_length: int = 512, num_channels: int = 1
    ) -> np.ndarray:
        """Generate a unified interface for time series data"""
