# -*- coding: utf-8 -*-
"""
This module defines the abstract base class for synthetic multivariate coupling mechanisms.

A coupling mechanism transforms a set of independently generated univariate time series
into a jointly dependent multivariate sample. Each mechanism targets a distinct region of
cross-variate dependency space, as described in the TiRex-2 paper (Podest et al., 2026).

The base class follows the same pattern as `BaseExcitation` in the excitation module.

Created on 2026/08/10 00:00:00
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/S2Generator
"""

import numpy as np
from abc import ABC, abstractmethod


class BaseCoupling(ABC):
    """Abstract base class for multivariate coupling mechanisms.

    Each coupling mechanism receives Q univariate time series z_1, ..., z_Q of length T
    and produces Q output variates x_1, ..., x_Q with a known dependency structure.

    Reference:
        Podest, P., et al. (2026). TiRex-2: Generalizing TiRex to Multivariate Data
        and Streaming. arXiv:2607.01204v1.
    """

    # Minimum shape requirements, overridden by subclasses. These serve both as
    # input validation bounds and (for min_variates) as the criterion the
    # CouplingPipeline uses to filter candidate mechanisms for a given Q.
    min_variates: int = 1
    min_length: int = 1

    def __init__(self, dtype: np.dtype = np.float64) -> None:
        """
        Initialize the base coupling mechanism.

        :param dtype: The numpy data type for generated data.
        """
        self._data_type = dtype

    def __str__(self) -> str:
        """Return the name of the coupling mechanism."""
        return self.__class__.__name__

    def __call__(
        self,
        rng: np.random.RandomState,
        series: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Call the `couple` method to apply the coupling transformation.

        :param rng: The random number generator with fixed seed.
        :param series: Input univariate series of shape (T, Q).
        :param kwargs: Additional mechanism-specific parameters.
        :return: Coupled multivariate series of shape (T, Q).
        """
        return self.couple(rng=rng, series=series, **kwargs)

    @property
    def dtype(self) -> np.dtype:
        """Get the current data type."""
        return self._data_type

    @abstractmethod
    def couple(
        self,
        rng: np.random.RandomState,
        series: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Transform independent univariate series into jointly dependent variates.

        :param rng: The random number generator with fixed seed.
        :param series: Input univariate series of shape (T, Q), where T is the
                       sequence length and Q is the number of variates.
        :param kwargs: Additional mechanism-specific parameters.
        :return: Coupled multivariate series of shape (T, Q).
        """
        pass

    def create_zeros(
        self,
        seq_length: int = 512,
        num_channels: int = 1,
    ) -> np.ndarray:
        """Construct an empty time series array of the specified length and dimension.

        :param seq_length: The length of the generated time series data.
        :param num_channels: The dimension of the generated time series data.
        :return: The zeros time series with the specified dimension and length.
        """
        return np.zeros(
            shape=(seq_length, num_channels),
            dtype=self._data_type,
        )

    @staticmethod
    def _validate_series(
        series: np.ndarray,
        min_variates: int = 1,
        min_length: int = 1,
    ) -> None:
        """Validate the input series array.

        :param series: Input series array of shape (T, Q).
        :param min_variates: Minimum number of variates required.
        :param min_length: Minimum sequence length required.
        :raises ValueError: If the series does not meet the requirements.
        """
        if series.ndim != 2:
            raise ValueError(
                f"series must be a 2D array of shape (T, Q), "
                f"got shape {series.shape}"
            )
        if series.shape[0] < min_length:
            raise ValueError(
                f"series length ({series.shape[0]}) must be at least {min_length}"
            )
        if series.shape[1] < min_variates:
            raise ValueError(
                f"series dimension ({series.shape[1]}) must be at least {min_variates}"
            )
