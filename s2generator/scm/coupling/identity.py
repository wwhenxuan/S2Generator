# -*- coding: utf-8 -*-
"""
Identity / pass-through coupling mechanism.

This is the simplest coupling: each output variate is exactly the corresponding input series.
It preserves univariate forecasting ability as a no-coupling control, ensuring the model
does not become biased toward assuming cross-variate structure when none is present.

Reference:
    Podest, P., et al. (2026). TiRex-2: Generalizing TiRex to Multivariate Data
    and Streaming. arXiv:2607.01204v1, Section 3.4, mechanism 1.

Created on 2026/08/10 00:00:00
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/S2Generator
"""

import numpy as np

from .base_coupling import BaseCoupling


class IdentityCoupling(BaseCoupling):
    """Identity / pass-through coupling: x_j(t) = z_j(t).

    Each output variate is exactly the corresponding input series. This mechanism
    serves as a no-coupling control to preserve univariate forecasting performance.
    """

    def __init__(self, dtype: np.dtype = np.float64) -> None:
        """Initialize the identity coupling mechanism."""
        super().__init__(dtype=dtype)

    def __str__(self) -> str:
        return "IdentityCoupling"

    def couple(
        self,
        rng: np.random.RandomState,
        series: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Return the input series unchanged (identity mapping).

        :param rng: The random number generator (unused, kept for interface consistency).
        :param series: Input univariate series of shape (T, Q).
        :param kwargs: Additional parameters (unused).
        :return: The same series, unchanged, of shape (T, Q).
        """
        self._validate_series(
            series, min_variates=self.min_variates, min_length=self.min_length
        )
        return series.astype(self._data_type, copy=False)


class UnivariatePassThrough(BaseCoupling):
    """Univariate pass-through: a single univariate output from a single input.

    This is used when the coupling pipeline samples a univariate-only mode,
    producing a single output variate without any cross-variate structure.
    This is equivalent to the "univariate" pass-through case in TiRex-2.
    """

    def __init__(self, dtype: np.dtype = np.float64) -> None:
        """Initialize the univariate pass-through mechanism."""
        super().__init__(dtype=dtype)

    def __str__(self) -> str:
        return "UnivariatePassThrough"

    def couple(
        self,
        rng: np.random.RandomState,
        series: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Return only the first variate of the input series.

        :param rng: The random number generator (unused, kept for interface consistency).
        :param series: Input univariate series of shape (T, Q).
        :param kwargs: Additional parameters (unused).
        :return: The first column of the input, shape (T, 1).
        """
        self._validate_series(
            series, min_variates=self.min_variates, min_length=self.min_length
        )
        result = series[:, :1].astype(self._data_type, copy=True)
        return result
