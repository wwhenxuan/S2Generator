# -*- coding: utf-8 -*-
"""
Cointegration coupling mechanism.

This mechanism creates variates with shared stochastic trends (random walks) and
stationary deviations, reproducing long-run equilibrium relationships between
nonstationary variates:

    x_j(t) = sum_k beta_{jk} * mu_k(t) + epsilon_j(t)

where mu_k(t) are shared random-walk trend components and epsilon_j(t) are
stationary AR(1) residuals.

Reference:
    Podest, P., et al. (2026). TiRex-2: Generalizing TiRex to Multivariate Data
    and Streaming. arXiv:2607.01204v1, Section 3.4, mechanism 4 & Appendix F.

Created on 2026/08/10 00:00:00
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/S2Generator
"""

from typing import Optional

import numpy as np

from .base_coupling import BaseCoupling


class Cointegration(BaseCoupling):
    """Cointegration coupling: shared random-walk trends with stationary AR(1) residuals.

    This mechanism reproduces long-run equilibria between nonstationary variates.
    Individual series may drift without bound while specific linear combinations
    remain stationary.
    """

    min_variates = 1
    min_length = 3

    def __init__(
        self,
        dtype: np.dtype = np.float64,
        min_trends: int = 1,
        max_trends: Optional[int] = None,
        ar_coeff_range: tuple = (-0.8, 0.8),
        trend_noise_std: float = 0.1,
        residual_noise_std: float = 0.2,
    ) -> None:
        """Initialize the cointegration coupling mechanism.

        :param dtype: The numpy data type for generated data.
        :param min_trends: Minimum number of shared random-walk trends.
        :param max_trends: Maximum number of shared trends.
                          If None, set to half the number of variates.
        :param ar_coeff_range: Range (min, max) for AR(1) coefficient sampling.
        :param trend_noise_std: Standard deviation of trend innovations.
        :param residual_noise_std: Standard deviation of residual innovations.
        """
        super().__init__(dtype=dtype)
        self._min_trends = min_trends
        self._max_trends = max_trends
        self._ar_coeff_range = ar_coeff_range
        self._trend_noise_std = trend_noise_std
        self._residual_noise_std = residual_noise_std

    def __str__(self) -> str:
        return "Cointegration"

    @property
    def min_trends(self) -> int:
        """Get the minimum number of trends."""
        return self._min_trends

    @property
    def max_trends(self) -> Optional[int]:
        """Get the maximum number of trends."""
        return self._max_trends

    def couple(
        self,
        rng: np.random.RandomState,
        series: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Apply cointegration coupling to the input series.

        The output variates share random-walk trends with stationary AR(1)
        deviations, mimicking cointegrated economic time series.

        :param rng: The random number generator with fixed seed.
        :param series: Input univariate series of shape (T, Q).
                      These serve as a source of randomness but are not directly
                      used in the output (the cointegration structure is
                      generated from scratch).
        :param kwargs: Additional parameters.
        :return: Cointegrated multivariate series of shape (T, Q).
        """
        self._validate_series(series, min_variates=self.min_variates, min_length=self.min_length)

        T, Q = series.shape
        max_t = (
            self._max_trends
            if self._max_trends is not None
            else max(1, Q // 2)
        )
        # Clamp so that 1 <= min_trends <= max_trends <= Q, keeping randint valid.
        max_t = max(1, min(max_t, Q))
        min_t = min(self._min_trends, max_t)
        n_trends = rng.randint(min_t, max_t + 1)

        # Generate shared random-walk trend components
        trends = self._generate_trends(rng, T, n_trends)

        # Generate stationary AR(1) residuals for each variate
        residuals = self._generate_residuals(rng, T, Q)

        # Generate loading matrix (how each trend loads onto each variate)
        loadings = rng.uniform(-1.5, 1.5, (Q, n_trends))

        # Combine trends and residuals
        result = trends @ loadings.T + residuals

        return result.astype(self._data_type, copy=False)

    def _generate_trends(
        self,
        rng: np.random.RandomState,
        T: int,
        n_trends: int,
    ) -> np.ndarray:
        """Generate shared random-walk trend components.

        :param rng: The random number generator.
        :param T: Length of the time series.
        :param n_trends: Number of independent trend components.
        :return: Array of shape (T, n_trends) with random-walk trends.
        """
        trends = np.zeros((T, n_trends), dtype=self._data_type)
        innovations = rng.normal(0, self._trend_noise_std, (T, n_trends))
        # Random walk: cumulative sum of innovations
        trends = np.cumsum(innovations, axis=0)
        return trends

    def _generate_residuals(
        self,
        rng: np.random.RandomState,
        T: int,
        Q: int,
    ) -> np.ndarray:
        """Generate stationary AR(1) residual processes for each variate.

        :param rng: The random number generator.
        :param T: Length of the time series.
        :param Q: Number of variates.
        :return: Array of shape (T, Q) with AR(1) residuals.
        """
        residuals = np.zeros((T, Q), dtype=self._data_type)
        ar_min, ar_max = self._ar_coeff_range

        for j in range(Q):
            phi = rng.uniform(ar_min, ar_max)
            noise = rng.normal(0, self._residual_noise_std, T)
            # Generate AR(1) process
            for t in range(1, T):
                residuals[t, j] = phi * residuals[t - 1, j] + noise[t]

        return residuals
