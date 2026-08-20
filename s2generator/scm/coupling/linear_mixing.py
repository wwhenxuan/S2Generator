# -*- coding: utf-8 -*-
"""
Linear mixing coupling mechanism.

This mechanism creates linearly mixed variates from independent sources:
    x(t) = A @ z(t)

where A is a mixing matrix whose singular-value spectrum is sampled from dominant,
uniform, or power-law regimes. This mimics shared latent drivers as in factor models.

Reference:
    Podest, P., et al. (2026). TiRex-2: Generalizing TiRex to Multivariate Data
    and Streaming. arXiv:2607.01204v1, Section 3.4, mechanism 3 & Appendix F.

Created on 2026/08/10 00:00:00
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/S2Generator
"""

from typing import Optional

import numpy as np

from .base_coupling import BaseCoupling


class LinearMixing(BaseCoupling):
    """Linear mixing: x(t) = A @ z(t).

    Observed series arise as linear combinations of a smaller number of
    underlying drivers. The induced correlation structure varies continuously
    with the spectrum of A.
    """

    # Available spectral regimes for the mixing matrix
    _SPECTRAL_REGIMES = ["dominant", "uniform", "power_law"]

    min_variates = 2
    min_length = 2

    def __init__(
        self,
        dtype: np.dtype = np.float64,
        spectral_regimes: Optional[list] = None,
    ) -> None:
        """Initialize the linear mixing coupling mechanism.

        :param dtype: The numpy data type for generated data.
        :param spectral_regimes: List of spectral regimes to sample from.
                                If None, all available regimes are used.
        """
        super().__init__(dtype=dtype)
        self._spectral_regimes = (
            spectral_regimes
            if spectral_regimes is not None
            else self._SPECTRAL_REGIMES
        )

    def __str__(self) -> str:
        return "LinearMixing"

    @property
    def spectral_regimes(self) -> list:
        """Get the available spectral regimes."""
        return self._spectral_regimes

    def couple(
        self,
        rng: np.random.RandomState,
        series: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Apply linear mixing to the input series.

        :param rng: The random number generator with fixed seed.
        :param series: Input univariate series of shape (T, Q).
        :param kwargs: Additional parameters.
        :return: Linearly mixed multivariate series of shape (T, Q).
        """
        self._validate_series(series, min_variates=self.min_variates, min_length=self.min_length)

        T, Q = series.shape
        regime = rng.choice(self._spectral_regimes)

        # Generate mixing matrix A with specified spectral regime
        A = self._generate_mixing_matrix(rng, Q, regime)

        # Apply mixing: x(t) = A @ z(t)
        result = (A @ series.T).T.astype(self._data_type, copy=False)

        return result

    def _generate_mixing_matrix(
        self,
        rng: np.random.RandomState,
        Q: int,
        regime: str,
    ) -> np.ndarray:
        """Generate a Q x Q mixing matrix with a specific singular-value spectrum.

        :param rng: The random number generator.
        :param Q: The dimension of the square mixing matrix.
        :param regime: The spectral regime ('dominant', 'uniform', 'power_law').
        :return: A Q x Q mixing matrix.
        """
        # Generate random orthogonal matrices for left and right singular vectors
        U = self._random_orthogonal(rng, Q)

        # Sample singular values according to the regime
        if regime == "dominant":
            # One dominant singular value, others small
            s = np.ones(Q)
            s[0] = Q  # Dominant component
            s[1:] = rng.uniform(0.01, 0.5, Q - 1)
            s = s / s.sum() * Q  # Normalize to trace = Q
        elif regime == "uniform":
            # Uniform singular values
            s = np.ones(Q)
        elif regime == "power_law":
            # Power-law decaying singular values
            exponent = rng.uniform(0.5, 2.0)
            s = 1.0 / (np.arange(1, Q + 1) ** exponent)
            s = s / s.sum() * Q
        else:
            s = np.ones(Q)

        # Construct mixing matrix: A = U @ diag(s) @ V^T
        # Use a random V as well for generality
        V = self._random_orthogonal(rng, Q)

        A = U @ np.diag(np.sqrt(np.maximum(s, 0.0))) @ V.T

        return A

    @staticmethod
    def _random_orthogonal(
        rng: np.random.RandomState, n: int
    ) -> np.ndarray:
        """Generate a random orthogonal matrix of size n x n.

        Uses QR decomposition of a random Gaussian matrix.

        :param rng: The random number generator.
        :param n: The size of the square matrix.
        :return: An n x n orthogonal matrix.
        """
        M = rng.randn(n, n)
        Q, R = np.linalg.qr(M)
        # Ensure determinant is +1 (no reflection)
        d = np.diag(np.sign(np.diag(R)))
        Q = Q @ d
        return Q
