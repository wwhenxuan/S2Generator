# -*- coding: utf-8 -*-
"""
Created on 2025/08/14 20:47:25
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/S2Generator
"""
import functools

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    DotProduct,
    ExpSineSquared,
    Kernel,
    RationalQuadratic,
    WhiteKernel,
)
from typing import Union, Optional, Any, Callable, List

from S2Generator.excitation.base_excitation import BaseExcitation


def get_exp_sine_squared(length: Optional[int] = 256) -> List[Kernel]:
    """
    Creates periodic kernels with various temporal patterns for time series modeling.

    Generates ExpSineSquared kernels covering multiple time frequencies:
    - Hourly (H), Half-hourly (0.5H), Quarter-hourly (0.25H)
    - Daily (D), Half-day (0.5D)
    - Weekly (W)
    - Monthly (M)
    - Quarterly (Q)
    - Yearly (Y)

    Periodicity parameters are scaled by input length to maintain consistent
    temporal relationships across different time series lengths.

    :param length: Reference length for periodicity scaling. Default assumes 256 time points.
    :type length: Optional[int]
    :return: List of periodic kernels with diverse temporal patterns
    :rtype: List[Kernel]
    """
    return [
        ExpSineSquared(periodicity=24 / length),  # Hourly cycle
        ExpSineSquared(periodicity=48 / length),  # Half-hourly cycle
        ExpSineSquared(periodicity=96 / length),  # Quarter-hourly cycle
        ExpSineSquared(periodicity=24 * 7 / length),  # Weekly hourly pattern
        ExpSineSquared(periodicity=48 * 7 / length),  # Weekly half-hourly pattern
        ExpSineSquared(periodicity=96 * 7 / length),  # Weekly quarter-hourly pattern
        ExpSineSquared(periodicity=7 / length),  # Daily cycle (week perspective)
        ExpSineSquared(periodicity=14 / length),  # Half-day cycle
        ExpSineSquared(periodicity=30 / length),  # Monthly daily pattern
        ExpSineSquared(periodicity=60 / length),  # Bi-monthly daily pattern
        ExpSineSquared(periodicity=365 / length),  # Yearly daily pattern
        ExpSineSquared(periodicity=365 * 2 / length),  # Biannual pattern
        ExpSineSquared(periodicity=4 / length),  # Weekly pattern (month perspective)
        ExpSineSquared(periodicity=26 / length),  # Bi-weekly pattern
        ExpSineSquared(periodicity=52 / length),  # Quarterly weekly pattern
        ExpSineSquared(periodicity=4 / length),  # Monthly pattern
        ExpSineSquared(periodicity=6 / length),  # Bimonthly pattern
        ExpSineSquared(periodicity=12 / length),  # Quarterly pattern
        ExpSineSquared(periodicity=4 / length),  # Quarterly pattern (year perspective)
        ExpSineSquared(periodicity=4 * 10 / length),  # Decade quarterly pattern
        ExpSineSquared(periodicity=10 / length),  # Yearly pattern
    ]


def get_dot_product(length: Optional[int] = 256) -> List[Kernel]:
    """
    Creates linear regression kernels with different intercept handling.

    The DotProduct kernel models linear relationships: k(x_i, x_j) = σ₀² + x_i · x_j

    :param length: Unused parameter (maintained for interface consistency)
    :type length: Optional[int]
    :return: Linear kernels with different intercept magnitudes
    :rtype: List[Kernel]
    """
    return [
        DotProduct(sigma_0=0.0),  # Linear kernel without intercept
        DotProduct(sigma_0=1.0),  # Linear kernel with unit intercept
        DotProduct(sigma_0=10.0),  # Linear kernel with large intercept
    ]


def get_rbf(length: Optional[int] = 256) -> List[Kernel]:
    """
    Creates Radial Basis Function (RBF) kernels with different length scales.

    The RBF kernel models smooth variations: k(x_i, x_j) = exp(-||x_i - x_j||²/(2l²))

    :param length: Unused parameter (maintained for interface consistency)
    :type length: Optional[int]
    :return: RBF kernels with varying smoothness scales
    :rtype: List[Kernel]
    """
    return [
        RBF(length_scale=0.1),  # Short-scale variations
        RBF(length_scale=1.0),  # Medium-scale variations
        RBF(length_scale=10.0),  # Long-scale variations
    ]


def get_rational_quadratic(length: Optional[int] = 256) -> List[Kernel]:
    """
    Creates Rational Quadratic kernels modeling multi-scale variations.

    Combines characteristics of multiple RBF kernels:
    k(x_i, x_j) = (1 + ||x_i - x_j||²/(2αl²))^(-α)

    :param length: Unused parameter (maintained for interface consistency)
    :type length: Optional[int]
    :return: RQ kernels with different scale mixture parameters
    :rtype: List[Kernel]
    """
    return [
        RationalQuadratic(alpha=0.1),  # Primarily short-scale variations
        RationalQuadratic(alpha=1.0),  # Balanced multi-scale variations
        RationalQuadratic(alpha=10.0),  # Primarily long-scale variations
    ]


def get_white_kernel(length: Optional[int] = 256) -> List[Kernel]:
    """
    Creates White Noise kernels modeling uncorrelated variations.

    Models independent noise: k(x_i, x_j) = noise_level if i == j else 0

    :param length: Unused parameter (maintained for interface consistency)
    :type length: Optional[int]
    :return: White noise kernels with different noise magnitudes
    :rtype: List[Kernel]
    """
    return [
        WhiteKernel(noise_level=0.1),  # Low noise
        WhiteKernel(noise_level=1.0),  # Medium noise
        WhiteKernel(noise_level=2.0),  # High noise
    ]


def get_constant_kernel(length: Optional[int] = 256) -> List[Kernel]:
    """
    Creates a Constant kernel modeling fixed offsets.

    Models constant values: k(x_i, x_j) = constant_value

    :param length: Unused parameter (maintained for interface consistency)
    :type length: Optional[int]
    :return: Constant kernel
    :rtype: List[Kernel]
    """
    return [ConstantKernel()]  # Models constant mean function


def random_binary_map(a: Kernel, b: Kernel) -> np.ndarray:
    """
    Applies a random binary operator (+ or \*) with equal probability
    on kernels ``a`` and ``b``.

    :param a: A GP kernel
    :param b: A GP kernel
    :return: The composite kernel `a + b` or `a * b`.
    """
    binary_maps = [lambda x, y: x + y, lambda x, y: x * y]
    return np.random.choice(binary_maps)(a, b)


def sample_from_gp_prior(
    kernel: Kernel, time_series: np.ndarray, random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Draw a sample from a GP prior.

    :param kernel: The GP covaraince kernel
    :param time_series: The input "time" points
    :param random_seed: The random seed for sampling, by default None
    :return: A time series sampled from the GP prior
    """
    if time_series.ndim == 1:
        time_series = time_series[:, None]

    assert time_series.ndim == 2
    gpr = GaussianProcessRegressor(kernel=kernel)
    ts = gpr.sample_y(time_series, n_samples=1, random_state=random_seed)

    return ts


def sample_from_gp_prior_efficient(
    kernel: Kernel,
    time_series: np.ndarray,
    random_seed: Optional[int] = None,
    method: str = "eigh",
) -> np.ndarray:
    """
    Draw a sample from a GP prior. An efficient version that allows specification
    of the sampling method. The default sampling method used in GaussianProcessRegressor
    is based on SVD which is significantly slower that alternatives such as `eigh` and
    `cholesky`.

    :param kernel: The GP covaraince kernel
    :param time_series: The input "time" points
    :param random_seed: The random seed for sampling, by default None
    :param method: The sampling method for multivariate_normal, by default `eigh`
    :return: A time series sampled from the GP prior
    """
    if time_series.ndim == 1:
        time_series = time_series[:, None]

    assert time_series.ndim == 2

    cov = kernel(time_series)
    ts = np.random.default_rng(seed=random_seed).multivariate_normal(
        mean=np.zeros(time_series.shape[0]), cov=cov, method=method
    )

    return ts


class KernelSynth(BaseExcitation):
    """Generate a synthetic time series from KernelSynth."""

    def __init__(
        self,
        min_kernels: Optional[int] = 1,
        max_kernels: Optional[int] = 5,
        exp_sine_squared: Optional[bool] = True,
        dot_product: Optional[bool] = True,
        rbf: Optional[bool] = True,
        rational_quadratic: Optional[bool] = True,
        white_kernel: Optional[bool] = True,
        constant_kernel: Optional[bool] = True,
        dtype: np.dtype = np.float64,
    ) -> None:
        """
        :param min_kernels: The minimum number of kernels to use.
        :param max_kernels: Max number of kernels for the input distribution in KernelSynth methods.
        :param exp_sine_squared: Boolean, whether to use the exponential square root of kernel.
        :param dot_product: Boolean, whether to use the dot product of kernel length and kernel.
        :param rbf: Boolean, whether to use the RBF kernel.
        :param rational_quadratic: Boolean, whether to use the rational quadratic kernel.
        :param white_kernel: Boolean, whether to use the white kernel.
        :param constant_kernel: Boolean, whether to use the constant kernel.
        :param dtype: The data type for the generated time series.
        """
        super().__init__(dtype=dtype)

        # The maximum and minimum number of cores to use
        # when generating time series
        self.min_kernels = min_kernels
        self.max_kernels = max_kernels

        # Whether to use certain filters
        self.exp_sine_squared = exp_sine_squared
        self.dot_product = dot_product
        self.rbf = rbf
        self.rational_quadratic = rational_quadratic
        self.white_kernel = white_kernel
        self.constant_kernel = constant_kernel

        # Select the data generation kernel that can be used based on user input
        self.bank_list = self.choice_bank_list

        # Store the current sampling length
        self.length = None

        # Record the various core libraries currently stored
        self._kernel_bank = None

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
        return "KernelSynth"

    @property
    def choice_bank_list(self) -> List[Callable]:
        bank_list = []
        if self.exp_sine_squared:
            bank_list.append(get_exp_sine_squared)
        if self.dot_product:
            bank_list.append(get_dot_product)
        if self.rbf:
            bank_list.append(get_rbf)
        if self.rational_quadratic:
            bank_list.append(get_rational_quadratic)
        if self.white_kernel:
            bank_list.append(get_white_kernel)
        if self.constant_kernel:
            bank_list.append(get_constant_kernel)

        return bank_list

    def set_length(self, length: int) -> None:
        """
        Sets the new length for the KernelSynth data generation object and updates kernel_bank.

        :param length: The length of the time series to be generated.
        :return: None
        """
        # Set a new length
        self.length = length

        # Update the kernel bank
        self._kernel_bank = self._update_kernel_bank(length=length)

    def _update_kernel_bank(self, length: Optional[int] = None) -> List[Kernel]:
        """
        Get and update all kernel in the bank list with inputs length.

        :param length: The length of the time series to be generated.
        :return: The updated kernel list.
        """

        # If the length is not specified, the length in the class attribute is used.
        length = self.length if length is None else length

        # Create a kernel bank with the new length
        # kernel_bank = [
        #     kernel_function(length=length) for kernel_function in self.bank_list
        # ]
        kernel_bank = []
        for kernel_function in self.bank_list:
            kernel_bank += kernel_function(length=length)

        return kernel_bank

    @property
    def kernel_bank(self) -> List:
        """External interface for accessing kernel_bank private properties"""
        if self._kernel_bank is None:
            # Unable to get kernel bank when `set_length` or `generate` method has not been executed
            raise ValueError
        return self._kernel_bank

    def generate_kernel_synth(
        self, rng: np.random.RandomState, length: Optional[int] = 256
    ) -> Union[np.ndarray[Any, Union[np.dtype[Any], Any]], None]:
        """
        Generate a synthetic time series from KernelSynth.

        :param rng: Random Number Generator
        :param length: The length of the time series, by default 256
        :return: A time series generated by KernelSynth
        """
        while True:
            time_series = np.linspace(0, 1, length)

            # Randomly select upto max_kernels kernels from the KERNEL_BANK
            selected_kernels = rng.choice(
                self.kernel_bank,
                rng.randint(self.min_kernels, self.max_kernels + 1),
                replace=True,
            )

            # Combine the sampled kernels using random binary operators
            kernel = functools.reduce(random_binary_map, selected_kernels)

            # Sample a time series from the GP prior
            try:
                ts = sample_from_gp_prior(kernel=kernel, time_series=time_series)
            except np.linalg.LinAlgError as err:
                print("Error caught:", err)
                continue

            # The timestamp is arbitrary
            return ts.squeeze()

    def generate(
        self, rng: np.random.RandomState, n_inputs_points: int = 512, input_dimension=1
    ) -> np.ndarray:
        """
        Generate a time series from KernelSynth, which comes from Chronos.

        :param rng: The random number generator in NumPy with fixed sedd.
        :param n_inputs_points: The number of points in the input distribution.
        :param input_dimension: The dimensionality of the input distribution.
        :return: The time series generated by KernelSynth.
        """
        # If the length of the input changes, the kernel bank needs to be adjusted.
        if self.length != n_inputs_points:
            # Record the first length information or update the latest length
            self.length = n_inputs_points
            # Create the latest library
            self._kernel_bank = self._update_kernel_bank(length=n_inputs_points)

        return np.vstack(
            [
                self.generate_kernel_synth(rng=rng, length=n_inputs_points)
                for _ in range(input_dimension)
            ]
        ).T
