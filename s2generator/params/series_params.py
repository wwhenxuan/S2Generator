# -*- coding: utf-8 -*-
"""
This module mainly stores the parameters used to control the generation of the stimulus time series.

Created on 2025/08/19 11:06:31
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/S2Generator
"""
import numpy as np

from typing import Optional, Dict, List


class SeriesParams(object):
    """Parameter Control The Generation of Excitation Time Series in S2 (Series-Symbol) Data Generation"""

    def __init__(
        self,
        # Management and control of multiple sampling methods
        mixed_distribution: Optional[float] = 0.10,
        autoregressive_moving_average: Optional[float] = 0.20,
        forecast_pfn: Optional[float] = 0.35,
        kernel_synth: Optional[float] = 0.25,
        intrinsic_mode_function: Optional[float] = 0.10,
        # Parameters controlling stimulus generation for sampling from mixture distributions
        min_centroids: Optional[int] = 3,
        max_centroids: Optional[int] = 8,
        rotate: Optional[bool] = False,
        gaussian: Optional[bool] = True,
        uniform: Optional[bool] = True,
        mixed_distribution_dict: Optional[Dict[str, float]] = None,
        mixed_distribution_list: Optional[List[float]] = None,
        # Parameters controlling the generation of excitations in the autoregressive moving average process
        p_min: Optional[int] = 1,
        p_max: Optional[int] = 3,
        q_min: Optional[int] = 1,
        q_max: Optional[int] = 5,
        upper_bound: float = 512,
        # Parameters controlling the time series scale combination of ForecastPFN
        is_sub_day: Optional[bool] = True,
        transition: Optional[bool] = True,
        start_time: Optional[str] = "1885-01-01",
        end_time: Optional[str] = None,
        random_walk: bool = False,
        # Controlling stimulus generation for kernel synth sampling
        min_kernels: Optional[int] = 1,
        max_kernels: Optional[int] = 5,
        exp_sine_squared: Optional[bool] = True,
        dot_product: Optional[bool] = True,
        rbf: Optional[bool] = True,
        rational_quadratic: Optional[bool] = True,
        white_kernel: Optional[bool] = True,
        constant_kernel: Optional[bool] = True,
        # Controlling the generation of eigenmode function combinations
        min_base_imfs: int = 1,
        max_base_imfs: int = 3,
        min_choice_imfs: int = 2,
        max_choice_imfs: int = 5,
        imfs_dict: Optional[Dict[str, float]] = None,
        imfs_list: Optional[List[float]] = None,
        min_duration: float = 0.5,
        max_duration: float = 10.0,
        min_amplitude: float = 0.01,
        max_amplitude: float = 10.0,
        min_frequency: float = 0.01,
        max_frequency: float = 8.0,
        noise_level: float = 0.01,
        # Types of management data generation
        dtype: np.dtype = np.float64,
    ):
        """
        :param mixed_distribution: The probability of generating using the mixed distribution sampling.
        :param autoregressive_moving_average: The probability of generating using the autoregressive moving average.
        :param forecast_pfn: The probability of generating using the forecast pattern.
        :param kernel_synth: The probability of generating using the kernel synthesis.
        :param intrinsic_mode_function: The probability of generating using the intrinsic mode function.
        :param min_centroids: Minimum number of distributions used in mixture distribution sampling.
        :param max_centroids: Maximum number of distributions used in mixture distribution sampling.
        :param rotate: Whether to rotate the generated distribution or not.
        :param gaussian: Whether to use Gaussian distribution in mixed distribution sampling.
        :param uniform: Whether to use uniform distribution in mixed distribution sampling.
        :param mixed_distribution_dict: A dictionary that determines the probability of a certain distribution being sampled in a mixed distribution.
        :param mixed_distribution_list: A list that determines the probability of a certain distribution being sampled from a mixed distribution.
        :param p_min: Minimum value for the AR(p) process in Autoregressive model.
        :param p_max: Maximum value for the AR(p) process in Autoregressive model.
        :param q_min: Minimum value for the MA(q) process in Moving Average model.
        :param q_max: Maximum value for the MA(q) process in Moving Average model.
        :param upper_bound: Upper bound for the ARMA(p, q) process.
        :param is_sub_day: Enable sub-daily frequency components (minutes/hours).
        :param transition: Enable smooth transitions between frequency components.
        :param start_time: Start timestamp for generated series (ISO format: YYYY-MM-DD).
        :param end_time: End timestamp for generated series. Uses current date if None.
        :param random_walk: Enable random walk transformation for non-stationary series.
        :param min_kernels: The minimum number of kernels to use.
        :param max_kernels: Max number of kernels for the input distribution in KernelSynth methods.
        :param exp_sine_squared: Boolean, whether to use the exponential square root of kernel.
        :param dot_product: Boolean, whether to use the dot product of kernel length and kernel.
        :param rbf: Boolean, whether to use the RBF kernel.
        :param rational_quadratic: Boolean, whether to use the rational quadratic kernel.
        :param white_kernel: Boolean, whether to use the white kernel.
        :param constant_kernel: Boolean, whether to use the constant kernel.
        :param min_base_imfs: Minimum number of fundamental IMF components to generate.
        :param max_base_imfs: Maximum number of fundamental IMF components to generate.
        :param min_choice_imfs: Minimum number of IntrinsicModeFunction to select from available types.
        :param max_choice_imfs: Maximum number of IntrinsicModeFunction to select from available types.
        :param probability_dict: Custom probability distribution for IMF types, with keys matching available IMF names.
        :param probability_list: Custom probability weights for IMF types in predefined order (overrides probability_dict).
        :param min_duration: Minimum signal duration in seconds.
        :param max_duration: Maximum signal duration in seconds.
        :param min_amplitude: Minimum amplitude for generated signal components.
        :param max_amplitude: Maximum amplitude for generated signal components.
        :param min_frequency: Minimum frequency for signal components (Hz).
        :param max_frequency: Maximum frequency for signal components (Hz).
        :param noise_level: Amplitude of Gaussian noise to add (relative to signal amplitude).
        """
        # Record the probability of generating using different incentive methods
        self._mixed_distribution = mixed_distribution
        self._autoregressive_moving_average = autoregressive_moving_average
        self._forecast_pfn = forecast_pfn
        self._kernel_synth = kernel_synth
        self._intrinsic_mode_function = intrinsic_mode_function

        # Control the probability array of various sampling methods and check the probability value of the sampling method
        # After calling `_check_excitation_methods` it will become an ndarray array
        self._prob_array: np.ndarray = self._check_excitation_methods()

        # Parameters controlling stimulus generation for sampling from mixture distributions
        self.min_centroids = min_centroids
        self.max_centroids = max_centroids
        self.rotate = rotate
        self.gaussian = gaussian
        self.uniform = uniform
        self.mixed_distribution_dict = mixed_distribution_dict
        self.mixed_distribution_list = mixed_distribution_list

        # Parameters controlling the generation of excitations in the autoregressive moving average process
        self.p_min = p_min
        self.p_max = p_max
        self.q_min = q_min
        self.q_max = q_max
        self.upper_bound = upper_bound

        # Parameters controlling the time series scale combination of ForecastPFN
        self.is_sub_day = is_sub_day
        self.transition = transition
        self.random_walk = random_walk
        self.start_time = start_time
        self.end_time = end_time
        self.random_walk = random_walk

        # Controlling stimulus generation for kernel synth sampling
        self.min_kernels = min_kernels
        self.max_kernels = max_kernels
        self.exp_sine_squared = exp_sine_squared
        self.dot_product = dot_product
        self.rbf = rbf
        self.rational_quadratic = rational_quadratic
        self.white_kernel = white_kernel
        self.constant_kernel = constant_kernel

        # Controlling the generation of eigenmode function combinations
        self.min_base_imfs = min_base_imfs
        self.max_base_imfs = max_base_imfs
        self.min_choice_imfs = min_choice_imfs
        self.max_choice_imfs = max_choice_imfs
        self.imfs_dict = imfs_dict
        self.imfs_list = imfs_list
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_amplitude = min_amplitude
        self.max_amplitude = max_amplitude
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self.noise_level = noise_level

        # Types of management data generation
        self.dtype = dtype

    def _check_excitation_methods(self) -> np.ndarray:
        """
        Check whether the probability values of different sampling methods of the input meet the
        basic requirements and redistribute the normalized probabilities.
        """
        # Determine whether the input data type is a numeric value
        for name, prob in zip(
            [
                "mixed_distribution",
                "autoregressive_moving_average",
                "forecast_pfn",
                "kernel_synth",
                "intrinsic_mode_function",
            ],
            [
                self._mixed_distribution,
                self._autoregressive_moving_average,
                self._forecast_pfn,
                self._kernel_synth,
                self._intrinsic_mode_function,
            ],
        ):
            # Detecting data types and formats
            if not isinstance(prob, float):
                raise TypeError(f"Probability of {name} must be a float!")

        # The case where the probability value is 0 can be ignored
        prob_array = np.array(
            [
                self._mixed_distribution,
                self._autoregressive_moving_average,
                self._forecast_pfn,
                self._kernel_synth,
                self._intrinsic_mode_function,
            ]
        )

        # Normalizes and returns the probability array of the sampling method of the stimulus time series
        return prob_array / np.sum(prob_array)

    @property
    def sampling_methods(self) -> List[str]:
        """Returns a list of various sampling methods"""
        return [
            "mixed_distribution",
            "autoregressive_moving_average",
            "forecast_pfn",
            "kernel_synth",
            "intrinsic_mode_function",
        ]

    @property
    def prob_array(self) -> np.ndarray:
        """Obtaining the sampling probability of time series data with different incentive methods."""
        if self._prob_array is None:
            # If the sampling probability in the class attribute has not yet been obtained, calculate it directly
            # But the if condition should not be called after instantiating the object
            prob_array = np.array(
                [
                    self._mixed_distribution,
                    self._autoregressive_moving_average,
                    self._forecast_pfn,
                    self._kernel_synth,
                    self._intrinsic_mode_function,
                ]
            )
        else:
            prob_array = self._prob_array

        return prob_array
