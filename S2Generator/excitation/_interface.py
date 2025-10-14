# -*- coding: utf-8 -*-
"""
This module is used to build a unified interface for generating time series using various different incentives.
It also manages the allocation of specific parameters for various data generation mechanisms.

Created on 2025/08/18 23:31:37
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/S2Generator
"""
import numpy as np

from typing import Optional, Union, List, Dict, Any, Tuple

from S2Generator.params import SeriesParams
from S2Generator.excitation import (
    MixedDistribution,
    AutoregressiveMovingAverage,
    ForecastPFN,
    KernelSynth,
    IntrinsicModeFunction,
)
from S2Generator.utils import z_score_normalization, max_min_normalization


class Excitation(object):
    """
    A unified interface for generating time series data is constructed through integrated parameter objects.
    A SeriesParams object is required for unified parameter control and management.
    """

    def __init__(self, series_params: Optional[SeriesParams] = None) -> None:
        """
        :param series_params: Parameter object that controls the stimulus time series generation process.
        """
        # Select hyperparameters set by user input
        self._series_params = (
            series_params if series_params is not None else SeriesParams()
        )

        # Create an algorithm instance object based on the hyperparameters entered by the user
        self._sampling_dict = self.create_sampling_dict(
            series_params=self._series_params
        )

    def __call__(
        self,
        rng: np.random.RandomState,
        n_inputs_points: int,
        input_dimension: Optional[int] = 1,
        normalization: Optional[str] = None,
        return_choice: Optional[bool] = None,
    ) -> Union[np.ndarray, List[str]]:
        """Call the `generate` method to stimulate time series generation"""
        return self.generate(
            rng=rng,
            n_inputs_points=n_inputs_points,
            input_dimension=input_dimension,
            normalization=normalization,
            return_choice=return_choice,
        )

    def __str__(self) -> str:
        """Get the name of the time series generator"""
        return "Excitation"

    def create_sampling_dict(
        self, series_params: Optional[SeriesParams] = None
    ) -> Dict[
        str,
        Union[
            MixedDistribution,
            AutoregressiveMovingAverage,
            ForecastPFN,
            KernelSynth,
            IntrinsicModeFunction,
        ],
    ]:
        """
        Create the sampling dictionary for different time series generation mechanisms.

        :param series_params: Parameter object that controls the stimulus time series generation process.
        :return: Dictionary that contains the sampling dictionary for different time series generation mechanisms.
        """
        # Determine whether the parameters are passed in. If not, use the default parameters.
        series_params = self.series_params if series_params is None else series_params

        # Traverse all the methods of the incentive time series data generation mechanism
        # to create objects and put them into the dictionary
        sampling_dict = {
            name: method
            for name, method in zip(
                self.sampling_methods,
                [
                    self.create_mixed_distribution(series_params=series_params),
                    self.create_autoregressive_moving_average(
                        series_params=series_params
                    ),
                    self.create_forecast_pfn(series_params=series_params),
                    self.create_kernel_synth(series_params=series_params),
                    self.create_intrinsic_mode_function(series_params=series_params),
                ],
            )
        }

        return sampling_dict

    def choice(
        self, rng: np.random.RandomState, input_dimension: int = 1
    ) -> np.ndarray:
        """
        Randomly select n specific methods based on the probability of selecting each
        method for generating the stimulus time series data.
        n is `input_dimension`, which is the dimension of the generated time series data.

        :param: rng: The random number generator in NumPy with fixed seed.
        :return: A numpy array of the random data generation methods.
        """
        return rng.choice(
            self.sampling_methods, size=input_dimension, p=self.prob_array
        )

    def create_mixed_distribution(
        self, series_params: Optional[SeriesParams] = None
    ) -> MixedDistribution:
        """
        Constructing excitation time series generation for sampling from mixture distributions.

        See [MixedDistribution](https://github.com/wwhenxuan/S2Generator/blob/main/S2Generator/excitation/mixed_distribution.py).

        :param series_params: The parameters for generating management incentive time series data.
        :return: The time series generation methods by mixture distributions.
        """
        series_params = self.series_params if series_params is None else series_params
        return MixedDistribution(
            min_centroids=series_params.min_centroids,
            max_centroids=series_params.max_centroids,
            rotate=series_params.rotate,
            gaussian=series_params.gaussian,
            uniform=series_params.uniform,
            probability_dict=series_params.mixed_distribution_dict,
            probability_list=series_params.mixed_distribution_list,
            dtype=self.dtype,
        )

    def create_autoregressive_moving_average(
        self, series_params: Optional[SeriesParams] = None
    ) -> AutoregressiveMovingAverage:
        """
        Constructing excitation time series generation for sampling from autoregressive moving averages process.

        See [AutoregressiveMovingAverage](https://github.com/wwhenxuan/S2Generator/blob/main/S2Generator/excitation/autoregressive_moving_average.py).

        :param series_params: The parameters for generating management incentive time series data.
        :return: The time series generation methods by autoregressive moving averages process.
        """
        series_params = self.series_params if series_params is None else series_params
        return AutoregressiveMovingAverage(
            p_min=series_params.p_min,
            p_max=series_params.p_max,
            q_min=series_params.q_min,
            q_max=series_params.q_max,
            upper_bound=series_params.upper_bound,
            dtype=self.dtype,
        )

    def create_forecast_pfn(
        self, series_params: Optional[SeriesParams] = None
    ) -> ForecastPFN:
        """
        Constructing excitation time series generation for sampling from ForecastPFN.

        See [ForecastPFN](https://github.com/wwhenxuan/S2Generator/blob/main/S2Generator/excitation/forecast_pfn.py).

        :param series_params: The parameters for generating management incentive time series data.
        :return: The time series generation methods by ForecastPFN.
        """
        series_params = self.series_params if series_params is None else series_params
        return ForecastPFN(
            is_sub_day=series_params.is_sub_day,
            transition=series_params.transition,
            start_time=series_params.start_time,
            end_time=series_params.end_time,
            random_walk=series_params.random_walk,
            dtype=self.dtype,
        )

    def create_kernel_synth(
        self, series_params: Optional[SeriesParams] = None
    ) -> KernelSynth:
        """
        Constructing excitation time series generation for sampling from KernelSynth.

        See [KernelSynth](https://github.com/wwhenxuan/S2Generator/blob/main/S2Generator/excitation/kernel_synth.py).

        :param series_params: The parameters for generating management incentive time series data.
        :return: The time series generation methods by KernelSynth.
        """
        series_params = self.series_params if series_params is None else series_params
        return KernelSynth(
            min_kernels=series_params.min_kernels,
            max_kernels=series_params.max_kernels,
            exp_sine_squared=series_params.exp_sine_squared,
            dot_product=series_params.dot_product,
            rbf=series_params.rbf,
            rational_quadratic=series_params.rational_quadratic,
            white_kernel=series_params.white_kernel,
            constant_kernel=series_params.constant_kernel,
            dtype=self.dtype,
        )

    def create_intrinsic_mode_function(
        self, series_params: Optional[SeriesParams] = None
    ) -> IntrinsicModeFunction:
        """
        Constructing excitation time series generation for sampling from intrinsic mode function.

        See [IntrinsicModeFunction](https://github.com/wwhenxuan/S2Generator/blob/main/S2Generator/excitation/intrinsic_mode_functions.py).

        :param series_params: The parameters for generating management incentive time series data.
        :return: The time series generation methods by intrinsic mode function.
        """
        series_params = self.series_params if series_params is None else series_params
        return IntrinsicModeFunction(
            min_base_imfs=series_params.min_base_imfs,
            max_base_imfs=series_params.max_base_imfs,
            min_choice_imfs=series_params.min_choice_imfs,
            max_choice_imfs=series_params.max_choice_imfs,
            probability_dict=series_params.imfs_dict,
            probability_list=series_params.imfs_list,
            min_duration=series_params.min_duration,
            max_duration=series_params.max_duration,
            min_amplitude=series_params.min_amplitude,
            max_amplitude=series_params.max_amplitude,
            min_frequency=series_params.min_frequency,
            max_frequency=series_params.max_frequency,
            dtype=self.dtype,
        )

    @property
    def series_params(self) -> SeriesParams:
        """Get the parameters for generating management incentive time series data."""
        return self._series_params

    @property
    def sampling_methods(self) -> List[str]:
        """Returns a list of various name of the different sampling methods"""
        return self.series_params.sampling_methods

    @property
    def sampling_object(self) -> List[Any]:
        """Returns a list of class objective of various sampling methods"""
        return list(self._sampling_dict.values())

    @property
    def sampling_dict(
        self,
    ) -> Dict[
        str,
        Union[
            MixedDistribution,
            AutoregressiveMovingAverage,
            ForecastPFN,
            KernelSynth,
            IntrinsicModeFunction,
        ],
    ]:
        """Returns a dictionary of the various sampling methods."""
        return self._sampling_dict

    @property
    def prob_array(self) -> np.ndarray:
        """Obtaining the sampling probability of time series data with different incentive methods."""
        return self.series_params.prob_array

    @property
    def dtype(self) -> np.dtype:
        """Obtaining the data type of the time series data with different incentive methods."""
        return self.series_params.dtype

    def generate(
        self,
        rng: np.random.RandomState,
        n_inputs_points: int,
        input_dimension: Optional[int] = 1,
        normalization: Optional[str] = None,
        return_choice: Optional[bool] = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Union[List[Any], np.ndarray]]]:
        """
        A unified interface for generating stimulus time series data.

        This interface allows for random calls to time series generation methods such as
        `MixedDistribution`, `AutoregressiveMovingAverage`, `ForecastPFN`, `KernelSynth`,
        and `IntrinsicModeFunction` to construct stimulus time series data.

        The selection of a particular method is achieved through random sampling.
        The probability of each method being sampled (selected) is a user-specified
        hyperparameter, as specified in the SeriesParams object.

        :param rng: The random number generator in NumPy with fixed seed.
        :param n_inputs_points: The length of time series data to be generated.
        :param input_dimension: The dimension of time series data to be generated.
        :param normalization: The normalization method to use, None for no normalization, choice in ["z-score", "max-min"].
        :param return_choice: If True, return a list of the selected methods.

        :return: The generated time series data and the selected methods (Optional).
        """
        # 1. Randomly select different sampling methods according to the specified probability
        choice_list = self.choice(rng=rng, input_dimension=input_dimension)

        # 2. Traverse the array to get the specific runnable instantiation object from the sampling dictionary
        time_series = np.hstack(
            [
                self.sampling_dict[name].generate(
                    rng=rng,
                    n_inputs_points=n_inputs_points,
                    input_dimension=1,
                )
                for name in choice_list
            ]
        )

        # 3. Whether to normalize the stimulus time series data
        if normalization is None:
            pass
        elif normalization == "z-score":
            for dim in range(input_dimension):
                time_series[:, dim] = z_score_normalization(x=time_series[:, dim])
        elif normalization == "max-min":
            for dim in range(input_dimension):
                time_series[:, dim] = max_min_normalization(x=time_series[:, dim])
        else:
            raise ValueError(
                "The normalization option must be 'z-score' or 'max-min' or None!"
            )

        if return_choice:
            # Whether to return a dictionary of sampling modes for each channel
            return time_series, choice_list
        return time_series
