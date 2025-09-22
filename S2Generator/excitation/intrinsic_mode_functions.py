# -*- coding: utf-8 -*-
"""
Created on 2025/08/12 13:40:16
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np

from pysdkit.data import (
    add_noise,
    generate_sin_signal,
    generate_cos_signal,
    generate_am_signal,
    generate_sawtooth_wave,
)
from pysdkit.utils import max_min_normalization

from typing import Optional, Union, Dict, List, Tuple, Callable
from S2Generator.excitation.base_excitation import BaseExcitation

# A dictionary of all available Eigenmodel functions
ALL_IMF_DICT = {
    "generate_sin_signal": generate_sin_signal,
    "generate_cos_signal": generate_cos_signal,
    "generate_am_signal": generate_am_signal,
    "generate_sawtooth_wave": generate_sawtooth_wave,
}


def _check_probability_dict(prob_dict: Dict[str, float]) -> Dict[str, float]:
    """
    Validates and normalizes a probability dictionary for intrinsic mode functions.

    This function checks:
    1. All keys exist in the global ALL_IMF_DICT
    2. Applies max-min normalization to probability values
    3. Returns a new dictionary with normalized probabilities

    :param prob_dict: Input dictionary with IMF names as keys and probabilities as values
    :type prob_dict: Dict[str, float]
    :return: Normalized probability dictionary with same keys
    :rtype: Dict[str, float]
    :raises ValueError: If any key is not found in ALL_IMF_DICT
    """
    prob_list = []

    # Validate all dictionary keys
    for key, value in prob_dict.items():
        if key not in ALL_IMF_DICT.keys():
            raise ValueError(f"Illegal key: {key} in `prob_dict`!")
        prob_list.append(value)

    # Normalize probabilities
    prob_array = max_min_normalization(x=np.array(prob_list))

    return {key: value for key, value in zip(prob_dict.keys(), prob_array)}


def _check_probability_list(prob_list: List[float]) -> Dict[str, float]:
    """
    Validates and normalizes a probability list for intrinsic mode functions.

    This function:
    1. Checks list length matches global ALL_IMF_DICT size
    2. Applies max-min normalization to probability values
    3. Returns dictionary with IMF keys and normalized probabilities

    :param prob_list: Probability values for IntrinsicModeFunction in predefined order
    :type prob_list: List[float]
    :return: Dictionary mapping IMF names to normalized probabilities
    :rtype: Dict[str, float]
    :raises ValueError: If input length is invalid
    """
    length = len(prob_list)
    total_imfs = len(ALL_IMF_DICT)

    # Validate list length
    if length > total_imfs or length <= 0:
        raise ValueError(
            f"Invalid `prob_list` length: {length}. Must be 1-{total_imfs}"
        )

    # Normalize and map to IMF keys
    prob_array = max_min_normalization(x=np.array(prob_list))
    return {
        imf: prob for imf, prob in zip(list(ALL_IMF_DICT.keys())[:length], prob_array)
    }


def _get_energy(signal: np.ndarray) -> Union[float, np.ndarray]:
    """
    Calculates the energy of a signal using mean absolute amplitude.

    Note: This is one of several possible energy computation methods.

    :param signal: Input signal array
    :type signal: np.ndarray
    :return: Energy measurement as mean absolute amplitude
    :rtype: float
    """
    return np.mean(np.abs(signal))


def get_adaptive_sampling_rate(duration: float, length: int) -> float:
    """
    Computes the minimum sampling rate required to achieve a target signal length.

    Formula: sampling_rate = ceil(signal_length / time_duration)

    :param duration: Time duration of the signal in seconds
    :type duration: float
    :param length: Desired number of samples in generated signal
    :type length: int
    :return: Minimum required sampling frequency (Hz)
    :rtype: float
    """
    return np.ceil(length / duration)


class IntrinsicModeFunction(BaseExcitation):
    """
    Generates excitation time series in the form of Intrinsic Mode Functions (IntrinsicModeFunction).

    This class creates composite signals by combining fundamental waveform components
    through Empirical Mode Decomposition (EMD)-like synthesis. The generated signals
    can simulate various real-world vibration patterns and non-stationary behaviors.

    See More Information please visit PySDKit: A Python library for signal decomposition algorithms.
    https://github.com/wwhenxuan/PySDKit
    """

    def __init__(
        self,
        min_base_imfs: int = 2,
        max_base_imfs: int = 4,
        min_choice_imfs: int = 1,
        max_choice_imfs: int = 5,
        probability_dict: Optional[Dict[str, float]] = None,
        probability_list: Optional[List[float]] = None,
        min_duration: float = 0.5,
        max_duration: float = 10.0,
        min_amplitude: float = 0.01,
        max_amplitude: float = 10.0,
        min_frequency: float = 0.01,
        max_frequency: float = 8.0,
        noise_level: float = 0.1,
        upper_energy: Optional[float] = 32,
        dtype: np.dtype = np.float64,
    ) -> None:
        """
        Initializes the IMF signal generator with configuration parameters.

        :param min_base_imfs: Minimum number of fundamental IMF components to generate
        :type min_base_imfs: int
        :param max_base_imfs: Maximum number of fundamental IMF components to generate
        :type max_base_imfs: int
        :param min_choice_imfs: Minimum number of IntrinsicModeFunction to select from available types
        :type min_choice_imfs: int
        :param max_choice_imfs: Maximum number of IntrinsicModeFunction to select from available types
        :type max_choice_imfs: int
        :param probability_dict: Custom probability distribution for IMF types,
                                 with keys matching available IMF names
        :type probability_dict: Optional[Dict[str, float]]
        :param probability_list: Custom probability weights for IMF types in
                                 predefined order (overrides probability_dict)
        :type probability_list: Optional[List[float]]
        :param min_duration: Minimum signal duration in seconds
        :type min_duration: float
        :param max_duration: Maximum signal duration in seconds
        :type max_duration: float
        :param min_amplitude: Minimum amplitude for generated signal components
        :type min_amplitude: float
        :param max_amplitude: Maximum amplitude for generated signal components
        :type max_amplitude: float
        :param min_frequency: Minimum frequency for signal components (Hz)
        :type min_frequency: float
        :param max_frequency: Maximum frequency for signal components (Hz)
        :type max_frequency: float
        :param noise_level: Amplitude of Gaussian noise to add (relative to signal amplitude)
        :type noise_level: float
        :param upper_energy: The upper energy of the generated signal.
        :param dtype: Numerical precision for generated signals
        :type dtype: np.dtype
        """
        super().__init__(dtype=dtype)

        # Configure fundamental IMF components
        self.min_base_imfs = min_base_imfs
        self.max_base_imfs = max_base_imfs
        self.base_imfs = [
            generate_sin_signal,
            generate_cos_signal,
        ]  # Core waveform generators

        # Configure IMF selection parameters
        self.min_choice_imfs = min_choice_imfs
        self.max_choice_imfs = max_choice_imfs

        # Process probability distributions for IMF selection
        (
            self.available_dict,
            self.available_list,
            self.available_probability,
        ) = self._processing_probability(
            probability_dict=probability_dict, probability_list=probability_list
        )

        # Configure temporal parameters
        self.min_duration = min_duration
        self.max_duration = max_duration

        # Configure amplitude parameters
        self.min_amplitude = min_amplitude
        self.max_amplitude = max_amplitude

        # Configure frequency parameters
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency

        # The max energy of the output signal
        self.upper_energy = upper_energy

        # Configure noise parameters
        self.noise_level = noise_level

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
        return self.__class__.__name__

    @property
    def all_imfs_dict(self) -> Dict[str, Callable]:
        """Get a dictionary of all available Eigen model functions"""
        return ALL_IMF_DICT

    @property
    def all_imfs_list(self) -> List[Callable]:
        """Get a list of all available eigenmode functions"""
        return list(self.all_imfs_dict.values())

    @property
    def default_probability_dict(self) -> Dict[str, float]:
        """Get the default probability dictionary when the user specifies parameters for the input"""
        return {
            "generate_sin_signal": 0.30,
            "generate_cos_signal": 0.30,
            "generate_am_signal": 0.20,
            "generate_sawtooth_wave": 0.20,
        }

    def _processing_probability(
        self,
        probability_dict: Optional[Dict[str, float]] = None,
        probability_list: Optional[List[float]] = None,
    ) -> Tuple[Dict[str, float], List[str], List[float]]:
        """
        Processes and validates input probability distributions for IMF selection.

        Handles four configuration scenarios:
        1. Both None: Uses default probability distribution
        2. Only dict provided: Validates and normalizes dictionary
        3. Only list provided: Validates and normalizes list
        4. Both provided: Prioritizes dictionary input

        :param probability_dict: Custom probability distribution with IMF names as keys
        :type probability_dict: Optional[Dict[str, float]]
        :param probability_list: Probability weights in predefined IMF order
        :type probability_list: Optional[List[float]]
        :return: Tuple containing:
            - available_dict: Normalized probability dictionary
            - available_list: IMF names in order
            - available_probability: Normalized probability values
        :rtype: Tuple[Dict[str, float], List[str], List[float]]
        :raises ValueError: If both inputs are None when required
        """
        # Handle different input configurations
        if probability_dict is None and probability_list is None:
            available_dict = self.default_probability_dict  # Use default distribution

        elif probability_dict is not None and probability_list is None:
            available_dict = _check_probability_dict(prob_dict=probability_dict)

        elif probability_dict is None and probability_list is not None:
            available_dict = _check_probability_list(prob_list=probability_list)

        elif probability_dict is not None and probability_list is not None:
            available_dict = _check_probability_dict(
                prob_dict=probability_dict
            )  # Dict takes priority

        else:
            raise ValueError("Must provide either probability_dict or probability_list")

        # Extract ordered IMF names and probabilities
        available_list = list(available_dict.keys())
        available_probability = list(available_dict.values())

        return available_dict, available_list, available_probability

    def _add_noise(self, imfs: np.ndarray, n_inputs_points: int) -> np.ndarray:
        """
        Generates adaptive Gaussian noise proportional to signal energy.

        Noise standard deviation calculation:
            STD = noise_level × signal_energy
            where signal_energy = mean(|imfs|)

        :param imfs: Matrix of intrinsic mode functions (time series)
        :type imfs: np.ndarray
        :param n_inputs_points: Number of time samples required
        :type n_inputs_points: int
        :return: Noise vector scaled to signal energy
        :rtype: np.ndarray
        """
        return add_noise(
            N=n_inputs_points,
            Mean=0,  # Zero-mean Gaussian noise
            STD=self.noise_level * _get_energy(signal=imfs),  # Energy-adaptive scaling
        )

    def get_random_duration(
        self, rng: np.random.RandomState, number: int
    ) -> np.ndarray:
        """
        Generates random durations for IMF components.

        Durations are uniformly distributed between min_duration and max_duration.

        :param rng: Seeded random number generator for reproducibility
        :type rng: np.random.RandomState
        :param number: Count of durations to generate (matches IMF count)
        :type number: int
        :return: Array of durations in seconds
        :rtype: np.ndarray
        """
        return rng.uniform(low=self.min_duration, high=self.max_duration, size=number)

    def get_random_amplitude(
        self, rng: np.random.RandomState, number: int
    ) -> np.ndarray:
        """
        Generates random amplitudes for IMF components.

        Amplitudes are uniformly distributed between min_amplitude and max_amplitude.

        :param rng: Seeded random number generator
        :type rng: np.random.RandomState
        :param number: Count of amplitudes to generate
        :type number: int
        :return: Array of amplitude values
        :rtype: np.ndarray
        """
        return rng.uniform(low=self.min_amplitude, high=self.max_amplitude, size=number)

    def get_random_frequency(
        self, rng: np.random.RandomState, number: int
    ) -> np.ndarray:
        """
        Generates random frequencies for IMF components.

        Frequencies are uniformly distributed between min_frequency and max_frequency.

        :param rng: Seeded random number generator
        :type rng: np.random.RandomState
        :param number: Count of frequencies to generate
        :type number: int
        :return: Array of frequency values in Hz
        :rtype: np.ndarray
        """
        return rng.uniform(low=self.min_frequency, high=self.max_frequency, size=number)

    def get_base_imfs(
        self, imfs: np.ndarray, rng: np.random.RandomState, n_inputs_points: int
    ) -> np.ndarray:
        """
        Generates fundamental IMF components (sine/cosine) and adds to the signal.

        The base IMFs form the core waveform structure:

        1. Randomly selects number of base components (between min_base_imfs and max_base_imfs)

        2. For each component:
            - Randomly selects waveform type (sine/cosine with equal probability)
            - Generates random amplitude, frequency, and duration
            - Computes adaptive sampling rate for time alignment
            - Adds component to the signal matrix

        :param imfs: Target signal matrix to populate with IMFs
        :type imfs: np.ndarray
        :param rng: Seeded random number generator for reproducibility
        :type rng: np.random.RandomState
        :param n_inputs_points: Required number of time samples
        :type n_inputs_points: int
        :return: Signal matrix with added fundamental components
        :rtype: np.ndarray
        """
        # Determine number of fundamental components to generate
        base_number = rng.randint(
            low=self.min_base_imfs, high=self.max_base_imfs + 1  # Inclusive upper bound
        )

        # Generate each fundamental component
        for base_function, amplitude, frequency, duration in zip(
            rng.choice(
                self.base_imfs, size=base_number, p=[0.5, 0.5]
            ),  # Equal probability
            self.get_random_amplitude(rng=rng, number=base_number),
            self.get_random_frequency(rng=rng, number=base_number),
            self.get_random_duration(rng=rng, number=base_number),
        ):
            # Calculate adaptive sampling rate for time alignment
            sampling_rate = get_adaptive_sampling_rate(
                duration=duration, length=n_inputs_points
            )

            # Generate component signal and add to matrix
            component = base_function(
                duration=duration,
                sampling_rate=sampling_rate,
                frequency=frequency,
                noise_level=0.0,  # No noise for fundamental components
            )[1][
                :n_inputs_points
            ]  # Truncate to required length

            imfs += amplitude * component

        return imfs

    def get_choice_imfs(
        self, imfs: np.ndarray, rng: np.random.RandomState, n_inputs_points: int
    ) -> np.ndarray:
        """
        Adds randomly selected IMF components from available types.

        Handles special case for AM signals which require additional parameters:
        1. Randomly selects number of components (between min_choice_imfs and max_choice_imfs)
        2. For each component:
            - Selects IMF type based on configured probability distribution
            - Generates random amplitude, frequency, and duration
            - For AM signals: generates random modulation parameters
            - For other signals: uses standard parameters
            - Adds component to signal matrix

        :param imfs: Target signal matrix being constructed
        :type imfs: np.ndarray
        :param rng: Seeded random number generator
        :type rng: np.random.RandomState
        :param n_inputs_points: Required number of time samples
        :type n_inputs_points: int
        :return: Signal matrix with added selected components
        :rtype: np.ndarray
        """
        # Determine number of additional components to add
        choice_number = rng.randint(low=self.min_choice_imfs, high=self.max_choice_imfs)

        # Generate each additional component
        for choice_function, amplitude, frequency, duration in zip(
            rng.choice(
                self.available_list, size=choice_number, p=self.available_probability
            ),
            self.get_random_amplitude(rng=rng, number=choice_number),
            self.get_random_frequency(rng=rng, number=choice_number),
            self.get_random_duration(rng=rng, number=choice_number),
        ):
            # Retrieve actual function from IMF dictionary
            func = ALL_IMF_DICT[choice_function]

            # Calculate adaptive sampling rate
            sampling_rate = get_adaptive_sampling_rate(
                duration=duration, length=n_inputs_points
            )

            # Handle AM signal special case
            if func == generate_am_signal:
                component = generate_am_signal(
                    duration=duration,
                    sampling_rate=sampling_rate,
                    mod_index=rng.randint(1, 4),  # Modulation index (1-3)
                    carrier_freq=rng.randint(50, 150),  # Carrier frequency (50-150 Hz)
                    modulating_freq=rng.randint(
                        1, 16
                    ),  # Modulating frequency (1-15 Hz)
                    noise_level=0.0,
                )[1][:n_inputs_points]
            else:
                # Standard signal generation
                component = func(
                    duration=duration,
                    sampling_rate=sampling_rate,
                    frequency=frequency,
                    noise_level=0.0,
                )[1][:n_inputs_points]

            imfs += amplitude * component

        return imfs

    def adjust_upper_energy(
        self, signal: np.ndarray, rng: np.random.RandomState
    ) -> np.ndarray:
        """
        Adjusts the upper energy of the signal.
        This operation is mainly used when the amplitude of the time series data generated by the method is too large.

        :param signal: Signal with high energy to be adjusted.
        :param rng: The random number generator in NumPy with fixed seed.
        :return: The adjusted signal with the lower energy.
        """
        # Calculate the energy of the current signal
        energy = np.mean(signal**2)

        # Get the random energy range
        upper_energy = (np.random.rand() + 0.05) * self.upper_energy

        # Returns the scaled energy value
        return signal * upper_energy / energy

    def generate(
        self,
        rng: np.random.RandomState,
        n_inputs_points: int = 512,
        input_dimension: int = 1,
    ) -> np.ndarray:
        """
        Generates multi-dimensional time series through IMF composition.

        Signal synthesis pipeline for each dimension:
        1. Initialize zero matrix
        2. Add fundamental waveform components (sine/cosine)
        3. Add randomly selected IMF components
        4. Add energy-adaptive Gaussian noise

        :param rng: Seeded random number generator
        :type rng: np.random.RandomState
        :param n_inputs_points: Number of time samples per dimension
        :type n_inputs_points: int
        :param input_dimension: Number of output dimensions (channels)
        :type input_dimension: int
        :return: Generated time series array of shape (n_inputs_points, input_dimension)
        :rtype: np.ndarray
        """
        # Initialize output matrix
        imfs = np.zeros(shape=(n_inputs_points, input_dimension), dtype=self.dtype)

        # Generate each dimension independently
        for i in range(input_dimension):
            # 1. Fundamental waveform components
            imfs[:, i] = self.get_base_imfs(
                imfs=imfs[:, i], rng=rng, n_inputs_points=n_inputs_points
            )

            # 2. Additional IMF components
            imfs[:, i] = self.get_choice_imfs(
                imfs=imfs[:, i], rng=rng, n_inputs_points=n_inputs_points
            )

            # 3. Energy-adaptive noise
            imfs[:, i] += self._add_noise(
                imfs=imfs[:, i], n_inputs_points=n_inputs_points
            )

            # 4. Scaling the energy of the imfs
            imfs[:, i] = self.adjust_upper_energy(imfs[:, i], rng=rng)

        return imfs
