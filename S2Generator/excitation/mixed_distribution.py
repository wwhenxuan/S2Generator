# -*- coding: utf-8 -*-
"""
This module generates excitation time series data from a mixture of multiple Gaussian or uniform distributions.
This module was subsequently integrated into the Excitation module as an interface.
In [`examples`](https://github.com/wwhenxuan/S2Generator/blob/main/examples/4-mixed_distribution.ipynb),
we provide examples demonstrating the module's usage.

Created on 2025/08/14 11:01:12
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/S2Generator
"""
import numpy as np
from scipy.stats import special_ortho_group
from pysdkit.utils import max_min_normalization

from typing import Optional, Union, Dict, Tuple, List
from S2Generator.excitation.base_excitation import BaseExcitation


class MixedDistribution(BaseExcitation):
    """Generate excitation time series data through mixed distribution."""

    def __init__(
        self,
        min_centroids: Optional[int] = 3,
        max_centroids: Optional[int] = 8,
        rotate: Optional[bool] = False,
        gaussian: Optional[bool] = True,
        uniform: Optional[bool] = True,
        probability_dict: Optional[Dict[str, float]] = None,
        probability_list: Optional[List[float]] = None,
        dtype: np.dtype = np.float64,
    ):
        """
        :param min_centroids: The min number of centroids for the input distribution.
        :param max_centroids: The max number of centroids for the input distribution.
        :param rotate: Whether to rotate or not.
        :param gaussian: Whether to use Gaussian distribution.
        :param uniform: Whether to use uniform distribution.
        :param probability_dict: A dictionary that determines the probability of a certain distribution
                                 being sampled in a mixed distribution.
        :param probability_list: A list that determines the probability of a certain distribution being
                                 sampled from a mixed distribution.
        :param dtype: The dtype of the generated data.
        """
        super().__init__(dtype=dtype)

        # Minimum and maximum number of mixed distributions
        self.min_centroids = min_centroids
        self.max_centroids = max_centroids

        # Whether to multiply the sampled time series data by the rotation matrix
        self.rotate = rotate

        # Whether to enable sampling of Gaussian process and uniform distribution process
        self.gaussian = gaussian
        self.uniform = uniform

        # Dictionary and list of stimulus probabilities
        self.probability_dict = probability_dict
        self.probability_list = probability_list

        # Get available dictionaries and lists
        (
            self._available_dict,
            self._available_list,
            self._available_prob,
        ) = self._get_available(
            probability_dict=probability_dict, probability_list=probability_list
        )

        # Sampling parameters of the record mixture distribution
        self.means, self.covariances, self.rotations = None, None, None

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
        return "MixedDistribution"

    @property
    def default_probability_dict(self) -> Dict[str, float]:
        """
        Provides default configuration for data generation,
        when the user does not specify a probability dictionary.
        """
        if self.gaussian is True and self.uniform is True:
            return {"gaussian": 0.5, "uniform": 0.5}
        elif self.gaussian is True and self.uniform is False:
            return {"gaussian": 1.0}
        elif self.gaussian is False and self.uniform is True:
            return {"uniform": 1.0}
        else:
            raise ValueError

    @property
    def available_dict(self) -> Dict[str, float]:
        """Get the probability dictionary of available samples."""
        return self._available_dict

    @property
    def available_list(self) -> List[str]:
        """Get a list of the distribution names used by available sampling, gaussian or uniform."""
        return self._available_list

    @property
    def available_prob(self) -> Union[List[float], np.ndarray]:
        """Get a list of the probability names used by available sampling, gaussian or uniform."""
        return self._available_prob

    def _get_available(
        self,
        probability_dict: Optional[Dict[str, float]] = None,
        probability_list: Optional[list[float]] = None,
    ) -> Tuple[Dict[str, float], List[str], Union[List[float], np.ndarray]]:
        """
        Handling user-supplied probability lists and probability dictionaries.
        The default configuration will be used when the user does not specify or specifies incorrectly.

        :param probability_dict: Probability dictionary of user inputs, defaults to None.
        :param probability_list: Probability list of user inputs, defaults to None.
        :return: Tuple of available probability dictionaries and list of available samples.
        """
        if probability_dict is None and probability_list is None:
            # When neither a dictionary nor a list is provided, the default configuration is returned.
            available_dict = self.default_probability_dict

        elif probability_dict is not None:
            # When the user provides a probability dictionary
            if "gaussian" not in probability_dict and "uniform" not in probability_dict:
                # If no key value is specified, the default configuration is returned.
                available_dict = self.default_probability_dict
            else:
                # Normalize the contents of the dictionary
                probability_array = max_min_normalization(
                    np.array(
                        [probability_dict["gaussian"], probability_dict["uniform"]]
                    )
                )
                available_dict = {
                    "gaussian": probability_array[0],
                    "uniform": probability_array[1],
                }

        elif probability_list is not None and probability_dict is None:
            # When the user provides a list of probabilities
            if len(probability_list) != 2:
                # When the list length does not meet the requirements
                available_dict = self.default_probability_dict
            else:
                probability_array = max_min_normalization(np.array([probability_list]))
                available_dict = {
                    "gaussian": probability_array[0],
                    "uniform": probability_array[1],
                }

        else:
            raise ValueError("Something wrong in probability_dict or probability_list!")

        # Get the contents of the dictionary
        available_list = list(available_dict.keys())
        available_prob = np.array(list(available_dict.values()))

        return available_dict, available_list, available_prob

    def generate_stats(
        self, rng: np.random.RandomState, input_dimension: int, n_centroids: int
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Generate parameters required for sampling from a mixture distribution.

        :param rng: The random number generator in NumPy with fixed seed.
        :param input_dimension: The number of input dimension.
        :param n_centroids: The number of centroids in mixed distribution.
        :return:
            - means: Mean array with np.ndarray;
            - *covariances*: Covariance array (Note: This actually generates variances,
               as each center generates a separate variance value in each dimension,
               so the covariance matrix is a diagonal matrix);
            - rotations: Rotation matrix list (each element is a np.ndarray, representing the rotation matrix for each center);
        """
        self.means = rng.randn(
            n_centroids, input_dimension
        )  # Means of the mixture distribution
        self.covariances = rng.uniform(
            0, 1, size=(n_centroids, input_dimension)
        )  # Variances of the mixture distribution

        # The rotation matrix is used to transform an independent Gaussian distribution
        # (i.e., each dimension is independent) into a Gaussian distribution with correlation.
        if self.rotate:
            self.rotations = [
                (
                    special_ortho_group.rvs(input_dimension)
                    if input_dimension > 1
                    else np.identity(1)
                )
                for _ in range(n_centroids)
            ]
        else:
            self.rotations = [np.identity(input_dimension) for _ in range(n_centroids)]

        return self.means, self.covariances, self.rotations

    @property
    def get_stats(self) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """Get the sampling parameters for the recorded mixture distribution."""
        return self.means, self.covariances, self.rotations

    @property
    def get_means(self) -> np.ndarray:
        """Get the sampling mean for a recorded mixture distribution."""
        return self.means

    @property
    def get_covariances(self) -> np.ndarray:
        """Get the sampling covariance for a recorded mixture distribution."""
        return self.covariances

    @property
    def get_rotations(self) -> np.ndarray:
        """Get the sampling rotations for a recorded mixture distribution."""
        return self.rotations

    def generate_gaussian(
        self,
        rng: np.random.RandomState,
        input_dimension: int,
        n_centroids: int,
        n_points_comp: np.ndarray,
    ) -> np.ndarray:
        """
        Generate time series of specified dimensions and lengths using a Gaussian mixture distribution.

        :param rng: The random number generator in NumPy with fixed seed.
        :param input_dimension: The number of input dimension.
        :param n_centroids: The number of centroids in mixed distribution.
        :param n_points_comp: The number of points in each dimension.
        :return: Time series of specified dimensions and lengths.
        """
        means, covariances, rotations = self.generate_stats(
            rng, input_dimension, n_centroids
        )
        return np.vstack(
            [
                rng.multivariate_normal(mean, np.diag(covariance), int(sample))
                @ rotation
                for (mean, covariance, rotation, sample) in zip(
                    means, covariances, rotations, n_points_comp
                )
            ]
        )

    def generate_uniform(
        self,
        rng: np.random.RandomState,
        input_dimension: int,
        n_centroids: int,
        n_points_comp: np.ndarray,
    ) -> np.ndarray:
        """
        Generate time series of specified dimensions and lengths using a uniform mixture distribution.

        :param rng: The random number generator in NumPy with fixed seed.
        :param input_dimension: The number of input dimension.
        :param n_centroids: The number of centroids in mixed distribution.
        :param n_points_comp: The number of points in each dimension.
        :return: Time series of specified dimensions and lengths.
        """
        means, covariances, rotations = self.generate_stats(
            rng, input_dimension, n_centroids
        )
        return np.vstack(
            [
                (
                    mean
                    + rng.uniform(-1, 1, size=(sample, input_dimension))
                    * np.sqrt(covariance)
                )
                @ rotation
                for (mean, covariance, rotation, sample) in zip(
                    means, covariances, rotations, n_points_comp
                )
            ]
        )

    def generate_once(
        self, rng: np.random.RandomState, n_inputs_points: int = 512
    ) -> Union[np.ndarray, None]:
        """
        Generate stimulus time series data for a single channel through a mixture distribution.

        :param rng: The random status generator in NumPy.
        :param n_inputs_points: The number of input points in this sampling.
        :return: The generated time series samples with mixture distribution.
        """
        # 1. Statistical parameters for mixture distribution sampling
        n_centroids = rng.randint(low=self.min_centroids, high=self.max_centroids)

        # 2. Randomly generate the weight values for each distribution
        weights = rng.uniform(0, 1, size=(n_centroids,))
        weights /= np.sum(weights)
        n_points_comp = rng.multinomial(n_inputs_points, weights)

        # 3. Decide which distribution to use for sampling
        dist_list = rng.choice(
            self.available_list, size=n_centroids, p=self.available_prob
        )

        # 4. Iterate over a list of mixed distributions to generate time series data
        for sampling_type in dist_list:
            if sampling_type == "gaussian":
                # Sample using a Gaussian mixture distribution
                return self.generate_gaussian(
                    rng=rng,
                    input_dimension=1,
                    n_centroids=n_centroids,
                    n_points_comp=n_points_comp,
                )
            elif sampling_type == "uniform":
                # Sample using a uniform mixture distribution
                return self.generate_uniform(
                    rng=rng,
                    input_dimension=1,
                    n_centroids=n_centroids,
                    n_points_comp=n_points_comp,
                )
            else:
                raise ValueError("Something wrong in sampling_type!")
        return None

    def generate(
        self,
        rng: np.random.RandomState,
        n_inputs_points: int = 512,
        input_dimension: int = 1,
    ) -> np.ndarray:
        """
        Generate time series of specified dimensions and lengths using a uniform or gaussian mixture distribution.

        :param rng: The random number generator of NumPy with fixed seed.
        :param n_inputs_points: The number of input points.
        :param input_dimension: The dimension of the time series.
        :return: The generated mixed distribution time series.
        """
        # Iterate over multiple channels to generate time series data
        time_series = np.hstack(
            [
                self.generate_once(rng=rng, n_inputs_points=n_inputs_points)
                for _ in range(input_dimension)
            ]
        )

        return time_series


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    mixed_distribution = MixedDistribution()

    time = mixed_distribution.generate(
        rng=np.random.RandomState(100), n_inputs_points=512, input_dimension=5
    )

    for i in range(5):
        plt.plot(time[:, i])
        plt.show()
