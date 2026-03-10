from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy import stats


class GaussianMixtureSimulator(object):
    def __init__(
        self,
        n_components=3,
        covariance_type="full",
        tol: float = 1e-3,
        reg_covarfloat: float = 1e-6,
        max_iter: int = 100,
        n_init: int = 1,
        init_params: str = "kmeans",
        random_state=42,
    ) -> None:
        """
        :param n_components: int, default=3. The number of mixture components.
        :param covariance_type: str, default='full'. The type of covariance parameters to use. Must be one of 'full', 'tied', 'diag', 'spherical'.
        :param tol: float, default=1e-3. The convergence threshold. EM iterations will stop when the lower bound average gain is below this threshold.
        :param reg_covarfloat: float, default=1e-6. Non-negative regularization added to the diagonal of covariance. Allows to assure that the covariance matrices are all positive.
        :param max_iter: int, default=100. The number of EM iterations to perform.
        :param n_init: int, default=1. The number of initializations to perform. The best results are kept.
        :param init_params: str, default='kmeans'. The method used to initialize the weights, the means and the precisions. String must be one of 'kmeans', 'k-means++', 'random', 'random_from_data'.
        :param random_state: int, default=42. The seed used by the random number generator.
        """
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state

        # Validate the covariance_type
        if covariance_type not in ["full", "tied", "diag", "spherical"]:
            raise ValueError(
                "Invalid covariance_type. Must be one of 'full', 'tied', 'diag', 'spherical'."
            )

        # Validate the init_params
        if init_params not in ["kmeans", "k-means++", "random", "random_from_data"]:
            raise ValueError(
                "Invalid init_params. Must be one of 'kmeans', 'k-means++', 'random', 'random_from_data'."
            )

        # Initialize the GaussianMixture model
        self.model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            tol=tol,
            reg_covar=reg_covarfloat,
            max_iter=max_iter,
            n_init=n_init,
            init_params=init_params,
            random_state=random_state,
        )

    def fit(self, time_series: np.ndarray) -> None:
        """
        Fit the Gaussian Mixture Model to the provided time series data.
        :param time_series: 1D np.ndarray. The input time series data.
        """

        # Check if the input time series is 1D numpy array
        time_series = self._check_inputs(time_series)

        # Reshape the time series data to fit the GMM input requirements
        time_series = time_series.reshape(-1, 1)

        self.model.fit(time_series)

    def transform(
        self, num_samples: int, seq_len: int, random_state: Optional[int] = None
    ) -> np.ndarray:
        """ """

        # 根据GMM的组件权重，随机选择每个样本属于哪个高斯组件
        component_indices = np.random.choice(
            self.n_components, size=(num_samples, seq_len), p=self.model.weights_
        )

        # 从选中的高斯组件中采样生成数据
        generated_series = np.zeros(shape=(num_samples, seq_len))
        for i in range(num_samples):
            for j in range(seq_len):
                comp = component_indices[i, j]
                mean = self.model.means_[comp, 0]
                cov = self.covariance(comp)
                generated_series[i, j] = np.random.normal(loc=mean, scale=np.sqrt(cov))

        return generated_series

    def _check_inputs(self, time_series: np.ndarray) -> np.ndarray:
        """
        Check if the input time series is a 1D numpy array and reshape it to fit the GMM input requirements.
        :param time_series: 1D np.ndarray. The input time series data.
        :return: Reshaped time series data.
        """
        if not isinstance(time_series, np.ndarray):
            raise ValueError("Input time_series must be a numpy array.")
        if time_series.ndim != 1:
            raise ValueError("Input time_series must be a 1D array.")

        return time_series.reshape(-1, 1)

    def weight(self, component_index: int) -> float:
        """
        Get the weight of a specific component in the GMM.
        :param component_index: int. The index of the component.
        :return: The weight of the specified component.
        """
        if component_index < 0 or component_index >= self.n_components:
            raise ValueError(
                f"Component index must be between 0 and {self.n_components - 1}."
            )

        return self.model.weights_[component_index]

    def weights(self) -> np.ndarray:
        """
        Get the weights of all components in the GMM.
        :return: The weights of all components.
        """
        return self.model.weights_

    def mean(self, component_index: int) -> float:
        """
        Get the mean of a specific component in the GMM.
        :param component_index: int. The index of the component.
        :return: The mean of the specified component.
        """
        if component_index < 0 or component_index >= self.n_components:
            raise ValueError(
                f"Component index must be between 0 and {self.n_components - 1}."
            )

        return self.model.means_[component_index][0]

    def means(self) -> np.ndarray:
        """
        Get the means of all components in the GMM.
        :return: The means of all components.
        """
        return self.model.means_.flatten()

    def covariance(self, component_index: int) -> float:
        """
        Get the covariance of a specific component in the GMM.
        :param component_index: int. The index of the component.
        :return: The covariance of the specified component.
        """
        if component_index < 0 or component_index >= self.n_components:
            raise ValueError(
                f"Component index must be between 0 and {self.n_components - 1}."
            )

        if self.covariance_type == "full":
            return self.model.covariances_[component_index][0][0]
        elif self.covariance_type == "tied":
            return self.model.covariances_[0][0][0]
        elif self.covariance_type == "diag":
            return self.model.covariances_[component_index][0]
        elif self.covariance_type == "spherical":
            return self.model.covariances_[component_index]

    def covariances(self) -> np.ndarray:
        """
        Get the covariances of all components in the GMM.
        :return: The covariances of all components.
        """
        if self.covariance_type == "full":
            return np.array([cov[0][0] for cov in self.model.covariances_])
        elif self.covariance_type == "tied":
            return np.array([self.model.covariances_[0][0][0]] * self.n_components)
        elif self.covariance_type == "diag":
            return self.model.covariances_.flatten()
        elif self.covariance_type == "spherical":
            return self.model.covariances_
