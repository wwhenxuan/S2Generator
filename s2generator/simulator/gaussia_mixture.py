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
    ):
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

    def fit(self, time_series: np.ndarray):
        """
        Fit the Gaussian Mixture Model to the provided time series data.
        :param time_series: np.ndarray, shape (n_samples, n_features). The input time series data.
        """
        self.model.fit(time_series)
