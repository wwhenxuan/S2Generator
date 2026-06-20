# -*- coding: utf-8 -*-
"""
This module replicates the similarity metric between time series datasets
used in the paper "Measuring Time-Series Dataset Similarity using Wasserstein Distance."
(Paper address: https://www.arxiv.org/abs/2507.22189)

Created on 2025/08/22 11:20:58
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/S2Generator
"""

import numpy as np
from numpy import signedinteger
import matplotlib.pyplot as plt

from typing import Optional, Union, Tuple, List, Any


def check_inputs(data: np.ndarray) -> bool:
    """
    Check if input data is valid, for the data type and the shape of ndarray.

    :param data: The input time series data in Numpy ndarray.
    :return: True if the input data is valid.
    """
    # check the data type
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy ndarray!")

    # check the shape of the input dataset
    if len(data.shape) != 2:
        raise ValueError(
            "The input data must be two-dimensional with shape [n_samples, n_length]!"
        )

    return True


def dataset_max_min_normalization(
    data: np.ndarray, epsilon: Optional[float] = 1e-5
) -> Union[np.ndarray, None]:
    """
    Apply the max-min normalization operation to multivariate time series datasets
    with the shape of [n_samples, n_length] in NumPy ndarray.

    :param data: The time series dataset with shape [n_samples, n_length].
    :param epsilon: The maximum absolute value to avoid division by zero.

    :return: The max-min normalized dataset with shape [n_samples, n_length].
    """
    # Assume data has the shape [n_samples, n_length]
    n_samples, n_length = data.shape

    for n in range(n_samples):
        # Traverse each sample
        series = data[n, :].copy()
        data[n, :] = (series - series.min()) / (series.max() - series.min() + epsilon)

    return data


def time_series_to_distribution(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert the input multivariate time series dataset to a normal distribution.

    :param data: The input multivariate time series dataset in NumPy ndarray with
                 [n_samples, n_length] shape.
    :return: The mean vector and covariance matrix of a multivariate time series dataset.
    """
    # Compute the mean vector (mean of each variable)
    mean_vector = np.mean(data, axis=0)

    # Calculate the covariance matrix (covariance between variables)
    cov_matrix = np.cov(data.T)

    # print("mean vector: ", mean_vector.shape)
    # print(mean_vector)
    # print("covariance matrix: ", cov_matrix.shape)
    # print(cov_matrix)

    return mean_vector, cov_matrix


def wasserstein_distance(
    x: np.ndarray,
    y: np.ndarray,
    mean_weight: Optional[float] = 0.5,
    covar_weight: Optional[float] = 0.5,
    return_all: Optional[bool] = False,
) -> Union[Tuple[Any, signedinteger[Any], Any], None, float]:
    """
    The Wasserstein distance is used to measure the similarity between two datasets.
    "Measuring Time-Series Dataset Similarity using Wasserstein Distance."
    (Paper address: https://www.arxiv.org/abs/2507.22189)

    This method models the time series dataset as a multivariate normal distribution
    and uses the Wasserstein similarity metric to measure the distribution distance between datasets,
    thereby characterizing the similarity or difference between the datasets.

    The Wasserstein distance is a metric that measures the distance between two probability distributions.
    This distance is widely used in optimal transmission problems, primarily to determine the minimum distance
    between two distributions by minimizing the coupling cost. The p-Wasserstein distance is defined as follows:

    :math:`W_p (\mu, \\nu) = \sideset{}{}{\mathrm{min}}_{\gamma \in \Gamma _{\mu, \\nu}} \left ( \int \left \| x - y \\right \| ^ p \gamma \left ( \mathrm d x, \mathrm d y \\right ) \\right ) ^ {1/p}`

    where :math:`\mu, \\nu \in \mathbb R ^ d` and math:`\Gamma` denotes the coupled set of :math:`\mu` and :math:`\\nu`.

    This paper focuses on the Wasserstein distance :math:`d_{\mathrm Ws}` between two multivariate normal distributions :math:`\mathcal D_X` and :math:`\mathcal D_Y`, which is defined as:

    :math:`d^2_{\mathrm W _s} \left ( \mathcal D_X, \mathcal D_Y \\right ) = \left \| \hat{\mathbf{\mu}}_X - \hat{\mathbf{\mu}}_Y \\right \| ^ 2 + \mathrm{tr} \left ( \hat{\mathbf{\Sigma}}_X + \hat{\mathbf{\Sigma}}_Y - 2 \sqrt{\hat{\mathbf{\Sigma}}_X \hat{\mathbf{\Sigma}}_Y} \\right )`

    Where the first term measures the distance between the mean vectors, and the second term captures the difference between the covariance matrices. If two datasets are to be close to each other, they must not only have similar mean vectors, but also similar covariate matrices.

    :param x: The first ndarray dataset in NumPy with [num_samples, seq_len].
    :param y: The second ndarray dataset in NumPy with [num_samples, seq_len].
    :param mean_weight: The weight of the second norm of mean vector.
    :param covar_weight: The weight of the second norm of covariance matrix.
    :param return_all: If True, return all the tuple of (Wasserstein distances, mean_value, covar_value).

    :return: The Wasserstein distance between x and y.
    """
    # 1. Check the data type and shape for the inputs
    if check_inputs(data=x) and check_inputs(data=y):
        # 2. Perform maximum and minimum normalization operations on two time series data sets
        x, y = dataset_max_min_normalization(data=x), dataset_max_min_normalization(
            data=y
        )

        # 3. Compute the mean vector and covariance matrix between two time series datasets
        mean_vector_x, cov_matrix_x = time_series_to_distribution(data=x)
        mean_vector_y, cov_matrix_y = time_series_to_distribution(data=y)

        # 4. Calculate the trace of the mean vector and the sum of squares of the covariance matrix
        mean_value = np.sum((mean_vector_x - mean_vector_y) ** 2)
        covar_value = np.trace(
            cov_matrix_x + cov_matrix_y - 2 * np.sqrt(cov_matrix_x * cov_matrix_y)
        )
        distance = np.sqrt(mean_weight * mean_value + covar_weight * covar_value)

        if return_all:
            return distance, mean_value, covar_value

        return distance

    return None


def wasserstein_distance_matrix(
    data_list: List[np.ndarray],
    mean_weight: Optional[float] = 0.5,
    covar_weight: Optional[float] = 0.5,
) -> np.ndarray:
    """
    Calculate the distances between multiple datasets in a list and form a distance matrix.

    :param data_list: The ndarray dataset list in NumPy with [n_samples, n_vars, n_length].
    :param mean_weight: The weight of the second norm of mean vector.
    :param covar_weight: The weight of the second norm of covariance matrix.
    :return: The Wasserstein distance matrix in for the input ndarray list.

    See Also
    --------
    wasserstein_distance
    """
    # Get the length of the input list
    num_datasets = len(data_list)

    # Create the zero matrix
    distance_matrix = np.zeros((num_datasets, num_datasets))

    for i in range(num_datasets):
        # Traverse each data set to calculate the distance
        for j in range(i + 1, num_datasets):
            # Handling different situations
            distance_matrix[i, j] = wasserstein_distance(
                x=data_list[i],
                y=data_list[j],
                mean_weight=mean_weight,
                covar_weight=covar_weight,
            )
            # Symmetrical processing based on diagonal lines
            distance_matrix[j, i] = distance_matrix[i, j]

    return distance_matrix


def plot_wasserstein_heatmap(
    distance_matrix: np.ndarray,
    dataset_list: Optional[List[str]] = None,
    figsize: Optional[Tuple[float, float]] = (6, 4),
    dpi: Optional[int] = 300,
    fontsize: Optional[int] = 8,
    cmap: Optional[str] = "coolwarm",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    The distance matrix formed by the Wasserstein distance between data sets is obtained through heat maps.

    :param distance_matrix: The wasserstein distance matrix between time series datasets.
    :param dataset_list: The list of dataset names.
    :param figsize: The figure size, defaults to (6, 4).
    :param dpi: The resolution of the figure, defaults to 300.
    :param fontsize: The font size, defaults to 8.
    :param cmap: The color mapping for the heatmap, defaults to "coolwarm".

    :return: The figure and axes objects, defaults to (plt.Figure, plt.Axes).
    """

    # 1. Make sure the matrix is symmetric (distance matrices are usually symmetric)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2

    # Set the diagonal to 0 (the distance from the point to itself is 0)
    np.fill_diagonal(distance_matrix, 0)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # 2. Use imshow to draw heatmap
    im = ax.imshow(distance_matrix, cmap=cmap, interpolation="nearest")

    # 3. Add a color bar
    cbar = plt.colorbar(im)
    cbar.set_label(label="Wasserstein Distance", rotation=90, labelpad=10)

    # 4. Add tags and titles
    num_datasets = distance_matrix.shape[0]

    if dataset_list is None:
        # Use sequence when no dataset name is specified
        ax.set_xticks(range(num_datasets))
        ax.set_yticks(range(num_datasets))

    else:
        # Add the specified dataset name
        ax.set_xticks(range(num_datasets), dataset_list, rotation=75)
        ax.set_yticks(range(num_datasets), dataset_list)

    # 5. Add a numerical annotation (optional)
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            text = ax.text(
                j,
                i,
                f"{distance_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=fontsize,
            )

    plt.tight_layout()

    return fig, ax


if __name__ == "__main__":
    X = np.ones((100, 5))
    Y = np.random.rand(100, 5)

    result = wasserstein_distance(x=X, y=Y)
    print(result)

    matrix = np.random.randn(10, 10)

    data_list = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    for i in range(len(data_list)):
        data_list[i] = data_list[i] * 3
    plot_wasserstein_heatmap(matrix, data_list)
    plt.show()
