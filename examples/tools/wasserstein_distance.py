r"""
Measuring Time-Series Dataset Similarity using Wasserstein Distance
===================================================================

The Wasserstein distance is used to measure the similarity between two datasets.
"Measuring Time-Series Dataset Similarity using Wasserstein Distance." (Paper address: https://www.arxiv.org/abs/2507.22189)

This method models the time series dataset as a multivariate normal distribution and uses the Wasserstein similarity metric to measure the distribution distance between datasets, thereby characterizing the similarity or difference between the datasets.

.. math::

   W_p (\mu, \nu) = {\mathrm{min}}_{\gamma \in \Gamma _{\mu, \nu}} \left ( \int \left \| x - y \right \| ^ p \gamma \left ( \mathrm d x, \mathrm d y \right ) \right ) ^ {1/p}

where, :math:`\mu, \nu \in \mathbb R ^ d` and :math:`\Gamma` denotes the coupled set of :math:`\mu` and :math:`\nu`.

This paper focuses on the Wasserstein distance :math:`d_{\mathrm Ws}` between two multivariate normal distributions :math:`\mathcal D_X` and :math:`\mathcal D_Y`, which is defined as:

.. math::

   d^2_{\mathrm W _s} \left ( \mathcal D_X, \mathcal D_Y \right ) = \left \| \hat{\mathbf{\mu}}_X - \hat{\mathbf{\mu}}_Y \right \| ^ 2 + \mathrm{tr} \left ( \hat{\mathbf{\Sigma}}_X + \hat{\mathbf{\Sigma}}_Y - 2 \sqrt{\hat{\mathbf{\Sigma}}_X \hat{\mathbf{\Sigma}}_Y} \right )

Where the first term measures the distance between the mean vectors, and the second term captures the difference between the covariance matrices. If two datasets are to be close to each other, they must not only have similar mean vectors, but also similar covariate matrices.

In s2generator, we reproduced this method and provided an interface to calculate the Wasserstein distance between two time series datasets. Users can use this interface to evaluate the similarity between different time series datasets, thereby better understanding and analyzing the relationships between them.

Furthermore, we also provided a function to visualize the distance matrix, which can more intuitively show the similarity between different datasets.
"""

import numpy as np
from s2generator.utils import (
    wasserstein_distance,
    wasserstein_distance_matrix,
    plot_wasserstein_heatmap,
)
from s2generator.utils._wasserstein_distance import time_series_to_distribution

# %%
# Wasserstein distance requires first calculating the mean vector and covariance matrix of the two datasets, and then calculating the distance value using the formula above. Users can control the importance of the mean vector and covariance matrix in the distance calculation by adjusting their weights, thus better assessing the similarity between different datasets.
#
# Here, we first generate the test data :math:`X, Y \in \mathbb{R}^{N \times L}` and calculate its multivariate time series distribution.

# Define test data parameters
N1 = 100  # Number of samples in test dataset
N2 = 50
L = 48  # Length of each time series

x = np.random.rand(N1, L)  # First dataset
y = np.random.rand(N2, L)  # Second dataset

# Calculate the distribution of a multivariate time series dataset
mean_vector, cov_matrix = time_series_to_distribution(x)

print(mean_vector.shape, cov_matrix.shape)

# %%
# Calculating the multivariate distribution of the time series dataset is the first step in calculating the Wasserstein distance. The complete calculation process is further integrated into our ``wasserstein_distance`` function, as shown below.
#
# It is worth noting that since this process involves the inner product of the p-norms of two vectors, we require that the sampled subsequences from both datasets have the same length.

# Call this function to calculate the distance
distances, mean_value, covar_value = wasserstein_distance(
    x, y, mean_weight=0.5, covar_weight=0.5, return_all=True
)

# The smaller the distance value, the closer the distributions of the two datasets are.
print("Wasserstein Distance:", distances)

# %%
# To calculate the distance between multiple datasets, we can store the datasets in a list and use the ``wasserstein_distance_matrix`` function to compute the distance matrix between them. Finally, we can use the ``plot_wasserstein_heatmap`` function to visualize the distance matrix, thus more intuitively showing the similarity between different datasets.

# Generate multiple random datasets
datasets = [np.random.randn(np.random.randint(10, N1), L) for _ in range(6)]
distance_matrix = wasserstein_distance_matrix(
    datasets, mean_weight=0.5, covar_weight=0.5
)

print("Distance Matrix:\n", distance_matrix)

plot_wasserstein_heatmap(distance_matrix=distance_matrix)
