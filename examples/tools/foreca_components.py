r"""
Forecastable Component Analysis (ForeCA)
========================================

Forecastable Component Analysis (Goerg, 2013) is a dimension-reduction
method for multivariate time series.  Unlike PCA (high variance) or ICA
(statistical independence), ForeCA searches for linear combinations

.. math::

   y_t = \mathbf{X}_t \mathbf{w}

that **minimise spectral entropy** — equivalently, that maximise the
Omega forecastability score.

This example builds a four-channel mixture: one easy sinusoid, one AR(1),
and two white-noise channels.  After ``ForeCA.fit``, the leading ForeC
should recover a highly forecastable signal while later components look
like noise.
"""

import numpy as np
from matplotlib import pyplot as plt

from s2generator.utils import ForeCA, omega, plot_foreca, plot_omega

rng = np.random.RandomState(1)
n = 400
t = np.arange(n)
sine = np.sin(2.0 * np.pi * t / 20.0)
ar = np.zeros(n)
eps = rng.randn(n)
for i in range(1, n):
    ar[i] = 0.85 * ar[i - 1] + eps[i]
noise1 = rng.randn(n)
noise2 = rng.randn(n)

# Mix the latent sources so no channel is trivially a ForeC.
sources = np.column_stack([sine, ar, noise1, noise2])
mix = rng.randn(4, 4)
X = sources @ mix

# %%
# Omega of the observed channels is moderate because each column is a
# linear mix of signal and noise.

print("Omega of observed channels (%):", np.round(omega(X, method="welch"), 2))
fig = plot_omega(omega(X, method="welch"), names=[f"X{i}" for i in range(X.shape[1])])
plt.show()

# %%
# Fit two forecastable components.  ``scores`` are mean-zero, unit-variance
# and uncorrelated, ordered from most to least forecastable.

model = ForeCA(n_comp=2, n_starts=4, method="welch", random_state=0)
scores = model.fit_transform(X)
print("ForeC Omega (%):", np.round(model.omega, 2))
print("scores shape:", scores.shape)
print("mean:", np.round(scores.mean(axis=0), 6))
print("std :", np.round(scores.std(axis=0, ddof=1), 6))

fig = plot_foreca(model)
plt.show()

# %%
# The first ForeC is visibly more structured than the raw mixtures.

fig, axes = plt.subplots(2, 1, figsize=(10, 4.5), sharex=True)
axes[0].plot(X[:, 0], color="royalblue", lw=0.9)
axes[0].set_ylabel("observed X0")
axes[1].plot(model.scores[:, 0], color="darkorange", lw=0.9)
axes[1].set_ylabel("ForeC1")
axes[1].set_xlabel("Time Steps")
for ax in axes:
    ax.grid(True, alpha=0.35)
fig.suptitle("Observed mixture vs leading ForeC", fontweight="bold")
fig.tight_layout()
plt.show()
