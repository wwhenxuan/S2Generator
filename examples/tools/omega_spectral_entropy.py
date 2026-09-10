r"""
Spectral entropy and the Omega forecastability score
====================================================

ForeCA (Goerg, 2013) measures how easy a stationary series is to forecast by
the Shannon entropy of its **normalized spectral density**. A flat spectrum
(white noise) has maximal entropy and is scored as unpredictable; a spectrum
with a single peak (a sinusoid) has low entropy and is almost perfectly
forecastable.

The forecastability score is

.. math::

   \Omega(x_t) = \bigl(1 - H_s(x_t)\bigr) \times 100 \in [0, 100],

where :math:`H_s` is the spectral entropy (normalized so that a discrete
uniform spectrum has entropy 1). This example compares white noise, an AR(1)
process, and a pure sinusoid.
"""

import numpy as np
from matplotlib import pyplot as plt

from s2generator.utils import (
    omega,
    plot_omega,
    plot_spectrum,
    spectral_entropy,
    univariate_spectrum,
)

rng = np.random.RandomState(0)
n = 512
t = np.arange(n)

noise = rng.randn(n)
ar = np.zeros(n)
eps = rng.randn(n)
for i in range(1, n):
    ar[i] = 0.9 * ar[i - 1] + eps[i]
sine = np.sin(2.0 * np.pi * t / 16.0)

# %%
# Omega is computed column-wise on a ``(T, K)`` array.  The sinusoid should
# sit well above the AR(1), which in turn sits above white noise.

series = np.column_stack([noise, ar, sine])
names = ["white noise", "AR(1) φ=0.9", "sinusoid"]
om = omega(series, method="welch")
print("Omega (%):", dict(zip(names, np.round(om, 2))))
print(
    "spectral entropy:",
    dict(zip(names, np.round(spectral_entropy(series, method="welch"), 3))),
)

fig = plot_omega(om, names=names)
plt.show()

# %%
# The corresponding Welch spectra make the ranking visible: noise is nearly
# flat, the AR(1) emphasizes low frequencies, and the sinusoid is a sharp peak.

fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
for ax, col, name in zip(axes, series.T, names):
    freqs, spec = univariate_spectrum(col, method="welch")
    ax.plot(freqs, spec, color="royalblue", lw=1.5)
    ax.set_ylabel(name, fontsize=10)
    ax.grid(True, alpha=0.35)
axes[-1].set_xlabel(r"Frequency $\lambda$ (rad)", fontsize=11)
axes[0].set_title("Welch spectra", fontweight="bold")
fig.tight_layout()
plt.show()

# %%
# ``plot_spectrum`` is the univariate helper used throughout the gallery.

freqs, spec = univariate_spectrum(sine, method="pgram")
fig = plot_spectrum(freqs, spec, log=True)
plt.show()
