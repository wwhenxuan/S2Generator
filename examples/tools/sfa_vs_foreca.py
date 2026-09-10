r"""
Slow Feature Analysis versus ForeCA
===================================

Slow Feature Analysis (SFA; Wiskott & Sejnowski, 2002) finds linear
combinations with the **smallest temporal derivative** — equivalently,
the largest lag-1 autocorrelation.  ForeCA instead maximises
**forecastability** (low spectral entropy).

Those objectives are not the same.  White noise is *slower* than an AR(1)
with a negative coefficient :math:`\varphi < 0`, but it is *less*
forecastable.  This example mixes three latent series and compares the
leading SFA and ForeCA directions.
"""

import numpy as np
from matplotlib import pyplot as plt

from s2generator.utils import (
    ForeCA,
    SlowFeatureAnalysis,
    omega,
    plot_foreca,
    plot_omega,
    plot_sfa,
)

rng = np.random.RandomState(2)
n = 400
slow = np.zeros(n)
fast_neg = np.zeros(n)
e1, e2, e3 = rng.randn(n), rng.randn(n), rng.randn(n)
for i in range(1, n):
    slow[i] = 0.92 * slow[i - 1] + e1[i]
    fast_neg[i] = -0.85 * fast_neg[i - 1] + e2[i]
noise = e3

sources = np.column_stack([slow, fast_neg, noise])
mix = rng.randn(3, 3)
X = sources @ mix


def lag1_acf(z: np.ndarray) -> float:
    z = z - z.mean()
    return float(np.corrcoef(z[:-1], z[1:])[0, 1])


# %%
# Latent sources already illustrate the mismatch: the slow AR(1) has both
# high lag-1 autocorrelation *and* high Omega; white noise is slower than
# the negative-AR series but has Omega near zero.

print(
    "latent lag-1 ACF :",
    {k: round(lag1_acf(v), 3) for k, v in zip(["slow AR", "neg AR", "WN"], sources.T)},
)
print("latent Omega (%) :", np.round(omega(sources, method="welch"), 2))

# %%
# SFA orders components from slowest to fastest.  ForeCA orders them from
# most to least forecastable.

sfa = SlowFeatureAnalysis(n_comp=3).fit(X)
fore = ForeCA(n_comp=2, n_starts=4, method="welch", random_state=0).fit(X)

print("SFA  lag-1 ACF   :", [round(lag1_acf(sfa.scores[:, i]), 3) for i in range(3)])
print("SFA  Omega (%)   :", np.round(omega(sfa.scores, method="welch"), 2))
print("ForeCA Omega (%) :", np.round(fore.omega, 2))
print("ForeCA lag-1 ACF :", [round(lag1_acf(fore.scores[:, i]), 3) for i in range(2)])

fig = plot_sfa(sfa)
plt.show()

fig = plot_foreca(fore)
plt.show()

# %%
# Side-by-side Omega of the first SFA feature versus the first ForeC.

fig = plot_omega(
    [float(omega(sfa.scores[:, 0], method="welch")), float(fore.omega[0])],
    names=["SFA-1", "ForeC-1"],
)
plt.show()
