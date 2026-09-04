r"""
DeepMIMO CSI for IQ simulation
==============================

`DeepMIMO <https://github.com/DeepMIMO/DeepMIMO>`_ is a **toolchain and database for ray-tracing datasets**, not a 5G/6G system simulator. It sits between site-specific propagation engines (Sionna RT, Wireless InSite, AODT) and the toolboxes that consume channels (Sionna, MATLAB 5G, NeoRadium). Authors publish a scenario once; anyone else loads the same geometry and material parameters in seconds instead of re-running hours of ray tracing and inventing a private file format.

Typical Python usage from the `project README <https://github.com/DeepMIMO/DeepMIMO>`_:

.. code-block:: python

   import deepmimo as dm

   dm.download("asu_campus_3p5")     # fetch a scenario from the public database
   dataset = dm.load("asu_campus_3p5")
   channels = dataset.compute_channels()  # [n_ue, n_rx, n_tx, n_sub]

Ray-tracer dumps can also be converted in place:

.. code-block:: python

   scenario_name = dm.convert("path_to_ray_tracing_output")

If you use DeepMIMO in a paper, cite Alkhateeb (2019), *DeepMIMO: A Generic Deep Learning Dataset for Millimeter Wave and Massive MIMO Applications*, `arXiv:1902.06435 <https://arxiv.org/abs/1902.06435>`_.

S2Generator does **not** vendor the full DeepMIMO package. We ship a compact CSI excerpt of one public scenario so ``IQSimulator`` can fit a two-sided Doppler spectrum without downloading the 160 MB Arrow file.
"""

# %%
# Scenario ``city_37_seoul_3p5``
# ------------------------------
#
# The source dataset is temporal CSI generated from DeepMIMO scenario ``city_37_seoul_3p5`` (Seoul, 3.5 GHz):
#
# * 10 fixed speeds: :math:`10, 20, \ldots, 100` km/h
# * 16 vehicle trajectories per speed (we keep users ``0`` and ``8``)
# * :math:`T = 128` samples at :math:`\Delta t = 1` ms
# * original CSI tensor ``(time, tx_antenna, subcarrier, real_imag) = (128, 32, 32, 2)``
#
# The packaged subset keeps **one TX antenna**, **four subcarriers** ``{0, 10, 21, 31}`` spread across the band, and **two users** at every speed. On disk this is a float32 cube of shape ``(10, 2, 4, 128, 2)`` plus a 1-D metadata vector — together about 80 KiB, well under 1 MB.
#
# ``load_deepmimo_iq`` flattens a slice of that cube to ``(N, 128, 2)``, which is the layout ``IQSimulator.fit`` expects.

import numpy as np
from matplotlib import pyplot as plt
from numpy.fft import fft, fftfreq, fftshift
from scipy.signal import hilbert

from s2generator.simulator import IQSimulator
from s2generator.utils.data import (
    AVAILABLE_DEEPMIMO_DATASETS,
    list_deepmimo_speeds,
    load_deepmimo_iq,
)
from s2generator.utils.visualization import plot_iq_analysis, plot_iq_series

print("packaged DeepMIMO scenarios:", AVAILABLE_DEEPMIMO_DATASETS)
print("speeds (km/h):", list_deepmimo_speeds())

all_traces = load_deepmimo_iq()  # every speed, user, and subcarrier
print("all traces:", all_traces.shape, all_traces.dtype)
print(
    "one speed + one subcarrier:", load_deepmimo_iq(speed_kmh=30, subcarrier=21).shape
)

# %%
# A single CSI snapshot as I/Q
# ----------------------------
#
# Each packaged trace is already a complex fading coefficient over 128 ms. The first plot is the 10 km/h user-0 / subcarrier-0 path; the analysis dashboard reports :math:`\mathrm{corr}(I,Q)` and :math:`\mathrm{corr}(\mathcal{H}\{I\}, Q)`. Baseband CSI is **not** a Hilbert pair, so the second correlation stays well below one.


def iq_correlations(iq: np.ndarray):
    i_s = np.asarray(iq[0], dtype=np.float64)
    q_s = np.asarray(iq[1], dtype=np.float64)
    h_q = np.imag(hilbert(i_s - np.mean(i_s)))
    return float(np.corrcoef(i_s, q_s)[0, 1]), float(np.corrcoef(h_q, q_s)[0, 1])


slow = load_deepmimo_iq(speed_kmh=10, subcarrier=0, user=0)[0]  # (128, 2)
iq_slow = slow.T  # [2, L] for the plot helpers
corr_iq, corr_h = iq_correlations(iq_slow)
print(f"10 km/h  corr(I, Q)={corr_iq:+.3f}  corr(H(I), Q)={corr_h:+.3f}")

plot_iq_series(iq_slow, overlay=True)
plt.show()
plot_iq_analysis(iq_slow, overlay=True)
plt.show()

# %%
# Faster motion, wider Doppler
# ----------------------------
#
# The same user / subcarrier at 100 km/h. The constellation is more mixed, the instantaneous phase wanders faster, and the two-sided PSD occupies a wider Doppler band.

fast = load_deepmimo_iq(speed_kmh=100, subcarrier=0, user=0)[0]
iq_fast = fast.T
corr_iq, corr_h = iq_correlations(iq_fast)
print(f"100 km/h  corr(I, Q)={corr_iq:+.3f}  corr(H(I), Q)={corr_h:+.3f}")

plot_iq_series(iq_fast, overlay=True)
plt.show()
plot_iq_analysis(iq_fast, overlay=True)
plt.show()

# %%
# Mean PSD versus speed
# ---------------------
#
# ``IQSimulator.estimate_mean_psd_from_channels`` is the same estimator used inside ``fit``: Hann-windowed, DC-dropped, averaged over the selected traces. Overlaying 10 / 40 / 100 km/h shows the Doppler spread growing with speed — this is the :math:`P[k]` that IQ generation copies onto a real stimulus.

estimator = IQSimulator()
fig, ax = plt.subplots(figsize=(9.5, 4.2))
for speed, color in ((10, "#1d4ed8"), (40, "#0f766e"), (100, "#c2410c")):
    traces = load_deepmimo_iq(speed_kmh=speed)
    psd = estimator.estimate_mean_psd_from_channels(traces, window="hann")
    freqs = fftshift(fftfreq(psd.size))
    values = fftshift(psd) / (psd.max() + 1e-18)
    ax.semilogy(freqs, values, color=color, lw=1.7, label=f"{speed} km/h")
    print(
        f"{speed:3d} km/h  f_cut={estimator.f_cut_from_psd(psd):.4f} cycles/sample  N={traces.shape[0]}"
    )

ax.set_xlim(-0.25, 0.25)
ax.set_xlabel("cycles / sample")
ax.set_ylabel("normalized mean PSD")
ax.set_title("DeepMIMO Seoul  |  mean two-sided Doppler vs speed")
ax.legend()
ax.grid(True, which="both", alpha=0.35)
fig.tight_layout()
plt.show()

# %%
# Subcarriers at a fixed speed
# ----------------------------
#
# The four packaged subcarriers ``{0, 10, 21, 31}`` are different frequency samples of the same link. At 40 km/h their I/Q traces share a similar Doppler width but not the same fading realisation.

fig, axes = plt.subplots(2, 2, figsize=(11, 6.2), sharex=True)
for ax, sc in zip(axes.ravel(), (0, 10, 21, 31)):
    traces = load_deepmimo_iq(speed_kmh=40, subcarrier=sc, user=0)
    iq = traces[0].T
    ax.plot(iq[0], color="royalblue", lw=1.2, label="I")
    ax.plot(iq[1], color="darkorange", lw=1.2, label="Q")
    corr_iq, corr_h = iq_correlations(iq)
    ax.set_title(f"subcarrier {sc}  |  corr(H(I), Q)={corr_h:+.2f}")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper right", fontsize=8)
axes[1, 0].set_xlabel("Time Steps")
axes[1, 1].set_xlabel("Time Steps")
fig.suptitle("40 km/h  |  packaged subcarriers", fontweight="bold")
fig.tight_layout()
plt.show()

# %%
# Loading recipe
# --------------
#
# .. code-block:: python
#
#    from s2generator.utils.data import load_deepmimo_iq, list_deepmimo_speeds
#    from s2generator.simulator import IQSimulator
#
#    list_deepmimo_speeds()                    # [10, 20, ..., 100]
#    channel = load_deepmimo_iq(speed_kmh=40)  # (N, 128, 2)
#
#    sim = IQSimulator(mode="baseband")
#    sim.fit(channel_ri=channel)
#    iq = sim.transform(seq_length=128)        # [2, 2, 128] from ARMA
#
# The original Arrow file (``repo/deepmimo_city_37_seoul_3p5/train.arrow``) is **not** required at runtime. To rebuild the npy subset after changing the sampling grid, run ``repo/deepmimo_city_37_seoul_3p5/extract_bundle.py``.
