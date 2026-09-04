r"""
IQ generation: phase from a real series, spectrum from a wireless channel
=========================================================================

A complex baseband fading coefficient :math:`z[t] = I[t] + j Q[t]` is **not** an analytic signal. Its Doppler spectrum is two-sided, and :math:`I` and :math:`Q` are typically uncorrelated. An analytic pair (one-sided spectrum, :math:`Q \approx \mathcal{H}\{I\}`) is a different, equally legitimate target. Stacking a Hilbert transform, a one-sided PSD match, and a hard low-pass filter cannot produce both: the three steps overwrite each other.

``IQSimulator`` therefore splits the two goals and transfers **two different things** from two different sources:

#. **Phase** comes from a real stimulus :math:`x[t]` — either a series you provide, or two independent ARMA channels drawn by ``AutoregressiveMovingAverage`` (the same generator as ``examples/excitation/arma.py``).
#. **Spectrum shape** comes from a reference wireless channel: the mean windowed periodogram :math:`P[k]` of DeepMIMO CSI snapshots.

This notebook follows that pipeline: mathematical sketch, default ARMA generation, a user-provided stimulus, then the effect of ``mode``, ``match_mix``, vehicle speed, and the optional low-pass stage. All figures use ``plot_iq_series`` / ``plot_iq_analysis``.
"""

# %%
# Algorithm
# ---------
#
# Let :math:`x[t]` be a real series of length at least the requested IQ length :math:`L`. A padded window is z-scored (and optionally low-passed). The Hilbert transform of that *padded* window is cropped back to :math:`L`, giving a complex seed
#
# .. math::
#
#    z[t] = I[t] + j\, Q[t], \qquad Q[t] \approx \mathcal{H}\{I\}[t]
#
# on the support of the window. After Hilbert the negative-frequency half of :math:`Z[k] = \mathrm{FFT}(z)` is essentially zero. Weak bins therefore receive interpolated unwrapped phase (``_infill_phase``); otherwise a two-sided magnitude match would be forced into a conjugate-symmetric, i.e. real-valued, phase.
#
# Magnitudes are then blended toward the reference periodogram:
#
# .. math::
#
#    |Z'[k]| = (1-m)\,|Z[k]| + m\,\sqrt{P[k]}, \qquad m \in [0,1].
#
# * **``mode="baseband"`` (default).** Negative frequencies are **kept**. :math:`P[k]` is the two-sided Doppler spectrum of a fading coefficient. :math:`I` and :math:`Q` are not a Hilbert pair: :math:`\mathrm{corr}(I,Q)\approx 0` and :math:`\mathrm{corr}(\mathcal{H}\{I\}, Q)` is typically modest.
# * **``mode="analytic"``.** :math:`P[k]` is projected onto positive frequencies, the match is applied, then the same analytic projection is applied and the lost energy is restored. :math:`\mathrm{corr}(\mathcal{H}\{I\}, Q)` is then close to one.
#
# When a reference PSD is fitted, bandwidth is owned by :math:`P[k]` and the optional Butterworth low-pass is **off** unless you set ``apply_lpf=True``.

import numpy as np
from matplotlib import pyplot as plt
from numpy.fft import fft, fftfreq, fftshift
from scipy.signal import hilbert

from s2generator.simulator import IQSimulator
from s2generator.utils.data import (
    generate_damped_oscillation,
    list_deepmimo_speeds,
    load_deepmimo_iq,
)
from s2generator.utils.visualization import plot_iq_analysis, plot_iq_series


def iq_correlations(iq: np.ndarray):
    """Return corr(I, Q) and corr(H{I}, Q) for a [2, L] real IQ pair."""
    i_s = np.asarray(iq[0], dtype=np.float64)
    q_s = np.asarray(iq[1], dtype=np.float64)
    h_q = np.imag(hilbert(i_s - np.mean(i_s)))
    corr_iq = float(np.corrcoef(i_s, q_s)[0, 1])
    corr_h = float(np.corrcoef(h_q, q_s)[0, 1])
    return corr_iq, corr_h


print("packaged speeds (km/h):", list_deepmimo_speeds())
channel = load_deepmimo_iq(speed_kmh=40)
print("CSI traces for 40 km/h:", channel.shape, channel.dtype)

# %%
# Default generation: two ARMA channels, no user series
# -----------------------------------------------------
#
# ``fit`` estimates the mean Hann-windowed periodogram of the packaged CSI. ``transform`` with ``time_series=None`` draws two independent ARMA paths and converts each column, returning shape ``[2, 2, L]`` (sample :math:`\times` I/Q :math:`\times` time).

sim = IQSimulator(mode="baseband", random_state=0)
sim.fit(channel_ri=channel)
iq_batch = sim.transform(seq_length=128, num_channels=2)
print("ARMA IQ batch:", iq_batch.shape)

iq0 = iq_batch[0]
plot_iq_series(iq0, overlay=True)
plt.show()
# plot_iq_series(iq0, overlay=False)
# plt.show()

corr_iq, corr_h = iq_correlations(iq0)
print(f"channel 0: corr(I, Q) = {corr_iq:+.3f}, corr(H(I), Q) = {corr_h:+.3f}")

plot_iq_analysis(iq0, overlay=True)
plt.show()

corr_iq, corr_h = iq_correlations(iq_batch[1])
print(f"channel 1: corr(I, Q) = {corr_iq:+.3f}, corr(H(I), Q) = {corr_h:+.3f}")
plot_iq_analysis(iq_batch[1], overlay=True)
plt.show()

# %%
# User-provided stimulus
# ----------------------
#
# A damped oscillation is converted with the same fitted 40 km/h PSD. The envelope and instantaneous phase still come from the real series; the two-sided Doppler occupancy comes from the channel.

stimulus = generate_damped_oscillation(
    seq_length=256,
    freq=4.0,
    damping_factor=0.8,
    sample_rate=64.0,
    amp=1.0,
    noise_std=0.0,
)
iq_damped = sim.transform(time_series=stimulus, seq_length=128)
print("damped IQ:", iq_damped.shape)
corr_iq, corr_h = iq_correlations(iq_damped)
print(f"corr(I, Q) = {corr_iq:+.3f}, corr(H(I), Q) = {corr_h:+.3f}")

plot_iq_series(iq_damped, overlay=True)
plt.show()
plot_iq_analysis(iq_damped, overlay=True)
plt.show()

# %%
# Hyperparameters
# ---------------
#
# ``mode``: baseband vs analytic
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The same stimulus and the same CSI yield a two-sided fading pair or a one-sided analytic pair. Watch the Hilbert panel and the sign of the Doppler spectrum.

for mode in ("baseband", "analytic"):
    conv = IQSimulator(mode=mode, random_state=0).fit(channel_ri=channel)
    iq = conv.transform(time_series=stimulus, seq_length=128)
    corr_iq, corr_h = iq_correlations(iq)
    print(f"mode={mode:8s}  corr(I, Q)={corr_iq:+.3f}  corr(H(I), Q)={corr_h:+.3f}")
    plot_iq_analysis(iq, overlay=True)
    plt.suptitle(f"IQ Signal Analysis  |  mode={mode}", fontweight="bold", y=0.985)
    plt.show()

# %%
# ``match_mix``: how hard the PSD is imposed
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# ``match_mix=0`` keeps the Hilbert-seed magnitudes (almost analytic). ``match_mix=1`` fully replaces them with :math:`\sqrt{P[k]}`.

fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.4), sharey=True)
for ax, mix in zip(axes, (0.0, 0.5, 1.0)):
    conv = IQSimulator(match_mix=mix, random_state=0).fit(channel_ri=channel)
    iq = conv.transform(time_series=stimulus, seq_length=128)
    z = iq[0] + 1j * iq[1]
    psd = np.abs(fft(z - np.mean(z))) ** 2
    psd = fftshift(psd) / (psd.max() + 1e-18)
    freqs = fftshift(fftfreq(z.size))
    ax.semilogy(freqs, psd, color="royalblue", lw=1.4)
    corr_iq, corr_h = iq_correlations(iq)
    ax.set_title(f"match_mix={mix:g}\ncorr(H(I), Q)={corr_h:+.2f}")
    ax.set_xlim(-0.5, 0.5)
    ax.grid(True, which="both", alpha=0.35)
axes[0].set_ylabel("normalized PSD")
axes[1].set_xlabel("cycles / sample")
fig.suptitle("Spectral mix toward the DeepMIMO Doppler shape", fontweight="bold")
fig.tight_layout()
plt.show()

# %%
# Vehicle speed: Doppler occupancy
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Faster trajectories in ``city_37_seoul_3p5`` spread energy over a wider two-sided Doppler band. Each speed uses the same damped stimulus so the comparison isolates :math:`P[k]`.

fig, ax = plt.subplots(figsize=(9.5, 4.2))
for speed, color in ((10, "#1d4ed8"), (40, "#0f766e"), (100, "#c2410c")):
    traces = load_deepmimo_iq(speed_kmh=speed)
    conv = IQSimulator(mode="baseband", random_state=0).fit(channel_ri=traces)
    iq = conv.transform(time_series=stimulus, seq_length=128)
    z = iq[0] + 1j * iq[1]
    psd = np.abs(fft(z - np.mean(z))) ** 2
    psd = fftshift(psd) / (psd.max() + 1e-18)
    freqs = fftshift(fftfreq(z.size))
    ax.semilogy(freqs, psd, color=color, lw=1.6, label=f"{speed} km/h")
    corr_iq, corr_h = iq_correlations(iq)
    print(
        f"{speed:3d} km/h  corr(I, Q)={corr_iq:+.3f}  corr(H(I), Q)={corr_h:+.3f}  "
        f"f_cut={conv.f_cut_from_psd(conv.target_psd):.3f} cycles/sample"
    )
ax.set_xlim(-0.25, 0.25)
ax.set_xlabel("cycles / sample")
ax.set_ylabel("normalized PSD")
ax.set_title("Two-sided Doppler vs vehicle speed")
ax.legend()
ax.grid(True, which="both", alpha=0.35)
fig.tight_layout()
plt.show()

# %%
# Optional low-pass
# ~~~~~~~~~~~~~~~~~
#
# With a fitted PSD the low-pass is off by default. Forcing ``apply_lpf=True`` smooths the stimulus *before* Hilbert and can trim high-frequency phase jitter, at the cost of a slightly rounder Doppler lobe.

for apply_lpf in (False, True):
    conv = IQSimulator(apply_lpf=apply_lpf, random_state=0).fit(channel_ri=channel)
    iq = conv.transform(time_series=stimulus, seq_length=128)
    corr_iq, corr_h = iq_correlations(iq)
    print(
        f"apply_lpf={apply_lpf}  corr(I, Q)={corr_iq:+.3f}  corr(H(I), Q)={corr_h:+.3f}"
    )
    plot_iq_series(iq, overlay=True)
    plt.gca().set_title(f"IQ Time Series  |  apply_lpf={apply_lpf}", fontweight="bold")
    plt.show()

# %%
# Minimal usage
# -------------
#
# .. code-block:: python
#
#    from s2generator.simulator import IQSimulator
#    from s2generator.utils.data import load_deepmimo_iq
#    from s2generator.utils.visualization import plot_iq_analysis
#
#    channel = load_deepmimo_iq(speed_kmh=40)
#    sim = IQSimulator(mode="baseband", random_state=0)
#    sim.fit(channel_ri=channel)
#
#    # two ARMA-driven IQ pairs, shape [2, 2, 128]
#    iq_batch = sim.transform(seq_length=128)
#
#    # or convert a series you already have → shape [2, 128]
#    iq = sim.transform(time_series=x, seq_length=128)
#    plot_iq_analysis(iq)
