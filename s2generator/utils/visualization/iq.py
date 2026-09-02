# -*- coding: utf-8 -*-
"""Visualize complex IQ (in-phase / quadrature) time series."""

from typing import Optional, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from numpy.fft import fft, fftfreq, fftshift
from scipy.signal import hilbert


_I_COLOR = "royalblue"
_Q_COLOR = "darkorange"


def _as_iq(time_series: Union[np.ndarray, complex]) -> Tuple[np.ndarray, np.ndarray]:
    """Parse ``[2, L]``, ``[L, 2]``, or a complex 1-D array into ``(I, Q)``."""
    data = np.asarray(time_series)
    if np.iscomplexobj(data):
        flat = np.asarray(data, dtype=np.complex128).reshape(-1)
        return np.real(flat), np.imag(flat)
    if data.ndim != 2:
        raise ValueError(
            "IQ series must be real with shape [2, seq_length] or [seq_length, 2], "
            f"or a complex 1-D array; got shape {data.shape}"
        )
    if data.shape[0] == 2:
        i_samples = np.asarray(data[0], dtype=np.float64).reshape(-1)
        q_samples = np.asarray(data[1], dtype=np.float64).reshape(-1)
    elif data.shape[1] == 2:
        i_samples = np.asarray(data[:, 0], dtype=np.float64).reshape(-1)
        q_samples = np.asarray(data[:, 1], dtype=np.float64).reshape(-1)
    else:
        raise ValueError(
            "IQ series must have one axis of length 2 "
            f"([2, seq_length] or [seq_length, 2]); got shape {data.shape}"
        )
    if i_samples.size != q_samples.size:
        raise ValueError("I and Q channels must have the same length")
    if i_samples.size < 2:
        raise ValueError("IQ series is too short to plot")
    return i_samples, q_samples


def _style_time_axis(ax: plt.Axes, seq_length: int) -> None:
    ax.set_xlim(0, seq_length - 1)
    ax.set_xlabel("Time Steps", fontsize=11.5)
    ax.grid(True, alpha=0.35)


def _plot_iq_overlay(
    ax: plt.Axes, i_samples: np.ndarray, q_samples: np.ndarray
) -> None:
    t = np.arange(i_samples.size)
    ax.plot(t, i_samples, color=_I_COLOR, lw=1.4, label="In-phase (I)")
    ax.plot(t, q_samples, color=_Q_COLOR, lw=1.4, label="Quadrature (Q)")
    ax.set_ylabel("Amplitude", fontsize=11.5)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.92)
    _style_time_axis(ax, i_samples.size)


def _plot_iq_stacked(
    ax_i: plt.Axes,
    ax_q: plt.Axes,
    i_samples: np.ndarray,
    q_samples: np.ndarray,
) -> None:
    t = np.arange(i_samples.size)
    ax_i.plot(t, i_samples, color=_I_COLOR, lw=1.4)
    ax_q.plot(t, q_samples, color=_Q_COLOR, lw=1.4)
    ax_i.set_ylabel("In-phase (I)", fontsize=11)
    ax_q.set_ylabel("Quadrature (Q)", fontsize=11)
    ax_i.grid(True, alpha=0.35)
    ax_i.set_xlim(0, i_samples.size - 1)
    ax_i.tick_params(labelbottom=False)
    _style_time_axis(ax_q, q_samples.size)


def plot_iq_series(
    time_series: np.ndarray,
    overlay: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 160,
) -> plt.Figure:
    """
    Plot an IQ pair as in-phase (royalblue) and quadrature (darkorange).

    :param time_series: Real array of shape ``[2, seq_length]`` or
                        ``[seq_length, 2]``. A complex 1-D array is also accepted.
    :param overlay: If True, draw I and Q on one axes. If False, stack them
                    as two rows that share the time axis.
    :param figsize: Optional figure size. Defaults to ``(12, 3.2)`` (overlay)
                    or ``(12, 5.2)`` (stacked).
    :param dpi: Figure resolution.
    :return: Matplotlib Figure.
    """
    i_samples, q_samples = _as_iq(time_series)

    if overlay:
        if figsize is None:
            figsize = (12, 3.2)
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        _plot_iq_overlay(ax, i_samples, q_samples)
        ax.set_title("IQ Time Series", fontweight="bold", fontsize=13)
        fig.tight_layout()
        return fig

    if figsize is None:
        figsize = (12, 5.2)
    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=figsize,
        dpi=dpi,
        sharex=True,
    )
    _plot_iq_stacked(axes[0], axes[1], i_samples, q_samples)
    axes[0].set_title("IQ Time Series", fontweight="bold", fontsize=13)
    fig.tight_layout()
    return fig


def plot_iq_analysis(
    time_series: np.ndarray,
    overlay: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 160,
) -> plt.Figure:
    """
    IQ dashboard: traces, magnitude, two-sided PSD, and Hilbert analysis.

    Layout (3 × 2)::

        IQ time series          I/Q constellation
        Magnitude |z|           Instantaneous phase
        Two-sided PSD           Hilbert: H{I} vs Q

    The Hilbert panel compares the quadrature channel with the Hilbert
    transform of the in-phase channel. For a true analytic pair they
    overlap; for baseband fading they typically do not.

    :param time_series: Real array of shape ``[2, seq_length]`` or
                        ``[seq_length, 2]``, or a complex 1-D array.
    :param overlay: How to draw I/Q in the top-left panel (same meaning as
                    :func:`plot_iq_series`).
    :param figsize: Optional figure size; defaults to ``(12.5, 9.6)``.
    :param dpi: Figure resolution.
    :return: Matplotlib Figure.
    """
    i_samples, q_samples = _as_iq(time_series)
    n = i_samples.size
    t = np.arange(n)
    z = i_samples + 1j * q_samples
    magnitude = np.abs(z)
    phase = np.unwrap(np.angle(z))
    analytic_i = hilbert(i_samples - np.mean(i_samples))
    hilbert_q = np.imag(analytic_i)

    z_centered = z - np.mean(z)
    psd = np.abs(fft(z_centered, n=n)) ** 2
    freqs = fftshift(fftfreq(n))
    psd = fftshift(psd)
    psd = psd / (psd.max() + 1e-18)

    iq_corr = float(np.corrcoef(i_samples, q_samples)[0, 1])
    hilbert_corr = float(np.corrcoef(hilbert_q, q_samples)[0, 1])

    if figsize is None:
        figsize = (12.5, 9.6)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = GridSpec(
        3,
        2,
        figure=fig,
        height_ratios=[1.15, 1.0, 1.12],
        hspace=0.38,
        wspace=0.28,
        left=0.07,
        right=0.98,
        top=0.95,
        bottom=0.07,
    )

    if overlay:
        ax_iq = fig.add_subplot(gs[0, 0])
        _plot_iq_overlay(ax_iq, i_samples, q_samples)
        ax_iq.set_title("IQ Time Series", fontweight="bold", fontsize=12)
    else:
        inner = gs[0, 0].subgridspec(2, 1, hspace=0.08)
        ax_i = fig.add_subplot(inner[0])
        ax_q = fig.add_subplot(inner[1], sharex=ax_i)
        _plot_iq_stacked(ax_i, ax_q, i_samples, q_samples)
        ax_i.set_title("IQ Time Series", fontweight="bold", fontsize=12)

    ax_const = fig.add_subplot(gs[0, 1])
    sc = ax_const.scatter(
        i_samples,
        q_samples,
        c=t,
        cmap="viridis",
        s=14,
        linewidths=0,
    )
    fig.colorbar(sc, ax=ax_const, fraction=0.046, pad=0.04, label="Time")
    ax_const.set_xlabel("In-phase (I)", fontsize=11)
    ax_const.set_ylabel("Quadrature (Q)", fontsize=11)
    ax_const.set_title("I/Q Constellation", fontweight="bold", fontsize=12)
    ax_const.set_aspect("equal", adjustable="datalim")
    ax_const.grid(True, alpha=0.35)
    ax_const.axhline(0.0, color="#94a3b8", lw=0.8)
    ax_const.axvline(0.0, color="#94a3b8", lw=0.8)

    ax_mag = fig.add_subplot(gs[1, 0])
    ax_mag.plot(t, magnitude, color="#0f766e", lw=1.5)
    ax_mag.set_title("Magnitude  $|z|$", fontweight="bold", fontsize=12)
    ax_mag.set_ylabel("Amplitude", fontsize=11.5)
    _style_time_axis(ax_mag, n)

    ax_phase = fig.add_subplot(gs[1, 1])
    ax_phase.plot(t, phase, color="#7c3aed", lw=1.5)
    ax_phase.set_title("Instantaneous Phase", fontweight="bold", fontsize=12)
    ax_phase.set_ylabel("Radians", fontsize=11.5)
    _style_time_axis(ax_phase, n)

    ax_psd = fig.add_subplot(gs[2, 0])
    ax_psd.semilogy(freqs, psd, color=_I_COLOR, lw=1.5)
    ax_psd.set_xlim(-0.5, 0.5)
    ax_psd.set_xlabel("Cycles / sample", fontsize=11.5)
    ax_psd.set_ylabel("Normalized PSD", fontsize=11.5)
    ax_psd.set_title("Two-sided Power Spectrum", fontweight="bold", fontsize=12)
    ax_psd.grid(True, which="both", alpha=0.35)

    ax_hilbert = fig.add_subplot(gs[2, 1])
    ax_hilbert.plot(
        t, hilbert_q, color=_I_COLOR, lw=1.4, label=r"Hilbert $\mathcal{H}\{I\}$"
    )
    ax_hilbert.plot(t, q_samples, color=_Q_COLOR, lw=1.4, label="Quadrature (Q)")
    ax_hilbert.set_title("Hilbert Analysis", fontweight="bold", fontsize=12)
    ax_hilbert.set_ylabel("Amplitude", fontsize=11.5)
    ax_hilbert.legend(loc="upper right", fontsize=8.5, framealpha=0.92)
    _style_time_axis(ax_hilbert, n)
    ax_hilbert.text(
        0.02,
        0.95,
        f"corr(I, Q) = {iq_corr:+.3f}\ncorr(H(I), Q) = {hilbert_corr:+.3f}",
        transform=ax_hilbert.transAxes,
        va="top",
        ha="left",
        fontsize=9.5,
        color="#0f172a",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, lw=0.4),
    )

    fig.suptitle("IQ Signal Analysis", fontweight="bold", fontsize=14, y=0.985)
    return fig
