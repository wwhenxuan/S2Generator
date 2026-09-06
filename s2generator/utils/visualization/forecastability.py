# -*- coding: utf-8 -*-
"""Plots for spectral entropy, Omega, ForeCA, and Slow Feature Analysis."""

from typing import Optional, Sequence, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt

from s2generator.utils.forecastability import ForeCA, SlowFeatureAnalysis


def plot_omega(
    values: Union[float, np.ndarray, Sequence[float]],
    names: Optional[Sequence[str]] = None,
    figsize: Tuple[int, int] = (8, 4),
    dpi: int = 128,
) -> plt.Figure:
    """
    Bar chart of forecastability scores in ``[0, 100]``.

    :param values: One or more Omega percentages.
    :param names: Optional tick labels.
    :param figsize: Figure size.
    :param dpi: Figure resolution.
    :return: Matplotlib figure.
    """
    omega = np.atleast_1d(np.asarray(values, dtype=float)).reshape(-1)
    labels = (
        list(names) if names is not None else [f"Series {i}" for i in range(omega.size)]
    )
    if len(labels) != omega.size:
        raise ValueError("names must match the number of Omega values.")

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.bar(np.arange(omega.size), omega, color="royalblue", edgecolor="k", lw=0.6)
    ax.set_xticks(np.arange(omega.size))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(r"$\Omega$ (%)", fontsize=11)
    ax.set_ylim(0.0, 100.0)
    ax.set_title("Forecastability", fontweight="bold", fontsize=13)
    ax.grid(True, axis="y", alpha=0.35)
    fig.tight_layout()
    return fig


def plot_spectrum(
    freqs: np.ndarray,
    spec: np.ndarray,
    log: bool = False,
    figsize: Tuple[int, int] = (8, 3.5),
    dpi: int = 128,
) -> plt.Figure:
    """
    Plot a univariate positive-frequency spectrum.

    :param freqs: Frequencies in radians (or any consistent unit).
    :param spec: Real non-negative spectral density, same length as ``freqs``.
    :param log: If True, use a log y-scale.
    :param figsize: Figure size.
    :param dpi: Figure resolution.
    :return: Matplotlib figure.
    """
    freqs = np.asarray(freqs, dtype=float).reshape(-1)
    spec = np.real(np.asarray(spec)).reshape(-1)
    if freqs.size != spec.size:
        raise ValueError("freqs and spec must have the same length.")

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(freqs, spec, color="royalblue", lw=1.6)
    ax.set_xlabel(r"Frequency $\lambda$", fontsize=11)
    ax.set_ylabel("Spectrum", fontsize=11)
    ax.set_title("Spectral density", fontweight="bold", fontsize=13)
    if log:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    return fig


def plot_foreca(
    model: ForeCA,
    figsize: Tuple[int, int] = (11, 6),
    dpi: int = 128,
) -> plt.Figure:
    """
    Plot fitted ForeC scores together with their Omega bars.

    :param model: A fitted :class:`~s2generator.utils.forecastability.ForeCA`.
    :param figsize: Figure size.
    :param dpi: Figure resolution.
    :return: Matplotlib figure with scores on the left and Omega on the right.
    """
    scores = model.scores
    omegas = np.atleast_1d(model.omega)
    t, n_comp = scores.shape

    fig, axes = plt.subplots(
        n_comp,
        2,
        figsize=figsize,
        dpi=dpi,
        gridspec_kw={"width_ratios": [3.2, 1.0]},
        squeeze=False,
        sharex=False,
    )
    for i in range(n_comp):
        axes[i, 0].plot(scores[:, i], color="royalblue", lw=1.1)
        axes[i, 0].set_ylabel(f"ForeC{i + 1}", fontsize=10)
        axes[i, 0].grid(True, alpha=0.35)
        axes[i, 0].set_xlim(0, max(t - 1, 1))
        axes[i, 1].barh([0], [omegas[i]], color="darkorange", edgecolor="k", height=0.5)
        axes[i, 1].set_xlim(0.0, 100.0)
        axes[i, 1].set_yticks([])
        axes[i, 1].set_xlabel(r"$\Omega$ (%)" if i == n_comp - 1 else "")
        axes[i, 1].grid(True, axis="x", alpha=0.35)
    axes[0, 0].set_title("Forecastable components", fontweight="bold", fontsize=12)
    axes[-1, 0].set_xlabel("Time Steps", fontsize=11)
    fig.tight_layout()
    return fig


def plot_sfa(
    model: SlowFeatureAnalysis,
    figsize: Tuple[int, int] = (11, 5),
    dpi: int = 128,
) -> plt.Figure:
    """
    Plot fitted slow-feature scores (slowest on top).

    :param model: A fitted :class:`~s2generator.utils.forecastability.SlowFeatureAnalysis`.
    :param figsize: Figure size.
    :param dpi: Figure resolution.
    :return: Matplotlib figure.
    """
    scores = model.scores
    t, n_comp = scores.shape
    fig, axes = plt.subplots(
        n_comp, 1, figsize=figsize, dpi=dpi, sharex=True, squeeze=False
    )
    for i in range(n_comp):
        axes[i, 0].plot(scores[:, i], color="royalblue", lw=1.1)
        axes[i, 0].set_ylabel(f"SF{i + 1}", fontsize=10)
        axes[i, 0].grid(True, alpha=0.35)
        axes[i, 0].set_xlim(0, max(t - 1, 1))
    axes[0, 0].set_title("Slow features (slow → fast)", fontweight="bold", fontsize=12)
    axes[-1, 0].set_xlabel("Time Steps", fontsize=11)
    fig.tight_layout()
    return fig
