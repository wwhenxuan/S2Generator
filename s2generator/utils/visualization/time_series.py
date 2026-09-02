# -*- coding: utf-8 -*-
"""Plot univariate, multivariate, and S2 excitation / response series."""

from typing import Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt


def plot_univariate_time_series(
    time_series: np.ndarray, figsize: Tuple[int, int] = (12, 3), dpi: int = 256
) -> plt.Figure:
    """
    Visualize a single univariate time series.

    :param time_series: 1D array of shape ``(L,)``, or 2D ``(1, L)`` / ``(L, 1)``.
    :param figsize: The size of the figure for the generated plot.
    :param dpi: Dots per inch (resolution) for the generated plot.
    :return: A matplotlib Figure object containing the time series plot.
    """
    series = np.asarray(time_series)
    if series.ndim == 2:
        if 1 in series.shape:
            series = series.reshape(-1)
        else:
            raise ValueError(
                "plot_univariate_time_series expects a 1D series; "
                f"got shape {series.shape}. Use plot_multivariate_time_series "
                "for multiple channels."
            )
    elif series.ndim != 1:
        raise ValueError(
            f"plot_univariate_time_series expects 1D input, got shape {series.shape}"
        )

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(series, color="royalblue")
    ax.set_title("Univariate Time Series", fontweight="bold", fontsize=13)
    ax.set_xlabel("Time Steps", fontsize=11.5)
    ax.set_ylabel("Value", fontsize=11.5)
    ax.grid(True)

    return fig


def plot_multivariate_time_series(
    time_series: np.ndarray,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 160,
    sharex: bool = True,
) -> plt.Figure:
    """
    Visualize multivariate / multi-sample time series as stacked row subplots.

    :param time_series: Array of shape ``[num_samples, seq_length]``
                        (e.g. CauKer output ``(d, L)``).
    :param figsize: Optional figure size; defaults to scale with ``num_samples``.
    :param dpi: Figure resolution.
    :param sharex: Whether subplots share the x-axis.
    :return: Matplotlib Figure with one row per sample.
    """
    data = np.asarray(time_series)
    if data.ndim != 2:
        raise ValueError(
            "plot_multivariate_time_series expects shape [num_samples, seq_length], "
            f"got {data.shape}"
        )
    n_samples, seq_length = data.shape
    if n_samples < 1:
        raise ValueError("time_series must contain at least one sample row")

    if figsize is None:
        figsize = (12, max(2.0, 1.8 * n_samples))

    fig, axes = plt.subplots(
        nrows=n_samples,
        ncols=1,
        figsize=figsize,
        dpi=dpi,
        sharex=sharex,
        squeeze=False,
    )
    for i in range(n_samples):
        ax = axes[i, 0]
        ax.plot(data[i], color="royalblue")
        ax.set_ylabel(f"Dim {i}", fontsize=10)
        ax.grid(True, alpha=0.35)
        ax.set_xlim(0, seq_length - 1 if seq_length > 1 else 1)
    axes[-1, 0].set_xlabel("Time Steps", fontsize=11)
    axes[0, 0].set_title("Multivariate Time Series", fontweight="bold", fontsize=12)
    fig.tight_layout()
    return fig


def plot_symbol_series(x: np.ndarray, y: np.ndarray) -> plt.Figure:
    """
    Visualize Series-Symbol (S2) excitation / response pairs.

    :param x: input sampling (excitation) series of shape ``(L, d_in)``.
    :param y: output (response) series of shape ``(L, d_out)``.
    :return: the plot figure of matplotlib
    """

    # Determine the shape and length of the data
    seq_length, input_dim = x.shape
    _, output_dim = y.shape
    max_dim = max(input_dim, output_dim)

    # Create a matplotlib plotting object
    fig, axes = plt.subplots(
        nrows=max_dim, ncols=2, figsize=(12, 2 * max_dim), sharex=True
    )

    # Plot the input sequence
    for i in range(input_dim):
        if max_dim == 1:
            ax = axes[0]
        else:
            ax = axes[i, 0]
        ax.plot(x[:, i], color="royalblue")
        ax.set_ylabel(f"Input Dim {i + 1}", fontsize=10)
        ax.set_xlim(0, seq_length)

    # Plot the output sequence
    for i in range(output_dim):
        if max_dim == 1:
            ax = axes[1]
        else:
            ax = axes[i, 1]
        ax.plot(y[:, i], color="royalblue")
        ax.set_ylabel(f"Output Dim {i + 1}", fontsize=10)
        ax.set_xlim(0, seq_length)

    # Add titles to the two columns of images
    if max_dim == 1:
        axes[0].set_title("Input Data", fontsize=12)
        axes[1].set_title("Output Data", fontsize=12)
    else:
        axes[0, 0].set_title("Input Data", fontsize=12)
        axes[0, 1].set_title("Output Data", fontsize=12)

    return fig
