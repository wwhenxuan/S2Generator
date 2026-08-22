# -*- coding: utf-8 -*-
"""
Created on 2025/01/25 00:02:43
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""

__all__ = [
    "plot_univariate_time_series",
    "plot_symbol_series",
    "plot_symbol",
    "plot_shapiro_wilk",
    "plot_simulator_statistics",
    "plot_adjacency_matrix",
    "plot_graph",
    "plot_multivariate_time_series",
    "plot_correlation",
]

from typing import Optional, Union, Dict, Any, Tuple, List

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.transforms import Bbox

from scipy import signal
from statsmodels.tsa.stattools import acf

from s2generator.symbol.base import Node, NodeList
from s2generator.symbol.print_symbol import symbol_to_markdown


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


def plot_symbol_series(x: np.ndarray, y: np.ndarray) -> plt.Figure:
    """
    Visualize Series-Symbol (S2) excitation / response pairs.

    :param x: input sampling (excitation) series of shape ``(L, d_in)``.
    :param y: output (response) series of shape ``(L, d_out)``.
    :return: the plot figure of matplotlib
    """

    # Determine the shape and length of the data
    seq_len, input_dim = x.shape
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
        ax.set_xlim(0, seq_len)

    # Plot the output sequence
    for i in range(output_dim):
        if max_dim == 1:
            ax = axes[1]
        else:
            ax = axes[i, 1]
        ax.plot(y[:, i], color="royalblue")
        ax.set_ylabel(f"Output Dim {i + 1}", fontsize=10)
        ax.set_xlim(0, seq_len)

    # Add titles to the two columns of images
    if max_dim == 1:
        axes[0].set_title("Input Data", fontsize=12)
        axes[1].set_title("Output Data", fontsize=12)
    else:
        axes[0, 0].set_title("Input Data", fontsize=12)
        axes[0, 1].set_title("Output Data", fontsize=12)

    return fig


def which_edges_out(
    artist: Union[plt.Text, Any], *, padding: Optional[int] = 0
) -> Dict[str, bool]:
    """
    Determine which edges of the canvas the artist is outside.

    :param artist: Additional safety margin in pixels (can be negative to indicate "almost outside").
    :param padding: number of pixels around the edge of the canvas.
    :return: Returns a dict: {'top', 'bottom', 'left', 'right'} -> True/False.
    """
    fig = artist.figure
    if fig is None:
        raise ValueError("artist has not been added to any figures")

    # Rendering the object
    renderer = fig.canvas.get_renderer()
    bbox = artist.get_window_extent(renderer=renderer)

    # Consider padding
    if padding:
        bbox = bbox.expanded(padding / fig.dpi, padding / fig.dpi)

    # Canvas pixel boundaries
    w, h = fig.canvas.get_width_height()
    canvas = Bbox([[0, 0], [w, h]])

    return {
        "left": bbox.xmin
        < canvas.xmin,  # The entire box is outside the left side of the canvas
        "right": bbox.xmax
        > canvas.xmax,  # The entire box is outside the right side of the canvas
        "bottom": bbox.ymin < canvas.ymin,  # The whole box is outside the canvas
        "top": bbox.ymax > canvas.ymax,  # The entire box is outside the canvas
    }


def create_symbol_figure(
    symbol: Union[str, List[str]], width: float, height: float, dpi: Optional[int] = 300
) -> Tuple[plt.Figure, plt.Axes, List[plt.Text]]:
    """
    Create a specific Figure object for visualization.

    :param symbol: The symbolic expression data to be visualized.
    :param width: The width of the drawn image may need to be adjusted multiple times.
    :param height: The height of the drawn image. If it is None, it will be automatically specified by the algorithm.
    :param dpi: The resolution of the visualization image.
    :return: The figure and axes objects.
    """
    # Create the Figure for matplotlib
    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)

    # Fill the entire picture
    # ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Completely blank, no axes are displayed
    ax.axis("off")

    if isinstance(symbol, str):
        text = ax.text(
            0.5, 0.5, symbol, ha="center", va="center", fontsize=14
        )  # Remove usetex=True
        text = [text]
    elif isinstance(symbol, list):
        number = len(symbol)
        # Determine the vertical coordinate position of each symbol visualization
        position = np.arange(0, number + 2)
        position = (position - position.min()) / (position.max() - position.min())
        position = position[1:-1]
        text = [
            ax.text(0.5, pos, s, ha="center", va="center", fontsize=14)
            for (s, pos) in zip(symbol, position[::-1])
        ]
    else:
        raise ValueError("symbol must be str or list")

    for spine in ax.spines.values():
        spine.set_visible(False)

    # Remove the x and y axis scales
    ax.set_xticks([])
    ax.set_yticks([])

    # Remove the tick labels (you can skip this step if you only want to hide the tick marks but keep the labels)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    return fig, ax, text


def plot_symbol(
    symbol: Union[str, Node, NodeList],
    width: Optional[int] = 20,
    height: Optional[int] = None,
    dpi: Optional[int] = 160,
    return_all: Optional[str] = False,
) -> Union[plt.Figure, Tuple[plt.Figure, plt.Axes, List[plt.Text]]]:
    """
    This function visualizes symbolic data.
    Since the input symbolic expression data varies, you may need to adjust the width multiple times in actual use.

    :param symbol: The symbolic data to be visualized.
    :param width: The width of the drawn image may need to be adjusted multiple times.
    :param height: The height of the drawn image. If it is None, it will be automatically specified by the algorithm.
    :param dpi: The resolution of the visualization image.
    :param return_all: Whether to return all visualization information
    :return: - True: return (Figure, Axis, List[Text]),
             - False: return Figure.
    """
    # Transform the symbol from string to markdown
    symbol_list = symbol_to_markdown(symbol)

    # Add y and subscript to each symbol
    symbol_list = [f"$ y_{i} = {sym} $" for (i, sym) in enumerate(symbol_list)]

    # Give the initial height and width values
    if height is None:
        height = 0.50 * len(symbol_list)

    # Visualizing symbols
    fig, ax, text = create_symbol_figure(symbol_list, width, height, dpi=dpi)

    # Whether to return all drawing information
    if return_all is True:
        return fig, ax, text
    return fig


def plot_shapiro_wilk(
    residuals: np.ndarray,
    bins: int = 13,
    dpi: int = 500,
    figsize: Tuple[int, int] = (12, 5),
) -> Tuple[plt.Figure, float, float]:
    """
    Plot the Shapiro-Wilk test for normality of the residuals.
    This method generates a Q-Q plot to visually assess whether the residuals
    of the fitted ARIMA model follow a normal distribution.

    :param residuals: Residuals from the fitted ARIMA model.
    :param bins: Number of bins for the histogram of residuals.
    :param dpi: Dots per inch (resolution) for the generated plot.
    :param figsize: Figure size for the generated plot.
    :return: A tuple containing the matplotlib Figure object, the Shapiro-Wilk statistic, and the p-value.
    """
    # Ensure the model has been fitted and the residuals have been calculated.
    if residuals is None:
        raise ValueError("Residuals must be provided before calling plot_shapiro_wilk.")

    # Convert residuals to a numpy array for consistency
    residuals = np.asarray(residuals)

    # Import necessary libraries
    from statsmodels.graphics.gofplots import qqplot
    from scipy.stats import shapiro

    # import seaborn as sns
    # sns.set_theme(style="ticks")

    # Perform Shapiro-Wilk normality test
    stat, p_value = shapiro(residuals)

    # Create visualization figure
    fig, ax = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    fig.subplots_adjust(wspace=0.16)

    # Plot histogram of the fitted residuals
    ax[0].hist(residuals, bins=bins, alpha=1, color="w", edgecolor="k", lw=1.2)

    # Plot Q-Q plot for normality test
    qqplot(
        residuals,
        line="s",
        ax=ax[1],
        markerfacecolor="white",
        markeredgecolor="k",
        markersize=7.5,
    )
    for line in ax[1].get_lines():
        if line.get_linestyle() == "-":
            line.set_color("#DC143C")
            line.set_linewidth(2.1)

    # Set titles and labels
    ax[0].grid(which="major", color="gray", linestyle="--", lw=0.5, alpha=0.8)
    ax[1].grid(which="major", color="gray", linestyle="--", lw=0.5, alpha=0.8)
    ax[0].set_xlabel("Standard Residual", fontsize=12.5)
    ax[0].set_ylabel("Frequency", fontsize=12.5)
    ax[1].set_xlabel("Theoretical Quantiles", fontsize=12.5)
    ax[1].set_ylabel("Sample Quantiles", fontsize=12.5)

    # Annotate the plots with statistics
    mean = np.round(np.mean(residuals), 4)
    std = np.round(np.std(residuals), 4)
    stat = np.round(stat, 4)
    p_value = np.round(p_value, 4)

    # Set the text annotations for the mean and std on the histogram
    ax[0].text(
        0.05,
        0.95,
        f"$\mu$ = {mean}\n$\sigma$ = {std}",
        transform=ax[0].transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        fontsize=13.5,
        color="k",
    )

    # Set the text annotations for the Shapiro-Wilk test on the Q-Q plot
    ax[1].text(
        0.05,
        0.95,
        f"$W$ = {stat}\n$p$ = {p_value}",
        transform=ax[1].transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        fontsize=13.5,
        color="k",
    )

    return fig, stat, p_value


def plot_simulator_statistics(
    original_series: np.ndarray,
    generated_series: np.ndarray,
    residuals: np.ndarray = None,
    nlags: int = 50,
    nperseg: int = 128,
    bins: int = 30,
    figsize: Tuple[int, int] = None,
) -> plt.Figure:
    """
    Compare the statistical properties of the original series and the generated series.

    This section mainly focuses on the basic characterization, distribution, autocorrelation function,
    power spectral density, and residual test of the input and generated time series.

    :param original_series: The original time series data.
    :param generated_series: The time series data generated by the model.
    :param residuals: The residuals from the model fit, if available.
    :param nlags: The number of lags to compute for the autocorrelation function.
    :param nperseg: The length of each segment for the Welch method in power spectral density estimation.
    :param bins: The number of bins to use for the histogram plots.
    :param figsize: The size of the figure for the generated plots. If None, it will be automatically determined.

    :return: A matplotlib Figure object containing the comparison plots.
    """
    # Ensure that the input data is a one-dimensional ndarray
    original_series = np.asarray(original_series).flatten()
    generated_series = np.asarray(generated_series).flatten()

    # Calculate the autocorrelation function
    acf_original = acf(original_series, nlags=nlags, fft=True)
    acf_generated = acf(generated_series, nlags=nlags, fft=True)

    # Calculate power spectral density
    f_original, Pxx_original = signal.welch(original_series, fs=1.0, nperseg=nperseg)
    f_generated, Pxx_generated = signal.welch(generated_series, fs=1.0, nperseg=nperseg)

    # Plot comparison
    # Here we need to determine if the residuals from the model fit have been passed in.
    # If residuals are present, a subplot can be added to display their statistical properties.
    if residuals is not None:
        fig, axes = plt.subplots(3, 2, figsize=(12, 10) if figsize is None else figsize)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 7) if figsize is None else figsize)

    # Original sequence and generated sequence
    axes[0, 0].plot(original_series, label="Original", alpha=0.7, color="royalblue")
    axes[0, 0].plot(generated_series, label="Generated", alpha=0.7, color="darkorange")
    axes[0, 0].set_title("Time Series Comparison", fontweight="bold", fontsize=13)
    axes[0, 0].set_xlabel("Time Steps", fontsize=11.5)
    axes[0, 0].set_ylabel("Value", fontsize=11.5)
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Histogram comparison between the original sequence and the generated sequence
    axes[0, 1].hist(
        original_series,
        bins=bins,
        alpha=0.5,
        density=True,
        label="Original",
        color="royalblue",
    )
    axes[0, 1].hist(
        generated_series,
        bins=bins,
        alpha=0.5,
        density=True,
        label="Generated",
        color="darkorange",
    )
    axes[0, 1].set_title("Histogram Comparison", fontweight="bold", fontsize=13)
    axes[0, 1].set_xlabel("Value", fontsize=11.5)
    axes[0, 1].set_ylabel("Density", fontsize=11.5)
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Comparison of autocorrelation functions
    axes[1, 0].plot(
        acf_original, label="Original", marker="o", markersize=3, color="royalblue"
    )
    axes[1, 0].plot(
        acf_generated, label="Generated", marker="x", markersize=3, color="darkorange"
    )
    axes[1, 0].set_title("Autocorrelation Function", fontweight="bold", fontsize=13)
    axes[1, 0].set_xlabel("Time Lag", fontsize=11.5)
    axes[1, 0].set_ylabel("Autocorrelation", fontsize=11.5)
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Power spectral density comparison
    axes[1, 1].semilogy(f_original, Pxx_original, label="Original", color="royalblue")
    axes[1, 1].semilogy(
        f_generated, Pxx_generated, label="Generated", color="darkorange"
    )
    axes[1, 1].set_title("Power Spectral Density", fontweight="bold", fontsize=13)
    axes[1, 1].set_xlabel("Frequency", fontsize=11.5)
    axes[1, 1].set_ylabel("Density Amplitude", fontsize=11.5)
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    # Plot the histogram of residuals and the Q-Q plot for the normality test.
    if residuals is not None:
        from statsmodels.graphics.gofplots import qqplot
        from scipy.stats import shapiro, norm

        axes[2, 0].grid(which="major", color="gray", lw=0.5, alpha=0.8)
        axes[2, 1].grid(which="major", color="gray", lw=0.5, alpha=0.8)

        # Perform Shapiro-Wilk normality test
        stat, p_value = shapiro(residuals)

        # Plot histogram of the fitted residuals
        axes[2, 1].hist(
            residuals,
            bins=bins,
            alpha=1,
            density=True,
            color="w",
            edgecolor="k",
            lw=1.2,
        )

        # Plot Q-Q plot for normality test
        qqplot(
            residuals,
            line="s",
            ax=axes[2, 0],
            markerfacecolor="white",
            markeredgecolor="k",
            markersize=7.5,
        )
        for line in axes[2, 0].get_lines():
            if line.get_linestyle() == "-":
                line.set_color("#DC143C")
                line.set_linewidth(2.1)

        # Set titles and labels
        axes[2, 1].set_title("Residuals Histogram", fontweight="bold", fontsize=13)
        axes[2, 0].set_title("Residuals Q-Q Plot", fontweight="bold", fontsize=13)

        axes[2, 1].set_xlabel("Standard Residual", fontsize=11.5)
        axes[2, 1].set_ylabel("Density", fontsize=11.5)
        axes[2, 0].set_xlabel("Theoretical Quantiles", fontsize=11.5)
        axes[2, 0].set_ylabel("Sample Quantiles", fontsize=11.5)

        # Annotate the plots with statistics
        mean = np.round(np.mean(residuals), 4)
        std = np.round(np.std(residuals), 4)
        stat = np.round(stat, 4)
        p_value = np.round(p_value, 4)

        # Generate probability density function curve data
        x = np.linspace(np.min(residuals), np.max(residuals), 1000)
        y = norm.pdf(x, loc=np.mean(residuals), scale=np.std(residuals))

        # Add density curve and perpendicular line to mean
        axes[2, 1].plot(x, y, color="#DC143C", linewidth=2)
        axes[2, 1].axvline(
            x=np.mean(residuals), color="#DC143C", linestyle="--", linewidth=1.5
        )

        # Set the text annotations for the mean and std on the histogram
        axes[2, 1].text(
            0.05,
            0.95,
            f"$\mu$ = {mean}\n$\sigma$ = {std}",
            transform=axes[2, 1].transAxes,
            verticalalignment="top",
            horizontalalignment="left",
            fontsize=11,
            color="k",
        )

        # Set the text annotations for the Shapiro-Wilk test on the Q-Q plot
        axes[2, 0].text(
            0.05,
            0.95,
            f"$W$ = {stat}\n$p$ = {p_value}",
            transform=axes[2, 0].transAxes,
            verticalalignment="top",
            horizontalalignment="left",
            fontsize=11,
            color="k",
        )

    plt.tight_layout()

    return fig


def plot_adjacency_matrix(
    adjacency_matrix: np.ndarray,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 160,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Visualize a binary adjacency matrix as a heatmap.

    Convention: ``adjacency[i, j] = 1`` means a directed edge ``i → j``.

    :param adjacency_matrix: Square ``(V, V)`` adjacency matrix.
    :param figsize: Figure size; defaults to a square scaled by ``V``.
    :param dpi: Figure resolution.
    :param ax: Optional existing axes to draw into.
    :return: Matplotlib Figure.
    """
    adj = np.asarray(adjacency_matrix)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(
            f"adjacency_matrix must be square (V, V), got shape {adj.shape}"
        )
    V = adj.shape[0]
    if figsize is None:
        side = max(4.0, 0.55 * V + 2.0)
        figsize = (side, side)

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    im = ax.imshow(adj, cmap="Blues", vmin=0, vmax=max(1, float(np.max(adj))))
    ax.set_xticks(np.arange(V))
    ax.set_yticks(np.arange(V))
    ax.set_xticklabels([f"{i}" for i in range(V)])
    ax.set_yticklabels([f"{i}" for i in range(V)])
    ax.set_xlabel("Target (j)", fontsize=11)
    ax.set_ylabel("Source (i)", fontsize=11)
    ax.set_title("Adjacency Matrix (i → j)", fontweight="bold", fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if created_fig:
        fig.tight_layout()
    return fig


def _draw_circular_digraph(ax: plt.Axes, adjacency_matrix: np.ndarray) -> None:
    """Draw a directed graph with a circular layout on ``ax``."""
    adj = np.asarray(adjacency_matrix).astype(bool)
    V = adj.shape[0]
    angles = np.linspace(0, 2 * np.pi, V, endpoint=False) - np.pi / 2
    radius = 1.0
    pos = {
        i: (radius * np.cos(angles[i]), radius * np.sin(angles[i])) for i in range(V)
    }

    # Edges
    for i in range(V):
        for j in range(V):
            if not adj[i, j] or i == j:
                continue
            x0, y0 = pos[i]
            x1, y1 = pos[j]
            # Shorten arrows so they stop at node boundaries
            dx, dy = x1 - x0, y1 - y0
            length = np.hypot(dx, dy)
            if length < 1e-8:
                continue
            shrink = 0.12
            sx, sy = x0 + shrink * dx / length, y0 + shrink * dy / length
            ex, ey = x1 - shrink * dx / length, y1 - shrink * dy / length
            ax.annotate(
                "",
                xy=(ex, ey),
                xytext=(sx, sy),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="#334155",
                    lw=1.4,
                    mutation_scale=12,
                ),
            )

    # Nodes
    xs = [pos[i][0] for i in range(V)]
    ys = [pos[i][1] for i in range(V)]
    ax.scatter(
        xs,
        ys,
        s=650,
        c="#e2e8f0",
        edgecolors="#0f172a",
        linewidths=1.5,
        zorder=3,
    )
    for i in range(V):
        ax.text(
            pos[i][0],
            pos[i][1],
            str(i),
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            zorder=4,
        )

    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Causal Graph (i → j)", fontweight="bold", fontsize=12)


def plot_graph(
    adjacency_matrix: np.ndarray,
    show_matrix: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 160,
) -> plt.Figure:
    """
    Visualize a DAG from its adjacency matrix with a circular layout.

    Convention: ``adjacency[i, j] = 1`` means a directed edge ``i → j``.
    When ``show_matrix`` is True, a second subplot shows the adjacency heatmap.

    :param adjacency_matrix: Square ``(V, V)`` adjacency matrix.
    :param show_matrix: If True, draw graph (left) and matrix (right).
    :param figsize: Optional figure size.
    :param dpi: Figure resolution.
    :return: Matplotlib Figure.
    """
    adj = np.asarray(adjacency_matrix)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(
            f"adjacency_matrix must be square (V, V), got shape {adj.shape}"
        )
    V = adj.shape[0]

    if show_matrix:
        if figsize is None:
            figsize = (max(9.0, 0.7 * V + 6.0), max(4.0, 0.55 * V + 2.5))
        fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
        _draw_circular_digraph(axes[0], adj)
        plot_adjacency_matrix(adj, ax=axes[1], dpi=dpi)
        fig.tight_layout()
        return fig

    if figsize is None:
        side = max(5.0, 0.55 * V + 3.0)
        figsize = (side, side)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    _draw_circular_digraph(ax, adj)
    fig.tight_layout()
    return fig


def plot_correlation(
    time_series: np.ndarray,
    measure: Union[str, List[str], Tuple[str, ...]] = "pearson",
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 160,
    cmap: Optional[str] = None,
    **kwargs,
) -> plt.Figure:
    """
    Visualize pairwise correlation / similarity / distance matrices.

    :param time_series: Multivariate series of shape ``[num_samples, seq_length]``
                        with ``num_samples >= 2``.
    :param measure: One measure, a space-separated string (e.g.
                    ``\"pearson wasserstein\"``), or a list of measure names.
                    Supported: ``pearson``, ``spearman``, ``autocorrelation``,
                    ``power_spectrum``, ``distribution``, ``wasserstein``.
    :param figsize: Optional figure size; scales with the number of measures.
    :param dpi: Figure resolution.
    :param cmap: Colormap override. Defaults to ``coolwarm`` for correlations
                 and ``viridis`` for Wasserstein distances.
    :param kwargs: Forwarded to ``multivariate_correlation`` (e.g. ``nlags``,
                   ``bins``, ``mean_weight``, ``covar_weight``).
    :return: Matplotlib Figure with one subplot per requested measure.
    """
    from s2generator.utils._multivariate_correlation import (
        multivariate_correlation,
        parse_correlation_measures,
    )

    measures = parse_correlation_measures(measure)
    computed = multivariate_correlation(time_series, measure=measures, **kwargs)
    if isinstance(computed, dict):
        matrices = computed
    else:
        matrices = {measures[0]: computed}

    n = len(measures)
    if figsize is None:
        figsize = (max(4.0, 4.2 * n), 4.0)

    fig, axes = plt.subplots(1, n, figsize=figsize, dpi=dpi, squeeze=False)
    titles = {
        "pearson": "Pearson Correlation",
        "spearman": "Spearman Correlation",
        "autocorrelation": "Autocorrelation Similarity",
        "power_spectrum": "Power-Spectrum Similarity",
        "distribution": "Distribution Similarity",
        "wasserstein": "Wasserstein Distance",
    }

    for col, name in enumerate(measures):
        ax = axes[0, col]
        mat = matrices[name]
        V = mat.shape[0]
        is_distance = name == "wasserstein"
        local_cmap = cmap or ("viridis" if is_distance else "coolwarm")
        if is_distance:
            vmin, vmax = 0.0, float(np.max(mat)) if np.max(mat) > 0 else 1.0
        else:
            finite = mat[np.isfinite(mat)]
            bound = float(np.max(np.abs(finite))) if finite.size else 1.0
            bound = max(bound, 1.0)
            vmin, vmax = -bound, bound

        im = ax.imshow(mat, cmap=local_cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(V))
        ax.set_yticks(np.arange(V))
        ax.set_xticklabels([str(i) for i in range(V)])
        ax.set_yticklabels([str(i) for i in range(V)])
        ax.set_xlabel("Series j", fontsize=10)
        ax.set_ylabel("Series i", fontsize=10)
        ax.set_title(titles.get(name, name), fontweight="bold", fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
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
    n_samples, seq_len = data.shape
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
        ax.set_xlim(0, seq_len - 1 if seq_len > 1 else 1)
    axes[-1, 0].set_xlabel("Time Steps", fontsize=11)
    axes[0, 0].set_title("Multivariate Time Series", fontweight="bold", fontsize=12)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    # Importing data generators, parameter controllers and visualization functions
    from s2generator.symbol import SeriesSymbolGenerator

    generator = SeriesSymbolGenerator()  # Create an instance

    rng = np.random.RandomState(0)  # Creating a random number object
    # Start generating symbolic expressions, sampling and generating series

    trees, x, y = generator.run(
        rng, input_dimension=2, output_dimension=10, n_inputs_points=20
    )

    trees_list = str(trees).split(" | ")
    for i, tree in enumerate(trees_list):
        print(i, tree)

    # Print the expressions
    fig = plot_symbol(trees)

    fig.savefig("test.png")
    plt.show()
