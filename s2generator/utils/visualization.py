# -*- coding: utf-8 -*-
"""
Created on 2025/01/25 00:02:43
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""

__all__ = [
    "plot_series",
    "plot_symbol",
    "plot_shapiro_wilk",
    "plot_simulator_statistics",
]

from typing import Optional, Union, Dict, Any, Tuple, List

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.transforms import Bbox

from scipy import signal
from statsmodels.tsa.stattools import acf

from s2generator import Node, NodeList
from s2generator.utils.print_symbol import symbol_to_markdown


def plot_series(x: np.ndarray, y: np.ndarray) -> plt.Figure:
    """
    Visualize S2 data

    :param x: input sampling series
    :param y: output generated series
    :return: the plot figure of matplotlib
    """

    # Determine the shape and length of the data
    (seq_len, input_dim) = x.shape
    (_, output_dim) = y.shape
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
    nlags: int = 50,
    nperseg: int = 128,
) -> None:
    """
    比较原始序列和生成序列的统计特性
    """
    # 确保输入的数据是一维的ndarray
    original_series = np.asarray(original_series).flatten()
    generated_series = np.asarray(generated_series).flatten()

    # 计算自相关函数
    acf_original = acf(original_series, nlags=nlags, fft=True)
    acf_generated = acf(generated_series, nlags=nlags, fft=True)

    # 计算功率谱密度
    f_original, Pxx_original = signal.welch(original_series, fs=1.0, nperseg=nperseg)
    f_generated, Pxx_generated = signal.welch(generated_series, fs=1.0, nperseg=nperseg)

    # 绘图对比
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))

    # 原始序列和生成序列
    axes[0, 0].plot(original_series, label="Original", alpha=0.7)
    axes[0, 0].plot(generated_series, label="Generated", alpha=0.7)
    axes[0, 0].set_title("Time Series Comparison")
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 原始序列与生成序列的直方图对比
    axes[0, 1].hist(original_series, bins=30, alpha=0.5, label="Original", color="blue")
    axes[0, 1].hist(generated_series, bins=30, alpha=0.5, label="Generated", color="orange")
    axes[0, 1].set_title("Histogram Comparison")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # 自相关函数对比
    axes[1, 0].plot(acf_original, label="Original", marker="o", markersize=3)
    axes[1, 0].plot(acf_generated, label="Generated", marker="x", markersize=3)
    axes[1, 0].set_title("Autocorrelation Function")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # 功率谱密度对比
    axes[1, 1].semilogy(f_original, Pxx_original, label="Original")
    axes[1, 1].semilogy(f_generated, Pxx_generated, label="Generated")
    axes[1, 1].set_title("Power Spectral Density")
    axes[1, 1].set_xlabel("Frequency")
    axes[1, 1].set_ylabel("PSD")
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import numpy as np

    # Importing data generators, parameter controllers and visualization functions
    from s2generator import Generator, Params, plot_series

    params = Params()  # Adjust the parameters here
    generator = Generator(params)  # Create an instance

    rng = np.random.RandomState(0)  # Creating a random number object
    # Start generating symbolic expressions, sampling and generating series

    trees, x, y = generator.run(
        rng, input_dimension=2, output_dimension=10, n_points=20
    )

    trees_list = str(trees).split(" | ")
    for i, tree in enumerate(trees_list):
        print(i, tree)

    # Print the expressions
    fig = plot_symbol(trees)

    fig.savefig("test.png")
    plt.show()
