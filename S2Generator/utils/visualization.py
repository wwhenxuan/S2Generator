# -*- coding: utf-8 -*-
"""
Created on 2025/01/25 00:02:43
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.transforms import Bbox

from typing import Optional, Union, Dict, Any, Tuple, List

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


if __name__ == "__main__":
    import numpy as np

    # Importing data generators, parameter controllers and visualization functions
    from S2Generator import Generator, Params, plot_series

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
