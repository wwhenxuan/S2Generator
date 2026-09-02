# -*- coding: utf-8 -*-
"""Visualize directed adjacency matrices and circular causal graphs."""

from typing import Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt


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
