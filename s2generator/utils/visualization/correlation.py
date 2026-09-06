# -*- coding: utf-8 -*-
"""Heatmaps of pairwise correlation, similarity, and distance matrices."""

from typing import List, Optional, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt


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
    from s2generator.utils.correlation import (
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
