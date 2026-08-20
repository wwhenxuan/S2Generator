# -*- coding: utf-8 -*-
"""
Shared helpers for deriving classification labels from time-series samples.

TiRex-2 (``coupling``) and CAUKER (``cauker``) are time-series generators: a
single sample is one multivariate series of shape ``(d, T)`` (or ``(T, Q)``).
To expose a classification interface analogous to :class:`ScmPriorPipeline`
(which discretizes a tabular target node into a many-class label), we collapse
each series into a scalar summary statistic and quantile-bin those statistics.

This yields balanced class labels at the batch level -- the time-series analogue
of RML2016-style labeled datasets (a stack of signals paired with a class label).
"""

import numpy as np


def summarize_series(x: np.ndarray) -> float:
    """Collapse a series of shape ``(d, T)`` or ``(T, Q)`` into a scalar statistic.

    Uses the global mean of all finite values; if the series is all-NaN/all-Inf,
    returns ``0.0``.

    :param x: A time-series array.
    :return: A scalar summary statistic.
    """
    x = np.asarray(x, dtype=np.float64)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return 0.0
    return float(finite.mean())


def discretize_labels(stats, n_classes: int) -> np.ndarray:
    """Discretize per-sample statistics into ``n_classes`` balanced classes.

    Each statistic is cut at the ``1/C, ..., (C-1)/C`` quantiles of the batch,
    so the resulting labels are naturally balanced (the same mechanism used for
    the many-class target in TabPFN-3).

    :param stats: Iterable of scalar statistics, one per sample.
    :param n_classes: Number of classes (>= 2).
    :return: Integer labels of shape ``(n_samples,)`` in ``[0, n_classes - 1]``.
    """
    stats = np.asarray(stats, dtype=np.float64)
    if n_classes < 2:
        raise ValueError(f"n_classes must be >= 2, got {n_classes}")
    qs = np.quantile(stats, np.arange(1, n_classes) / n_classes)
    y = np.digitize(stats, qs)
    return np.clip(y, 0, n_classes - 1).astype(np.int64)


def label_single(s: float, n_classes: int) -> int:
    """Deterministic class label for a single statistic (not balanced).

    Maps the statistic through a logistic squashing function into ``[0, 1)`` and
    floors it into ``n_classes`` bins. This is deterministic and reproducible,
    but class balance is only meaningful across a batch (see
    :func:`discretize_labels`), so use the batch-level labelling for datasets.

    :param s: A scalar summary statistic.
    :param n_classes: Number of classes (>= 2).
    :return: An integer class label in ``[0, n_classes - 1]``.
    """
    if n_classes < 2:
        raise ValueError(f"n_classes must be >= 2, got {n_classes}")
    p = 1.0 / (1.0 + np.exp(-s))
    return int(np.clip(np.floor(p * n_classes), 0, n_classes - 1))
