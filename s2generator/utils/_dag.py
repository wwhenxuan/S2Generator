# -*- coding: utf-8 -*-
"""
Shared DAG utilities for the SCM generators.

This module provides the conversion from a user-supplied numpy adjacency
matrix to the internal ``(parents, roots, edges)`` representation used by
``CaukerPipeline`` and ``ScmPriorPipeline``.

An adjacency matrix ``A`` of shape ``(V, V)`` is interpreted as a directed
graph where ``A[i, j] != 0`` denotes an edge ``i -> j``. The matrix is
validated to be a square, loop-free directed acyclic graph (DAG) before
conversion.

Created on 2026/08/18
@author: Ruizhe Wang
@email: changewam6@gmail.com
@url: https://github.com/wwhenxuan/S2Generator
"""

from typing import List, Tuple

import numpy as np


def adjacency_to_dag(
    adjacency: np.ndarray,
) -> Tuple[List[List[int]], List[int], List[Tuple[int, int]]]:
    """Convert a numpy adjacency matrix to the ``(parents, roots, edges)`` form.

    ``adjacency[i, j] != 0`` denotes a directed edge from node ``i`` to node
    ``j``. The matrix must be a square 2D array describing a loop-free DAG;
    otherwise a :class:`ValueError` is raised.

    :param adjacency: Binary adjacency matrix of shape ``(V, V)``.
    :return: Tuple of ``(parents, roots, edges)`` where ``parents[child]`` is
             the sorted list of parent indices, ``roots`` is the list of nodes
             with in-degree 0, and ``edges`` is the list of ``(parent, child)``
             tuples.
    :raises ValueError: If ``adjacency`` is not a 2D square array, contains
                        self-loops, or is not acyclic.
    """
    adjacency = np.asarray(adjacency)

    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(
            f"adjacency must be a square 2D array, got shape {adjacency.shape}"
        )

    V = adjacency.shape[0]

    # Self-loops are not allowed in a DAG.
    if np.any(np.diag(adjacency) != 0):
        raise ValueError("adjacency must not contain self-loops")

    # Build the parent lists and edge list.
    parents: List[List[int]] = [[] for _ in range(V)]
    edges: List[Tuple[int, int]] = []
    for i in range(V):
        for j in range(V):
            if adjacency[i, j] != 0:
                parents[j].append(i)
                edges.append((i, j))

    # Sort parents for deterministic output.
    parents = [sorted(p) for p in parents]

    # Acyclicity check via Kahn's topological ordering.
    in_degree = [len(parents[i]) for i in range(V)]
    queue = [i for i in range(V) if in_degree[i] == 0]
    visited = 0
    while queue:
        node = queue.pop(0)
        visited += 1
        for child in range(V):
            if node in parents[child]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

    if visited != V:
        raise ValueError("adjacency must be acyclic (a DAG)")

    roots = [i for i in range(V) if len(parents[i]) == 0]

    return parents, roots, edges
