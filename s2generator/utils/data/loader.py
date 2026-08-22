# -*- coding: utf-8 -*-
"""
Bundled real time-series slices shipped with S2Generator.

The files under this package are 4096-step excerpts of public forecasting
benchmarks (ETT, exchange rate, weather, electricity). They are intended for
examples and tests of the TiRex-2 coupling pipeline, not as a full training
corpus.

Created on 2026/08/22
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/S2Generator
"""

from __future__ import annotations

from importlib.resources import files
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "AVAILABLE_UNIVARIATE_DATASETS",
    "AVAILABLE_MULTIVARIATE_DATASETS",
    "list_datasets",
    "load_univariate",
    "load_multivariate",
]

# Canonical names -> on-disk resource
_CSV_DATASETS: Dict[str, str] = {
    "ETTh1": "ETTh1.csv",
    "ETTh2": "ETTh2.csv",
    "ETTm1": "ETTm1.csv",
    "ETTm2": "ETTm2.csv",
    "exchange_rate": "exchange_rate.csv",
    "weather": "weather.csv",
}
_NPY_DATASETS: Dict[str, str] = {
    "electricity": "electricity.npy",
}

_ALIASES: Dict[str, str] = {
    "etth1": "ETTh1",
    "etth2": "ETTh2",
    "ettm1": "ETTm1",
    "ettm2": "ETTm2",
    "exchange": "exchange_rate",
    "exchange_rate": "exchange_rate",
    "weather": "weather",
    "electricity": "electricity",
}

AVAILABLE_UNIVARIATE_DATASETS: Tuple[str, ...] = (
    "ETTh1",
    "ETTh2",
    "ETTm1",
    "ETTm2",
    "exchange_rate",
    "weather",
    "electricity",
)
AVAILABLE_MULTIVARIATE_DATASETS: Tuple[str, ...] = (
    "ETTh1",
    "ETTh2",
    "ETTm1",
    "ETTm2",
    "exchange_rate",
    "weather",
)

_OT_COLUMN = "OT"


def list_datasets(kind: str = "all") -> List[str]:
    """List bundled dataset names.

    :param kind: ``\"univariate\"``, ``\"multivariate\"``, or ``\"all\"``.
    :return: Canonical dataset names in a stable order.
    """
    if kind == "univariate":
        return list(AVAILABLE_UNIVARIATE_DATASETS)
    if kind == "multivariate":
        return list(AVAILABLE_MULTIVARIATE_DATASETS)
    if kind == "all":
        return list(AVAILABLE_UNIVARIATE_DATASETS)
    raise ValueError(
        "kind must be 'univariate', 'multivariate', or 'all', " f"got {kind!r}"
    )


def _canonical_name(name: str) -> str:
    """Resolve a user-facing dataset name to the canonical identifier."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("dataset name must be a non-empty string")
    key = name.strip()
    if key in AVAILABLE_UNIVARIATE_DATASETS:
        return key
    alias = _ALIASES.get(key.lower())
    if alias is not None:
        return alias
    raise ValueError(
        f"unknown dataset {name!r}; choose from {AVAILABLE_UNIVARIATE_DATASETS} "
        f"(aliases: Exchange, Weather, Electricity)"
    )


def _resource_path(filename: str):
    """Locate a packaged data file (works after wheel install)."""
    return files("s2generator.utils.data").joinpath(filename)


def load_univariate(name: str) -> np.ndarray:
    """Load a bundled univariate series as a 1D NumPy array.

    For ETT / exchange_rate / weather this returns the ``OT`` column of the
    packaged CSV. For electricity this returns the packaged ``.npy`` array.

    :param name: Dataset name (see :data:`AVAILABLE_UNIVARIATE_DATASETS`).
    :return: Float array of shape ``(4096,)``.
    """
    canonical = _canonical_name(name)
    if canonical in _NPY_DATASETS:
        with _resource_path(_NPY_DATASETS[canonical]).open("rb") as handle:
            series = np.load(handle)
        return np.asarray(series, dtype=np.float64).reshape(-1)

    frame = load_multivariate(canonical)
    if _OT_COLUMN not in frame.columns:
        raise KeyError(
            f"dataset {canonical!r} has no {_OT_COLUMN!r} column; "
            f"got {list(frame.columns)}"
        )
    return frame[_OT_COLUMN].to_numpy(dtype=np.float64)


def load_multivariate(name: str) -> pd.DataFrame:
    """Load a bundled multivariate CSV as a pandas DataFrame.

    The DataFrame includes the ``date`` column and all numeric channels
    (including ``OT``). Electricity is univariate-only and raises ``ValueError``.

    :param name: Dataset name (see :data:`AVAILABLE_MULTIVARIATE_DATASETS`).
    :return: DataFrame with 4096 rows.
    """
    canonical = _canonical_name(name)
    if canonical in _NPY_DATASETS:
        raise ValueError(
            "electricity is shipped as a univariate .npy slice (OT only); "
            "use load_univariate('electricity') instead"
        )
    filename = _CSV_DATASETS[canonical]
    with _resource_path(filename).open("rb") as handle:
        return pd.read_csv(handle)
