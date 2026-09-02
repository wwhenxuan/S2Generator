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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

__all__ = [
    "AVAILABLE_UNIVARIATE_DATASETS",
    "AVAILABLE_MULTIVARIATE_DATASETS",
    "AVAILABLE_DEEPMIMO_DATASETS",
    "list_datasets",
    "list_deepmimo_speeds",
    "load_univariate",
    "load_multivariate",
    "load_deepmimo_iq",
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

# Compact DeepMIMO Seoul CSI subset: (n_speed, n_user, n_sub, T, RI)
_DEEPMIMO_FILES: Dict[str, Tuple[str, str]] = {
    "city_37_seoul_3p5": ("city_37_seoul_3p5.npy", "city_37_seoul_3p5_meta.npy"),
}
_DEEPMIMO_ALIASES: Dict[str, str] = {
    "city_37_seoul_3p5": "city_37_seoul_3p5",
    "city_37_seoul": "city_37_seoul_3p5",
    "seoul": "city_37_seoul_3p5",
    "seoul_3p5": "city_37_seoul_3p5",
}
_DEEPMIMO_N_SPEED = 10
_DEEPMIMO_N_USER = 2
_DEEPMIMO_N_SUB = 4

AVAILABLE_DEEPMIMO_DATASETS: Tuple[str, ...] = ("city_37_seoul_3p5",)


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


def _canonical_deepmimo_name(name: str) -> str:
    """Resolve a DeepMIMO scenario alias to the packaged identifier."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("DeepMIMO dataset name must be a non-empty string")
    key = name.strip()
    if key in AVAILABLE_DEEPMIMO_DATASETS:
        return key
    alias = _DEEPMIMO_ALIASES.get(key.lower())
    if alias is not None:
        return alias
    raise ValueError(
        f"unknown DeepMIMO dataset {name!r}; choose from {AVAILABLE_DEEPMIMO_DATASETS}"
    )


def _load_deepmimo_bundle(
    name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the packaged CSI cube and the 1-D metadata vector.

    :param name: Scenario name or alias.
    :return: ``(csi, speeds_kmh, user_ids, subcarriers)``.
    """
    canonical = _canonical_deepmimo_name(name)
    data_name, meta_name = _DEEPMIMO_FILES[canonical]
    with _resource_path(data_name).open("rb") as handle:
        csi = np.asarray(np.load(handle), dtype=np.float32)
    with _resource_path(meta_name).open("rb") as handle:
        meta = np.asarray(np.load(handle), dtype=np.float32).reshape(-1)
    expected = _DEEPMIMO_N_SPEED + _DEEPMIMO_N_USER + _DEEPMIMO_N_SUB
    if meta.size < expected:
        raise ValueError(f"DeepMIMO metadata is too short: {meta.shape}")
    speeds = meta[:_DEEPMIMO_N_SPEED]
    users = meta[_DEEPMIMO_N_SPEED : _DEEPMIMO_N_SPEED + _DEEPMIMO_N_USER]
    subcarriers = meta[_DEEPMIMO_N_SPEED + _DEEPMIMO_N_USER : expected]
    if csi.ndim != 5 or csi.shape[-1] != 2:
        raise ValueError(f"unexpected DeepMIMO CSI shape {csi.shape}")
    return csi, speeds, users, subcarriers


def _as_int_set(
    value: Optional[Union[int, float, Sequence[Union[int, float]]]],
    allowed: np.ndarray,
    label: str,
) -> np.ndarray:
    """Map a scalar / sequence filter onto packaged integer labels."""
    allowed_int = np.asarray(np.round(allowed), dtype=np.int64)
    if value is None:
        return np.arange(allowed_int.size, dtype=np.int64)
    if isinstance(value, (int, float, np.integer, np.floating)):
        candidates: Iterable[Union[int, float]] = (value,)
    else:
        candidates = value
    index = []
    lookup = {int(item): pos for pos, item in enumerate(allowed_int)}
    for item in candidates:
        key = int(round(float(item)))
        if key not in lookup:
            raise ValueError(
                f"{label} {item!r} is not in the packaged set {allowed_int.tolist()}"
            )
        index.append(lookup[key])
    return np.asarray(index, dtype=np.int64)


def list_deepmimo_speeds(name: str = "city_37_seoul_3p5") -> List[int]:
    """List packaged vehicle speeds (km/h) for a DeepMIMO CSI subset.

    :param name: Scenario name (see :data:`AVAILABLE_DEEPMIMO_DATASETS`).
    :return: Speeds in ascending order.
    """
    _, speeds, _, _ = _load_deepmimo_bundle(name)
    return [int(round(float(item))) for item in speeds]


def load_deepmimo_iq(
    name: str = "city_37_seoul_3p5",
    speed_kmh: Optional[Union[int, float, Sequence[Union[int, float]]]] = None,
    subcarrier: Optional[Union[int, float, Sequence[Union[int, float]]]] = None,
    user: Optional[Union[int, float, Sequence[Union[int, float]]]] = None,
) -> np.ndarray:
    """Load packaged DeepMIMO CSI as real/imag stacks for :class:`IQSimulator`.

    The on-disk cube has shape ``(n_speed, n_user, n_sub, T, 2)``.  Filters
    select a subset of speeds, original user ids, and original subcarrier
    indices; the remaining traces are flattened to ``(N, T, 2)``.

    :param name: Scenario name (see :data:`AVAILABLE_DEEPMIMO_DATASETS`).
    :param speed_kmh: One speed, a list of speeds, or ``None`` for every speed.
    :param subcarrier: Original subcarrier index (``0, 10, 21, 31``) or ``None``.
    :param user: Original user id within a speed group (``0`` or ``8``) or ``None``.
    :return: Float64 array of shape ``(N, seq_length, 2)`` with ``seq_length=128``.
    """
    csi, speeds, users, subcarriers = _load_deepmimo_bundle(name)
    speed_idx = _as_int_set(speed_kmh, speeds, "speed_kmh")
    user_idx = _as_int_set(user, users, "user")
    sub_idx = _as_int_set(subcarrier, subcarriers, "subcarrier")
    sliced = csi[np.ix_(speed_idx, user_idx, sub_idx)]
    n_trace = int(np.prod(sliced.shape[:3]))
    traces = sliced.reshape(n_trace, sliced.shape[3], sliced.shape[4])
    return np.asarray(traces, dtype=np.float64)
