# -*- coding: utf-8 -*-
"""
Created on 2026/06/21 22:59:38
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/S2Generator
"""

import copy
import inspect
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Union, Optional, List, Sequence

import numpy as np

from s2generator.simulator.arima import ARIMASimulator
from s2generator.simulator.markov_switching import MarkovSwitchingSimulator
from s2generator.simulator.kalman_filtering import KalmanFilterSimulator
from s2generator.simulator.wiener_filter import WienerFilterSimulator
from s2generator.simulator.gaussian_mixture import GaussianMixtureSimulator

SimulatorType = Union[
    WienerFilterSimulator,
    KalmanFilterSimulator,
    MarkovSwitchingSimulator,
    ARIMASimulator,
    GaussianMixtureSimulator,
]

_SUPPORTED_SIMULATORS = (
    WienerFilterSimulator,
    KalmanFilterSimulator,
    MarkovSwitchingSimulator,
    ARIMASimulator,
    GaussianMixtureSimulator,
)


def _fit_simulator(
    simulator: SimulatorType,
    time_series: np.ndarray,
    select_order: bool = False,
) -> SimulatorType:
    """Fit a single-channel simulator, forwarding ``select_order`` when supported."""
    fit_params = inspect.signature(simulator.fit).parameters
    if "select_order" in fit_params:
        simulator.fit(time_series, select_order=select_order)
    else:
        simulator.fit(time_series)
    return simulator


def _fit_channel_worker(
    channel_index: int,
    time_series: np.ndarray,
    simulator_template: SimulatorType,
    select_order: bool,
) -> tuple:
    """Worker function for parallel per-channel fitting."""
    simulator = copy.deepcopy(simulator_template)
    _fit_simulator(simulator, time_series=time_series, select_order=select_order)
    return channel_index, simulator


def _supports_shared_excitation(simulator: SimulatorType) -> bool:
    """Return True when a fitted simulator can be driven by a supplied noise path."""
    return hasattr(simulator, "invoke") and callable(getattr(simulator, "invoke"))


def _excitation_padding(simulator: SimulatorType) -> int:
    """Return the number of warm-up samples required by ``invoke``."""
    if isinstance(simulator, WienerFilterSimulator):
        return simulator.filter_order
    if isinstance(simulator, KalmanFilterSimulator):
        return simulator.state_order
    return 0


def _apply_revin(simulator: SimulatorType, series: np.ndarray) -> np.ndarray:
    """Inverse-transform a generated series when reversible normalization was used."""
    if getattr(simulator, "revin", False):
        if isinstance(simulator, GaussianMixtureSimulator):
            return series * simulator.input_std + simulator.input_mean
        mean = getattr(simulator, "mean", None)
        std = getattr(simulator, "std", None)
        if mean is not None and std is not None:
            return series * std + mean
    return series


class MultivariateSimulator(object):
    """
    Simulate multivariate time series by fitting one univariate simulator per channel.

    Existing simulators in S2Generator are designed for univariate series. This wrapper
    fits an independent model on each channel and, during generation, excites all
    ``invoke``-capable channels with the **same white-noise excitation**, so the outputs
    become cross-correlated even though each channel is modeled separately.

    When the user passes a single simulator template, every channel is fitted with a
    deep-copied instance of that template. When the user passes a list, channels are
    matched to list entries in order; extra channels default to ``WienerFilterSimulator``.

    Parallel fitting is supported through ``n_jobs`` for high-dimensional inputs.
    """

    def __init__(
        self,
        simulator: Union[SimulatorType, List[SimulatorType]],
        default_simulator: Optional[WienerFilterSimulator] = None,
        n_jobs: int = -1,
    ) -> None:
        """
        :param simulator: A single simulator template or an ordered list of templates.
        :param default_simulator: Simulator template used for channels beyond the list
            length when ``simulator`` is a list. Defaults to ``WienerFilterSimulator()``.
        :param n_jobs: Number of parallel workers used in ``fit``. ``-1`` uses all CPUs,
            ``1`` disables parallelism.

        :return: None
        """
        if isinstance(simulator, list):
            if len(simulator) == 0:
                raise ValueError(
                    "When passing a list, at least one simulator is required."
                )
            for item in simulator:
                self._validate_simulator_template(item)
            self._simulator_templates: List[SimulatorType] = list(simulator)
            self._single_template: Optional[SimulatorType] = None
        else:
            self._validate_simulator_template(simulator)
            self._single_template = simulator
            self._simulator_templates = []

        self.default_simulator = (
            default_simulator
            if default_simulator is not None
            else WienerFilterSimulator()
        )
        self._validate_simulator_template(self.default_simulator)

        self.n_jobs = n_jobs
        self.n_channels: Optional[int] = None
        self.simulators: List[SimulatorType] = []
        self.simulated_series: Optional[np.ndarray] = None
        self.time_series: Optional[np.ndarray] = None

    def fit(
        self, time_series: np.ndarray, select_order: Optional[bool] = False
    ) -> None:
        """
        Fit one simulator per channel on the multivariate input series.

        :param time_series: Multivariate series with shape ``[seq_len, n_channels]``.
        :param select_order: Forwarded to simulators whose ``fit`` supports it.

        :return: None
        """
        time_series = self.check_inputs(time_series=time_series)
        self.time_series = time_series
        self.n_channels = time_series.shape[1]

        templates = [
            self._simulator_template_for_channel(channel_index)
            for channel_index in range(self.n_channels)
        ]

        self.simulators = self._fit_channels_parallel(
            time_series=time_series,
            templates=templates,
            select_order=bool(select_order),
        )

    def transform(
        self, num_samples: int, seq_len: int, random_state: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate multivariate samples by exciting fitted channel models.

        For simulators that expose ``invoke`` (Wiener / Kalman), all channels in a sample
        share the same white-noise excitation. Other simulators fall back to their own
        ``transform`` implementation for the corresponding channel.

        :param num_samples: Number of multivariate sample paths to generate.
        :param seq_len: Length of each generated sequence.
        :param random_state: Random seed for reproducibility.

        :return: Generated series with shape ``[num_samples, seq_len, n_channels]``.
        """
        if not self.simulators:
            raise ValueError(
                "No fitted channel simulators found; please call `fit` first."
            )

        seed = random_state
        rng = np.random.RandomState(seed=seed)
        n_channels = len(self.simulators)
        simulated = np.zeros((num_samples, seq_len, n_channels), dtype=np.float64)

        invoke_indices = [
            idx
            for idx, simulator in enumerate(self.simulators)
            if _supports_shared_excitation(simulator)
        ]
        max_padding = max(
            (_excitation_padding(self.simulators[idx]) for idx in invoke_indices),
            default=0,
        )

        for sample_index in range(num_samples):
            sample_seed = None if seed is None else int(seed) + sample_index
            sample_rng = np.random.RandomState(seed=sample_seed)

            shared_noise = None
            if invoke_indices:
                shared_noise = sample_rng.normal(
                    loc=0.0,
                    scale=1.0,
                    size=seq_len + max_padding,
                )

            for channel_index, simulator in enumerate(self.simulators):
                if _supports_shared_excitation(simulator):
                    padding = _excitation_padding(simulator)
                    channel_noise = shared_noise[-(seq_len + padding) :]
                    channel_series = simulator.invoke(white_noise=channel_noise)
                    channel_series = _apply_revin(simulator, channel_series)
                else:
                    channel_seed = (
                        None
                        if sample_seed is None
                        else sample_seed * 1009 + channel_index
                    )
                    channel_series = simulator.transform(
                        num_samples=1,
                        seq_len=seq_len,
                        random_state=channel_seed,
                    )[0]

                simulated[sample_index, :, channel_index] = channel_series

        self.simulated_series = simulated
        return self.simulated_series

    def check_inputs(self, time_series: np.ndarray) -> np.ndarray:
        """
        Validate a multivariate input array.

        :param time_series: Array with shape ``[seq_len, n_channels]``.

        :return: Validated ``float64`` array with shape ``[seq_len, n_channels]``.
        """
        if not isinstance(time_series, np.ndarray):
            raise ValueError("Input time_series must be a numpy ndarray.")

        if time_series.ndim != 2:
            raise ValueError(
                "Input time_series must be 2-dimensional with shape [seq_len, n_channels]."
            )

        time_series = np.asarray(time_series, dtype=np.float64)
        if time_series.shape[1] < 1:
            raise ValueError(
                "Multivariate time series must contain at least one channel."
            )

        if np.isnan(time_series).any():
            raise ValueError("Input time_series must not contain NaN values.")

        if np.all(np.std(time_series, axis=0) < 1e-8):
            raise ValueError(
                "At least one channel has zero variance; multivariate fitting is impossible."
            )

        return time_series

    def _simulator_template_for_channel(self, channel_index: int) -> SimulatorType:
        """Resolve the simulator template used for a given channel index."""
        if self._single_template is not None:
            return copy.deepcopy(self._single_template)

        if channel_index < len(self._simulator_templates):
            return copy.deepcopy(self._simulator_templates[channel_index])

        return copy.deepcopy(self.default_simulator)

    def _fit_channels_parallel(
        self,
        time_series: np.ndarray,
        templates: Sequence[SimulatorType],
        select_order: bool,
    ) -> List[SimulatorType]:
        """Fit all channels, using thread or process pools when ``n_jobs != 1``."""
        n_channels = time_series.shape[1]
        fitted: List[Optional[SimulatorType]] = [None] * n_channels
        workers = self._resolve_n_jobs(n_channels)

        if workers == 1 or n_channels == 1:
            for channel_index in range(n_channels):
                simulator = copy.deepcopy(templates[channel_index])
                _fit_simulator(
                    simulator,
                    time_series=time_series[:, channel_index],
                    select_order=select_order,
                )
                fitted[channel_index] = simulator
            return fitted  # type: ignore[return-value]

        executor_cls = (
            ProcessPoolExecutor
            if os.name != "nt" and n_channels >= 4
            else ThreadPoolExecutor
        )

        with executor_cls(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _fit_channel_worker,
                    channel_index,
                    time_series[:, channel_index].copy(),
                    templates[channel_index],
                    select_order,
                )
                for channel_index in range(n_channels)
            ]

            for future in as_completed(futures):
                channel_index, simulator = future.result()
                fitted[channel_index] = simulator

        return fitted  # type: ignore[return-value]

    def _resolve_n_jobs(self, n_channels: int) -> int:
        """Resolve the effective worker count from ``n_jobs``."""
        if self.n_jobs == 1 or n_channels <= 1:
            return 1

        cpu_count = os.cpu_count() or 1
        if self.n_jobs < 0:
            return min(n_channels, cpu_count)
        return min(n_channels, self.n_jobs)

    @staticmethod
    def _validate_simulator_template(simulator: SimulatorType) -> None:
        """Ensure the provided object is a supported simulator template."""
        if not isinstance(simulator, _SUPPORTED_SIMULATORS):
            raise TypeError(
                "simulator must be one of "
                "WienerFilterSimulator, KalmanFilterSimulator, "
                "MarkovSwitchingSimulator, ARIMASimulator, or GaussianMixtureSimulator."
            )
