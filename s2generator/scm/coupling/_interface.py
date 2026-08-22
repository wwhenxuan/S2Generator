# -*- coding: utf-8 -*-
"""
Unified interface for the synthetic data generation pipelines.

This module provides top-level pipeline classes that orchestrate the
full dataset construction process:

1. **CouplingPipeline** (TiRex-2): augmentation → coupling → post-processing
2. **CaukerPipeline** (CAUKER): GP kernel composition → SCM propagation
3. **SCMPriorPipeline** (TabPFN-3): DAG-based SCM prior for tabular data (N×P + target)

Reference:
    - Podest, P., et al. (2026). TiRex-2. arXiv:2607.01204v1.
    - Xie, S., et al. (2025). CAUKER. arXiv:2508.02879v3.
    - Prior Labs Team (2026). TabPFN-3. arXiv:2605.13986v2.

Created on 2026/08/10 00:00:00
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/S2Generator
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .base_coupling import BaseCoupling
from .identity import IdentityCoupling, UnivariatePassThrough
from .functional import FunctionalCoupling
from .linear_mixing import LinearMixing
from .cointegration import Cointegration
from .linear_scm import LinearSCM
from .nonlinear_scm import NonlinearSCM
from .postprocessing import PostProcessor

from s2generator.augmentation import (
    amplitude_modulation,
    censor_augmentation,
    spike_injection,
)

from ...utils._label import discretize_labels, label_single, summarize_series

from ..cauker import (
    _build_kernel_bank,
    _sample_composite_kernel,
    _sample_gp,
)


class CouplingPipeline(object):
    """Unified pipeline for synthetic multivariate coupling.

    This pipeline implements the full TiRex-2 data generation process:
    independently generated univariate series → augmentation → coupling →
    post-processing → multivariate training sample.

    The pipeline is designed as a procedural prior: each stage is independently
    randomized per example, yielding a combinatorial enlargement of the effective
    training distribution.

    Note:
        Coupling mixes channels (linear mixing, SCMs, cointegration, etc.).
        Real or concatenated series often live on very different scales, so a
        high-energy channel would dominate the mixture and drown out the rest.
        Set ``normalize=True`` in :meth:`__call__` to z-score each of the Q
        columns of the ``(T, Q)`` input, then multiply every channel by an
        energy coefficient. The coefficient is either a random draw from
        ``U[scale_min, scale_max]`` (defaults 0.5 and 2.0) or a user-supplied
        ``channel_scales`` vector. The extra scale keeps the series from all
        collapsing onto a standard normal after z-scoring.

        When ``normalize=False`` (the default) the pipeline does not rescale
        anything. The caller **must** already have balanced per-channel energy
        before passing ``series`` in; otherwise mixing is scale-dominated.
    """

    def __init__(
        self,
        mechanisms: Optional[Dict[str, BaseCoupling]] = None,
        mechanism_probabilities: Optional[Dict[str, float]] = None,
        post_processor: Optional[PostProcessor] = None,
        patch_size: int = 32,
        horizon: int = 0,
        scale_min: float = 0.5,
        scale_max: float = 2.0,
        dtype: np.dtype = np.float64,
    ) -> None:
        """Initialize the coupling pipeline.

        :param mechanisms: Dictionary of coupling mechanism instances.
                          If None, default mechanisms are created.
        :param mechanism_probabilities: Sampling probability for each mechanism.
                                       If None, uniform sampling is used.
        :param post_processor: PostProcessor instance for observational transforms.
                              If None, a default PostProcessor is created.
        :param patch_size: Patch size for post-processing operations.
        :param horizon: Forecast horizon for future masking.
        :param scale_min: Lower bound of the per-channel energy multiplier
                          sampled after z-score normalization when
                          ``normalize=True`` and ``channel_scales`` is not given.
        :param scale_max: Upper bound of the per-channel energy multiplier
                          sampled after z-score normalization when
                          ``normalize=True`` and ``channel_scales`` is not given.
        :param dtype: The numpy data type for generated data.
        """
        if scale_max < scale_min:
            raise ValueError(
                f"scale_max ({scale_max}) must be >= scale_min ({scale_min})"
            )
        self._dtype = dtype
        self._patch_size = patch_size
        self._horizon = horizon
        self._scale_min = float(scale_min)
        self._scale_max = float(scale_max)

        # Initialize coupling mechanisms
        if mechanisms is None:
            self._mechanisms = self._create_default_mechanisms()
        else:
            self._mechanisms = mechanisms

        # Initialize mechanism probabilities
        if mechanism_probabilities is None:
            n = len(self._mechanisms)
            self._mechanism_probs = {name: 1.0 / n for name in self._mechanisms}
        else:
            self._mechanism_probs = mechanism_probabilities

        # Initialize post-processor
        self._post_processor = (
            post_processor
            if post_processor is not None
            else PostProcessor(
                patch_size=patch_size,
                horizon=horizon,
                dtype=dtype,
            )
        )

    def __str__(self) -> str:
        return "CouplingPipeline"

    def __call__(
        self,
        rng: np.random.RandomState,
        series: np.ndarray,
        mechanism: Optional[str] = None,
        horizon: Optional[int] = None,
        adjacency: Optional[np.ndarray] = None,
        apply_postprocessing: bool = True,
        return_metadata: bool = False,
        normalize: bool = False,
        channel_scales: Optional[Union[Sequence[float], np.ndarray]] = None,
    ) -> Union[
        np.ndarray,
        Tuple[np.ndarray, Dict[str, Any]],
    ]:
        """Run the full coupling pipeline.

        Note:
            Why normalize. Mechanisms such as linear mixing and the SCMs form
            each variate from a combination of the input channels. If those
            channels have mismatched amplitudes, the largest one dominates the
            coupled output. Normalization equalizes per-channel energy so that
            mixing reflects the intended relationships rather than raw scale.

            How it is done. With ``normalize=True`` each of the Q columns of
            ``series`` (shape ``(T, Q)``) is z-scored along time. A per-channel
            energy multiplier is then applied: a random sample from
            ``U[scale_min, scale_max]`` when ``channel_scales`` is omitted, or
            the user-provided length-Q tuple/list/array otherwise. The
            multiplier is there so that z-scored channels do not all sit on
            N(0, 1).

            When not to use it. With ``normalize=False`` this method leaves the
            array unchanged. The caller **must** preprocess ``series`` first
            (balance each channel's energy / scale) before calling the
            pipeline. Passing ``channel_scales`` in this mode is rejected.

        :param rng: The random number generator with fixed seed.
        :param series: Input univariate series of shape (T, Q).
        :param mechanism: Name of the coupling mechanism to use.
                         If None, one is randomly sampled.
        :param horizon: Override for forecast horizon.
        :param adjacency: Optional (Q, Q) binary graph describing the parent
                          structure over the Q variates. Only used by the SCM
                          mechanisms (``linear_scm`` / ``nonlinear_scm``); the
                          other mechanisms ignore it.
        :param apply_postprocessing: Whether to apply post-processing transforms.
        :param return_metadata: If True, also return metadata about the
                               generation process.
        :param normalize: If True, z-score each of the Q channels before
                          coupling. After z-scoring, each channel is multiplied
                          by an energy scale so the series do not all collapse
                          onto a standard normal. If False, the input is used
                          as-is and the caller is responsible for balancing
                          per-channel energy.
        :param channel_scales: Optional length-Q tuple, list, or array of
                               per-channel energy multipliers. Used only when
                               ``normalize=True``. If omitted, a scale is drawn
                               independently for each channel from
                               ``U[scale_min, scale_max]``. If provided, those
                               values are used and no random offset is added.
        :return: Coupled multivariate series of shape (T, Q), and optionally
                 a metadata dictionary.
        """
        T, Q = series.shape
        series, applied_scales = self._prepare_input_series(
            rng=rng,
            series=series,
            normalize=normalize,
            channel_scales=channel_scales,
        )
        metadata: Dict[str, Any] = {
            "input_shape": (T, Q),
            "n_variates": Q,
            "sequence_length": T,
            "normalized": bool(normalize),
            "channel_scales": applied_scales,
        }

        # Stage 1: Sample and apply coupling mechanism
        if mechanism is None:
            mechanism = self._sample_mechanism(rng, n_variates=Q)
        metadata["coupling_mechanism"] = mechanism
        metadata["custom_adjacency"] = adjacency is not None

        coupling = self._mechanisms[mechanism]
        coupled = coupling.couple(rng=rng, series=series, adjacency=adjacency)
        metadata["coupled_shape"] = coupled.shape

        # Stage 2: Post-processing
        if apply_postprocessing:
            h = horizon if horizon is not None else self._horizon
            coupled = self._post_processor(rng=rng, series=coupled, horizon=h)
            metadata["post_processed"] = True
        else:
            metadata["post_processed"] = False

        if return_metadata:
            return coupled, metadata
        return coupled

    def _prepare_input_series(
        self,
        rng: np.random.RandomState,
        series: np.ndarray,
        normalize: bool,
        channel_scales: Optional[Union[Sequence[float], np.ndarray]],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Optionally z-score and re-scale each channel of an input series.

        Operates on arrays of shape (T, Q), treating each of the Q columns as
        an independent channel. When ``normalize`` is False the series is
        returned unchanged and the caller is expected to have already set
        per-channel energy. When ``normalize`` is True each channel is first
        z-scored, then multiplied by either the user-supplied
        ``channel_scales`` or a random draw from ``U[scale_min, scale_max]``.

        :param rng: Random number generator used for the random energy offset.
        :param series: Input series of shape (T, Q).
        :param normalize: Whether to z-score and re-scale channels.
        :param channel_scales: Optional length-Q energy multipliers. Ignored
                               (and rejected) when ``normalize`` is False.
        :return: Prepared series of shape (T, Q) and the applied per-channel
                 scales, or ``None`` when ``normalize`` is False.
        """
        prepared = np.asarray(series, dtype=self._dtype)
        if prepared.ndim != 2:
            raise ValueError(
                f"series must be a 2-D array of shape (T, Q), got ndim={prepared.ndim}"
            )
        _, Q = prepared.shape

        if not normalize:
            if channel_scales is not None:
                raise ValueError(
                    "channel_scales is only used when normalize=True; "
                    "with normalize=False scale each channel before calling "
                    "the pipeline"
                )
            return prepared, None

        mean = prepared.mean(axis=0, keepdims=True)
        std = prepared.std(axis=0, keepdims=True)
        std = np.where(std < 1e-12, 1.0, std)
        prepared = (prepared - mean) / std

        if channel_scales is None:
            scales = rng.uniform(self._scale_min, self._scale_max, size=Q).astype(
                self._dtype, copy=False
            )
        else:
            scales = np.asarray(channel_scales, dtype=self._dtype).reshape(-1)
            if scales.size != Q:
                raise ValueError(
                    f"channel_scales must have length Q={Q}, got {scales.size}"
                )
        prepared = prepared * scales[np.newaxis, :]
        return prepared, scales

    def _sample_mechanism(
        self,
        rng: np.random.RandomState,
        n_variates: Optional[int] = None,
    ) -> str:
        """Sample a coupling mechanism according to the configured probabilities.

        :param rng: The random number generator.
        :param n_variates: If given, mechanisms that require more variates than
                           this are excluded before sampling (e.g. linear mixing
                           and the SCMs require at least 2 variates).
        :return: Name of the selected coupling mechanism.
        """
        names = list(self._mechanism_probs.keys())
        probs = np.array([self._mechanism_probs[n] for n in names], dtype=float)

        # Restrict to mechanisms compatible with the available number of variates.
        if n_variates is not None:
            valid_names, valid_probs = [], []
            for name, p in zip(names, probs):
                min_v = getattr(self._mechanisms[name], "min_variates", 1)
                if min_v <= n_variates:
                    valid_names.append(name)
                    valid_probs.append(p)
            names = valid_names
            probs = np.array(valid_probs, dtype=float)

        if len(names) == 0 or probs.sum() <= 0.0:
            raise ValueError(
                "no coupling mechanism is available for the requested number of "
                "variates (check mechanism_probabilities and min_variates)"
            )
        probs = probs / probs.sum()  # Normalize to sum to 1
        return rng.choice(names, p=probs)

    def _create_default_mechanisms(self) -> Dict[str, BaseCoupling]:
        """Create the default set of coupling mechanisms.

        Includes all seven mechanisms from TiRex-2 Section 3.4:
        identity, univariate, functional, linear mixing, cointegration,
        linear SCM, and nonlinear SCM.

        :return: Dictionary of coupling mechanism instances.
        """
        return {
            "identity": IdentityCoupling(dtype=self._dtype),
            "univariate": UnivariatePassThrough(dtype=self._dtype),
            "functional": FunctionalCoupling(dtype=self._dtype),
            "linear_mixing": LinearMixing(dtype=self._dtype),
            "cointegration": Cointegration(dtype=self._dtype),
            "linear_scm": LinearSCM(dtype=self._dtype),
            "nonlinear_scm": NonlinearSCM(dtype=self._dtype),
        }

    def generate(
        self,
        rng: np.random.RandomState,
        n_inputs_points: int,
        input_dimension: Optional[int] = None,
        mechanism: Optional[str] = None,
        horizon: Optional[int] = None,
        adjacency: Optional[np.ndarray] = None,
        apply_augmentation: bool = True,
        apply_postprocessing: bool = True,
        n_classes: Optional[int] = None,
        return_metadata: bool = False,
        normalize: bool = False,
        channel_scales: Optional[Union[Sequence[float], np.ndarray]] = None,
    ) -> Union[
        np.ndarray,
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, Dict[str, Any]],
        Tuple[np.ndarray, np.ndarray, Dict[str, Any]],
    ]:
        """Generate a coupled multivariate sample from scratch.

        This method first generates independent univariate series from a
        zero-mean Gaussian-process pool (randomly composed kernels, per the
        TiRex-2 synthetic pipeline), then optionally applies the Stage-1
        augmentation
        (piecewise-linear amplitude trends, quantile censoring, synthetic
        spikes), couples them, and finally post-processes the result.

        :param rng: The random number generator.
        :param n_inputs_points: Length of the time series to generate.
        :param input_dimension: Number of variates. If None, randomly
                               sampled from {1, ..., 12} (or taken from the
                               adjacency size when ``adjacency`` is given).
        :param mechanism: Coupling mechanism name or None for random.
        :param horizon: Forecast horizon.
        :param adjacency: Optional (Q, Q) binary graph describing the parent
                          structure over the Q variates. Only used by the SCM
                          mechanisms (``linear_scm`` / ``nonlinear_scm``); the
                          other mechanisms ignore it.
        :param apply_augmentation: Whether to apply the Stage-1 augmentation.
        :param apply_postprocessing: Whether to apply post-processing.
        :param n_classes: If given, also return a classification label for the
                          generated sample, derived deterministically from the
                          series summary statistic. For balanced labels use
                          ``generate_batch``.
        :param return_metadata: Whether to return metadata.
        :param normalize: Forwarded to :meth:`__call__`. If True, z-score and
                          re-scale each generated channel before coupling.
        :param channel_scales: Forwarded to :meth:`__call__`. Optional
                               per-channel energy multipliers used only when
                               ``normalize`` is True.
        :return: Generated multivariate time series. If ``n_classes`` is given,
                 returns ``(series, label)``. A metadata dict is appended when
                 ``return_metadata`` is True.
        """
        # Resolve the number of variates, honoring a user-supplied graph size.
        if adjacency is not None:
            adjacency = np.asarray(adjacency)
            if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
                raise ValueError(
                    "adjacency must be a square (Q, Q) matrix, got "
                    f"shape {adjacency.shape}"
                )
            graph_Q = adjacency.shape[0]
            if input_dimension is None:
                input_dimension = graph_Q
            elif input_dimension != graph_Q:
                raise ValueError(
                    f"input_dimension ({input_dimension}) does not match "
                    f"adjacency size ({graph_Q})"
                )
        elif input_dimension is None:
            input_dimension = rng.randint(1, 13)  # V ~ U{1, ..., 12}

        # Generate base univariate series from the GP-based synthetic pool
        # (TiRex-2: zero-mean GP with randomly composed kernels).
        base_series = self._generate_base_series(rng, n_inputs_points, input_dimension)

        # Stage 1: independent augmentation of each univariate series.
        if apply_augmentation:
            base_series = self._augment_series(rng, base_series)

        result = self(
            rng=rng,
            series=base_series,
            mechanism=mechanism,
            horizon=horizon,
            adjacency=adjacency,
            apply_postprocessing=apply_postprocessing,
            return_metadata=return_metadata,
            normalize=normalize,
            channel_scales=channel_scales,
        )

        if return_metadata:
            series, metadata = result
            metadata["augmented"] = apply_augmentation
        else:
            series = result

        if n_classes is not None:
            label = label_single(summarize_series(series), n_classes)
            if return_metadata:
                metadata["n_classes"] = n_classes
                return series, label, metadata
            return series, label

        if return_metadata:
            return series, metadata
        return series

    def generate_batch(
        self,
        rng: np.random.RandomState,
        n_samples: int,
        n_inputs_points: int,
        input_dimension: Optional[int] = None,
        mechanism: Optional[str] = None,
        horizon: Optional[int] = None,
        adjacency: Optional[np.ndarray] = None,
        apply_augmentation: bool = True,
        apply_postprocessing: bool = True,
        n_classes: Optional[int] = None,
    ) -> List[Any]:
        """Generate a batch of coupled multivariate samples from scratch.

        Mirrors :meth:`CaukerPipeline.generate_batch`: each sample is a coupled
        multivariate time series, and the coupling mechanism is randomized per
        sample (unless ``mechanism`` is fixed).

        :param rng: The random number generator.
        :param n_samples: Number of samples to generate.
        :param n_inputs_points: Length of each time series.
        :param input_dimension: Number of variates per sample (None => random,
                                or taken from ``adjacency`` when given).
        :param mechanism: Coupling mechanism name or None for random per sample.
        :param horizon: Forecast horizon.
        :param adjacency: Optional (Q, Q) binary graph reused for every sample;
                          only used by the SCM mechanisms.
        :param apply_augmentation: Whether to apply the Stage-1 augmentation.
        :param apply_postprocessing: Whether to apply post-processing.
        :param n_classes: If given, assign each sample a balanced class label by
                          quantile-binning the per-sample summary statistic.
        :return: List of coupled series, each of shape (T, Q). If ``n_classes``
                 is given, returns a list of ``(series, label)`` tuples instead.
        """
        dataset = []
        for _ in range(n_samples):
            x = self.generate(
                rng=rng,
                n_inputs_points=n_inputs_points,
                input_dimension=input_dimension,
                mechanism=mechanism,
                horizon=horizon,
                adjacency=adjacency,
                apply_augmentation=apply_augmentation,
                apply_postprocessing=apply_postprocessing,
                return_metadata=False,
            )
            dataset.append(x)

        if n_classes is not None:
            stats = [summarize_series(x) for x in dataset]
            labels = discretize_labels(stats, n_classes)
            return [(x, int(y)) for x, y in zip(dataset, labels)]
        return dataset

    @staticmethod
    def _generate_base_series(
        rng: np.random.RandomState,
        T: int,
        Q: int,
    ) -> np.ndarray:
        """Generate base univariate series for coupling.

        Following the TiRex-2 univariate pool, each base series is drawn from
        a zero-mean Gaussian process whose kernel is randomly composed from a
        fixed bank under {+, x}. This replaces the earlier random-walk /
        AR(1) / noise / sine placeholders with the paper's GP-based synthetic
        pipeline.

        :param rng: The random number generator.
        :param T: Sequence length.
        :param Q: Number of variates.
        :return: Base series of shape (T, Q).
        """
        kernel_bank = _build_kernel_bank()
        t_grid = np.linspace(0, 1, T, dtype=np.float64)
        mean = np.zeros(T, dtype=np.float64)

        columns = [
            _sample_gp(rng, mean, _sample_composite_kernel(rng, kernel_bank), t_grid)
            for _ in range(Q)
        ]
        return np.column_stack(columns).astype(np.float64)

    def _augment_series(
        self,
        rng: np.random.RandomState,
        series: np.ndarray,
    ) -> np.ndarray:
        """Apply the Stage-1 augmentation to each univariate series.

        Implements the first stage of the TiRex-2 coupling pipeline: each
        series is independently perturbed with piecewise-linear amplitude
        trends, quantile censoring, and synthetic spikes (Gaussian, triangular,
        or rectangular kernels). All transforms preserve length.

        :param rng: The random number generator.
        :param series: Base univariate series of shape (T, Q).
        :return: Augmented series of shape (T, Q).
        """
        T, Q = series.shape
        result = np.zeros_like(series, dtype=self._dtype)

        for j in range(Q):
            col = series[:, j].astype(float).copy()

            # Piecewise-linear amplitude trends
            if T >= 2:
                n_changepoints = max(2, min(5, T))
                col = amplitude_modulation(
                    col, num_changepoints=n_changepoints, rng=rng
                )

            # Quantile censoring
            col = censor_augmentation(col, rng=rng)

            # Synthetic spikes
            col = spike_injection(col, rng=rng)

            result[:, j] = col

        return result

    @property
    def mechanisms(self) -> Dict[str, BaseCoupling]:
        """Get the coupling mechanism dictionary."""
        return self._mechanisms

    @property
    def mechanism_probabilities(self) -> Dict[str, float]:
        """Get the mechanism sampling probabilities."""
        return self._mechanism_probs

    @property
    def post_processor(self) -> PostProcessor:
        """Get the post-processor instance."""
        return self._post_processor

    @property
    def dtype(self) -> np.dtype:
        """Get the data type."""
        return self._dtype

    @property
    def scale_min(self) -> float:
        """Lower bound of the random per-channel energy multiplier."""
        return self._scale_min

    @property
    def scale_max(self) -> float:
        """Upper bound of the random per-channel energy multiplier."""
        return self._scale_max
