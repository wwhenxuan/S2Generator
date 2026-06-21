# -*- coding: utf-8 -*-
"""
Created on 2026/06/21 23:00:00
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/S2Generator
"""

from typing import Optional, Tuple, List, Union

import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.stats.diagnostic import acorr_ljungbox

import warnings

warnings.filterwarnings("ignore")


class GaussianMixtureSimulator(object):
    """
    Simulate time series using a Markov-switching Gaussian mixture model.

    The original static Gaussian mixture treats every observation as an i.i.d.
    draw from a finite mixture and therefore cannot capture temporal dependence.
    This simulator instead follows the same white-noise excitation viewpoint as
    the other modules in ``s2generator.simulator``:

        S_t ~ Markov(S_{t-1})
        y_t = mu_{S_t} + sigma_{S_t} * w_t,   w_t ~ N(0, 1)

    Each mixture component corresponds to a latent regime with its own mean and
    variance. A Markov transition matrix governs regime persistence, so generated
    sequences preserve segment-like structure instead of independent point-wise
    sampling.

    Fitting is performed with ``statsmodels.tsa.regime_switching.MarkovRegression``
    (order = 0). After fitting, ``transform`` simulates a regime path and excites
    each regime with fresh Gaussian white noise.
    """

    _VALID_COVARIANCE_TYPES = ("full", "tied", "diag", "spherical")

    def __init__(
        self,
        n_components: int = 3,
        covariance_type: str = "full",
        tol: float = 1e-3,
        reg_covarfloat: float = 1e-6,
        max_iter: int = 100,
        n_init: int = 1,
        init_params: str = "kmeans",
        random_state: Optional[int] = 42,
        max_n_components: Optional[int] = None,
        switching_variance: bool = True,
        trend: str = "c",
        signif: float = 0.05,
        not_white_alarm: bool = False,
        revin: bool = True,
    ) -> None:
        """
        :param n_components: Number of Gaussian mixture components / latent regimes.
        :param covariance_type: Kept for API compatibility. Univariate emissions use a
            scalar variance per component regardless of this setting.
        :param tol: Convergence tolerance passed to the statsmodels optimizer.
        :param reg_covarfloat: Minimum variance floor used during simulation.
        :param max_iter: Maximum number of EM / optimization iterations.
        :param n_init: Number of random restarts attempted during fitting.
        :param init_params: Kept for API compatibility with the legacy sklearn-based
            implementation. The Markov-switching fitter manages its own initialization.
        :param random_state: Random seed for reproducible simulation.
        :param max_n_components: Upper bound used when ``select_order=True`` in ``fit``.
            Defaults to ``n_components`` when None.
        :param switching_variance: Whether each component has its own variance.
        :param trend: Trend specification passed to ``MarkovRegression``.
        :param signif: Significance level used in residual diagnosis.
        :param not_white_alarm: Whether to raise when residuals fail the white-noise test.
        :param revin: Whether to apply reversible normalization before fitting and after
            generation.

        :return: None
        """
        if n_components < 1:
            raise ValueError("n_components must be at least 1.")
        if covariance_type not in self._VALID_COVARIANCE_TYPES:
            raise ValueError(
                "Invalid covariance_type. Must be one of "
                "'full', 'tied', 'diag', 'spherical'."
            )
        if init_params not in ["kmeans", "k-means++", "random", "random_from_data"]:
            raise ValueError(
                "Invalid init_params. Must be one of "
                "'kmeans', 'k-means++', 'random', 'random_from_data'."
            )

        self.n_components = n_components
        self.max_n_components = (
            max_n_components if max_n_components is not None else n_components
        )
        self.covariance_type = covariance_type
        self.tol = tol
        self.reg_covarfloat = reg_covarfloat
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.random_state = random_state
        self.switching_variance = switching_variance
        self.trend = trend
        self.signif = signif
        self.not_white_alarm = not_white_alarm
        self.revin = revin

        self.input_mean, self.input_std = None, None
        self.k_regimes = None

        self.mr_model = None
        self.model = None
        self.residuals = None
        self.smoothed_probabilities = None
        self.simulated_series = None

        self.rng = np.random.RandomState(seed=random_state)

    def fit(
        self, time_series: np.ndarray, select_order: Optional[bool] = False
    ) -> None:
        """
        Fit a Markov-switching Gaussian mixture model to the input time series.

        :param time_series: Input series with shape 1D [seq_len, ] or 2D [num_samples, seq_len].
        :param select_order: If True, select the number of components by BIC.

        :return: None
        """
        time_series = self.check_inputs(time_series=time_series)
        endog = np.asarray(time_series, dtype=np.float64)

        if self.revin:
            self.input_mean = np.mean(endog)
            self.input_std = np.std(endog)
            endog = (endog - self.input_mean) / self.input_std

        if select_order:
            self.k_regimes = self.select_n_components(endog=endog)
        else:
            self.k_regimes = self.n_components

        self.model = self._fit_markov_regression(endog=endog, k_regimes=self.k_regimes)

        self.residuals = np.asarray(self.model.resid)
        self.smoothed_probabilities = np.asarray(
            self.model.smoothed_marginal_probabilities
        )

        mean_p_value, is_white = self.residual_diagnosis(signif=self.signif)
        if not is_white and self.not_white_alarm:
            raise ValueError(
                "Warning: Model residuals may not be white noise "
                f"(mean p-value={mean_p_value:.4f} < significance level={self.signif}), "
                "please re-evaluate the number of mixture components."
            )

    def transform(
        self, num_samples: int, seq_len: int, random_state: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate new time series by exciting the fitted regime dynamics with white noise.

        :param num_samples: Number of independent sample paths to generate.
        :param seq_len: Length of each generated sequence.
        :param random_state: Random seed for reproducibility. Uses the instance seed if None.

        :return: Generated series with shape [num_samples, seq_len].
        """
        if self.model is None:
            raise ValueError(
                "The model must be fitted before calling transform; please call `fit` first."
            )

        seed = random_state if random_state is not None else self.random_state
        simulated_series = np.zeros((num_samples, seq_len), dtype=np.float64)

        for i in range(num_samples):
            sample_seed = None if seed is None else int(seed) + i
            simulated_series[i, :] = self._simulate(
                seq_len=seq_len, random_state=sample_seed
            )

        if self.revin:
            self.simulated_series = simulated_series * self.input_std + self.input_mean
        else:
            self.simulated_series = simulated_series

        return self.simulated_series

    def select_n_components(self, endog: np.ndarray) -> int:
        """
        Select the number of mixture components using the BIC criterion.

        :param endog: One-dimensional normalized input series.

        :return: Number of components with the lowest BIC among successfully fitted models.
        """
        best_bic = np.inf
        best_k = 1

        for k_regimes in range(1, self.max_n_components + 1):
            if len(endog) <= k_regimes + 2:
                continue
            try:
                result = self._fit_markov_regression(endog=endog, k_regimes=k_regimes)
                if result.bic < best_bic:
                    best_bic = result.bic
                    best_k = k_regimes
            except Exception:
                continue

        return best_k

    def residual_diagnosis(
        self, lags: int = 20, signif: float = None
    ) -> Tuple[float, bool]:
        """
        Perform a Ljung-Box white-noise test on the fitted model residuals.

        :param lags: Number of lags used in the Ljung-Box test.
        :param signif: Significance level. Defaults to ``self.signif`` when None.

        :return: Mean p-value and a boolean indicating whether all p-values exceed ``signif``.
        """
        if self.residuals is None:
            raise ValueError(
                "The model must be fitted before calling residual_diagnosis."
            )

        lb_test = acorr_ljungbox(self.residuals, lags=lags, return_df=True)
        lb_p_values = lb_test["lb_pvalue"]
        threshold = signif if signif is not None else self.signif

        return lb_p_values.mean(), np.all(lb_p_values > threshold)

    def model_summary(self) -> str:
        """
        Return the textual summary of the fitted Markov-switching model.

        :return: Summary string produced by statsmodels.
        """
        if self.model is None:
            raise ValueError("The model must be fitted before calling model_summary.")
        return self.model.summary().as_text()

    def check_inputs(self, time_series: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """
        Check whether the input time series satisfies the modeling requirements.

        :param time_series: Input with shape 1D [seq_len, ] or 2D [num_samples, seq_len].

        :return: Validated one-dimensional ``np.ndarray``.
        """
        if not isinstance(time_series, (pd.Series, np.ndarray)):
            raise ValueError(
                "Input time_series must be a pandas Series or numpy ndarray."
            )

        if len(time_series.shape) > 2:
            raise ValueError(
                "Input time_series must be 1-dimensional with [seq_len, ] or "
                "2-dimensional with [num_samples, seq_len]."
            )

        if len(time_series.shape) == 2:
            time_series = time_series.flatten()

        time_series = np.asarray(time_series, dtype=np.float64)

        min_length = max(10, self.n_components + 2)
        if len(time_series) < min_length:
            raise ValueError(
                f"Input time_series must have at least {min_length} data points."
            )

        if np.isnan(time_series).any():
            raise ValueError("Input time_series must not contain NaN values.")

        if np.std(time_series) < 1e-8:
            raise ValueError(
                "The time series variance is 0 (all values are the same), "
                "making it impossible to fit the mixture model."
            )

        return time_series

    def _check_inputs(self, time_series: np.ndarray) -> np.ndarray:
        """Backward-compatible alias of :meth:`check_inputs`."""
        return self.check_inputs(time_series=time_series)

    def weight(self, component_index: int) -> float:
        """
        Get the stationary mixture weight of a specific component.

        :param component_index: Index of the component.
        :return: Stationary regime probability of the specified component.
        """
        stationary_probs = self._stationary_probabilities()
        if component_index < 0 or component_index >= self.k_regimes:
            raise ValueError(
                f"Component index must be between 0 and {self.k_regimes - 1}."
            )
        return float(stationary_probs[component_index])

    def weights(self) -> np.ndarray:
        """
        Get the stationary mixture weights of all components.

        :return: Stationary regime probabilities for all components.
        """
        return self._stationary_probabilities()

    def mean(self, component_index: int) -> float:
        """
        Get the mean of a specific Gaussian component.

        :param component_index: Index of the component.
        :return: Mean of the specified component.
        """
        means = self.means()
        if component_index < 0 or component_index >= self.k_regimes:
            raise ValueError(
                f"Component index must be between 0 and {self.k_regimes - 1}."
            )
        return float(means[component_index])

    def means(self) -> np.ndarray:
        """
        Get the means of all Gaussian components.

        :return: Component means with shape [k_regimes, ].
        """
        if self.model is None:
            raise ValueError("The model must be fitted before calling means.")
        const, _ = self._extract_regime_parameters()
        return const

    def covariance(self, component_index: int) -> float:
        """
        Get the variance of a specific Gaussian component.

        :param component_index: Index of the component.
        :return: Variance of the specified component.
        """
        covariances = self.covariances()
        if component_index < 0 or component_index >= self.k_regimes:
            raise ValueError(
                f"Component index must be between 0 and {self.k_regimes - 1}."
            )
        return float(covariances[component_index])

    def covariances(self) -> np.ndarray:
        """
        Get the variances of all Gaussian components.

        :return: Component variances with shape [k_regimes, ].
        """
        if self.model is None:
            raise ValueError("The model must be fitted before calling covariances.")
        _, variance = self._extract_regime_parameters()
        return variance

    @property
    def param_names(self) -> List[str]:
        """Return the names of the parameters in the fitted model."""
        if self.model is None:
            raise ValueError("The model must be fitted before calling param_names.")
        return self.model.model.param_names

    @property
    def params(self) -> np.ndarray:
        """Return the parameter values of the fitted model."""
        if self.model is None:
            raise ValueError("The model must be fitted before calling params.")
        return np.asarray(self.model.params)

    @property
    def param_items(self) -> List[Tuple[str, float]]:
        """Return a list of ``(parameter name, parameter value)`` tuples."""
        if self.model is None:
            raise ValueError("The model must be fitted before calling param_items.")
        return list(zip(self.param_names, self.params))

    @property
    def regime_transition(self) -> np.ndarray:
        """
        Return the fitted Markov transition probability matrix.

        Entry ``[j, i, 0]`` denotes ``P(S_t = j | S_{t-1} = i)``.
        """
        if self.model is None:
            raise ValueError(
                "The model must be fitted before calling regime_transition."
            )
        return np.asarray(self.model.regime_transition)

    def _fit_markov_regression(self, endog: np.ndarray, k_regimes: int):
        """
        Fit ``MarkovRegression`` with optional random restarts.

        :param endog: One-dimensional normalized input series.
        :param k_regimes: Number of latent regimes / mixture components.

        :return: Fitted statsmodels result object.
        """
        best_result = None
        best_llf = -np.inf

        for init_idx in range(self.n_init):
            seed = (
                None if self.random_state is None else int(self.random_state) + init_idx
            )
            self.mr_model = MarkovRegression(
                endog=endog,
                k_regimes=k_regimes,
                order=0,
                trend=self.trend,
                switching_trend=True,
                switching_variance=self.switching_variance,
            )
            try:
                result = self.mr_model.fit(
                    disp=False,
                    maxiter=self.max_iter,
                    start_params=None,
                )
            except Exception:
                continue

            if result.llf > best_llf:
                best_llf = result.llf
                best_result = result

        if best_result is None:
            raise ValueError(
                "Failed to fit the Markov-switching Gaussian mixture model. "
                "Please check the input series or reduce the number of components."
            )

        return best_result

    def _extract_regime_parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parse regime-specific means and innovation variances from the fitted model.

        :return: Tuple ``(const, variance)`` with shape ``[k_regimes, ]``.
        """
        k_regimes = self.model.k_regimes
        mr_model = self.model.model
        fitted_params = self.model.params

        const = np.zeros(k_regimes, dtype=np.float64)
        for regime in range(k_regimes):
            const[regime] = fitted_params[mr_model.parameters[regime, "exog"]][0]

        variance = fitted_params[mr_model.parameters["variance"]]
        variance = np.atleast_1d(np.asarray(variance, dtype=np.float64)).reshape(-1)
        if variance.size == 1:
            variance = np.repeat(variance[0], k_regimes)

        variance = np.maximum(variance, self.reg_covarfloat)
        return const, variance

    def _stationary_probabilities(self) -> np.ndarray:
        """
        Compute the stationary distribution of the fitted Markov chain.

        :return: Stationary regime probabilities with shape [k_regimes, ].
        """
        if self.model is None:
            raise ValueError(
                "The model must be fitted before accessing mixture weights."
            )

        transition = self.regime_transition[:, :, 0]
        k_regimes = transition.shape[0]
        pi = np.asarray(self.model.initial_probabilities, dtype=np.float64)

        for _ in range(1000):
            pi_next = np.zeros(k_regimes, dtype=np.float64)
            for j in range(k_regimes):
                pi_next[j] = np.sum(pi * transition[j, :])
            if np.allclose(pi_next, pi, atol=1e-10):
                break
            pi = pi_next

        pi = np.maximum(pi, 0.0)
        return pi / np.sum(pi)

    def _simulate(self, seq_len: int, random_state: Optional[int] = None) -> np.ndarray:
        """
        Simulate a single sample path from the fitted Markov-switching Gaussian model.

        White noise excites each regime-specific Gaussian emission while a Markov chain
        governs regime persistence:

            y_t = mu_{S_t} + sigma_{S_t} * w_t

        :param seq_len: Length of the simulated sequence.
        :param random_state: Random seed for reproducibility.

        :return: Simulated series with shape [seq_len, ].
        """
        rng = np.random.RandomState(seed=random_state)
        const, variance = self._extract_regime_parameters()
        transition = self.regime_transition[:, :, 0]
        initial_prob = np.asarray(self.model.initial_probabilities, dtype=np.float64)

        current_regime = rng.choice(self.k_regimes, p=initial_prob)
        simulated = np.zeros(seq_len, dtype=np.float64)

        for t in range(seq_len):
            if t > 0:
                current_regime = rng.choice(
                    self.k_regimes, p=transition[:, current_regime]
                )

            white_noise = rng.normal(loc=0.0, scale=1.0)
            simulated[t] = (
                const[current_regime] + np.sqrt(variance[current_regime]) * white_noise
            )

        return simulated
