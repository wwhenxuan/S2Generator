# -*- coding: utf-8 -*-
"""
Created on 2026/02/12 12:53:15
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""

from typing import Union, Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.api import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from s2generator.utils._tools import eacf_rlike
from s2generator.utils.visualization import plot_shapiro_wilk
from s2generator.simulator.low_pass_filter import (
    maybe_attach_lowpass,
    maybe_apply_lowpass,
)

import warnings

warnings.filterwarnings("ignore")


class ARIMASimulator(object):
    """
    Simulate time series data using ARIMA model.

    Any stationary signal can be viewed as the response obtained by exciting a linear system with white noise.

    Through differencing operations, we can transform a non-stationary time series into a stationary one.

    Based on these two points, we can use the ARIMA model to generate non-stationary time series data.
    Compared to previous data generation methods, we can further fit the statistical characteristics of real time series data through the ARIMA model, thereby generating more realistic time series data.

    Since this generation method involves the fitting and training of the ARIMA model, linear operations may trigger exceptions such as `LinAlgError`, resulting in generation failure.
    This issue is generally related to the input time series data and the order of the ARIMA model. We have investigated the common input data problems as follows:

    1. The data is completely constant (variance = 0);
    2. The length of the input time series is too short;
    3. There are obvious extreme values or outliers in the input sequence after standardization;
    4. An excessively high order setting (p,q) leads to matrix dimension mismatch or singularity.

    In addition, the `ARIMA` implementation in `statsmodels` has limited ability to handle certain ill-conditioned matrices (e.g., nearly singular matrices).
    Even if the data appears normal, LU decomposition may still fail due to floating-point precision issues.
    """

    def __init__(
        self,
        max_p: int = 5,
        max_d: int = 2,
        max_q: int = 5,
        signif: float = 0.05,
        not_white_alarm: bool = True,
        revin: bool = True,
        random_state: Optional[int] = 42,
        lowpass: bool = False,
        lowpass_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        :param max_p: Maximum AR order (p) to consider when fitting the ARIMA model.
        :param max_d: Maximum differencing order (d) to consider when fitting the ARIMA model.
        :param max_q: Maximum MA order (q) to consider when fitting the ARIMA model.
        :param signif: Significance level for the ADF test to determine stationarity.
        :param not_white_alarm: Whether to issue a warning when the residuals of the fitted model are not white noise.
        :param revin: Should reversible normalization be performed on time series data?
        :param random_state: Random state for reproducibility when generating new time series data.
        :param lowpass: Whether to apply low-pass post-processing after ``transform``.
        :param lowpass_kwargs: Optional kwargs forwarded to ``LowPassFilter``.
        """
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q

        # Significance level of ADF test
        self.signif = signif

        # Whether to issue a warning when residuals are not white noise
        self.not_white_alarm = not_white_alarm

        # Should reversible normalization be performed on time series data?
        # If True, the generated time series data will be normalized to have zero mean and unit variance,
        # and the original mean and variance will be recorded for potential inverse transformation.
        self.revin = revin
        self.mean, self.std = None, None

        # Record the parameters of the model fit
        self.d_order = None
        self.p_order, self.q_order = None, None

        # Record the fitted model
        self.model = None
        # Record the residual results of the model fitting
        self.residuals = None

        # The random state for reproducibility
        self.random_state = random_state

        self.lowpass = lowpass
        self.lowpass_kwargs = lowpass_kwargs
        self._lowpass_filter = None
        self.time_series = None

    def fit(
        self, time_series: np.ndarray, select_order: Optional[bool] = False
    ) -> None:
        """
        Fit the ARIMA model to the provided time series data.

        :param time_series: The input time series data to fit the ARIMA model.
        :param select_order: Whether to automatically select the (p, d, q) order.

        :return: None.
        """
        # Check the input time series data
        time_series = self.check_inputs(time_series=time_series)
        self.time_series = np.asarray(time_series, dtype=np.float64)

        # Optionally reverse the time series data to generate data in reverse order
        if self.revin:
            self.mean, self.std = time_series.mean(), time_series.std()
            time_series = (
                time_series - self.mean
            ) / self.std  # Normalize the time series data

        # First, difference the time series to make it stationary
        stationary_series, self.d_order = self.diff_stationary(time_series=time_series)

        if select_order:
            # Use the AIC and BIC criteria to select the optimal (p,q) combination.
            self.p_order, self.q_order = self.select_arma_order(
                stationary_series=stationary_series
            )
        else:
            self.p_order, self.q_order = self.max_p, self.max_q

        # Fit the ARIMA model
        self.model = ARIMA(
            time_series, order=(self.p_order, self.d_order, self.q_order)
        ).fit()

        # Get the residuals from the fitted model
        self.residuals = self.model.resid

        # Perform residual diagnosis
        mean_p_value, is_white = self.residual_diagnosis(signif=self.signif)

        if not is_white and self.not_white_alarm:
            raise ValueError(
                f"Warning: Model residuals may not be white noise (mean p-value={mean_p_value:.4f} < significance level={self.signif}), please re-evaluate the model order or parameters."
            )

        maybe_attach_lowpass(
            self,
            enabled=self.lowpass,
            kwargs=self.lowpass_kwargs,
            reference=self.time_series,
        )

    def transform(
        self, num_samples: int, seq_len: int, random_state: Optional[int] = None
    ) -> np.ndarray:
        """
        Transform the input time series data using the fitted ARIMA model.

        :param num_samples: Number of samples to generate.
        :param seq_len: Length of each generated sequence.

        :return: Transformed time series data with shape (num_samples, seq_len).
        """
        # Check if the model has been fitted
        if not hasattr(self, "model") or self.model is None:
            raise ValueError("The model must be fitted before calling transform.")

        # Generate new time series data
        generated_series = self.model.simulate(
            nsimulations=seq_len,
            repetitions=num_samples,
            random_state=(
                random_state if random_state is not None else self.random_state
            ),
        )

        simulated = (
            generated_series.values.T * self.std + self.mean
            if self.revin
            else generated_series.values.T
        )
        return maybe_apply_lowpass(self, simulated)

    @property
    def param_names(self) -> List[str]:
        """Return the names of the parameters in the fitted ARIMA model."""
        if not hasattr(self, "model"):
            raise ValueError("The model must be fitted before calling param_names.")

        return self.model.param_names

    @property
    def params(self) -> Union[np.ndarray, pd.Series]:
        """Return the parameter values of the fitted ARIMA model."""
        if not hasattr(self, "model"):
            raise ValueError("The model must be fitted before calling params.")

        return self.model.params

    @property
    def param_items(self) -> List[Tuple[str, float]]:
        """Return a list of (parameter name, parameter value) tuples for the fitted ARIMA model."""
        if not hasattr(self, "model"):
            raise ValueError("The model must be fitted before calling param_items.")

        return list(zip(self.param_names, self.params))

    def check_inputs(self, time_series: Union[pd.Series, np.ndarray]) -> pd.Series:
        """
        Check if the input time series data is valid.

        :param time_series: The input time series data to check.

        :return: Validated time series data as a pandas Series.
        """

        # Check if the input is a pandas Series or numpy ndarray
        if not isinstance(time_series, (pd.Series, np.ndarray)):
            raise ValueError(
                "Input time series must be a pandas Series or numpy ndarray."
            )

        # Check the shape of the input time series
        if len(time_series.shape) > 2:
            raise ValueError(
                "Input time series must be 1-dimensional with [seq_len, ] or 2-dimensional with [num_samples, seq_len]."
            )

        # Two-dimensional data needs to be flattened into one-dimensional data for use.
        elif len(time_series.shape) == 2:
            time_series = time_series.flatten()

        # Check if the length of the time series is sufficient
        if len(time_series) < 10:
            raise ValueError("Input time series must have at least 10 data points.")

        # Check if the time series contains NaN values
        if pd.isnull(time_series).any():
            raise ValueError("Input time series must not contain NaN values.")

        # std = np.std(time_series)
        std = np.std(time_series)
        if (
            std < 1e-8
        ):  # A very small threshold to check if the variance is effectively zero
            raise ValueError(
                "The time series variance is 0 (all values ​​are the same), making it impossible to fit the ARIMA model."
            )

        return pd.Series(time_series)

    def select_arma_order(
        self, stationary_series: Union[pd.Series, np.ndarray]
    ) -> Tuple[int, int]:
        """
        Select ARMA(p,q) order for a stationary series (i.e., ARIMA's p,q).

        :param stationary_series: The input stationary time series inputs data.

        :return: A tuple containing the selected (p, q) order.
        """

        # Use AIC and BIC criteria to select the optimal (p,q) combination

        # Initialize AIC and BIC
        best_aic = np.inf
        best_bic = np.inf
        best_order_aic = (0, 0)
        best_order_bic = (0, 0)

        # Iterate over all possible (p,q) combinations
        for p in range(self.max_p + 1):
            for q in range(self.max_q + 1):
                if p == 0 and q == 0:
                    # Skip the (p,q)=(0,0) case
                    continue
                try:
                    # Fit ARMA model
                    model = ARIMA(stationary_series, order=(p, 0, q))
                    results = model.fit()
                    if results.aic < best_aic:
                        best_aic = results.aic
                        best_order_aic = (p, q)
                    if results.bic < best_bic:
                        best_bic = results.bic
                        best_order_bic = (p, q)
                except:
                    continue

        return best_order_bic

    def diff_stationary(
        self, time_series: Union[pd.Series, np.ndarray]
    ) -> Tuple[Union[pd.Series, np.ndarray], int]:
        """
        Successive differencing makes the time series stationary, returning the stationary series and the order of differencing.

        :param time_series: The input time series data to difference.
        :return: A tuple containing the stationary series and the order of differencing.
        """

        # Check the input series
        is_stationary = self.adf_test(time_series=time_series)

        diff_count = 0

        diff_series = time_series.copy()

        while not is_stationary and diff_count < self.max_d:
            diff_series = diff_series.diff().dropna()  # Perform differencing
            is_stationary = self.adf_test(time_series=diff_series)
            diff_count += 1

        # Return the stationary series and the order of differencing
        return diff_series, diff_count

    def adf_test(self, time_series: Union[pd.Series, np.ndarray]) -> bool:
        """Perform the Augmented Dickey-Fuller (ADF) test to check if the time series is stationary."""

        # After differencing, there may be NaN values, which need to be removed
        adf_result = adfuller(time_series.dropna())
        p_value = adf_result[1]

        if p_value < self.signif:
            return True
        else:
            return False

    def model_summary(self) -> str:
        """Return the summary information of the fitted model."""
        if not hasattr(self, "model"):
            raise ValueError("The model must be fitted before calling model_summary.")

        return self.model.summary().as_text()

    def residual_diagnosis(
        self, lags: int = 20, signif: float = None
    ) -> Tuple[float, bool]:
        """
        Perform residual diagnosis for the fitted ARIMA model.
        This method evaluates the white noise nature of the model residuals by performing the Ljung-Box test.
        If the residuals are considered white noise, it indicates a good model fit.
        If not white noise, the model order or parameters may need to be re-evaluated.
        The method returns the mean of the test statistics as well as
        a boolean indicating whether the residuals are white noise.
        If the significance level is not provided, the instance's significance level is used, with a default of 0.05.

        :param signif: Significance level for the Ljung-Box test. If None, uses the instance's signif.

        :return: A tuple containing the mean p-value from the Ljung-Box test and a boolean indicating if all p-values exceed the significance level.
        """
        # Ensure the model has been fitted and residuals have been computed
        if not hasattr(self, "residuals"):
            raise ValueError(
                "The model must be fitted before calling residual_diagnosis."
            )

        # Perform the Ljung-Box test
        lb_test = acorr_ljungbox(self.residuals, lags=lags, return_df=True)
        lb_p_values = lb_test["lb_pvalue"]

        return lb_p_values.mean(), np.all(
            lb_p_values > signif if signif is not None else self.signif
        )

    @staticmethod
    def acf(
        time_series: Union[pd.Series, np.ndarray],
        nlags: Optional[int] = None,
        qstat: Optional[bool] = False,
        fft: Optional[bool] = True,
        missing: Optional[str] = "none",
    ) -> np.ndarray:
        """
        Calculate the autocorrelation function (ACF) of the given time series data.
        The acf at lag 0 (ie., 1) is returned.

        For very long time series it is recommended to use fft convolution instead. When fft is False uses a simple,
        direct estimator of the autocovariances that only computes the first nlag + 1 values.
        This can be much faster when the time series is long and only a small number of autocovariances are needed.

        If adjusted is true, the denominator for the autocovariance is adjusted for the loss of data.

        :param time_series: The input time series data.
        :param nlags: Number of lags to return autocorrelation for. If not provided, uses min(10 * np.log10(nobs), nobs - 1). The returned value includes lag 0 (ie., 1) so size of the acf vector is (nlags + 1,).
        :param qstat: If True, returns the Ljung-Box q statistic for each autocorrelation coefficient. See q_stat for more information.
        :param fft: If True, computes the ACF via FFT.
        :param missing: A string in [“none”, “raise”, “conservative”, “drop”] specifying how the NaNs are to be treated. “none” performs no checks.
                        “raise” raises an exception if NaN values are found. “drop” removes the missing observations and then estimates the autocovariances treating the non-missing as contiguous.
                        “conservative” computes the autocovariance using nan-ops so that nans are removed when computing the mean and cross-products that are used to estimate the autocovariance.
                         When using “conservative”, n is set to the number of non-missing observations.

        :return: The autocorrelation function values.
        """
        return acf(x=time_series, nlags=nlags, qstat=qstat, fft=fft, missing=missing)

    @staticmethod
    def plot_acf(
        time_series: Union[pd.Series, np.ndarray],
        lags: Optional[int] = 20,
        figsize: Optional[Tuple[int, int]] = (10, 4),
        dpi: int = 128,
    ) -> plt.Figure:
        """
        Plot the autocorrelation function (ACF) of the given time series data.
        :param time_series: The input time series data.
        :param lags: Number of lags to include in the plot.
        :param figsize: Figure size for the plot.

        :return: The matplotlib Figure object containing the ACF plot.
        """
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        plot_acf(time_series, lags=lags, ax=ax)
        return fig

    @staticmethod
    def pacf(
        time_series: Union[pd.Series, np.ndarray],
        nlags: Optional[int] = None,
        method: Optional[str] = "ywunbiased",
        alpha: Optional[float] = None,
    ) -> np.ndarray:
        """
        Partial autocorrelation estimate for a given time series.

        :param time_series: The input time series data.
        :param nlags: Number of lags to return autocorrelation for. If not provided, uses min(10 * np.log10(nobs), nobs - 1). The returned value includes lag 0 (ie., 1) so size of the acf vector is (nlags + 1,).
        :param method: The method to use for estimating the partial autocorrelations. Options are 'ywunbiased', 'ywmle', 'ols'.
        :param alpha: If provided, the confidence intervals for the partial autocorrelations are also returned at the given significance level.

        :return: The partial autocorrelation function values.
        """
        return pacf(x=time_series, nlags=nlags, method=method, alpha=alpha)

    @staticmethod
    def plot_pacf(
        time_series: Union[pd.Series, np.ndarray],
        lags: Optional[int] = 20,
        figsize: Optional[Tuple[int, int]] = (10, 4),
        dpi: int = 128,
    ) -> plt.Figure:
        """
        Plot the partial autocorrelation function (PACF) of the given time series data.
        :param time_series: The input time series data.
        :param lags: Number of lags to include in the plot.
        :param figsize: Figure size for the plot.

        :return: The matplotlib Figure object containing the PACF plot.
        """
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        plot_pacf(time_series, lags=lags, ax=ax)
        return fig

    @staticmethod
    def eacf(
        time_series: Union[pd.Series, np.ndarray],
        symbolize: bool = True,
        max_ar: int = 5,
        max_ma: int = 6,
    ) -> Tuple[np.ndarray, float, pd.DataFrame]:
        """
        Calculate the Extended Autocorrelation Function (EACF) matrix for a given time series.

        The EACF matrix is ​​used to identify the appropriate order (p, q) of the ARMA model for the time series.

        The rows of the matrix represent the AR order (p), and the columns represent the MA order (q).

        Each element in the matrix indicates whether the autocorrelation coefficient of the residual series is significant for a given combination of (p, q).

        This method also returns summary information about the EACF matrix, including possible combinations of (p, q).

        :param time_series: The input time series data.
        :param symbolize: Whether to symbolize the EACF matrix as "x" (significant) and "o" (not significant).
        :param max_ar: Maximum AR order (p) to compute in the EACF matrix.
        :param max_ma: Maximum MA order (q) to compute in the EACF matrix.

        :return: A tuple containing the EACF matrix, summary information, and possible (p, q) combinations.
        """
        # Calculate the EACF matrix
        eacf_matrix, threshold, eacf_df = eacf_rlike(
            time_series=time_series, max_ar=max_ar, max_ma=max_ma
        )

        if symbolize:
            # Symbolize the EACF matrix
            # 'x' indicates significant, 'o' indicates not significant
            eacf_matrix = np.where(np.abs(eacf_matrix) >= threshold, "x", "o")

        return eacf_matrix, threshold, eacf_df

    def plot_eacf(
        self,
        time_series: Union[pd.Series, np.ndarray],
        max_ar: int = 5,
        max_ma: int = 6,
        dpi: int = 128,
        figsize: Tuple[int, int] = (8, 6),
    ) -> plt.Figure:
        """
        Plot the Extended Autocorrelation Function (EACF) matrix for the given time series data.

        :param time_series: The input time series data.
        :param max_ar: Maximum AR order (p) to compute in the EACF matrix.
        :param max_ma: Maximum MA order (q) to compute in the EACF matrix.
        :param dpi: Dots per inch for the figure.
        :param figsize: Figure size for the plot.

        :return: The matplotlib Figure object containing the EACF plot.
        """
        eacf_matrix, threshold, _ = self.eacf(
            time_series=time_series,
            symbolize=True,
            max_ar=max_ar,
            max_ma=max_ma,
        )

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.imshow(eacf_matrix != "o", cmap="Greys", interpolation="nearest")
        ax.set_xticks(np.arange(max_ma + 1))
        ax.set_yticks(np.arange(max_ar + 1))
        ax.set_xticklabels(np.arange(max_ma + 1))
        ax.set_yticklabels(np.arange(max_ar + 1))
        ax.set_xlabel("MA Order (q)")
        ax.set_ylabel("AR Order (p)")
        ax.set_title("Extended Autocorrelation Function (EACF)")

        # Annotate the matrix with 'x' and 'o'
        for i in range(eacf_matrix.shape[0]):
            for j in range(eacf_matrix.shape[1]):
                ax.text(
                    j,
                    i,
                    eacf_matrix[i, j],
                    ha="center",
                    va="center",
                    color="red" if eacf_matrix[i, j] == "x" else "blue",
                )

        return fig

    def plot_shapiro_wilk(
        self, bins: int = 13, dpi: int = 500, figsize: Tuple[int, int] = (12, 5)
    ) -> Tuple[plt.Figure, float, float]:
        """
        Plot the Shapiro-Wilk test for normality of the residuals.
        This method generates a Q-Q plot to visually assess whether the residuals
        of the fitted ARIMA model follow a normal distribution.
        """
        return plot_shapiro_wilk(
            residuals=self.residuals, bins=bins, dpi=dpi, figsize=figsize
        )
