r"""
Time series generation through fitting and simulation of the ARIMA model.
=========================================================================

In signal processing, any stationary signal can be regarded as the output of a linear time-invariant (LTI) system excited by Gaussian white noise of :math:`\mathcal N (0, 1)`. Differencing operations can eliminate the prominent trend components in non-stationary signals and transform them into stationary signals.

Based on the above two viewpoints, we regard the ARIMA model as **a linear system with difference operations** that can be trained, and propose a time series generation method. Compared with the time series generation approach in S2Generator, which relies on the theory of complex dynamical systems, the main advantage of the proposed method is that the ARIMA model can be trained and fitted using the input time series data, thereby generating time series that are highly similar to the input data in statistical characteristics (autocorrelation and power spectral density). It no longer generates time series by randomly parameterizing a certain dynamical system.

This notebook will introduce how to generate time series data through ARIMA model fitting and simulation in the following order:

#. Elaboration of the basic mathematical principles.
#. Testing the fitting performance on sinusoidal signals with trends and noise.
#. Introduction to the order selection methods for ARIMA models.
#. Introduction to model fitting and diagnostic methods.
"""

# %%
# ARIMA (Autoregressive Integrated Moving Average) is a classic linear time series forecasting model, denoted as :math:`\text{ARIMA}(p,d,q)`. It is used to deal with non-stationary time series: the series is first transformed into a stationary one via **d-th order differencing**, and then an :math:`\text{AR}(p)` (Autoregressive) + :math:`\text{MA}(q)` (Moving Average) combined model is fitted to the stationary series.
#
# Let the original input sequence be :math:`x_t`. After differencing, the stationary sequence is obtained as :math:`y_t = \nabla^d x_t`. The expression of its ARMA model is:
#
# .. math::
#
#    {{y}_{t}}={{\phi }_{1}}{{y}_{t-1}}+{{\phi }_{2}}{{y}_{t-2}}+\cdots +{{\phi }_{p}}{{y}_{t-p}}+{{e}_{t}}-{{\theta }_{1}}{{e}_{t-1}}-{{\theta }_{2}}{{e}_{t-2}}-\cdots -{{\theta }_{q}}{{e}_{t-q}},
#
# where :math:`p` and :math:`q` represent the orders of the AR and MA models, respectively, :math:`\phi_p` and :math:`\theta_q` are the parameters of the AR and MA processes \cite{ARMA}, and :math:`e_t \sim \mathcal N(0,1)` denotes the observed white noise sequence.
#
# We first perform stationarity tests on the input sequence and apply successive differencing operations to render it stationary. Subsequently, we fit the differenced stationary sequence using the :math:`\text{ARMA}(p, q)` model as a linear time‑invariant system. Finally, we use the fitted ARIMA model to generate new time series data.
#
# Specifically, we regard the :math:`\text{ARMA}(p, q)` model as a combination of the autoregressive model :math:`\text{AR}(p)` and the moving average process :math:`\text{MA}(q)`. This process can be written separately as:
#
# .. math::
#
#    \mathrm{AR}(p): {{y}_{t}}={{\phi }_{1}}{{y}_{t-1}}+{{\phi }_{2}}{{y}_{t-2}}+\cdots +{{\phi }_{p}}{{y}_{t-p}} = \sum_{i=1}^p \phi_i y_{t-i},
#
# .. math::
#
#    \mathrm{MA}(q): {{y}_{t}}={{e}_{t}}-{{\theta }_{1}}{{e}_{t-1}}-{{\theta }_{2}}{{e}_{t-2}}-\cdots -{{\theta }_{q}}{{e}_{t-q}} = e_t - \sum_{j=1}^q \theta_j e_{t-j}.

# %%
# To more intuitively understand the input-output relationship of the ARMA model as a linear time-invariant system, we will write it in the more common basic form used in signal processing in this notebook:
#
# .. math::
#
#    x(n) + \sum_{i=1}^p a_i x(n-i) = e(n) + \sum_{j=1}^q b_j e(n-j), \quad (*)
#
# This expression is equivalent to the formulation of the :math:`\text{ARMA}(p, q)` described above. We further define its coefficient matrix as follows:
#
# .. math::
#
#    A(z) = [1, a_1 z^{-1}, a_2 z^{-2}, \cdots, a_p z^{-p}], \quad B(z) = [1, b_1 z^{-1}, b_2 z^{-2}, \cdots, b_q z^{-q}].
#
# where :math:`z^{-i}` denotes a delay of :math:`i` steps. Its matrix form can be written as:
#
# .. math::
#
#    A(z) x(n) = B(z) e(n), \quad (*)
#
# From this, we can construct a simple model filter as shown in the figure below, whose transfer function can be defined as
#
# .. math::
#
#    H(z) = \frac{B(z)}{A(z)} = \frac{1 + b_1 z^{-1} + b_2 z^{-2} + \cdots + b_q z^{-q}}{1 + a_1 z^{-1} + a_2 z^{-2} + \cdots + a_p z^{-p}}.
#
# <p align="center">
# <img width="50%" align="middle" src="https://raw.githubusercontent.com/wwhenxuan/S2Generator/master/docs/source/_static/arma_linear_system.jpg?raw=true">
# </p>
#
# This model can be regarded as a simple learnable adaptive filter, which transforms the input white noise sequence into time series data highly similar to the fitted data in terms of statistical properties.
# To be precise, according to the principle of power spectrum estimation, this filter converts the input white noise signal into time series data with the same power spectral density (PSD) as the fitted data.
#
# Assuming that the PSD of the white noise :math:`e(n)` is :math:`\sigma^2`, the PSD of the output signal :math:`x(n)` can be expressed according to the frequency response characteristics of a linear time-invariant system as:
#
# .. math::
#
#    S_x(f) = \sigma^2 |H(e^{j2\pi f})|^2 = \sigma^2 \frac{|B(e^{j2\pi f})|^2}{|A(e^{j2\pi f})|^2},
#
# here, :math:`S_x(f)` represents the PSD of the output signal, which will match the PSD of the input time series data for fitting the ARMA model. In other words, this method guarantees statistical consistency between the generated time series and our training data in terms of autocorrelation and power spectral density.

# %%
# Below, we will further demonstrate how to generate time series data using an ARIMA model.
#
# We will first import the necessary libraries and generate non-stationary sinusoidal time series data as an example.

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from s2generator.simulator import ARIMASimulator
from s2generator.utils import generate_arma_samples, generate_nonstationary_sine

# Generating non-stationary samples
num_samples = 5
seq_length = 512
target_freq = 4.0
sample_rate = 100
nonstationary_samples = np.stack(
    [
        generate_nonstationary_sine(
            seq_length=seq_length,
            freq=target_freq,
            sample_rate=sample_rate,
        )
        for _ in range(num_samples)
    ],
    axis=0,
)
print(f"Non-stationary sample shape: {nonstationary_samples.shape}")

# Visualize the first non-stationary sample
plt.figure(figsize=(10, 2), dpi=100)
plt.plot(nonstationary_samples[0], color="royalblue")

# %%
# We will then instantiate the ARIMA simulator and fit the generated non-stationary sinusoidal signal.
#
# Since we need to perform a hyperparameter search (order determination) on the ARIMA model based on specific input data, the following code will take some time to execute.

# Instantiate the ARIMA simulator
simulator = ARIMASimulator(max_p=5, max_d=2, max_q=5)

# Fit the model; FIXME: consider adding a progress bar during fitting
simulator.fit(time_series=nonstationary_samples, select_order=True)

print(simulator.model_summary())

# The optimal model order determined by BIC
p, d, q = simulator.p_order, simulator.d_order, simulator.q_order
print(f"Selected ARIMA Order: p={p}, d={d}, q={q}")

# Generate ARIMA samples
arma_samples = simulator.transform(
    num_samples=num_samples, seq_length=seq_length, random_state=4
)

print("ARIMA samples shape:", arma_samples.shape)

fig, ax = plt.subplots(5, 2, figsize=(10, 6), dpi=300, sharex=True)

for i in range(num_samples):
    ax[i, 0].plot(nonstationary_samples[i], color="royalblue")
    ax[i, 1].plot(arma_samples[i], color="orange")

plt.tight_layout()
ax[0, 0].set_title("Non-stationary Sine Samples")
ax[0, 1].set_title("Generated ARIMA Samples")

# %%
# The application of the :math:`\text{ARIMA}(p, d, q)` model hinges on model selection and fitting diagnosis.
#
# We provide a variety of methods to help users select the appropriate model order, including AIC, BIC, ACF, PACF, and EACF.
# For the AR model, the PACF plot can be used to select the order :math:`p`; for the MA model, the ACF plot can be used to determine the order :math:`q`; for the ARMA model, the ACF and PACF plots can be combined to choose suitable :math:`p` and :math:`q`.However, since the EACF method requires subjective manual judgment, AIC and BIC are generally adopted for model order selection in automatic order determination.
#
# As the ARIMA model includes the differencing order :math:`d`, the time series must first undergo differencing operations until it becomes stationary before model selection.Only then can the methods above be used to select :math:`p` and :math:`q`.

# Perform stationarity test on the input signal
time_series = pd.Series(nonstationary_samples[0])
simulator.adf_test(time_series)

# %%
# The ADF stationarity test results show that the time series is non-stationary, so we need to differulate it to make it stationary.

# The `diff_stationary` method is called to perform difference stationarization on the time series.
stationary_series, diff_count = simulator.diff_stationary(time_series=time_series)
print(f"Number of differences applied to achieve stationarity: {diff_count}")

# Perform a stationarity test on the stationarized signal.
print(simulator.adf_test(stationary_series))

# Visualize the original and differenced stationary time series
fig, ax = plt.subplots(2, 1, figsize=(10, 5), dpi=100, sharex=True)
ax[0].plot(time_series, color="royalblue")
ax[0].set_title("Original Non-stationary Time Series")
ax[1].plot(stationary_series, color="orange")
ax[1].set_title("Differenced Stationary Time Series")

# %%
# After obtaining a stationary signal, our next step is to determine the orders p and q of the ARMA model. For manual, subjective selection, we can use the EACF method. For automatic order selection, we typically use the AIC and BIC methods.
#
# The main advantage of EACF is that it does not require model fitting; it determines the order solely based on the residuals of the AR process. For specific usage instructions, please see: https://rdrr.io/cran/TSA/src/R/eacf.R

# Show the EACF matrix for determining ARMA orders
eacf_matrix, threshold, eacf_df = simulator.eacf(
    time_series=stationary_series, symbolize=True, max_ar=6, max_ma=6
)
print("EACF Matrix:")
print(eacf_matrix)
print("\nEACF DataFrame:")
print(eacf_df)

# %%
# Generally, for automated algorithms, we select the order of the ARMA model using the AIC and BIC methods. Specifically, we can call the following function. This method is also encapsulated in the ``fit`` interface.

# Automatically select the order of the ARMA model using AIC and BIC methods.
aic_order = simulator.select_arma_order(stationary_series=stationary_series)

# %%
# Once the model order is determined, the model can be fitted, and the model's residuals can be diagnosed.
#
# Ideally, we want the model's residuals to be a white noise sequence; therefore, we can judge the model's fit by performing a white noise test on the residuals.
#
# The ARMASimulator class provides two residual diagnosis methods: Ljung-Box and Shapiro-Wilk.
#
# We first generate an ideal white noise sequence and demonstrate how to diagnose using the above methods.

from statsmodels.stats.diagnostic import acorr_ljungbox
from s2generator.utils import plot_shapiro_wilk

# Generate an ideal white noise sequence
white_noise = np.random.normal(loc=0.0, scale=1.0, size=512)

plt.figure(figsize=(10, 2), dpi=100)
plt.plot(white_noise, color="royalblue")
plt.title("Generated White Noise Series")

# Use the Ljung-Box method to perform white noise testing on the model residuals.
lb_p = acorr_ljungbox(white_noise, return_df=True)
print("Ljung-Box Test p-values for residuals:")
print(np.mean(lb_p["lb_pvalue"]))
print("\nLjung-Box Test Results:")
print(np.all(lb_p["lb_pvalue"] > 0.05))

# Perform Shapiro-Wilk normality test on ideal white noise
# Here we need to observe the test statistic and p-value of the Q-Q plot on the right.
plot_shapiro_wilk(white_noise)

# Use the Ljung-Box method to perform white noise testing on the model residuals.
lb_p_values, results = simulator.residual_diagnosis()

print("Ljung-Box Test p-values for residuals:")
print(lb_p_values)
print("\nLjung-Box Test Results:")
print(results)

# Perform Shapiro-Wilk normality test on the model residuals
fig = simulator.plot_shapiro_wilk()
