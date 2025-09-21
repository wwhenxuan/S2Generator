#!/usr/bin/env python
# coding: utf-8
"""
Excitation Generation via ForecastPFN
========================================

In this example, we demonstrate how to generate incentive signals based on the method of time series combination proposed in `ForecastPFN <https://arxiv.org/abs/2311.01933>`_. The paper primarily trained a zero-shot time series prediction model using a prior feature fitting network approach on a large amount of synthetic data.

In the S2Generator, we have improved the data generation method from the original code and provided a Python-based object-oriented programming interface to make parameter management and invocation easier.

The original code link is: https://github.com/abacusai/ForecastPFN.

This method considers the time series :math:`y_t` to be composed of trends and a large number of cyclical components with different frequencies. These cyclical components include weeks, months, and years. Finally, some random noise is added to the generated sequence data. For the target generated time series :math:`y_t`, we can view it as a combination of trends and cycles:

.. math::
   y_t = \phi(t) \cdot z_t = \mathrm{trend}(t) \cdot \mathrm{seasonal}(t) \cdot z_t,

where, :math:`z_t` is the noise time series and can be represented as

.. math::
   z_t = 1 + m_{\mathrm{noise}}(z - \\bar{z}),

where, :math:`z_t` is the noise time series and can be represented as

.. math::
   z_t = 1 + m_{\mathrm{noise}}(z - \\bar{z}),

where, :math:`z \sim \mathrm{Weibull}(1, k)`, :math:`\\bar{z} = (\mathrm{ln}2) ^ {1 / k}`.

The trend component is made up of a linear and exponential component with coefficients:

.. math::
   \mathrm{trend}(t) = (1 + m_{\mathrm{linear}} \cdot t + c_{\mathrm{linear}})(m_{\mathrm{exp}} \cdot c_{\mathrm{exp} ^ t}),

where, :math:`m_{\mathrm{linear}}, m_{\mathrm{exp}} ~ \mathcal{N}(-0.01, 0.5)`, :math:`c_{\mathrm{linear}} \sim \mathcal{N}(0, 0.01)`, :math:`c_{\mathrm{exp}} \sim \mathcal N (1, 0.0005)`.

For the periodic components, we model them based on the basic idea of Fourier decomposition, representing the periodic components as a combination of different sine and cosine signals, and combining them according to weeks, months, and years. Specifically, it can be expressed as:

.. math::
   \mathrm{seasonal}(t) = \mathrm{seasonal_{week}}(t) \cdot \mathrm{seasonal_{month}}(t) \cdot \mathrm{seasonal_{year}}(t),

where, the seasonal can be represented as

.. math::
   \mathrm{seasonal}_{\\nu}(t) = 1 + m_{\\nu} \sum_{f = 1}^{\left \lfloor p_{\\nu / 2} \\right \\rfloor }\left [ c_{f, \\nu} \mathrm{sin} \left ( 2 \pi f \\frac{t}{p_{\\nu}} \\right ) + d_{f, \\nu} \mathrm{cos} \left ( 2 \pi f \\frac{t}{p_{\\nu}} \\right ) \\right ].

We set each :math:`m_{\\nu} \in \left \{ \mathrm{week}, \mathrm{month}, \mathrm{year} \\right \}`, separately for daily, weekly, and monthly, as follows. :math:`\\nu = 0` unless specified:

where, the seasonal can be represented as:

.. math::
   \mathrm{seasonal}_{\\nu}(t) = 1 + m_{\\nu} \sum_{f = 1}^{\left \lfloor p_{\\nu / 2} \\right \\rfloor }\left [ c_{f, \\nu} \mathrm{sin} \left ( 2 \pi f \\frac{t}{p_{\\nu}} \\right ) + d_{f, \\nu} \mathrm{cos} \left ( 2 \pi f \\frac{t}{p_{\\nu}} \\right ) \\right ].

We set each :math:`m_{\\nu} \in \left \{ \mathrm{week}, \mathrm{month}, \mathrm{year} \\right \}`, separately for daily, weekly, and monthly, as follows. :math:`\\nu = 0` unless specified:

 - Daily: :math:`m_{\mathrm{week}} \sim \mathcal{U}([0, 1])`, :math:`m_{\mathrm{month}} \sim \mathcal{U}([0, 2])`, :math:`p_{\mathrm{week}} = 7`, :math:`p_{\mathrm{week}} = 30.5`.
 - Weekly: :math:`m_{\mathrm{week}} \sim \mathcal{U}([0, 3])`, :math:`m_{\mathrm{month}} \sim \mathcal{U}([0, 1])`, :math:`p_{\mathrm{week}} = 2`, :math:`p_{\mathrm{week}} = 52`.
 - Monthly: :math:`m_{\mathrm{year}} \sim \sim \mathcal{U}([0, 5])`, :math:`p_{\mathrm{year}} = 12`.

"""

# %%


import numpy as np
from matplotlib import pyplot as plt
from S2Generator.excitation import ForecastPFN
# sphinx_gallery_thumbnail_path = 'source/_static/background.png'

# Create the instance for ForecastPFN
forecast_pfn = ForecastPFN(
    start_time="1885-01-01", end_time=None
)  # We use real year, month, and day information as timestamps for data generation

# Create the random number generator
rng = np.random.RandomState(0)

# Generate the excitation through `generate` method
time_series = forecast_pfn.generate(
    rng=np.random.RandomState(0), input_dimension=1, n_inputs_points=256
)

print(
    f"The Excitation Method: {str(forecast_pfn)} and Generate the Time Series Data with Shape: {time_series.shape}"
)

# %%
# We recorded the specific parameters generated during the data generation process and encapsulated them into specific local class attributes. We can further examine the scale, offset, and noise components after invoking the `generate` method.

# %%


# Visualization for the excitation
fig, ax = plt.subplots(figsize=(9, 2), dpi=120)

ax.plot(time_series, color="royalblue")

# Check the params for the basis configure
print("Amplitude scaling configuration:", forecast_pfn.scale_config)
print("Baseline offset configuration:", forecast_pfn.offset_config)
print("Noise generation configuration:", forecast_pfn.noise_config)

# %%
# We also provide an interface that can directly observe all parameter components within it.

# %%


# Get the time series configuration
time_series_config = forecast_pfn.time_series_config

print("Comprehensive configuration for time series generation:")
print(
    "Compact string representation encoding key parameters:",
    forecast_pfn.time_series_config,
)
print("scale:", time_series_config.scale)
print("offset:", time_series_config.offset)
print("noise:", time_series_config.noise_config)

# %%
# Further breakdown allows us to observe the specific magnitudes of each frequency component.

# %%


print("Annual seasonality component weight(s):", forecast_pfn.annual)
print("Weekly seasonality component weight(s)", forecast_pfn.weekly)
print("Hourly seasonality component weight(s)", forecast_pfn.hourly)
print("Minute-level seasonality component weight(s)", forecast_pfn.minutely)

# %%
# We can also generate multi-channel time series data by specifying the number of generated channel dimensions using the `generate` method.

# %%


# Generate the multi-dimension time series data
time_series = forecast_pfn.generate(
    rng=np.random.RandomState(666), input_dimension=4, n_inputs_points=512
)
print(
    f"The Excitation Method: {str(forecast_pfn)} and Generate the Time Series Data with Shape: {time_series.shape}"
)

# Visualize the multi-dimension time series data
fig, ax = plt.subplots(4, 1, figsize=(12, 6), dpi=120, sharex=True)
for i in range(4):
    ax[i].plot(time_series[:, i], color="royalblue")
