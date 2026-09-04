s2generator.augmentation
========================

Transforms applied to generated or observed traces: trends, mixup,
resampling, spikes, amplitude / frequency modulation, and related
helpers.

.. currentmodule:: s2generator.augmentation

.. autosummary::
   :nosignatures:

   amplitude_modulation
   censor_augmentation
   empirical_mode_modulation
   frequency_perturbation
   time_series_upsampling
   time_series_downsampling
   spike_injection
   wiener_filter
   add_linear_trend
   add_piecewise_linear_trend
   add_nonlinear_trend
   value_flipping
   time_series_mixup

.. autofunction:: amplitude_modulation

.. autofunction:: censor_augmentation

.. autofunction:: empirical_mode_modulation

.. autofunction:: frequency_perturbation

.. autofunction:: time_series_upsampling

.. autofunction:: time_series_downsampling

.. autofunction:: spike_injection

.. autofunction:: wiener_filter

.. autofunction:: add_linear_trend

.. autofunction:: add_piecewise_linear_trend

.. autofunction:: add_nonlinear_trend

.. autofunction:: value_flipping

.. autofunction:: time_series_mixup
