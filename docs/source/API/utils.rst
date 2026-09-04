s2generator.utils
=================

Loaders, synthetic generators, distances, decompositions, and plots.

.. currentmodule:: s2generator.utils

Data
----

.. autosummary::
   :nosignatures:

   list_datasets
   list_deepmimo_speeds
   load_univariate
   load_multivariate
   load_deepmimo_iq
   generate
   save_s2data
   load_s2data
   save_table

.. autofunction:: list_datasets

.. autofunction:: list_deepmimo_speeds

.. autofunction:: load_univariate

.. autofunction:: load_multivariate

.. autofunction:: load_deepmimo_iq

.. autofunction:: generate

.. autofunction:: save_s2data

.. autofunction:: load_s2data

.. autofunction:: save_table

Synthetic traces
----------------

.. autosummary::
   :nosignatures:

   generate_arma_samples
   generate_nonstationary_sine
   generate_variable_frequency_sine
   generate_sine_with_local_frequency_changes
   generate_triangle_wave
   generate_square_wave
   generate_sawtooth_wave
   generate_damped_oscillation
   generate_chirp_signal
   generate_impulse_signal
   generate_step_signal
   generate_ramp_signal
   generate_exponential_signal
   generate_logarithmic_signal
   generate_stock_price
   generate_electrocardiogram
   generate_electroencephalogram

.. autofunction:: generate_arma_samples

.. autofunction:: generate_nonstationary_sine

Plots
-----

.. autosummary::
   :nosignatures:

   plot_univariate_time_series
   plot_multivariate_time_series
   plot_symbol_series
   plot_symbol
   plot_iq_series
   plot_iq_analysis
   plot_graph
   plot_adjacency_matrix
   plot_correlation
   plot_simulator_statistics
   plot_shapiro_wilk
   plot_wasserstein_heatmap

.. autofunction:: plot_symbol_series

.. autofunction:: plot_iq_series

.. autofunction:: plot_iq_analysis

.. autofunction:: plot_graph

.. autofunction:: plot_correlation

Analysis
--------

.. autosummary::
   :nosignatures:

   STL
   STLResult
   MovingDecomp
   PrintStatus
   wasserstein_distance
   wasserstein_distance_matrix
   multivariate_correlation
   yule_walker
   eacf_rlike
   z_score_normalization
   max_min_normalization

.. autoclass:: STL
   :members:
   :show-inheritance:

.. autoclass:: STLResult
   :members:
   :show-inheritance:

.. autoclass:: MovingDecomp
   :members:
   :show-inheritance:

.. autoclass:: PrintStatus
   :members:
   :show-inheritance:

.. autofunction:: wasserstein_distance

.. autofunction:: wasserstein_distance_matrix

.. autofunction:: multivariate_correlation

.. autofunction:: yule_walker
