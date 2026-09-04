r"""
The logging for :math:`S^2` Generator
=====================================

The core of the :math:`S^2` data generation mechanism is to randomly construct a large number of symbolic expressions (complex systems) :math:`f(\cdot)` and stimulus time series :math:`X`, and obtain the response of the complex system by inputting the stimulus into the complex system:

.. math::

   Y = f(X).

However, in this process, since the generated symbolic expression :math:`f(\cdot)` has a domain, for example, the domain of :math:`f(x) = \mathrm{ln} (x)` is :math:`x \in (0, + \infty ]`, the domain of :math:`f(x) = \frac{1}{x}` is :math:`x \in \left \{ x | x \ne 0 \right \}`. Although we usually replace :math:`f(x) = \mathrm{ln}(x)` and :math:`f(x) = \frac{1}{x}` with :math:`f(x) = \mathrm{ln}(|x|)` and :math:`f(x) = \frac{1}{x + \varepsilon}` respectively when constructing symbolic expressions to increase the range of their domains without changing their symbolic operation logic, there are still many cases where the values fall outside the domain. When this happens, we will abandon the time series data and generate a new :math:`X` for resampling.

In addition, since we use the power operation pow and exponent exp when constructing the complex system :math:`f(\cdot)`, numerical explosion may occur when performing numerical sampling. To this end, we will limit the value of the response time series to a certain range to improve the quality of the basic representation of the time series data.

For these two reasons, we provide a status monitoring module for the data generation process. This module allows you to intuitively determine whether the stimulus time series data is successfully sampled and how many times it has been successfully sampled. You can specify the ``print_status`` and ``logging_path`` parameters in ``SeriesSymbolGenerator`` to print and log status information during the execution of the data generation algorithm.
"""

import numpy as np

# Importing data generators, parameter controllers and visualization functions
from s2generator.symbol import SeriesSymbolGenerator, SeriesParams, SymbolParams
from s2generator.utils import plot_symbol_series

# Create an instance and print the status for the generation
generator = SeriesSymbolGenerator(print_status=True)


def _run_until_valid(start_seed, **kwargs):
    trees = x = y = None
    for seed in range(int(start_seed), int(start_seed) + 80):
        trial = np.random.RandomState(seed)
        trees, x, y = generator.run(trial, **kwargs)
        if x is not None and y is not None:
            return trees, x, y, seed
    raise RuntimeError("SeriesSymbolGenerator.run returned no valid sample")


# Start generating symbolic expressions, sampling and generating series
trees, x, y, seed = _run_until_valid(
    0, input_dimension=1, output_dimension=1, seq_length=256
)

# Visualize the time series
fig = plot_symbol_series(x, y)

# Add the params `logging_path` to save the stats
generator = SeriesSymbolGenerator(print_status=True, logging_path=".")

# Start generating symbolic expressions, sampling and generating series
trees, x, y, seed = _run_until_valid(
    1, input_dimension=1, output_dimension=1, seq_length=256
)

# Visualize the time series
fig = plot_symbol_series(x, y)

# We can also generate the multivariate input and output time series
trees, x, y, seed = _run_until_valid(
    seed + 1, input_dimension=4, output_dimension=4, seq_length=256
)

# Visualize the time series
fig = plot_symbol_series(x, y)
