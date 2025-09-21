#!/usr/bin/env python
# coding: utf-8
"""
The Demo of :math:`S^2` Generator for Series-Symbol Data Generation
===================================================================

Time series data serves as the external manifestation of complex dynamical systems. This method aims to generate diverse complex systems represented by symbolic expressions :math:`f(\cdot)` — through unconstrained construction. It simultaneously generates excitation time series :math:`X \in \mathbb{R} ^ {M \\times L}`, which are then fed into the complex systems to produce their responses :math:`Y=f(X) \in \mathbb{R} ^ {N \\times L}`. Here, :math:`M`, :math:`N` and :math:`L` denote the number of input channels, output channels, and series length, respectively.

**Note: Because the values of the stimulus time series can inflate or fall outside the domain of complex systems, the following examples may not work if your Python version and library version are inconsistent with ours. In this case, please adjust the random seed value in the random number generator.**
"""



# %%

import numpy as np
import sys
import os
# sphinx_gallery_thumbnail_path = 'source/_static/background.png'

sys.path.append(os.path.abspath(".."))

# Importing data generators, parameter controllers and visualization functions
from S2Generator import Generator, SeriesParams, SymbolParams, plot_series, print_hello

print_hello()

# %%

# Adjust the parameters here
# Create a parameter controls the generation of the excitation time series
series_params = SeriesParams()

# Create a parameter controls the generation of the symbolic expression (complex systems)
symbol_params = SymbolParams()

# %%
# The core of the :math:`S^2` data generation mechanism is to randomly construct a large number of symbolic expressions (complex systems) :math:`f(\cdot)` and stimulus time series :math:`X`, and obtain the response of the complex system by inputting the stimulus into the complex system:
#
# .. math::
#    Y = f(X)
#
# where, the sampling multivariate time series :math:`X = \left [ x_1, x_2, \cdots, x_m \right ]  \in \mathbb{R}^{M \times L}` and the generated multivariate time series :math:`Y = \left [ y_1, y_2, \cdots, y_m \right ]  \in \mathbb{R}^{N \times L}`. :math:`M` and :math:`N` are the input and output dimension for the time series, :math:`L` is the length of points of the time series data.
#
# After constructing the input parameters, we can complete this process end-to-end by creating a data generation object and executing the `run` method.

# Create an instance
generator = Generator(series_params=series_params, symbol_params=symbol_params)

# Creating a random number object
rng = np.random.RandomState(0)

# Start generating symbolic expressions, sampling and generating series
trees, x, y = generator.run(
    rng, input_dimension=1, output_dimension=1, n_inputs_points=256
)
# Print the expressions
print(trees)

# Visualize the time series
fig = plot_series(x, y)


# %%

# Try to generate the 2-channels and longer time series
trees, x, y = generator.run(
    rng,
    input_dimension=2,
    output_dimension=2,
    n_inputs_points=512,
    output_normalize="z-score",
)
print(trees)
fig = plot_series(x, y)


# %%

# Try to generate the 3-channels time series
trees, x, y = generator.run(
    rng,
    input_dimension=3,
    output_dimension=3,
    n_inputs_points=512,
    output_normalize="z-score",
)
print(trees)
fig = plot_series(x, y)


# %%

# Save the plotting time series
fig.savefig("../images/ID3_OD3.jpg", dpi=300, bbox_inches="tight")

# %%
