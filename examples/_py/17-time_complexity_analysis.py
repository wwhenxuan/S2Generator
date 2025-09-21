#!/usr/bin/env python
# coding: utf-8
"""
Time Complexity Analysis for The :math:`S^2` Data Generation
==================================================================

In this section, we provide a detailed analysis and proof of the time complexity of the :math:`S^2` data generation mechanism. Our theoretical analysis shows that the time complexity of data generation is proportional to the length :math:`L` of the time series. We will then verify the specific time required for data generation using multiple sets of different lengths to validate our theoretical analysis.

We define the specific symbol explanation and its  as follows. Then we use the divide-and-conquer approach to analyze the complexity of our :math:`S^2` data generation mechanism.

+-----------+----------------------------------------------------------------------+
| symbol    | explanation                                                          |
+===========+======================================================================+
| :math:`L` | The length of time series                                            |
+-----------+----------------------------------------------------------------------+
| :math:`M` | Number of input channels                                             |
+-----------+----------------------------------------------------------------------+
| :math:`N` | Number of output channels                                            |
+-----------+----------------------------------------------------------------------+
| :math:`k` | Total number of mixed distributions used                             |
+-----------+----------------------------------------------------------------------+
| :math:`p` | Autoregressive model order in ARMA model                             |
+-----------+----------------------------------------------------------------------+
| :math:`q` | Moving average model order in ARMA model                             |
+-----------+----------------------------------------------------------------------+
| :math:`P` | Probability of choosing a sampling method                            |
+-----------+----------------------------------------------------------------------+
| :math:`b` | Number of binary operators used to construct symbolic expressions    |
+-----------+----------------------------------------------------------------------+
| :math:`u` | The number of unary operators used to construct symbolic expressions |
+-----------+----------------------------------------------------------------------+

1. **Symbolic Expression Generation**: We construct symbolic expressions using a tree structure as a medium. When we have :math:`b` binary operators, we further insert :math:`(b + 1)` leaf nodes (the process from (a) to (b) in **Figure 3** in our paper). Therefore, after inserting :math:`u` unary operators (**Figure 3** (c)), the total number of nodes in the tree is :math:`n = 2b + u + 1`. Because there are many ways to construct a tree, we consider the time complexity of constructing a balanced tree. Therefore, for :math:`N` symbols constructed, the specific complexity of this process is:

.. math::
   O(N\\times n\mathrm{log}n)


2. **Sampling series generation**: When we want to generate a sampling time series with :math:`M` channels, each channel has a probability of :math`:`P` to be sampled using a mixture distribution and a probability of :math:`(1-P)` to be sampled using an ARMA model. When the sampling length of the series is :math:`L`, the complexity of generating :math:`k` mixture distribution and ARMA (:math:`p`, :math:`q`) series is :math:`O(kL)` and :math:`O(L(p+q))`. Therefore, the time complexity of this process can be quantified as:

.. math::
   O \left ( ML \\times [Pk + (1-P)(p+q)] \\right )


3. **Sampling through symbolic expressions and series**: We simplify the specific operational details of this process and only consider the time complexity of operations with variables. For a series of length L, we have :math:`N` symbolic expressions to be sampled, and each symbol has an average of :math:`\\frac{M+1}{2}` variables (Each symbolic expression may contain any number of variables from 1 to M, so here we take :math:`\\frac{M+1}{2}=\\frac{(1+2+\cdots+M)}{M}` as the average probability). Then the process can be quantified as:

.. math::
   O(N \cdot \\frac{M+1}{2}\cdot L)

To sum up, since other variables that affect the :math:`S^2` sampling process are usually small, it can be intuitively understood that the time complexity of the entire sampling process is proportional to the length :math:`L`.
"""
# %%
import time
import numpy as np
from tqdm import tqdm

from S2Generator import Generator, SymbolParams
# sphinx_gallery_thumbnail_path = 'source/_static/background.png'

# %%
# In the following experiment, we generate time series data with a length interval of 16 and calculate the specific time required.

# %%


# Create the generator instance
generator = Generator(symbol_params=SymbolParams(max_trials=16))

# The number of generated data
try_count = 1

length_array = np.arange(16, 528, 16)

time_array = np.zeros_like(length_array)

# Variables with different time lengths
for index, n_points in enumerate(length_array):
    start = time.time()
    for seed in tqdm(range(try_count)):
        # Create the random number generator
        rng = np.random.RandomState(seed)

        # Start generating data
        generator.run(
            rng=rng, input_dimension=1, output_dimension=1, n_inputs_points=n_points
        )

    # Record the time required for this length
    end = time.time()
    time_array[index] = end - start

    # Print status information
    print(f"Generate Length: {n_points}, Time: {end - start}!")

# %%
# From the above experimental results, we can generally see that the time required for data generation is generally proportional to the length of the time series.
# However, consider that in many cases, when we construct the symbolic expression :math:`f(\cdot)` and input the stimulus time series :math:`X` into the system to obtain the corresponding :math:`Y=f(X)`, the values ​​of the stimulus time series may fall outside the domain of the symbolic expression :math:`f(\cdot)`, resulting in sampling failure. This phenomenon will affect the sampling time.

# %%


from matplotlib import pyplot as plt

fig, ax = plt.subplots(figsize=(8, 3), dpi=180)

ax.plot(
    length_array,
    time_array,
    color="royalblue",
    marker="o",
    markersize=5,
    markerfacecolor="white",
)
ax.grid("--", color="gray", alpha=0.4)
ax.set_xlabel("The length of the generated time series", fontsize=12)
ax.set_ylabel("The Time Complexity Analysis", fontsize=12)

# %%
