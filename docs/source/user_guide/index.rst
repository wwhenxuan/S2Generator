User guide
==========

What S2Generator is
-------------------

Time series are treated as the **external response of a complex
dynamical system**.  S2Generator builds a symbolic expression
:math:`f(\cdot)` (the system) and an excitation series :math:`X`, then
records

.. math::

   Y = f(X).

The pair :math:`(f, X, Y)` is the series–symbol datum used to pre-train
time series foundation models.  Five excitation families
(mixed distributions, ARMA, ForecastPFN, KernelSynth, intrinsic mode
functions) sit beside learnable simulators, SCM coupling pipelines, and
augmentation transforms.

Install
-------

From PyPI:

.. code-block:: bash

   pip install s2generator

Runtime dependencies include NumPy, SciPy, Matplotlib, Pandas, and
Statsmodels.

From a clone (editable, for development):

.. code-block:: bash

   git clone https://github.com/wwhenxuan/S2Generator.git
   cd S2Generator
   pip install -e .

Generate a series–symbol pair
-----------------------------

.. code-block:: python

   import numpy as np

   from s2generator.symbol import SeriesSymbolGenerator, SeriesParams, SymbolParams
   from s2generator.utils import plot_symbol_series

   rng = np.random.RandomState(0)
   generator = SeriesSymbolGenerator(
       series_params=SeriesParams(),
       symbol_params=SymbolParams(),
   )
   symbols, inputs, outputs = generator.run(
       rng, input_dimension=1, output_dimension=1, seq_length=256
   )
   print(symbols)
   fig = plot_symbol_series(inputs, outputs)

Bind a user-specified expression with
:class:`~s2generator.symbol.CustomSymbolGenerator`:

.. code-block:: python

   from s2generator.symbol import CustomSymbolGenerator

   custom = CustomSymbolGenerator("(x_0 add sin(x_0))")
   symbols, inputs, outputs = custom.run(rng, seq_length=256)

The gallery under :doc:`../auto_examples/index` walks through excitation,
simulators (including :class:`~s2generator.simulator.IQSimulator`),
SCM coupling, and augmentation.

Citation
--------

If you use the series–symbol generator, please cite:

.. code-block:: bibtex

   @misc{wang2025syntheticseriessymboldatageneration,
         title={Synthetic Series-Symbol Data Generation for Time Series Foundation Models},
         author={Wenxuan Wang and Kai Wu and Yujian Betterest Li and Dan Wang and Xiaoyu Zhang},
         year={2025},
         eprint={2510.08445},
         archivePrefix={arXiv},
         primaryClass={cs.LG},
         url={https://arxiv.org/abs/2510.08445},
   }
