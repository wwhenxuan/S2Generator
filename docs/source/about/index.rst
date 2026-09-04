About
=====

Project
-------

S2Generator is a Python library for **unrestricted synthetic time series**
paired with symbolic representations of the underlying system.  A
non-stationary trace is treated as the response of a complex dynamical
system: sample an excitation :math:`X`, evaluate a symbolic map
:math:`f`, and keep :math:`Y = f(X)` together with the expression.

The public surface splits into:

* :mod:`s2generator.symbol` — expression trees and generators
* :mod:`s2generator.excitation` — stimulus families
* :mod:`s2generator.simulator` — fitted white-noise-to-signal models
* :mod:`s2generator.scm` — multivariate coupling (TiRex-2, CauKer, TabPFN-3)
* :mod:`s2generator.augmentation` — post-hoc transforms
* :mod:`s2generator.utils` — plots, bundled slices, and helpers

Install and the shortest ``run`` call are in the
:doc:`../user_guide/index`; the gallery is under
:doc:`../auto_examples/index`.

Papers
------

* Wang et al., *Synthetic Series-Symbol Data Generation for Time Series
  Foundation Models*, NeurIPS 2025,
  `arXiv:2510.08445 <https://arxiv.org/abs/2510.08445>`_.
* Alkhateeb, *DeepMIMO: A Generic Deep Learning Dataset for Millimeter
  Wave and Massive MIMO Applications*,
  `arXiv:1902.06435 <https://arxiv.org/abs/1902.06435>`_
  (CSI excerpts used by :class:`~s2generator.simulator.IQSimulator`).

Citing S2Generator
------------------

Cite the NeurIPS paper above when a result depends on the series–symbol
generator.  For the software itself, name the GitHub repository and the
**version you actually ran**:

.. code-block:: text

   @software{s2generator,
     author  = {{S2Generator developers}},
     title   = {{S2Generator}: series-symbol data generation in {Python}},
     url     = {https://github.com/wwhenxuan/S2Generator},
     version = {X.Y.Z},
   }

.. admonition:: TODO (author)
   :class: note

   Replace ``X.Y.Z`` in the ``@software`` snippet with the version you
   want advertised, or point to a JOSS / software paper if one appears.

Developers
----------

.. admonition:: TODO (author)
   :class: note

   Add the public developers list (name, role, homepage) here.  The
   narrative acknowledgements currently live on
   :doc:`../development/index`.

Related projects
----------------

* `SymTime <https://github.com/wwhenxuan/SymTime>`_ — foundation model
  pre-trained on :math:`S^2` data.
* `PySDKit <https://github.com/wwhenxuan/PySDKit>`_ — signal
  decomposition toolkit that shares this documentation theme.

.. admonition:: TODO (author)
   :class: note

   Add any further related projects or organisation links here.

License
-------

S2Generator is released under the MIT license.  See the ``LICENSE`` file
in the repository.
