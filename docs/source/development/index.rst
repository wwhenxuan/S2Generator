Development
===========

This page is the contributor guide for S2Generator.

Project layout
--------------

Implementation lives under ``s2generator/``:

* ``s2generator/symbol`` — expression trees and generators
* ``s2generator/excitation`` — stimulus families
* ``s2generator/simulator`` — fitted simulators
* ``s2generator/scm`` — multivariate coupling
* ``s2generator/augmentation`` — transforms
* ``s2generator/utils`` — plots and bundled data

Tests live in ``tests/``.  Gallery examples live in ``examples/`` as
**Python files**, not notebooks.

Gallery examples
----------------

Sphinx-Gallery executes every ``examples/<section>/*.py``.  Follow the
PySDKit / sphinx-gallery layout:

1. A module docstring that starts with a level-1 RST title (``====``).
2. Narrative in ``# %%`` comment blocks (markdown-like RST).
3. Code cells that call ``plt.show()`` so figures are captured.

Do **not** add ``sys.path`` hacks; ``docs/source/conf.py`` already puts
the repository root on ``sys.path``.

Convert a leftover notebook with:

.. code-block:: bash

   python docs/ipynb_to_gallery.py --keep-ipynb

Participate
-----------

Using S2Generator
~~~~~~~~~~~~~~~~~

Although large-scale time series pre-training datasets exist, the field
still faces scarcity and imbalance compared with vision or language.
S2Generator treats traces as manifestations of complex systems and
generates high-quality pairs without a fixed corpus.

We would greatly appreciate it if you could incorporate data generated
by S2Generator into the pre-training of your foundation models.  You
are welcome to submit your work to be featured in this documentation.

Bug reports
~~~~~~~~~~~

If you discover a bug, open an issue on
`GitHub <https://github.com/wwhenxuan/S2Generator/issues>`_.
Pull requests against the default branch are also welcome.

Acknowledge
-----------

.. image:: https://raw.githubusercontent.com/wwhenxuan/S2Generator/master/images/correspondencce.jpg?raw=true
   :alt: correspondence
   :align: center

`whenxuan <https://wwhenxuan.github.io/>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The success of this project would not have been possible without the
support and contributions of many individuals.

First and foremost, thanks to `Kai Wu <https://web.xidian.edu.cn/kwu/>`_
from the School of Artificial Intelligence, Xidian University.  He
introduced the viewpoint of complex dynamical systems in time series,
which led to S2Generator and the foundation model
`SymTime <https://github.com/wwhenxuan/SymTime>`_.

Thanks to `Dan Wang <https://web.xidian.edu.cn/danwang/>`_ from the
School of Telecommunications Engineering, Xidian University, for
guidance during undergraduate research.

Thanks to `Rezhe Wang <https://github.com/changewam>`_ for the technical
documentation of S2Generator and `PySDKit <https://github.com/wwhenxuan/PySDKit>`_.

Thanks to Yifan Wu for differential equations and unit tests.

Thanks to Baixiang Wang for encouragement in programming and research.

Thanks to Mengyao Zhang for designing the S2Generator logo.
