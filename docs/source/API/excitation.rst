s2generator.excitation
======================

Stimulus series that drive a symbolic system :math:`f(\cdot)`.
Five generators share a common :class:`~s2generator.excitation.Excitation`
facade; :class:`~s2generator.symbol.SeriesParams` holds their knobs.

.. currentmodule:: s2generator.excitation

.. autosummary::
   :nosignatures:

   Excitation
   MixedDistribution
   AutoregressiveMovingAverage
   ForecastPFN
   KernelSynth
   IntrinsicModeFunction

.. autoclass:: Excitation
   :members:
   :show-inheritance:

.. autoclass:: MixedDistribution
   :members:
   :show-inheritance:

.. autoclass:: AutoregressiveMovingAverage
   :members:
   :show-inheritance:

.. autoclass:: ForecastPFN
   :members:
   :show-inheritance:

.. autoclass:: KernelSynth
   :members:
   :show-inheritance:

.. autoclass:: IntrinsicModeFunction
   :members:
   :show-inheritance:
