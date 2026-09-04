s2generator.simulator
=====================

Learnable white-noise-to-signal mappings. Each simulator **fits**
dynamics from an observed series and **transform** draws new traces by
exciting the fitted system.

.. currentmodule:: s2generator.simulator

.. autosummary::
   :nosignatures:

   ARIMASimulator
   WienerFilterSimulator
   HammersteinWienerSimulator
   KalmanFilterSimulator
   MarkovSwitchingSimulator
   GaussianMixtureSimulator
   MultivariateSimulator
   IQSimulator
   LowPassFilter
   apply_lowpass

.. autoclass:: ARIMASimulator
   :members:
   :show-inheritance:

.. autoclass:: WienerFilterSimulator
   :members:
   :show-inheritance:

.. autoclass:: HammersteinWienerSimulator
   :members:
   :show-inheritance:

.. autoclass:: KalmanFilterSimulator
   :members:
   :show-inheritance:

.. autoclass:: MarkovSwitchingSimulator
   :members:
   :show-inheritance:

.. autoclass:: GaussianMixtureSimulator
   :members:
   :show-inheritance:

.. autoclass:: MultivariateSimulator
   :members:
   :show-inheritance:

.. autoclass:: IQSimulator
   :members:
   :show-inheritance:

.. autoclass:: LowPassFilter
   :members:
   :show-inheritance:

.. autofunction:: apply_lowpass
