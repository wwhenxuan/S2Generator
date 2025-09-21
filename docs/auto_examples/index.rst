:orphan:

Examples
============================

This gallery contains examples demonstrating the usage of S2Generator for signal generation and analysis.

.. centered:: `View code on GitHub <https://github.com/wwhenxuan/S2Generator/tree/main/examples>`_


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Time series data serves as the external manifestation of complex dynamical systems. This method aims to generate diverse complex systems represented by symbolic expressions f(\cdot) — through unconstrained construction. It simultaneously generates excitation time series X \in \mathbb{R} ^ {M \times L}, which are then fed into the complex systems to produce their responses Y=f(X) \in \mathbb{R} ^ {N \times L}. Here, M, N and L denote the number of input channels, output channels, and series length, respectively.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_1-basic_demo_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_1-basic_demo.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">The Demo of S^2 Generator for Series-Symbol Data Generation</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The core of the S^2 data generation mechanism is to randomly construct a large number of symbolic expressions (complex systems) f(\cdot) and stimulus time series X, and obtain the response of the complex system by inputting the stimulus into the complex system:">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_12-logging_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_12-logging.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">The logging for S^2 Generator</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="To better observe the cyclical and trend characteristics of generated time series data, we have built-in time series decomposition algorithms. This example focuses on a seasonal and trend decomposition algorithm: Seasonal-Trend decomposition using LOESS. This method allows us to better observe the important components of the time series while filtering out noise.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_14-stl_decomposition_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_14-stl_decomposition.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">The Seasonal-Trend decomposition using LOESS</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this section, we provide a detailed analysis and proof of the time complexity of the S^2 data generation mechanism. Our theoretical analysis shows that the time complexity of data generation is proportional to the length L of the time series. We will then verify the specific time required for data generation using multiple sets of different lengths to validate our theoretical analysis.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_17-time_complexity_analysis_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_17-time_complexity_analysis.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Time Complexity Analysis for The S^2 Data Generation</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This module will detail the generation and manipulation of our excitation time series data. The diversity of our S^2 data generation mechanisms is primarily due to two factors: (1) the diversity of generated symbolic expressions (complex systems) f(\cdot); and (2) the diversity of generated excitation time series data X.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_3-excitation_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_3-excitation.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">The Generation for Excitation Time Series</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this example we will go into detail and show how to generate motivating time series data from a mixture distribution.   This method came from the SNIP in the domain of Symbolic Regression. It has subsequently further extended in SymTime to make it applicable to time series data.   In this method, we will first randomly generate the number of mixed distributions k \in [k_{min}, k_{max}]. For each mixed distribution we then determine its specific type, and choose whether to use the normal distribution \mathcal{N}(\mu, \sigma^2) or the uniform distribution \mathcal{U}(\mathrm{a}, \mathrm{b}) according to the specified probability p_{\mathrm{select}}. Normally, the default value is p_{\mathrm{select}} = 0.5.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_4-mixed_distribution_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_4-mixed_distribution.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Excitation Generation via Mixed Distribution</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this example we will go into detail and show how to generate motivating time series data from a Autoregressive Moving Average (ARMA) model with random params.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_5-arma_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_5-arma.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Excitation Generation via Autoregressive Moving Average (ARMA)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this example, we demonstrate how to generate incentive signals based on the method of time series combination proposed in ForecastPFN. The paper primarily trained a zero-shot time series prediction model using a prior feature fitting network approach on a large amount of synthetic data.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_6-forecast_pfn_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_6-forecast_pfn.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Excitation Generation via ForecastPFN</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="For a time series zero-shot forecasting model, high-quality training data is essential. To further supplement the training dataset, Chronos proposed KernelSynth, a method to generate synthetic time series using Gaussian processes (GPs). The model struction of Chronos are show as fellow:">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_7-kernel_synth_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_7-kernel_synth.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Excitation Generation via KernelSynth</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This method is primarily inspired by adaptive decomposition algorithms in signal processing. For the first few methods, the excitation time series sampled from a mixed distribution exhibit strong randomness. The excitation time series sampled from an ARMA(p, q) model primarily reflect the continuity and temporal dependencies of time series data (mainly in terms of trend characteristics). In contrast, time series generated by methods such as ForecastPFN and KernelSynth incorporate both trend and periodic characteristics of time series. Therefore, to specifically capture the periodicity of time series data, we generate excitation time series data based on the perspective of intrinsic mode functions, leveraging the PySDKit project.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_8-intrinsic_mode_function_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_8-intrinsic_mode_function.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Excitation Generation via Intrinsic Mode Function</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="在 exmaple 1 中我们已经展示了如何通过 Generator 对象生成时间序列和符号表达式数据，并通过 SeriesParams 和 SymbolParams 对象来传入特定的参数来进一步调控其生成过程。">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_9-save_and_load_s2data_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_9-save_and_load_s2data.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">How to Save and Load S^2 data</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/1-basic_demo
   /auto_examples/12-logging
   /auto_examples/14-stl_decomposition
   /auto_examples/17-time_complexity_analysis
   /auto_examples/3-excitation
   /auto_examples/4-mixed_distribution
   /auto_examples/5-arma
   /auto_examples/6-forecast_pfn
   /auto_examples/7-kernel_synth
   /auto_examples/8-intrinsic_mode_function
   /auto_examples/9-save_and_load_s2data


.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-gallery

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download all examples in Python source code: auto_examples_python.zip </auto_examples/auto_examples_python.zip>`

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download all examples in Jupyter notebooks: auto_examples_jupyter.zip </auto_examples/auto_examples_jupyter.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
