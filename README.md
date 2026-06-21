<img width="100%" align="middle" src="https://raw.githubusercontent.com/wwhenxuan/S2Generator/main/docs/source/_static/background.png?raw=true">

---

<div align="center">

[![PyPI version](https://badge.fury.io/py/s2generator.svg)](https://pypi.org/project/s2generator/)  ![License](https://img.shields.io/github/license/wwhenxuan/PySDKit) [![Python](https://img.shields.io/badge/python-3.9+-blue?logo=python)](https://www.python.org/) [![Downloads](https://pepy.tech/badge/s2generator)](https://pepy.tech/project/s2generator) [![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[Installation](#Installation) | [Examples](https://github.com/wwhenxuan/S2Generator/tree/main/examples) | [Docs]() | [Acknowledge]() | [Citation](#Citation)

</div>

Based on the important perspective that time series are external manifestations of complex dynamical systems, 
we propose a bimodal generative mechanism for time series data that integrates both symbolic and series modalities. 
This mechanism enables the unrestricted generation of a vast number of complex systems represented as symbolic expressions $f(\cdot)$ and excitation time series $X$. 
By inputting the excitation into these complex systems, we obtain the corresponding response time series $Y=f(X)$. 
This method allows for the unrestricted creation of high-quality time series data for pre-training the time series foundation models.

### 🔥 News

**[Jun. 2026]** We extend the learnable white-noise-to-signal simulator family with [KalmanFilterSimulator](https://github.com/wwhenxuan/S2Generator/blob/main/s2generator/simulator/kalman_filtering.py) (state-space AR + Kalman filtering) and [MarkovSwitchingSimulator](https://github.com/wwhenxuan/S2Generator/blob/main/s2generator/simulator/markov_switching.py) (Markov-switching autoregression for regime-dependent dynamics).

**[Feb. 2026]** Since all stationary time series can be obtained by exciting a linear time-invariant system with white noise, we propose [a learnable series generation method](https://github.com/wwhenxuan/S2Generator/blob/main/s2generator/simulator/arima.py) based on the ARIMA model. This method ensures the generated series is highly similar to the inputs in autocorrelation and power spectrum density.

**[Sep. 2025]** Our paper "Synthetic Series-Symbol Data Generation for Time Series Foundation Models" has been accepted by **NeurIPS 2025**, where **[*SymTime*](https://arxiv.org/abs/2502.15466)** pre-trained on the $S^2$ synthetic dataset achieved SOTA results in fine-tuning of forecasting, classification, imputation and anomaly detection tasks.

## 🚀 Installation <a id="Installation"></a>

We have highly encapsulated the algorithm and uploaded the code to PyPI:
~~~
pip install s2generator
~~~

We used [`NumPy`](https://numpy.org/), [`Pandas`](https://pandas.pydata.org/), and [`Scipy`](https://scipy.org/) to build the data science environment, [`Matplotlib`](https://matplotlib.org/) for data visualization, and [`Statsmodels`](https://www.statsmodels.org/stable/index.html) for time series analysis and statistical processing.

## ✨ Usage

We provide a unified data generation interface [`Generator`](https://github.com/wwhenxuan/S2Generator/blob/main/s2generator/generators.py), two parameter modules [`SeriesParams`](https://github.com/wwhenxuan/S2Generator/blob/main/s2generator/params/series_params.py) and [`SymbolParams`](https://github.com/wwhenxuan/S2Generator/blob/main/s2generator/params/symbol_params.py), as well as auxiliary modules for the generation of excitation time series and complex system. We first specify the parameters or use the default parameters to create parameter objects, and then pass them into our `Generator` respectively. finally, we can start data generation through the `run` method after instantiation.

~~~python
import numpy as np

# Importing data generators object
from s2generator import Generator, SeriesParams, SymbolParams, plot_series

# Creating a random number object
rng = np.random.RandomState(0)

# Create the parameter control modules
series_params = SeriesParams()
symbol_params = SymbolParams()  # specify specific parameters here or use the default parameters

# Create an instance
generator = Generator(series_params=series_params, symbol_params=symbol_params)

# Start generating symbolic expressions, sampling and generating series
symbols, inputs, outputs = generator.run(
    rng, input_dimension=1, output_dimension=1, n_inputs_points=256
)

# Print the expressions
print(symbols)
# Visualize the time series
fig = plot_series(inputs, outputs)
~~~

> (73.5 add (x_0 mul (((9.38 mul cos((-0.092 add (-6.12 mul x_0)))) add (87.1 mul arctan((-0.965 add (0.973 mul rand))))) sub (8.89 mul exp(((4.49 mul log((-29.3 add (-86.2 mul x_0)))) add (-2.57 mul ((51.3 add (-55.6 mul x_0)))**2)))))))

<img width="100%" align="middle" src="https://raw.githubusercontent.com/wwhenxuan/S2Generator/main/docs/source/_static/ID1_OD1.jpg?raw=true">

The input and output dimensions of the multivariate time series and the length of the sampling sequence can be adjusted in the `run` method.

~~~python
rng = np.random.RandomState(512)  # Change the random seed

# Try to generate the multi-channels time series
symbols, inputs, outputs = generator.run(rng, input_dimension=2, output_dimension=2, n_inputs_points=336)

print(symbols)
fig = plot_series(inputs, outputs)
~~~

> (-9.45 add ((((0.026 mul rand) sub (-62.7 mul cos((4.79 add (-6.69 mul x_1))))) add (-0.982 mul sqrt((4.2 add (-0.14 mul x_0))))) sub (0.683 mul x_1))) | (67.6 add ((-9.0 mul x_1) add (2.15 mul sqrt((0.867 add (-92.1 mul x_1))))))
>
> Two symbolic expressions are connected by " | ".

<img width="100%" align="middle" src="https://raw.githubusercontent.com/wwhenxuan/S2Generator/main/docs/source/_static/ID2_OD2.jpg?raw=true">

## 🧮 Algorithm <img width="25%" align="right" src="https://github.com/wwhenxuan/S2Generator/blob/main/docs/source/_static/trees.png?raw=true">

The advantage of $S^2$ data lies in its diversity and unrestricted generation capacity. 
On the one hand, we can build a complex system with diversity based on binary trees (right); 
on the other hand, we combine 5 different methods to generate excitation series, as follows:

- [`MixedDistribution`](https://github.com/wwhenxuan/S2Generator/blob/main/s2generator/excitation/mixed_distribution.py): Sampling from a mixture of distributions can show the random of time series;
- [`ARMA`](https://github.com/wwhenxuan/S2Generator/blob/main/s2generator/excitation/autoregressive_moving_average.py): The sliding average and autoregressive processes can show obvious temporal dependencies;
- [`ForecastPFN`](https://github.com/wwhenxuan/S2Generator/blob/main/s2generator/excitation/forecast_pfn.py) and [`KernelSynth`](https://github.com/wwhenxuan/S2Generator/blob/main/s2generator/excitation/kernel_synth.py): The decomposition and combination methods can reflect the dynamics of time series;
- [`IntrinsicModeFunction`](https://github.com/wwhenxuan/S2Generator/blob/main/s2generator/excitation/intrinsic_mode_functions.py): The excitation generated by the modal combination method has obvious periodicity.

By generating diverse complex systems and combining multiple excitation generation methods, 
we can obtain high-quality, diverse time series data without any constraints. 
For detailed on the data generation process, please refer to our [paper](https://arxiv.org/abs/2502.15466) or [documentation]().

## 🎖️ Citation <a id="Citation"></a>

If you find this $S^2$ data generation method helpful, please cite the following paper:

~~~latex
@misc{wang2025syntheticseriessymboldatageneration,
      title={Synthetic Series-Symbol Data Generation for Time Series Foundation Models}, 
      author={Wenxuan Wang and Kai Wu and Yujian Betterest Li and Dan Wang and Xiaoyu Zhang},
      year={2025},
      eprint={2510.08445},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.08445}, 
}
~~~