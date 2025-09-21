# User giude

Based on the important perspective that time series are external manifestations of complex dynamical systems, 
we propose a bimodal generative mechanism for time series data that integrates both symbolic and series modalities. 
This mechanism enables the unrestricted generation of a vast number of complex systems represented as symbolic expressions $f(\cdot)$ and excitation time series $X$. 
By inputting the excitation into these complex systems, we obtain the corresponding response time series $Y=f(X)$. 
This method allows for the unrestricted creation of high-quality time series data for pre-training the time series foundation models.

<div align="center">

[Installation](#Installation) | {doc}`Examples <../../auto_examples/index>` | {doc}`Acknowledge <../development/index>` | [Citation](#Citation)

</div>

### 🔥 News

**[Sep. 2025]** Our paper "Synthetic Series-Symbol Data Generation for Time Series Foundation Models" has been accepted by **NeurIPS 2025**, where **[*SymTime*](https://arxiv.org/abs/2502.15466)** pre-trained on the $S^2$ synthetic dataset achieved SOTA results in fine-tuning of forecasting, classification, imputation and anomaly detection tasks.

(Installation)=
## 🚀 Installation <a id="Installation"></a>

We have highly encapsulated the algorithm and uploaded the code to PyPI. Users can download the code through `pip`.

~~~
pip install s2generator
~~~

We only used [`NumPy`](https://numpy.org/), [`Scipy`](https://scipy.org/) and [`matplotlib`](https://matplotlib.org/) when developing the project.

## ✨ Usage

We provide a unified data generation interface [`Generator`](https://github.com/wwhenxuan/S2Generator/blob/main/S2Generator/generators.py), two parameter modules [`SeriesParams`](https://github.com/wwhenxuan/S2Generator/blob/main/S2Generator/params/series_params.py) and [`SymbolParams`](https://github.com/wwhenxuan/S2Generator/blob/main/S2Generator/params/symbol_params.py), as well as auxiliary modules for the generation of excitation time series and complex system. We first specify the parameters or use the default parameters to create parameter objects, and then pass them into our `Generator` respectively. finally, we can start data generation through the `run` method after instantiation.

~~~python
import numpy as np

# Importing data generators object
from S2Generator import Generator, SeriesParams, SymbolParams, plot_series

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

<img width="100%" align="middle" src="https://raw.githubusercontent.com/wwhenxuan/S2Generator/main/images/ID1_OD1.jpg?raw=true">

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

<img width="100%" align="middle" src="https://raw.githubusercontent.com/wwhenxuan/S2Generator/main/images/ID2_OD2.jpg?raw=true">

## 🧮 Algorithm 

The advantage of $S^2$ data lies in its diversity and unrestricted generation capacity. <img width="25%" align="right" src="https://github.com/wwhenxuan/S2Generator/blob/main/images/trees.png?raw=true">

On the one hand, we can build a complex system with diversity based on binary trees (right); 

on the other hand, we combine 5 different methods to generate excitation series, as follows:

- [`MixedDistribution`](https://github.com/wwhenxuan/S2Generator/blob/main/S2Generator/excitation/mixed_distribution.py): Sampling from a mixture of distributions can show the random of time series;
- [`ARMA`](https://github.com/wwhenxuan/S2Generator/blob/main/S2Generator/excitation/autoregressive_moving_average.py): The sliding average and autoregressive processes can show obvious temporal dependencies;
- [`ForecastPFN`](https://github.com/wwhenxuan/S2Generator/blob/main/S2Generator/excitation/forecast_pfn.py) and [`KernelSynth`](https://github.com/wwhenxuan/S2Generator/blob/main/S2Generator/excitation/kernel_synth.py): The decomposition and combination methods can reflect the dynamics of time series;
- [`IntrinsicModeFunction`](https://github.com/wwhenxuan/S2Generator/blob/main/S2Generator/excitation/intrinsic_mode_functions.py): The excitation generated by the modal combination method has obvious periodicity.

By generating diverse complex systems and combining multiple excitation generation methods, 
we can obtain high-quality, diverse time series data without any constraints. 
For detailed on the data generation process, please refer to our [paper](https://arxiv.org/abs/2502.15466).

(Citation)=
## 🎖️ Citation <a id="Citation"></a>

If you find this $S^2$ data generation method helpful, please cite the following paper:

~~~latex
@misc{wang2025mitigatingdatascarcitytime,
      title={Mitigating Data Scarcity in Time Series Analysis: A Foundation Model with Series-Symbol Data Generation}, 
      author={Wenxuan Wang and Kai Wu and Yujian Betterest Li and Dan Wang and Xiaoyu Zhang and Jing Liu},
      year={2025},
      eprint={2502.15466},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.15466}, 
}
~~~