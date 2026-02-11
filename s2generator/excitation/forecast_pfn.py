import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import date, datetime
from pandas import DatetimeIndex
from pandas.tseries.frequencies import to_offset
from scipy.stats import beta

from dataclasses import dataclass
from typing import Optional, Union, Dict, Tuple, List, Any

from s2generator.excitation.base_excitation import BaseExcitation


@dataclass
class ComponentScale:
    """
    Represents scaling factors for time series components.

    Each attribute controls the amplitude scaling of a specific temporal component:
    - base: Constant baseline scaling
    - linear: Linear trend scaling (slope)
    - exp: Exponential growth/decay factor
    - a: Annual seasonality scaling
    - m: Monthly seasonality scaling
    - w: Weekly seasonality scaling
    - h: Hourly seasonality scaling
    - minute: Minute-level seasonality scaling

    Scaling formula:
        component_value = base + linear*t + exp^t + a*annual + m*monthly + ...
    """

    base: float  #: Constant baseline offset
    linear: float = None  #: Linear trend coefficient (slope)
    exp: float = None  #: Exponential growth/decay base (e.g., 1.01 for 1% growth)
    a: np.ndarray = None  #: Annual seasonality scaling vector
    m: np.ndarray = None  #: Monthly seasonality scaling vector
    w: np.ndarray = None  #: Weekly seasonality scaling vector
    h: np.ndarray = None  #: Hourly seasonality scaling vector
    minute: np.ndarray = None  #: Minute-level seasonality scaling vector


@dataclass
class ComponentNoise:
    """
    Configures noise properties for time series generation.

    Models multiplicative Weibull-distributed noise with scaling:
        noise_term = (1 + scale * (weibull_noise - E[noise]))

    Weibull distribution parameters:
        k: Shape parameter (k > 0)
        median: Median value of the distribution

    Special case:
        scale = 0 → No noise (deterministic series)
    """

    k: float  #: Weibull shape parameter (k > 0)
    median: float  #: Median value of Weibull distribution
    scale: float  #: Noise scaling factor (0 = no noise)


@dataclass
class SeriesConfig:
    """
    Comprehensive configuration for time series generation.

    Combines:
    1. scale: Amplitude scaling components (ComponentScale)
    2. offset: Baseline offset components (ComponentScale)
    3. noise_config: Stochastic noise parameters (ComponentNoise)

    The string representation encodes key parameters in compact format:
        "L{linear}E{exp}A{annual}M{monthly}W{weekly}"
    - Linear (L): 1000 × linear
    - Exponential (E): 10000 × (exp - 1)
    - Annual (A): 100 × annual_scale
    - Monthly (M): 100 × monthly_scale
    - Weekly (W): 100 × weekly_scale
    """

    scale: ComponentScale  #: Amplitude scaling configuration
    offset: ComponentScale  #: Baseline offset configuration
    noise_config: ComponentNoise  #: Noise generation configuration

    def __str__(self):
        """Compact string representation encoding key parameters"""
        return (
            f"L{1000 * self.scale.linear:+02.0f}"
            f"E{10000 * (self.scale.exp - 1):+02.0f}"
            f"A{100 * self.scale.a:02.0f}"
            f"M{100 * self.scale.m:02.0f}"
            f"W{100 * self.scale.w:02.0f}"
        )


def weibull_noise(
    rng: np.random.RandomState,
    k: Optional[float] = 2,
    length: Optional[int] = 1,
    median: Optional[float] = 1,
) -> np.ndarray:
    """
    Function to generate weibull noise with a fixed median.
    Its main feature is that it achieves a fixed median output by adjusting the scale parameter.
    The probability density function of the Weibull distribution is:
    f(x; λ, k) = (k/λ)(x/λ)^{k-1}e^{-(x/λ)^k} for x ≥ 0
    To achieve a fixed median, the function inversely solves for the scale parameter λ using the median formula:
    lamda = median / (np.log(2) ** (1 / k))

    :param rng: Random number generator used to generate random values.
    :param k: Shape parameter, determines the shape of the distribution:
              1. k < 1: Decreasing failure rate;
              2. k = 1: Exponential distribution;
              3. k > 1: Increasing failure rate.
    :param length: Shape parameter, determines the shape of the distribution:
    :param median: Mandatory median (50% quantile).
    :return:
    """
    # we set lambda so that median is a given value
    lamda = median / (np.log(2) ** (1 / k))

    return lamda * rng.weibull(k, length)


def shift_axis(
    days: pd.DatetimeIndex, shift: Optional[pd.DatetimeIndex] = None
) -> pd.DatetimeIndex:
    """
    Used to adjust the relative position of a time series (or other numerical series),
    specifically to shift the series proportionally to a new reference point.

    :param days: pd.DatetimeIndex containing the time series to shift.
    :param shift: Shift parameter, determines the shape of the distribution:
    :return: pd.DatetimeIndex containing the shifted time series.
    """
    if shift is None:
        return days
    return days - shift * days[-1]


def get_random_walk_series(
    rng: np.random.RandomState, length: int, movements: Optional[List[int]] = None
):
    """
    Function to generate a random walk series with a specified length.
    This is a random process model widely used in finance, physics, statistics and other fields.

    :param rng: Random number generator used to generate random values.
    :param length: Shape parameter, determines the shape of the distribution:
    :param movements: Shape parameter, possible step sizes:
                      1. Default: Binary Random Walk (±1);
                      2. Customizable (e.g., [-2, 0, 2]).
    :return: pd.DatetimeIndex containing the random walk series.
    """
    if movements is None:
        movements = [-1, 1]

    random_walk = list()
    random_walk.append(rng.choice(movements))
    for i in range(1, length):
        movement = rng.choice(movements)
        value = random_walk[i - 1] + movement
        random_walk.append(value)

    return np.array(random_walk)


def sample_scale(rng: np.random.RandomState = None) -> Union[np.ndarray, float]:
    """
    Function to sample scale such that it follows 60-30-10 distribution
    i.e. 60% of the times it is very low, 30% of the times it is moderate and
    the rest 10% of the times it is high.

    :param rng: The random number generator in NumPy with fixed seed.
    :return: The sampled scale for noise generation.
    """
    if rng is None:
        # When no random number generator is specified
        rand = np.random.rand()

        # very low noise
        if rand <= 0.6:
            return np.random.uniform(0, 0.1)
        # moderate noise
        elif rand <= 0.9:
            return np.random.uniform(0.2, 0.4)
        # high noise
        else:
            return np.random.uniform(0.6, 0.8)

    else:
        # When a random number generator is specified
        rand = rng.rand()

        # very low noise
        if rand <= 0.6:
            return rng.uniform(0, 0.1)
        # moderate noise
        elif rand <= 0.9:
            return rng.uniform(0.2, 0.4)
        # high noise
        else:
            return rng.uniform(0.6, 0.8)


def get_transition_coefficients(context_length: int) -> np.ndarray:
    """
    Transition series refers to the linear combination of 2 series
    S1 and S2 such that the series S represents S1 for a period and S2
    for the remaining period. We model S as S = (1 - f) * S1 + f * S2
    Here f = 1 / (1 + e^{-k (x-m)}) where m = (a + b) / 2 and k is chosen
    such that f(a) = 0.1 (and hence f(b) = 0.9). a and b refer to
    0.2 * CONTEXT_LENGTH and 0.8 * CONTEXT_LENGTH

    :param: context_length: The length of time series to be generated.
    :return: np.ndarray of transition coefficients.
    """
    # a and b are chosen with 0.2 and 0.8 parameters
    a, b = 0.2 * context_length, 0.8 * context_length

    # fixed to this value
    f_a = 0.1

    m = (a + b) / 2
    k = 1 / (a - m) * np.log(f_a / (1 - f_a))
    coeff = 1 / (1 + np.exp(-k * (np.arange(1, context_length + 1) - m)))

    return coeff


def make_series_trend(
    series: SeriesConfig, dates: pd.DatetimeIndex
) -> np.ndarray[Any, np.dtype[Any]]:
    """
    Function to generate the trend(t) component of synthetic series.

    :param series: series config for generating trend of synthetic series
    :param dates: dates for which data is present
    :return: trend component of synthetic series
    """
    values = np.full_like(dates, series.scale.base, dtype=np.float32)

    days = (dates - dates[0]).days
    if series.scale.linear is not None:
        values += shift_axis(days, series.offset.linear) * series.scale.linear
    if series.scale.exp is not None:
        values *= np.power(series.scale.exp, shift_axis(days, series.offset.exp))

    return values


def get_freq_component(
    rng: np.random.RandomState,
    dates_feature: pd.Index,
    n_harmonics: Union[int, float],
    n_total: Union[int, float],
) -> Union[np.ndarray, Any]:
    """
    Method to get systematic movement of values across time
    :param dates_feature: the component of date to be used for generating
    the seasonal movement is different. For example, for monthly patterns
    in a year we will use months of a date, while for day-wise patterns in
    a month, we will use days as the feature

    :param rng: The random number generator in NumPy with fixed seed.
    :param n_harmonics: number of harmonics to include.
                        For example, for monthly trend, we use 12/2 = 6 harmonics
    :param n_total: total cycle length
    :return: numpy array of shape dates_feature.shape containing

    sinusoidal value for a given point in time
    """
    harmonics = list(range(1, n_harmonics + 1))

    # initialize sin and cosine coefficients with 0
    sin_coef = np.zeros(n_harmonics)
    cos_coef = np.zeros(n_harmonics)

    # choose coefficients inversely proportional to the harmonic
    for idx, harmonic in enumerate(harmonics):
        sin_coef[idx] = rng.normal(scale=1 / harmonic)
        cos_coef[idx] = rng.normal(scale=1 / harmonic)

    # normalize the coefficients such that their sum of squares is 1
    coef_sq_sum = np.sqrt(np.sum(np.square(sin_coef)) + np.sum(np.square(cos_coef)))
    sin_coef /= coef_sq_sum
    cos_coef /= coef_sq_sum

    # construct the result for systematic movement which
    # comprises of patterns of varying frequency
    return_val = 0
    for idx, harmonic in enumerate(harmonics):
        return_val += sin_coef[idx] * np.sin(
            2 * np.pi * harmonic * dates_feature / n_total
        )
        return_val += cos_coef[idx] * np.cos(
            2 * np.pi * harmonic * dates_feature / n_total
        )

    return return_val


def make_series_seasonal(
    rng: np.random.RandomState, series: SeriesConfig, dates: pd.DatetimeIndex
) -> Any:
    """
    Function to generate the seasonal(t) component of synthetic series.
    It represents the systematic pattern-based movement over time

    :param rng: The random number generator in NumPy with fixed seed.
    :param series: series config used for generating values
    :param dates: dates on which the data needs to be calculated
    """
    seasonal = 1

    seasonal_components = defaultdict(lambda: 1)
    if series.scale.minute is not None:
        seasonal_components["minute"] = 1 + series.scale.minute * get_freq_component(
            rng=rng, dates_feature=dates.minute, n_harmonics=10, n_total=60
        )
        seasonal *= seasonal_components["minute"]
    if series.scale.h is not None:
        seasonal_components["h"] = 1 + series.scale.h * get_freq_component(
            rng=rng, dates_feature=dates.hour, n_harmonics=10, n_total=24
        )
        seasonal *= seasonal_components["h"]
    if series.scale.a is not None:
        seasonal_components["a"] = 1 + series.scale.a * get_freq_component(
            rng=rng, dates_feature=dates.month, n_harmonics=6, n_total=12
        )
        seasonal *= seasonal_components["a"]
    if series.scale.m is not None:
        seasonal_components["m"] = 1 + series.scale.m * get_freq_component(
            rng=rng, dates_feature=dates.day, n_harmonics=10, n_total=30.5
        )
        seasonal *= seasonal_components["m"]
    if series.scale.w is not None:
        seasonal_components["w"] = 1 + series.scale.w * get_freq_component(
            rng=rng, dates_feature=dates.dayofweek, n_harmonics=4, n_total=7
        )
        seasonal *= seasonal_components["w"]

    seasonal_components["seasonal"] = seasonal
    return seasonal_components


def make_series(
    rng: np.random.RandomState,
    series: SeriesConfig,
    freq: pd.DateOffset,
    periods: int,
    start: pd.Timestamp,
    options: dict,
    random_walk: bool,
) -> Dict[str, Union[pd.Series, np.ndarray, pd.DataFrame, DatetimeIndex]]:
    """
    make series of the following form
    series(t) = trend(t) * seasonal(t)
    """
    start = freq.rollback(start)
    dates = pd.date_range(start=start, periods=periods, freq=freq)
    scaled_noise_term = 0
    values_seasonal = {}

    if random_walk:
        values = get_random_walk_series(rng=rng, length=len(dates))
    else:
        values_trend = make_series_trend(series=series, dates=dates)
        values_seasonal = make_series_seasonal(rng=rng, series=series, dates=dates)

        values = values_trend * values_seasonal["seasonal"]

        weibull_noise_term = weibull_noise(
            rng=rng,
            k=series.noise_config.k,
            median=series.noise_config.median,
            length=len(values),
        )

        # approximating estimated value from median
        noise_expected_val = series.noise_config.median

        # expected value of this term is 0
        # for no noise, scale is set to 0
        scaled_noise_term = series.noise_config.scale * (
            weibull_noise_term - noise_expected_val
        )

    dataframe_data = {
        **values_seasonal,
        "values": values,
        "noise": 1 + scaled_noise_term,
        "dates": dates,
    }

    return dataframe_data


class ForecastPFN(BaseExcitation):
    """
    Generates excitation time series by simulating combinations of trends, seasonality, and noise.

    This implementation is inspired by ForecastPFN: Synthetically-Trained Zero-Shot Forecasting
    (https://arxiv.org/abs/2311.01933) with significant enhancements:
    1. Unified data generation interface
    2. Extended hyperparameter configuration
    3. Flexible temporal component weighting
    4. Improved time range handling
    5. Customizable frequency components

    Key innovations over the original implementation:
    - Support for sub-daily frequencies
    - Configurable transition behaviors
    - Component-specific scaling/offset/noise
    - Random walk option for non-stationary series
    """

    def __init__(
        self,
        is_sub_day: Optional[bool] = True,
        transition: Optional[bool] = True,
        start_time: Optional[str] = "1885-01-01",
        end_time: Optional[str] = None,
        random_walk: bool = False,
        dtype: np.dtype = np.float64,
    ) -> None:
        """
        Initializes the time series generator with configuration parameters.

        :param is_sub_day: Enable sub-daily frequency components (minutes/hours)
        :type is_sub_day: Optional[bool]
        :param transition: Enable smooth transitions between frequency components
        :type transition: Optional[bool]
        :param start_time: Start timestamp for generated series (ISO format: YYYY-MM-DD)
        :type start_time: Optional[str]
        :param end_time: End timestamp for generated series. Uses current date if None.
        :type end_time: Optional[str]
        :param random_walk: Enable random walk transformation for non-stationary series
        :type random_walk: bool
        :param dtype: Numerical precision for output series
        :type dtype: np.dtype
        """
        super().__init__(dtype=dtype)
        self.is_sub_day = is_sub_day
        self.transition = transition

        # Configuration for temporal components
        self.frequencies = None  # Frequency tuples (name, days)
        self.frequency_names = None  # Human-readable frequency names
        self.freq_and_index = None  # Mapping between names and indices

        # Initialize frequency and transition settings
        self.set_freq_variables(is_sub_day=self.is_sub_day)
        self.set_transition(transition=self.transition)

        # Process time range parameters
        self.user_start_time = start_time
        self.user_end_time = (
            end_time if end_time is not None else datetime.now().strftime("%Y-%m-%d")
        )

        # Convert to ordinal dates for efficient date arithmetic
        self.base_start = date.fromisoformat(start_time).toordinal()
        self.base_end = date.fromisoformat(self.user_end_time).toordinal()

        # Non-stationary series configuration
        self.random_walk = random_walk

        # Initialize frequency component weights
        self._annual: Optional[Union[np.ndarray, float]] = 0.0
        self._monthly: Optional[Union[np.ndarray, float]] = 0.0
        self._weekly: Optional[Union[np.ndarray, float]] = 0.0
        self._hourly: Optional[Union[np.ndarray, float]] = 0.0
        self._minutely: Optional[Union[np.ndarray, float]] = 0.0

        # Component configuration parameters
        self._scale_config: Optional[ComponentScale] = None  # Amplitude scaling
        self._offset_config: Optional[ComponentScale] = None  # Baseline offsets
        self._noise_config: Optional[ComponentNoise] = None  # Noise parameters

        # Global series configuration
        self._time_series_config: Optional[SeriesConfig] = None

    def __call__(
        self,
        rng: np.random.RandomState,
        n_inputs_points: int = 512,
        input_dimension: int = 1,
    ) -> np.ndarray:
        """Call the `generate` method to stimulate time series generation"""
        return self.generate(
            rng=rng, n_inputs_points=n_inputs_points, input_dimension=input_dimension
        )

    def __str__(self) -> str:
        """Get the name of the time series generator"""
        return self.__class__.__name__

    def set_freq_variables(self, is_sub_day: Optional[bool] = None) -> None:
        """
        Configures frequency components based on temporal resolution.

        For sub-daily series (is_sub_day=True):
          - Includes minute, hourly, daily, weekly, monthly components
          - Excludes yearly components for stability

        For daily+ series (is_sub_day=False):
          - Includes daily, weekly, monthly components

        :param is_sub_day: Enable sub-daily components. Uses class default if None.
        :type is_sub_day: Optional[bool]
        """
        # Use class default if parameter not provided
        if is_sub_day is None:
            is_sub_day = self.is_sub_day

        if is_sub_day:
            # Sub-daily configuration (minutes to months)
            self.frequencies = [
                ("min", 1 / 1440),  # Minutes (1/1440 days)
                ("h", 1 / 24),  # Hours (1/24 days)
                ("D", 1),  # Days
                ("W", 7),  # Weeks
                ("MS", 30),  # Months (~30 days)
            ]
            self.frequency_names = ["minute", "hourly", "daily", "weekly", "monthly"]
            self.freq_and_index = (
                ("minute", 0),
                ("hourly", 1),
                ("daily", 2),
                ("weekly", 3),
                ("monthly", 4),
            )
        else:
            # Daily+ configuration (days to months)
            self.frequencies = [
                ("D", 1),  # Days
                ("W", 7),  # Weeks
                ("MS", 30),  # Months
            ]
            self.frequency_names = ["daily", "weekly", "monthly"]
            self.freq_and_index = (
                ("daily", 0),
                ("weekly", 1),
                ("monthly", 2),
            )

    def set_transition(self, transition: bool) -> None:
        """
        Enables/disables smooth transitions between frequency components.

        When enabled, creates gradual changes between seasonal patterns rather
        than abrupt shifts. Particularly useful for simulating business cycles
        or economic regime changes.

        :param transition: Enable component transition smoothing.
        :type transition: bool
        """
        self.transition = transition

    def reset_frequency_components(self) -> None:
        """Reset the frequency components recorded in the current class"""
        self._annual, self._monthly, self._weekly, self._hourly, self._minutely = (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

    def set_frequency_components(
        self, rng: np.random.RandomState, frequency: str
    ) -> None:
        """
        Configures frequency component weights based on input frequency type.

        Sets randomized component weights optimized for different temporal patterns:
        - min: Optimized for minute-level data (high minute variation, low hourly)
        - h: Optimized for hourly data (low minute variation, high hourly)
        - D: Optimized for daily data (high weekly seasonality, low monthly)
        - W: Optimized for weekly data (balanced monthly/annual seasonality)
        - MS: Optimized for monthly starts (low weekly, moderate annual seasonality)
        - YE: Optimized for year-end data (low weekly, high annual seasonality)

        Note: Weight ranges are empirically determined and can be modified as class properties.

        :param rng: The random number generator in NumPy with fixed seed.
        :param frequency: Temporal frequency identifier specifying the data type
        :type frequency: str
        :raises NotImplementedError: For unsupported frequency identifiers
        """
        # TODO: Make range limits configurable as class properties
        if frequency == "min":
            # Minute-level data: emphasize minutely variations
            self._minutely = rng.uniform(0.0, 1.0)  # High weight
            self._hourly = rng.uniform(0.0, 0.2)  # Low weight
        elif frequency == "h":
            # Hourly data: emphasize hourly patterns
            self._minutely = rng.uniform(0.0, 0.2)  # Low weight
            self._hourly = rng.uniform(0.0, 1)  # High weight
        elif frequency == "D":
            # Daily data: emphasize weekly seasonality
            self._weekly = rng.uniform(0.0, 1.0)  # High weight
            self._monthly = rng.uniform(0.0, 0.2)  # Low weight
        elif frequency == "W":
            # Weekly data: balanced monthly/annual patterns
            self._monthly = rng.uniform(0.0, 0.3)  # Moderate weight
            self._annual = rng.uniform(0.0, 0.3)  # Moderate weight
        elif frequency == "MS":
            # Month-start data: emphasize annual seasonality
            self._weekly = rng.uniform(0.0, 0.1)  # Low weight
            self._annual = rng.uniform(0.0, 0.5)  # Moderate weight
        elif frequency == "YE":
            # Year-end data: emphasize annual patterns
            self._weekly = rng.uniform(0.0, 0.2)  # Low weight
            self._annual = rng.uniform(0.0, 1)  # High weight
        else:
            raise NotImplementedError(
                f"Unsupported frequency type: {frequency}. "
                "Valid options: ['min', 'h', 'D', 'W', 'MS', 'YE']"
            )

    def get_component_scale_config(
        self,
        base: float,
        linear: Optional[float] = None,
        exp: Optional[float] = None,
        annual: Optional[np.ndarray] = None,
        monthly: Optional[np.ndarray] = None,
        weekly: Optional[np.ndarray] = None,
        hourly: Optional[np.ndarray] = None,
        minutely: Optional[np.ndarray] = None,
    ) -> ComponentScale:
        """
        Creates scaling configuration for time series components.

        This function follows the ForecastPFN architecture to define amplitude scaling:
        1. Uses class-level component weights as defaults when parameters are None
        2. Allows custom override of specific components
        3. Handles both fundamental (base, linear, exp) and seasonal components

        :param base: Constant baseline scaling factor
        :type base: float
        :param linear: Linear trend coefficient (slope). Default: class-stored value
        :type linear: Optional[float]
        :param exp: Exponential growth/decay base. Default: class-stored value
        :type exp: Optional[float]
        :param annual: Annual seasonality scaling vector. Default: self._annual
        :type annual: Optional[np.ndarray]
        :param monthly: Monthly seasonality scaling vector. Default: self._monthly
        :type monthly: Optional[np.ndarray]
        :param weekly: Weekly seasonality scaling vector. Default: self._weekly
        :type weekly: Optional[np.ndarray]
        :param hourly: Hourly seasonality scaling vector. Default: self._hourly
        :type hourly: Optional[np.ndarray]
        :param minutely: Minute-level seasonality scaling vector. Default: self._minutely
        :type minutely: Optional[np.ndarray]
        :return: Component scaling configuration
        :rtype: ComponentScale
        :note: The base, linear, and exp parameters could be moved to class attributes
        """
        # TODO: Consider making base, linear, exp class-level parameters
        config = ComponentScale(
            base=base,
            linear=linear,
            exp=exp,
            a=self._annual if annual is None else annual,
            m=self._monthly if monthly is None else monthly,
            w=self._weekly if weekly is None else weekly,
            h=self._hourly if hourly is None else hourly,
            minute=self._minutely if minutely is None else minutely,
        )

        return config

    @staticmethod
    def get_component_noise_config(
        k: float, median: float, scale: float
    ) -> ComponentNoise:
        """
        Creates noise configuration for time series generation.

        Parameters define multiplicative Weibull-distributed noise:
        :param k: Shape parameter for Weibull distribution (k > 0).

            - k < 1: Decreasing failure rate
            - k = 1: Exponential distribution (constant failure rate)
            - k > 1: Increasing failure rate

        :type k: float
        :param median: Median value of Weibull distribution (location parameter)
        :type median: float
        :param scale: Noise scaling factor where:

            - 0 = no noise (deterministic series)
            - 0-0.1 = low noise
            - 0.1-0.5 = moderate noise
            - >0.5 = high noise

        :type scale: float
        :return: Noise configuration object
        :rtype: ComponentNoise
        :note: These parameters could be set as class attributes for consistency
        """
        # TODO: Consider making these parameters class-level constants
        config = ComponentNoise(k=k, median=median, scale=scale)
        return config

    def get_time_series_config(
        self,
        scale_config: ComponentScale = None,
        offset_config: ComponentScale = None,
        noise_config: ComponentNoise = None,
    ) -> SeriesConfig:
        """
        Creates comprehensive configuration for time series generation.

        Combines three essential aspects of time series modeling:
        1. Scale: Component amplitude scaling
        2. Offset: Baseline offsets
        3. Noise: Stochastic properties

        Uses class-stored configurations when parameters are None.

        :param scale_config: Amplitude scaling configuration. Default: self._scale_config
        :type scale_config: Optional[ComponentScale]
        :param offset_config: Baseline offset configuration. Default: self._offset_config
        :type offset_config: Optional[ComponentScale]
        :param noise_config: Noise generation parameters. Default: self._noise_config
        :type noise_config: Optional[ComponentNoise]
        :return: Complete series generation configuration
        :rtype: SeriesConfig
        """
        config = SeriesConfig(
            scale=self._scale_config if scale_config is None else scale_config,
            offset=self._offset_config if offset_config is None else offset_config,
            noise_config=self._noise_config if noise_config is None else noise_config,
        )
        return config

    def generate_series(
        self,
        rng: np.random.RandomState,
        length=100,
        freq_index: int = None,
        start: pd.Timestamp = None,
        options: Optional[dict] = None,
        random_walk: bool = False,  # TODO: 是否可以添加为类属性
    ) -> Dict[str, Union[pd.Series, np.ndarray, pd.DataFrame, DatetimeIndex]]:
        """
        Function to construct synthetic series configs and generate synthetic series.

        :param rng: The random number generator in NumPy with fixed seed.
        :param length: The length of time series to generate.
        :param freq_index: The frequency of time series to generate.
        :param start: The start date of time series to generate.
        :param options: Options dict for generating series.
        :param random_walk: Whether to generate random walk or not.
        :return: The generated series dict.
        """
        if options is None:
            options = {}

        if freq_index is None:
            # Randomly pick a timestamp frequency from the existing list
            freq_index = rng.choice(len(self.frequencies))

        # Get the frequency information of the timestamp
        freq, timescale = self.frequencies[freq_index]

        # Reset various frequency components in class attributes
        self.reset_frequency_components()

        # Reselect the frequency components of various class attributes
        self.set_frequency_components(rng=rng, frequency=freq)

        if start is None:
            # Check if the user specified a start timestamp
            # start = pd.Timestamp(date.fromordinal(np.random.randint(BASE_START, BASE_END)))
            start = pd.Timestamp(
                date.fromordinal(
                    int(
                        (self.base_start - self.base_end) * beta.rvs(5, 1)
                        + self.base_end
                    )
                )
            )

        # Construct the data structure of each frequency component
        self._scale_config = self.get_component_scale_config(
            base=1.0,
            linear=rng.normal(loc=0.0, scale=0.01),
            exp=rng.normal(loc=1.0, scale=0.005 / timescale),
            annual=self._annual,
            monthly=self._monthly,
            weekly=self._weekly,
            hourly=self._hourly,
            minutely=self._minutely,
        )

        # Constructing time scale offset configurations for each frequency component
        self._offset_config = self.get_component_scale_config(
            base=0.0,
            linear=rng.normal(loc=-0.1, scale=0.5),
            exp=rng.normal(loc=-0.1, scale=0.5),
            annual=self._annual,
            monthly=self._monthly,
            weekly=self._weekly,
        )

        # Build a configuration for generating random noise sequences
        self._noise_config = self.get_component_noise_config(
            k=rng.uniform(low=1.0, high=5.0), median=1.0, scale=sample_scale(rng=rng)
        )

        # Build an overall configuration for generating time series data
        self._time_series_config = self.get_time_series_config(
            scale_config=self._scale_config,
            offset_config=self._offset_config,
            noise_config=self._noise_config,
        )

        return make_series(
            rng=rng,
            series=self._time_series_config,
            freq=to_offset(freq),
            periods=length,
            start=start,
            options=options,
            random_walk=random_walk,
        )

    def _select_ndarray_from_dict(
        self,
        rng: np.random.RandomState,
        length: int = 100,
        freq_index: int = None,
        start: pd.Timestamp = None,
        options: Optional[dict] = None,
    ) -> np.ndarray:
        """
        This method is internal and does not support external calls.
        Generates two time series data segments. Why two segments? See the `get_transition_coefficients` function for details.
        Select the desired time series data from the dictionary generated by the `make_series` function.

        Transition series refers to the linear combination of 2 series
        S1 and S2 such that the series S represents S1 for a period and S2
        for the remaining period.

        Function to construct synthetic series configs and generate synthetic series.

        :param rng: The random number generator in NumPy with fixed seed.
        :param length: The length of time series to generate.
        :param freq_index: The frequency of time series to generate.
        :param start: The start date of time series to generate.
        :param options: Options dict for generating series.
        :return: The selected time series data with `np.ndarray`.
        """
        series1 = self.generate_series(
            rng=rng,
            length=length,
            freq_index=freq_index,
            start=start,
            options=options,
            random_walk=self.random_walk,
        )

        series2 = self.generate_series(
            rng=rng,
            length=length,
            freq_index=freq_index,
            start=start,
            options=options,
            random_walk=self.random_walk,
        )

        if self.transition:
            coefficients = get_transition_coefficients(context_length=length)
            values = (
                coefficients * series1["values"]
                + (1 - coefficients) * series2["values"]
            )
        else:
            values = series1["values"]

        return values

    def generate(
        self,
        rng: np.random.RandomState,
        n_inputs_points: int = 512,
        input_dimension: int = 1,
        freq_index: Optional[int] = None,
        start: Optional[pd.Timestamp] = None,
        options: Optional[Dict] = None,
        random_walk: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Generates time series using the ForecastPFN algorithm.

        Implementation based on:
        ForecastPFN: Synthetically-Trained Zero-Shot Forecasting
        (https://arxiv.org/abs/2311.01933)

        The algorithm combines:
        1. Multiple seasonal components (annual, monthly, weekly, etc.)
        2. Trend components (linear, exponential)
        3. Noise components (Weibull-distributed)
        4. Optional random walk transformation

        :param rng: Seeded random number generator for reproducibility.
        :param n_inputs_points: Number of time points to generate per dimension
        :param input_dimension: Number of independent time series dimensions to generate
        :param freq_index: Index of frequency configuration to use (0-4 for sub-daily)
        :param start: Custom start timestamp for the series. Uses class default if None.
        :param options: Additional generation options (reserved for future extensions)
        :param random_walk: Enable random walk transformation. Overrides class default.
        :return: Generated time series array of shape (n_inputs_points, input_dimension)
        """
        # Handle random walk override
        if random_walk is not None:
            self.random_walk = random_walk

        # Initialize output array
        time_series = self.create_zeros(
            n_inputs_points=n_inputs_points, input_dimension=input_dimension
        )

        # Generate each dimension independently
        for i in range(input_dimension):
            time_series[:, i] = self._select_ndarray_from_dict(
                rng=rng,
                length=n_inputs_points,
                freq_index=freq_index,
                start=start,
                options=options,
            )
        return time_series

    @property
    def annual(self) -> Union[float, np.ndarray]:
        """Annual seasonality component weight(s)"""
        return self._annual

    @property
    def monthly(self) -> Union[float, np.ndarray]:
        """Monthly seasonality component weight(s)"""
        return self._monthly

    @property
    def weekly(self) -> Union[float, np.ndarray]:
        """Weekly seasonality component weight(s)"""
        return self._weekly

    @property
    def hourly(self) -> Union[float, np.ndarray]:
        """Hourly seasonality component weight(s)"""
        return self._hourly

    @property
    def minutely(self) -> Union[float, np.ndarray]:
        """Minute-level seasonality component weight(s)"""
        return self._minutely

    @property
    def scale_config(self) -> ComponentScale:
        """Amplitude scaling configuration for time series components"""
        return self._scale_config

    @property
    def offset_config(self) -> ComponentScale:
        """Baseline offset configuration for time series components"""
        return self._offset_config

    @property
    def noise_config(self) -> ComponentNoise:
        """Noise generation configuration for time series"""
        return self._noise_config

    @property
    def time_series_config(self) -> SeriesConfig:
        """Comprehensive configuration for time series generation"""
        return self._time_series_config


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    forecast_pfn = ForecastPFN()

    for i in range(2):
        time_series = forecast_pfn.generate(
            rng=np.random.RandomState(i), n_inputs_points=256, input_dimension=1
        )
        plt.plot(time_series)
        plt.show()
        # plt.savefig(f"../../data/forecast_pfn_{i}.jpg", dpi=300, bbox_inches="tight")

    print(str(forecast_pfn))
