# -*- coding: utf-8 -*-
"""
Created on 2026/02/12 12:53:15
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
from typing import Union, Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings

warnings.filterwarnings("ignore")


class ARIMAGenerator(object):
    """Generate time series data using ARIMA model."""

    def __init__(
        self,
        max_p: int = 5,
        max_d: int = 2,
        max_q: int = 5,
        signif: float = 0.05,
        not_white_alarm: bool = True,
        random_state: Optional[int] = 42,
    ) -> None:
        """
        :param order: A tuple specifying the (p, d, q) order of the ARIMA model.
        """
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q

        # ADF检验的显著性水平
        self.signif = signif

        # 是否在残差非白噪声时发出警告
        self.not_white_alarm = not_white_alarm

        # 记录模型的拟合的参数
        self.d_order = None
        self.p_order, self.q_order = None, None

        # 记录拟合后的模型
        self.model = None
        # 记录模型拟合的残差结果
        self.residuals = None

        # 随机数种子
        self.random_state = random_state

    def fit(
        self, time_series: np.ndarray, select_order: Optional[bool] = False
    ) -> None:
        """
        Fit the ARIMA model to the provided time series data.

        :param time_series: The input time series data to fit the ARIMA model.
        :param select_order: Whether to automatically select the (p, d, q) order.

        :return: None.
        """
        # 首先对时间序列进行差分使其平稳
        stationary_series, self.d_order = self.diff_stationary(
            time_series=pd.Series(time_series)
        )

        if select_order:
            # 使用AIC和BIC准则选择最优的(p,q)组合
            self.p_order, self.q_order = self.select_arma_order(
                stationary_series=stationary_series
            )
        else:
            self.p_order, self.q_order = self.max_p, self.max_q

        # 拟合ARIMA模型
        self.model = ARIMA(
            time_series, order=(self.p_order, self.d_order, self.q_order)
        ).fit()

        # 获取模型拟合的残差结果
        self.residuals = self.model.resid

        # 进行残差诊断
        mean_p_value, is_white = self.residual_diagnosis(signif=self.signif)
        if not is_white and self.not_white_alarm:
            print(
                f"警告: 模型残差可能不是白噪声 (平均p值={mean_p_value:.4f} < 显著性水平={self.signif})，请重新评估模型的阶数或参数。"
            )

    def transform(
        self, num_samples: int, seq_len: int, random_state: Optional[int] = None
    ) -> np.ndarray:
        """
        Transform the input time series data using the fitted ARIMA model.

        :param num_samples: Number of samples to generate.
        :param seq_len: Length of each generated sequence.

        :return: Transformed time series data.
        """
        # 判断模型是否已经拟合
        if not hasattr(self, "model"):
            raise ValueError("The model must be fitted before calling transform.")

        # 生成新的时间序列数据
        generated_series = self.model.simulate(
            nsimulations=seq_len,
            repetitions=num_samples,
            random_state=(
                random_state if random_state is not None else self.random_state
            ),
        )

        return generated_series.values

    def check_inputs(self, time_series: Union[pd.Series, np.ndarray]) -> None:
        """检查输入的时间序列数据是否合法"""
        if not isinstance(time_series, (pd.Series, np.ndarray)):
            raise ValueError(
                "Input time series must be a pandas Series or numpy ndarray."
            )
        if len(time_series) < 10:
            raise ValueError("Input time series must have at least 10 data points.")

    def select_arma_order(
        self, stationary_series: Union[pd.Series, np.ndarray]
    ) -> Tuple[int, int, int]:
        """对平稳序列选择ARMA(p,q)阶数（即ARIMA的p,q）"""

        # 使用AIC和BIC准则选择最优的(p,q)组合

        # 对AIC和BIC进行初始化
        best_aic = np.inf
        best_bic = np.inf
        best_order_aic = (0, 0)
        best_order_bic = (0, 0)

        # 遍历所有可能的(p,q)组合
        for p in range(self.max_p + 1):
            for q in range(self.max_q + 1):
                if p == 0 and q == 0:
                    # 跳过(p,q)=(0,0)的情况
                    continue
                try:
                    print(f"尝试ARIMA({p},{0},{q})")
                    # 开始拟合ARMA模型
                    # FIXME: 这里是否可以选择使用EACF方法来选择最优的(p,q)组合？
                    model = ARIMA(stationary_series, order=(p, 0, q))
                    results = model.fit()
                    if results.aic < best_aic:
                        best_aic = results.aic
                        best_order_aic = (p, q)
                    if results.bic < best_bic:
                        best_bic = results.bic
                        best_order_bic = (p, q)
                except:
                    continue

        print(f"基于AIC最优(p,q): {best_order_aic} (AIC={best_aic:.2f})")
        print(f"基于BIC最优(p,q): {best_order_bic} (BIC={best_bic:.2f})")
        return best_order_bic

    def diff_stationary(
        self, time_series: Union[pd.Series, np.ndarray]
    ) -> Tuple[Union[pd.Series, np.ndarray], int]:
        """逐次差分使时间序列平稳，返回平稳序列和差分阶数"""

        # 对输入的序列进行检验
        is_stationary = self.adf_test(time_series=time_series)

        diff_count = 0

        diff_series = time_series.copy()

        while not is_stationary and diff_count < self.max_d:
            diff_series = diff_series.diff().dropna()  # d阶差分
            is_stationary = self.adf_test(time_series=diff_series)
            diff_count += 1

        # 返回平稳序列和差分阶数
        return diff_series, diff_count

    def adf_test(self, time_series: Union[pd.Series, np.ndarray]) -> bool:
        """ADF检验，返回是否平稳"""

        # 差分后会有NaN，需删除
        adf_result = adfuller(time_series.dropna())
        p_value = adf_result[1]

        if p_value < self.signif:
            return True
        else:
            return False

    def model_summary(self) -> str:
        """返回拟合模型的摘要信息"""
        if not hasattr(self, "model"):
            raise ValueError("The model must be fitted before calling model_summary.")

        return self.model.summary().as_text()

    def residual_diagnosis(
        self, lags: int = 20, signif: float = None
    ) -> Tuple[float, bool]:
        """
        Perform residual diagnosis for the fitted ARIMA model.
        该方法通过执行Ljung-Box检验来评估模型残差的白噪声性质。
        如果残差被认为是白噪声，则表明模型拟合良好。
        如果为非白噪声，则可能需要重新评估模型的阶数或参数。
        该方法将返回模型的检验统计量的平均值以及
        一个能够判断残差是否为白噪声的布尔值。
        如果显著性水平未提供，则使用实例的显著性水平，默认值为0.05。

        :param signif: Significance level for the Ljung-Box test. If None, uses the instance's signif.

        :return: A tuple containing the mean p-value from the Ljung-Box test and a boolean indicating if all p-values exceed the significance level.
        """
        # 确保模型已经被拟合并且残差已经被计算
        if not hasattr(self, "residuals"):
            raise ValueError(
                "The model must be fitted before calling residual_diagnosis."
            )

        # 执行Ljung-Box检验
        lb_test = acorr_ljungbox(self.residuals, lags=lags, return_df=True)
        lb_p_values = lb_test["lb_pvalue"]

        return lb_p_values.mean(), all(lb_p_values > signif)

    def plot_shapiro_wilk(
        self, bins: int = 13, dpi: int = 500, save_path: str = None
    ) -> Tuple[plt.Figure, float, float]:
        """
        Plot the Shapiro-Wilk test for normality of the residuals.
        This method generates a Q-Q plot to visually assess whether the residuals
        of the fitted ARIMA model follow a normal distribution.
        """
        # 确保模型已经被拟合并且残差已经被计算
        if not hasattr(self, "residuals"):
            raise ValueError(
                "The model must be fitted before calling plot_shapiro_wilk."
            )

        # 导入必要的库
        from statsmodels.graphics.gofplots import qqplot
        from scipy.stats import shapiro

        # import seaborn as sns
        # sns.set_theme(style="ticks")

        # 执行Shapiro-Wilk正态性检验
        stat, p_value = shapiro(self.residuals)

        # 创建可视化的图形
        fig, ax = plt.subplots(1, 2, figsize=(12.1, 5))
        fig.subplots_adjust(wspace=0.16)

        # 绘制拟合残差的直方图
        ax[0].hist(self.residuals, bins=bins, alpha=1, color="w", edgecolor="k", lw=1.2)

        # 绘制正态分布检验的Q-Q图
        qqplot(
            self.residuals,
            line="s",
            ax=ax[1],
            markerfacecolor="white",
            markeredgecolor="k",
            markersize=7.5,
        )
        for line in ax[1].get_lines():
            if line.get_linestyle() == "-":
                line.set_color("#DC143C")
                line.set_linewidth(2.1)

        # Set titles and labels
        ax[0].grid(which="major", color="gray", linestyle="--", lw=0.5, alpha=0.8)
        ax[1].grid(which="major", color="gray", linestyle="--", lw=0.5, alpha=0.8)
        ax[0].set_xlabel("Standard Residual", fontsize=12.5)
        ax[0].set_ylabel("Frequency", fontsize=12.5)
        ax[1].set_xlabel("Theoretical Quantiles", fontsize=12.5)
        ax[1].set_ylabel("Sample Quantiles", fontsize=12.5)

        # Annotate the plots with statistics
        mean = np.round(np.mean(self.residuals), 4)
        std = np.round(np.std(self.residuals), 4)
        stat = np.round(stat, 4)
        p_value = np.round(p_value, 4)

        # Set the text annotations for the mean and std on the histogram
        ax[0].text(
            0.05,
            0.95,
            f"$\mu$ = {mean}\n$\sigma$ = {std}",
            transform=ax[0].transAxes,
            verticalalignment="top",
            horizontalalignment="left",
            fontsize=13.5,
            color="k",
        )

        # Set the text annotations for the Shapiro-Wilk test on the Q-Q plot
        ax[1].text(
            0.05,
            0.95,
            f"$W$ = {stat}\n$p$ = {p_value}",
            transform=ax[1].transAxes,
            verticalalignment="top",
            horizontalalignment="left",
            fontsize=13.5,
            color="k",
        )

        return fig, stat, p_value
