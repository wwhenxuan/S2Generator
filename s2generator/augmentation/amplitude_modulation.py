# -*- coding: utf-8 -*-
"""
Created on 2026/03/04 22:52:40
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/S2Generator
"""

import numpy as np
from scipy.interpolate import interp1d, lagrange  # 改用lagrange函数


def linear_interpolation(
    x_known: np.ndarray, y_known: np.ndarray, x_new: np.ndarray
) -> np.ndarray:
    """
    线性插值函数
    :param x_known: 已知离散点的x坐标（时间点），numpy数组
    :param y_known: 已知离散点的y坐标，numpy数组
    :param x_new: 需要插值的新x坐标，numpy数组或单个数值
    :return: 插值后的y_new值，与x_new形状相同
    """
    # 输入校验
    if len(x_known) != len(y_known):
        raise ValueError("x_known和y_known的长度必须相等")
    if np.any(np.diff(x_known) <= 0):
        raise ValueError("x_known必须是严格递增的序列")

    # 创建线性插值器
    linear_interp = interp1d(x_known, y_known, kind="linear", fill_value="extrapolate")
    # 计算插值结果
    y_new = linear_interp(x_new)
    return y_new


def cubic_spline_interpolation(
    x_known: np.ndarray, y_known: np.ndarray, x_new: np.ndarray
) -> np.ndarray:
    """
    三次样条插值函数
    :param x_known: 已知离散点的x坐标（时间点），numpy数组
    :param y_known: 已知离散点的y坐标，numpy数组
    :param x_new: 需要插值的新x坐标，numpy数组或单个数值
    :return: 插值后的y_new值，与x_new形状相同
    """
    # 输入校验
    if len(x_known) != len(y_known):
        raise ValueError("x_known和y_known的长度必须相等")
    if len(x_known) < 3:
        raise ValueError("三次样条插值需要至少3个已知点")
    if np.any(np.diff(x_known) <= 0):
        raise ValueError("x_known必须是严格递增的序列")

    # 创建三次样条插值器
    spline_interp = interp1d(x_known, y_known, kind="cubic", fill_value="extrapolate")
    # 计算插值结果
    y_new = spline_interp(x_new)
    return y_new


def lagrange_interpolation(x_known: np.ndarray, y_known: np.ndarray, x_new: np.ndarray):
    """
    拉格朗日插值函数（兼容所有scipy版本）
    :param x_known: 已知离散点的x坐标（时间点），numpy数组
    :param y_known: 已知离散点的y坐标，numpy数组
    :param x_new: 需要插值的新x坐标，numpy数组或单个数值
    :return: 插值后的y_new值，与x_new形状相同
    """
    # 输入校验
    if len(x_known) != len(y_known):
        raise ValueError("x_known和y_known的长度必须相等")
    if len(x_known) < 2:
        raise ValueError("拉格朗日插值需要至少2个已知点")
    if len(np.unique(x_known)) != len(x_known):
        raise ValueError("x_known中不能有重复的坐标点")

    # 创建拉格朗日插值多项式（兼容旧版本scipy）
    lagrange_poly = lagrange(x_known, y_known)
    # 计算插值结果（polyval支持单个值或数组输入）
    y_new = np.polyval(lagrange_poly, x_new)
    return y_new


def amplitude_modulation(
    time_series: np.ndarray,
    num_changepoints: int = 5,
    mean_amplitude: float = 1.0,
    amplitude_variation: float = 1.0,
) -> np.ndarray:
    """
    Perform amplitude modulation on the input time series.
    This method applies a random amplitude modulation to the time series, which can help to enhance the diversity of the data and improve the robustness of models trained on it.

    :param time_series: Input time series, a 1D numpy array

    :return: Amplitude modulated time series, a 1D numpy array of the same length as the input series.
    """
