# -*- coding: utf-8 -*-
"""
Created on 2026/02/13 13:04:42
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/S2Generator

Simulator module for learning white-noise-to-signal mappings from input data.

Core idea
---------
The simulators in this package treat an observed time series as the output of a
learnable dynamical system driven by stochastic excitation. From the perspective
of modern signal processing and complex-system modeling, any useful signal can
often be represented as the response obtained by passing white noise through a
linear or piecewise-linear system whose parameters encode the second-order
statistics of the input.

Given an input sequence, each simulator **fits** the parameters of such a system
(autocorrelation structure, state-space dynamics, ARIMA coefficients, or
regime-switching laws). After fitting, **transform** generates new samples by
exciting the learned system with fresh white noise (and, where applicable, latent
regime transitions), thereby synthesizing time series with statistical properties
similar to the original data.

Available implementations
-------------------------
- ``WienerFilterSimulator``: Yule-Walker / whitening filter formulation
- ``KalmanFilterSimulator``: companion state-space AR model with Kalman filtering
- ``ARIMASimulator``: differencing + ARMA model with maximum-likelihood fitting
- ``MarkovSwitchingSimulator``: Markov-switching autoregression for piecewise dynamics

All simulators expose a unified ``fit`` / ``transform`` interface for downstream use.
"""

__all__ = [
    "ARIMASimulator",
    "KalmanFilterSimulator",
    "MarkovSwitchingSimulator",
    "WienerFilterSimulator",
]

from .arima import ARIMASimulator

from .kalman_filtering import KalmanFilterSimulator

from .markov_switching import MarkovSwitchingSimulator

from .wiener_filter import WienerFilterSimulator
