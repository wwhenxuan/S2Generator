# -*- coding: utf-8 -*-
"""Tests for ForeCA / Omega visualization helpers."""

import unittest

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt

from s2generator.utils import (
    ForeCA,
    SlowFeatureAnalysis,
    omega,
    plot_foreca,
    plot_omega,
    plot_sfa,
    plot_spectrum,
    univariate_spectrum,
)


class TestForecastabilityPlots(unittest.TestCase):
    """Figures return the expected number of axes / artists."""

    def setUp(self) -> None:
        rng = np.random.RandomState(0)
        t = np.linspace(0.0, 8.0 * np.pi, 160, endpoint=False)
        self.X = np.column_stack([np.sin(t), rng.randn(160), rng.randn(160)])

    def tearDown(self) -> None:
        plt.close("all")

    def test_plot_omega_bars(self) -> None:
        """One bar is drawn per Omega value."""
        om = omega(self.X, method="pgram")
        fig = plot_omega(om, names=["sine", "n1", "n2"])
        self.assertEqual(len(fig.axes), 1)
        self.assertEqual(len(fig.axes[0].patches), 3)

    def test_plot_spectrum_line(self) -> None:
        """Univariate spectrum is a single line."""
        freqs, spec = univariate_spectrum(self.X[:, 0], method="pgram")
        fig = plot_spectrum(freqs, spec)
        self.assertEqual(len(fig.axes), 1)
        self.assertGreaterEqual(len(fig.axes[0].lines), 1)

    def test_plot_foreca_layout(self) -> None:
        """n_comp score rows plus n_comp Omega bars."""
        model = ForeCA(n_comp=2, n_starts=2, method="pgram", random_state=0).fit(self.X)
        fig = plot_foreca(model)
        self.assertEqual(len(fig.axes), 4)

    def test_plot_sfa_rows(self) -> None:
        """One row per slow feature."""
        model = SlowFeatureAnalysis(n_comp=2).fit(self.X)
        fig = plot_sfa(model)
        self.assertEqual(len(fig.axes), 2)
