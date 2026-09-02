# -*- coding: utf-8 -*-
"""Tests for IQ visualization helpers."""

import unittest

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt

from s2generator.utils.visualization import plot_iq_analysis, plot_iq_series


class TestPlotIQ(unittest.TestCase):
    """Plot I/Q traces and the analysis dashboard."""

    def setUp(self) -> None:
        """Build a short analytic-like IQ pair of shape [2, seq_length]."""
        t = np.linspace(0.0, 1.0, 128, endpoint=False)
        self.iq = np.vstack([np.cos(2 * np.pi * 3 * t), np.sin(2 * np.pi * 3 * t)])

    def tearDown(self) -> None:
        """Close figures created by the plot helpers."""
        plt.close("all")

    def test_plot_iq_series_overlay(self) -> None:
        """Overlay mode draws I and Q on a single axes."""
        fig = plot_iq_series(self.iq, overlay=True)
        self.assertEqual(len(fig.axes), 1)
        self.assertEqual(len(fig.axes[0].lines), 2)

    def test_plot_iq_series_stacked(self) -> None:
        """Stacked mode draws I and Q on two shared-x rows."""
        fig = plot_iq_series(self.iq, overlay=False)
        self.assertEqual(len(fig.axes), 2)

    def test_plot_iq_series_accepts_length_last(self) -> None:
        """[seq_length, 2] should be accepted as well as [2, seq_length]."""
        fig = plot_iq_series(self.iq.T, overlay=True)
        self.assertEqual(len(fig.axes), 1)

    def test_plot_iq_analysis_layout(self) -> None:
        """Analysis figure should contain the time, constellation, mag, phase, PSD, Hilbert panels."""
        fig = plot_iq_analysis(self.iq, overlay=True)
        # constellation colorbar adds one extra axes
        self.assertGreaterEqual(len(fig.axes), 6)

    def test_invalid_shape_raises(self) -> None:
        """A series that is not IQ-shaped should raise ValueError."""
        with self.assertRaises(ValueError):
            plot_iq_series(np.zeros((3, 64)))


if __name__ == "__main__":
    unittest.main()
