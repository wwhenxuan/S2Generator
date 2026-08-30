# -*- coding: utf-8 -*-
"""Unit tests for HammersteinWienerSimulator."""

import unittest

import numpy as np
from scipy import signal

from s2generator.simulator import HammersteinWienerSimulator, WienerFilterSimulator
from s2generator.simulator.hammerstein_wiener_filter import (
    apply_polynomial,
    fit_polynomial_ridge,
)


def _make_nonlinear_target(n: int = 800, seed: int = 0) -> np.ndarray:
    """AR-colored noise with a mild asymmetric static distortion (stable skew)."""
    rng = np.random.RandomState(seed)
    e = rng.normal(0.0, 1.0, size=n + 40)
    z = signal.lfilter(b=[1.0], a=[1.0, -0.6, 0.25], x=e)[40:]
    z = (z - np.mean(z)) / (np.std(z) + 1e-12)
    # Bounded odd/even mix: skew without pathological outliers
    y = np.tanh(z) + 0.55 * z + 0.35 * (z**2) / (1.0 + z**2)
    return y.astype(np.float64)


def _wasserstein1(a: np.ndarray, b: np.ndarray) -> float:
    """1-D Wasserstein-1 via sorted samples (equal length after resample)."""
    a = np.sort(np.asarray(a, dtype=np.float64).ravel())
    b = np.sort(np.asarray(b, dtype=np.float64).ravel())
    n = min(a.size, b.size)
    if n == 0:
        return np.inf
    # interpolate to common length
    qa = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, a.size), a)
    qb = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, b.size), b)
    return float(np.mean(np.abs(qa - qb)))


def stats_skew(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x - np.mean(x)
    s = float(np.std(x))
    if s < 1e-12:
        return 0.0
    z = x / s
    return float(np.mean(z**3))


class TestHammersteinWienerSimulator(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.RandomState(0)
        self.target = _make_nonlinear_target(640, seed=1)

    def test_create_instance(self) -> None:
        for order in (3, 5, 7):
            for revin in (True, False):
                sim = HammersteinWienerSimulator(
                    filter_order=order, revin=revin, random_state=0
                )
                self.assertIsInstance(sim, HammersteinWienerSimulator)
                self.assertEqual(str(sim), "HammersteinWienerSimulator")

    def test_invalid_constructor(self) -> None:
        with self.assertRaises(ValueError):
            HammersteinWienerSimulator(filter_order=1)
        with self.assertRaises(ValueError):
            HammersteinWienerSimulator(input_degree=-1)

    def test_polynomial_helpers(self) -> None:
        x = np.linspace(-1.0, 1.0, 50)
        coeffs = np.array([0.5, -0.2, 0.1])
        y = apply_polynomial(x, coeffs)
        np.testing.assert_allclose(y, 0.5 - 0.2 * x + 0.1 * x**2, rtol=1e-10)

        fitted = fit_polynomial_ridge(x, y, degree=2, ridge=1e-10)
        np.testing.assert_allclose(fitted, coeffs, rtol=1e-5, atol=1e-5)

    def test_check_inputs(self) -> None:
        sim = HammersteinWienerSimulator(filter_order=4)
        with self.assertRaises(ValueError):
            sim.check_inputs([1, 2, 3])  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            sim.check_inputs(np.ones((2, 3, 4)))
        with self.assertRaises(ValueError):
            sim.check_inputs(np.ones(5))  # too short
        with self.assertRaises(ValueError):
            sim.check_inputs(np.zeros(100))
        flat = sim.check_inputs(np.random.RandomState(0).randn(2, 80))
        self.assertEqual(flat.ndim, 1)
        self.assertEqual(flat.size, 160)

    def test_transform_before_fit_raises(self) -> None:
        sim = HammersteinWienerSimulator(filter_order=4)
        with self.assertRaises(ValueError):
            sim.transform(num_samples=1, seq_length=64)
        with self.assertRaises(ValueError):
            _ = sim.coeffs

    def test_fit_transform_shapes(self) -> None:
        sim = HammersteinWienerSimulator(
            filter_order=5, input_degree=3, output_degree=3, random_state=0
        )
        sim.fit(self.target)
        self.assertEqual(len(sim.coeffs), 5)
        self.assertEqual(len(sim.input_coeffs), 4)
        self.assertEqual(len(sim.output_coeffs), 4)
        self.assertGreater(sim.sigma_sq, 0.0)
        self.assertIsNotNone(sim.residuals)
        self.assertIsNotNone(sim._input_xq)
        self.assertIsNotNone(sim._output_xq)

        out = sim.transform(num_samples=4, seq_length=200, random_state=0)
        self.assertEqual(out.shape, (4, 200))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_reproducibility(self) -> None:
        a = HammersteinWienerSimulator(filter_order=5, random_state=7)
        b = HammersteinWienerSimulator(filter_order=5, random_state=7)
        a.fit(self.target)
        b.fit(self.target)
        y1 = a.transform(num_samples=2, seq_length=128, random_state=11)
        y2 = b.transform(num_samples=2, seq_length=128, random_state=11)
        np.testing.assert_array_equal(y1, y2)

    def test_invoke_padding(self) -> None:
        sim = HammersteinWienerSimulator(filter_order=4, random_state=0)
        sim.fit(self.target)
        with self.assertRaises(ValueError):
            sim.invoke(np.ones(4))
        noise = np.random.RandomState(0).randn(128 + 4)
        y = sim.invoke(noise)
        self.assertEqual(y.shape, (128,))
        self.assertTrue(np.all(np.isfinite(y)))

    def test_lowpass_option_finite(self) -> None:
        sim = HammersteinWienerSimulator(
            filter_order=5,
            lowpass=True,
            lowpass_kwargs={"cutoff": 0.35},
            random_state=0,
        )
        sim.fit(self.target)
        y = sim.transform(num_samples=2, seq_length=128, random_state=0)
        self.assertTrue(np.all(np.isfinite(y)))
        self.assertIsNotNone(sim._lowpass_filter)

    def test_hw_beats_wiener_on_amplitude_law(self) -> None:
        """
        On a nonlinear target, HW should match the amplitude distribution
        (Wasserstein-1) better than linear Wiener, and skewness at least as well.
        """
        target = _make_nonlinear_target(1200, seed=2)
        target_skew = stats_skew(target)

        hw = HammersteinWienerSimulator(
            filter_order=6, input_degree=3, output_degree=3, random_state=0
        )
        wiener = WienerFilterSimulator(filter_order=6, random_state=0)
        hw.fit(target)
        wiener.fit(target)

        y_hw = hw.transform(num_samples=6, seq_length=len(target), random_state=3)
        y_w = wiener.transform(num_samples=6, seq_length=len(target), random_state=3)

        w1_hw = np.mean([_wasserstein1(target, y_hw[i]) for i in range(6)])
        w1_w = np.mean([_wasserstein1(target, y_w[i]) for i in range(6)])
        self.assertLess(w1_hw, w1_w)

        err_hw = np.mean([abs(stats_skew(y_hw[i]) - target_skew) for i in range(6)])
        err_w = np.mean([abs(stats_skew(y_w[i]) - target_skew) for i in range(6)])
        self.assertLessEqual(err_hw, err_w + 1e-6)


if __name__ == "__main__":
    unittest.main()
