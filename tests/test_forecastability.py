# -*- coding: utf-8 -*-
"""
Tests for ForeCA-style forecastability utilities: spectral entropy, Omega,
multivariate spectra, whitening, Slow Feature Analysis, and ForeCA.
"""

from __future__ import annotations

import unittest

import numpy as np

from s2generator.utils import (
    ForeCA,
    SlowFeatureAnalysis,
    discrete_entropy,
    initialize_weightvector,
    mvspectrum,
    normalize_mvspectrum,
    omega,
    spectral_entropy,
    spectrum_of_linear_combination,
    sqrt_matrix,
    whiten,
)


def _ar1(phi: float, n: int, rng: np.random.RandomState) -> np.ndarray:
    x = np.zeros(n)
    eps = rng.randn(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    return x


class TestDiscreteEntropy(unittest.TestCase):
    """Plug-in Shannon entropy of a discrete pmf."""

    def test_uniform_is_log_n(self) -> None:
        """Uniform over n outcomes has entropy log2(n)."""
        n = 8
        h = discrete_entropy(np.full(n, 1.0 / n), base=2.0)
        self.assertAlmostEqual(h, np.log2(n), places=10)

    def test_one_hot_is_zero(self) -> None:
        """A delta pmf has no uncertainty."""
        self.assertAlmostEqual(
            discrete_entropy(np.array([1.0, 0.0, 0.0])), 0.0, places=10
        )

    def test_rejects_negative_and_unnormalized(self) -> None:
        """Invalid probability vectors raise ValueError."""
        with self.assertRaises(ValueError):
            discrete_entropy(np.array([-0.1, 1.1]))
        with self.assertRaises(ValueError):
            discrete_entropy(np.array([0.2, 0.2]))

    def test_prior_moves_toward_uniform(self) -> None:
        """Mixing a peaked pmf with a uniform prior increases entropy."""
        peaked = np.array([0.9, 0.1, 0.0, 0.0])
        h0 = discrete_entropy(peaked, prior_weight=0.0)
        h1 = discrete_entropy(peaked, prior_weight=0.4)
        self.assertGreater(h1, h0)


class TestOmegaAndSpectralEntropy(unittest.TestCase):
    """Omega ranking: white noise < AR < sinusoid."""

    def test_sine_more_forecastable_than_white_noise(self) -> None:
        """A pure sinusoid has much larger Omega than Gaussian noise."""
        rng = np.random.RandomState(1)
        noise = rng.randn(256)
        sine = np.sin(np.linspace(0.0, 24 * np.pi, 256, endpoint=False))
        om_n = omega(noise, method="pgram")
        om_s = omega(sine, method="pgram")
        self.assertLess(om_n, 25.0)
        self.assertGreater(om_s, 70.0)
        self.assertGreater(om_s, om_n)

    def test_ar_stronger_phi_is_more_forecastable(self) -> None:
        """AR(1) with phi=0.9 is more forecastable than phi=0.5."""
        rng = np.random.RandomState(2)
        weak = _ar1(0.5, 512, rng)
        strong = _ar1(0.9, 512, rng)
        self.assertGreater(
            float(omega(strong, method="welch")),
            float(omega(weak, method="welch")),
        )

    def test_multivariate_is_columnwise(self) -> None:
        """A (T, K) array is scored independently per column."""
        rng = np.random.RandomState(3)
        t = 256
        x = np.column_stack(
            [
                np.sin(np.linspace(0.0, 20 * np.pi, t, endpoint=False)),
                rng.randn(t),
            ]
        )
        om = omega(x, method="pgram")
        self.assertEqual(om.shape, (2,))
        self.assertGreater(om[0], om[1])

    def test_spectral_entropy_nonnegative_and_unit_interval(self) -> None:
        """Normalized spectral entropy of white noise sits in (0, 1]."""
        rng = np.random.RandomState(4)
        h = spectral_entropy(rng.randn(200), method="pgram")
        self.assertGreaterEqual(h, 0.0)
        self.assertLessEqual(h, 1.0)

    def test_bad_shape_raises(self) -> None:
        """3-D input is rejected."""
        with self.assertRaises(ValueError):
            omega(np.zeros((4, 4, 4)))


class TestMvspectrum(unittest.TestCase):
    """Hermitian multivariate spectra and quadratic-form identity."""

    def test_hermitian(self) -> None:
        """Each frequency slice of a multivariate spectrum is Hermitian."""
        rng = np.random.RandomState(5)
        x = rng.randn(128, 3)
        spec = mvspectrum(x, method="pgram")
        self.assertEqual(spec.shape[1:], (3, 3))
        for f in range(spec.shape[0]):
            np.testing.assert_allclose(spec[f], spec[f].conj().T, atol=1e-10)

    def test_quadratic_form_matches_univariate(self) -> None:
        """w' S(lambda) w tracks the spectrum of y = X w (pgram, same FFT grid)."""
        rng = np.random.RandomState(6)
        x = rng.randn(200, 3)
        w = np.array([0.5, -0.2, 0.8])
        spec = mvspectrum(x, method="pgram")
        fy_quad = spectrum_of_linear_combination(spec, w)
        fy_direct = mvspectrum(x @ w, method="pgram")
        corr = np.corrcoef(fy_quad, fy_direct)[0, 1]
        self.assertGreater(corr, 0.95)

    def test_normalize_univariate_sums_to_half(self) -> None:
        """Positive-frequency univariate spectrum sums to 0.5 after normalize."""
        rng = np.random.RandomState(7)
        spec = mvspectrum(rng.randn(100), method="pgram")
        ns = normalize_mvspectrum(spec)
        self.assertAlmostEqual(float(np.real(ns).sum()), 0.5, places=10)


class TestWhiten(unittest.TestCase):
    """ZCA whitening maps data to zero mean and identity covariance."""

    def test_mean_zero_identity_cov(self) -> None:
        """Whitened columns are standardized and uncorrelated."""
        rng = np.random.RandomState(8)
        mix = rng.randn(3, 3)
        x = rng.randn(400, 3) @ mix
        result = whiten(x)
        np.testing.assert_allclose(result.U.mean(axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(np.cov(result.U, rowvar=False), np.eye(3), atol=1e-8)

    def test_already_white_is_idempotent(self) -> None:
        """A second whitening of U is essentially the identity map."""
        rng = np.random.RandomState(9)
        x = rng.randn(300, 2)
        u = whiten(x).U
        again = whiten(u)
        np.testing.assert_allclose(again.U, u - u.mean(axis=0), atol=1e-8)

    def test_sqrt_matrix_squares_back(self) -> None:
        """sqrt(A) sqrt(A) recovers a SPD covariance."""
        rng = np.random.RandomState(10)
        a = rng.randn(4, 4)
        spd = a.T @ a + np.eye(4)
        root = sqrt_matrix(spd)
        np.testing.assert_allclose(root @ root, spd, atol=1e-8)


class TestSlowFeatureAnalysis(unittest.TestCase):
    """Linear SFA fit / transform and lag-1 autocorrelation order."""

    def setUp(self) -> None:
        rng = np.random.RandomState(11)
        t = np.linspace(0.0, 4.0 * np.pi, 300, endpoint=False)
        slow = np.sin(t)
        fast = rng.randn(300)
        mix = rng.randn(2, 2)
        self.X = np.column_stack([slow, fast]) @ mix
        self.model = SlowFeatureAnalysis(n_comp=2).fit(self.X)

    def test_fit_transform_shape_and_methods(self) -> None:
        """fit, transform, and fit_transform agree and return (T, n_comp)."""
        scores = self.model.fit_transform(self.X)
        self.assertEqual(scores.shape, (self.X.shape[0], 2))
        np.testing.assert_allclose(scores, self.model.transform(self.X), atol=1e-10)
        np.testing.assert_allclose(scores, self.model.scores, atol=1e-12)

    def test_scores_uncorrelated(self) -> None:
        """SFA scores are contemporaneously uncorrelated."""
        corr = np.corrcoef(self.model.scores, rowvar=False)
        np.testing.assert_allclose(corr, np.eye(2), atol=1e-6)

    def test_lag1_acf_decreases(self) -> None:
        """The first (slow) feature has larger lag-1 autocorrelation than the last."""

        def lag1(z: np.ndarray) -> float:
            z = z - z.mean()
            return float(np.corrcoef(z[:-1], z[1:])[0, 1])

        acf_slow = lag1(self.model.scores[:, 0])
        acf_fast = lag1(self.model.scores[:, -1])
        self.assertGreater(acf_slow, acf_fast)

    def test_too_few_series_raises(self) -> None:
        """Univariate input is rejected."""
        with self.assertRaises(ValueError):
            SlowFeatureAnalysis().fit(np.arange(50.0))


class TestForeCA(unittest.TestCase):
    """EM ForeCA: scores, Omega order, and transform consistency."""

    def setUp(self) -> None:
        rng = np.random.RandomState(12)
        t = 256
        sine = np.sin(np.linspace(0.0, 16 * np.pi, t, endpoint=False))
        ar = _ar1(0.8, t, rng)
        noise = rng.randn(t)
        self.X = np.column_stack([sine, ar, noise])
        self.model = ForeCA(n_comp=2, n_starts=3, method="pgram", random_state=0).fit(
            self.X
        )

    def test_n_comp_too_large_raises(self) -> None:
        """n_comp cannot exceed the number of columns."""
        with self.assertRaises(ValueError):
            ForeCA(n_comp=5).fit(self.X)

    def test_univariate_raises(self) -> None:
        """A single column is not enough for ForeCA."""
        with self.assertRaises(ValueError):
            ForeCA(n_comp=1).fit(self.X[:, :1])

    def test_scores_standardized_and_uncorrelated(self) -> None:
        """ForeCs have mean 0, variance 1, and off-diagonal correlations ~ 0."""
        s = self.model.scores
        np.testing.assert_allclose(s.mean(axis=0), 0.0, atol=1e-8)
        np.testing.assert_allclose(s.std(axis=0, ddof=1), 1.0, atol=1e-6)
        corr = np.corrcoef(s, rowvar=False)
        np.testing.assert_allclose(corr, np.eye(s.shape[1]), atol=1e-6)

    def test_omega_is_nonincreasing(self) -> None:
        """Components are stored from most to least forecastable."""
        om = np.asarray(self.model.omega)
        self.assertTrue(np.all(np.diff(om) <= 1e-8))
        self.assertGreater(om[0], om[-1])

    def test_transform_matches_fit_transform(self) -> None:
        """transform after fit equals fit_transform on the training data."""
        other = ForeCA(n_comp=2, n_starts=3, method="pgram", random_state=0)
        scores = other.fit_transform(self.X)
        np.testing.assert_allclose(scores, other.transform(self.X), atol=1e-10)
        np.testing.assert_allclose(scores, self.model.scores, atol=1e-8)

    def test_loadings_and_weightvectors_shapes(self) -> None:
        """Loadings / weightvectors are (K, n_comp)."""
        k = self.X.shape[1]
        self.assertEqual(self.model.loadings.shape, (k, 2))
        self.assertEqual(self.model.weightvectors.shape, (k, 2))
        self.assertEqual(self.model.whitening.shape, (k, k))

    def test_initialize_weightvector_unit_norm(self) -> None:
        """Random initializers return a unit-norm vector of length K."""
        rng = np.random.RandomState(0)
        w = initialize_weightvector(4, method="rnorm", random_state=rng)
        self.assertEqual(w.size, 4)
        self.assertAlmostEqual(float(np.linalg.norm(w)), 1.0, places=10)
        self.assertGreaterEqual(w[0], 0.0)


if __name__ == "__main__":
    unittest.main()
