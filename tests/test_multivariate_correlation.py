# -*- coding: utf-8 -*-
"""
Unit tests for multivariate correlation / similarity matrices.

Created on 2026/03/22
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/S2Generator
"""

import unittest

import numpy as np

from s2generator.utils import multivariate_correlation
from s2generator.utils.correlation import (
    AVAILABLE_CORRELATION_MEASURES,
    parse_correlation_measures,
    pearson_correlation_matrix,
    spearman_correlation_matrix,
    autocorrelation_similarity_matrix,
    power_spectrum_similarity_matrix,
    distribution_similarity_matrix,
    wasserstein_distance_correlation_matrix,
)


class TestParseCorrelationMeasures(unittest.TestCase):
    """Tests for measure-name parsing and alias resolution."""

    def test_single_string(self) -> None:
        """A single measure name should parse to a one-element canonical list."""
        self.assertEqual(parse_correlation_measures("pearson"), ["pearson"])

    def test_space_separated_string(self) -> None:
        """Space-separated names and aliases should parse in order."""
        self.assertEqual(
            parse_correlation_measures("pearson wasserstein acf"),
            ["pearson", "wasserstein", "autocorrelation"],
        )

    def test_comma_separated_string(self) -> None:
        """Comma-separated names and aliases should parse in order."""
        self.assertEqual(
            parse_correlation_measures("pearson,psd,dist"),
            ["pearson", "power_spectrum", "distribution"],
        )

    def test_list_input_with_aliases(self) -> None:
        """A list of aliases should resolve to canonical measure names."""
        self.assertEqual(
            parse_correlation_measures(["corr", "ws", "spearman"]),
            ["pearson", "wasserstein", "spearman"],
        )

    def test_deduplicate_preserving_order(self) -> None:
        """Duplicate measure names should be dropped while preserving first-seen order."""
        self.assertEqual(
            parse_correlation_measures("pearson acf pearson corr"),
            ["pearson", "autocorrelation"],
        )

    def test_unknown_measure_raises(self) -> None:
        """An unknown measure name should raise ValueError."""
        with self.assertRaises(ValueError):
            parse_correlation_measures("not_a_real_measure")

    def test_empty_measure_raises(self) -> None:
        """An empty measure specification should raise ValueError."""
        with self.assertRaises(ValueError):
            parse_correlation_measures("   ")

    def test_invalid_type_raises(self) -> None:
        """A non-string, non-list measure argument should raise TypeError."""
        with self.assertRaises(TypeError):
            parse_correlation_measures(123)  # type: ignore[arg-type]

    def test_list_with_non_string_raises(self) -> None:
        """A mixed-type measure list should raise TypeError."""
        with self.assertRaises(TypeError):
            parse_correlation_measures(["pearson", 1])  # type: ignore[list-item]


class TestMultivariateCorrelation(unittest.TestCase):
    """Tests for pairwise correlation / similarity estimators."""

    def setUp(self) -> None:
        """Prepare fixtures used by the Multivariate Correlation tests."""
        self.rng = np.random.RandomState(0)
        self.n_samples = 4
        self.seq_length = 128
        base = self.rng.randn(self.seq_length)
        noise = self.rng.randn(self.n_samples, self.seq_length)
        self.data = np.vstack(
            [
                base,
                0.85 * base + 0.15 * noise[1],
                noise[2],
                -0.7 * base + 0.3 * noise[3],
            ]
        )

    def _assert_square_symmetric(
        self, mat: np.ndarray, n: int, *, diagonal: float | None = 1.0
    ) -> None:
        """Assert that a matrix is finite, square, symmetric, and has the expected diagonal."""
        self.assertIsInstance(mat, np.ndarray)
        self.assertEqual(mat.shape, (n, n))
        self.assertTrue(np.all(np.isfinite(mat)))
        self.assertTrue(np.allclose(mat, mat.T, atol=1e-8), msg="matrix not symmetric")
        if diagonal is not None:
            self.assertTrue(
                np.allclose(np.diag(mat), diagonal, atol=1e-8),
                msg=f"diagonal expected {diagonal}",
            )

    def test_input_validation(self) -> None:
        """Correlation helpers should reject arrays that are too small or have the wrong rank."""
        with self.assertRaises(ValueError):
            pearson_correlation_matrix(self.data[:1])
        with self.assertRaises(ValueError):
            pearson_correlation_matrix(np.zeros((3, 1)))
        with self.assertRaises(ValueError):
            pearson_correlation_matrix(np.zeros((2, 2, 2)))

    def test_pearson_matches_numpy(self) -> None:
        """Pearson matrices should match numpy.corrcoef on the same channels."""
        mat = pearson_correlation_matrix(self.data)
        expected = np.corrcoef(self.data)
        self._assert_square_symmetric(mat, self.n_samples, diagonal=1.0)
        self.assertTrue(np.allclose(mat, expected))
        # Strongly related channels should have large |corr|
        self.assertGreater(mat[0, 1], 0.8)
        self.assertLess(mat[0, 3], -0.4)

    def test_spearman_two_series_scalar_path(self) -> None:
        """Spearman correlation of two series should return a 2x2 matrix with unit diagonal."""
        pair = self.data[:2]
        mat = spearman_correlation_matrix(pair)
        self._assert_square_symmetric(mat, 2, diagonal=1.0)
        self.assertGreater(mat[0, 1], 0.5)

    def test_spearman_multi_series(self) -> None:
        """Spearman correlation of several series should return a square symmetric matrix."""
        mat = spearman_correlation_matrix(self.data)
        self._assert_square_symmetric(mat, self.n_samples, diagonal=1.0)

    def test_autocorrelation_similarity(self) -> None:
        """Autocorrelation similarity should return a square symmetric matrix with unit diagonal."""
        mat = autocorrelation_similarity_matrix(self.data, nlags=20)
        self._assert_square_symmetric(mat, self.n_samples, diagonal=1.0)

    def test_power_spectrum_similarity(self) -> None:
        # Two sinusoids with same frequency should be highly similar in PSD space
        """Series that share a frequency should be more similar in PSD space than unrelated ones."""
        t = np.linspace(0, 4 * np.pi, 256)
        x = np.vstack(
            [
                np.sin(2 * t),
                np.sin(2 * t + 0.3),
                np.sin(7 * t),
            ]
        )
        mat = power_spectrum_similarity_matrix(x)
        self._assert_square_symmetric(mat, 3, diagonal=1.0)
        self.assertGreater(mat[0, 1], mat[0, 2])

    def test_distribution_similarity(self) -> None:
        """Distribution similarity should return a square symmetric matrix with unit diagonal."""
        mat = distribution_similarity_matrix(self.data, bins=16)
        self._assert_square_symmetric(mat, self.n_samples, diagonal=1.0)

    def test_wasserstein_distance_matrix(self) -> None:
        """Wasserstein distance matrices should be symmetric, non-negative, and zero on identical channels."""
        mat = wasserstein_distance_correlation_matrix(self.data)
        self._assert_square_symmetric(mat, self.n_samples, diagonal=0.0)
        self.assertTrue(np.all(mat >= 0))
        # Identical channels → near-zero distance
        twin = np.vstack([self.data[0], self.data[0]])
        twin_mat = wasserstein_distance_correlation_matrix(twin)
        self.assertLess(twin_mat[0, 1], 1e-6)

    def test_wasserstein_short_series_fallback(self) -> None:
        """Short series should still produce a valid non-negative Wasserstein distance matrix."""
        short = self.rng.randn(3, 16)
        mat = wasserstein_distance_correlation_matrix(short)
        self._assert_square_symmetric(mat, 3, diagonal=0.0)
        self.assertTrue(np.all(mat >= 0))

    def test_multivariate_correlation_single_returns_array(self) -> None:
        """A single measure name should return a numpy correlation matrix."""
        mat = multivariate_correlation(self.data, measure="pearson")
        self.assertIsInstance(mat, np.ndarray)
        self._assert_square_symmetric(mat, self.n_samples, diagonal=1.0)

    def test_multivariate_correlation_multi_returns_dict(self) -> None:
        """Multiple measures should return a dict of named matrices."""
        result = multivariate_correlation(
            self.data, measure="pearson wasserstein distribution"
        )
        self.assertIsInstance(result, dict)
        self.assertEqual(set(result.keys()), {"pearson", "wasserstein", "distribution"})
        self._assert_square_symmetric(result["pearson"], self.n_samples, diagonal=1.0)
        self._assert_square_symmetric(
            result["wasserstein"], self.n_samples, diagonal=0.0
        )
        self._assert_square_symmetric(
            result["distribution"], self.n_samples, diagonal=1.0
        )

    def test_multivariate_correlation_list_and_kwargs(self) -> None:
        """List measure names should accept extra kwargs such as nlags and bins."""
        result = multivariate_correlation(
            self.data,
            measure=["acf", "distribution"],
            nlags=10,
            bins=8,
        )
        self.assertEqual(set(result.keys()), {"autocorrelation", "distribution"})
        for mat in result.values():
            self._assert_square_symmetric(mat, self.n_samples, diagonal=1.0)

    def test_all_available_measures_run(self) -> None:
        """Every registered correlation measure should run and return a valid matrix."""
        result = multivariate_correlation(
            self.data, measure=list(AVAILABLE_CORRELATION_MEASURES)
        )
        self.assertEqual(set(result.keys()), set(AVAILABLE_CORRELATION_MEASURES))
        for name, mat in result.items():
            diag = 0.0 if name == "wasserstein" else 1.0
            self._assert_square_symmetric(mat, self.n_samples, diagonal=diag)

    def test_does_not_mutate_input(self) -> None:
        """Verify that the call does not modify the input array in place."""
        original = self.data.copy()
        _ = multivariate_correlation(
            self.data, measure=["pearson", "wasserstein", "distribution"]
        )
        self.assertTrue(np.array_equal(self.data, original))


if __name__ == "__main__":
    unittest.main()
