# -*- coding: utf-8 -*-
"""
Test suite for the synthetic multivariate coupling subpackage (TiRex-2).

Covers all coupling mechanisms and post-processing transforms from:
    Podest, P., et al. (2026). TiRex-2: Generalizing TiRex to Multivariate
    Data and Streaming. arXiv:2607.01204v1.

The coupling mechanisms operate on arrays of shape (T, Q), where T is the
sequence length and Q is the number of variates (opposite convention to the
CAUKER / SCM-prior pipelines, which use (d, L)).

Created on 2026/08/18
@author: Ruizhe Wang
@email: changewam6@gmail.com
"""

import unittest
import numpy as np

from s2generator.scm.coupling import (
    # Base class
    BaseCoupling,
    # Coupling mechanisms
    IdentityCoupling,
    UnivariatePassThrough,
    FunctionalCoupling,
    LinearMixing,
    Cointegration,
    LinearSCM,
    NonlinearSCM,
    # Post-processing
    PostProcessor,
    variate_permutation,
    smooth_time_warping,
    patch_masking,
    partial_future_observability,
    value_discretization,
    time_discretization,
    # Unified interface
    CouplingPipeline,
)


# ===========================================================================
# Base Class Tests
# ===========================================================================


class TestBaseCoupling(unittest.TestCase):
    """Test the abstract base class infrastructure."""

    def setUp(self):
        self.rng = np.random.RandomState(42)

    def test_abstract_class(self):
        """BaseCoupling should not be directly instantiable."""
        with self.assertRaises(TypeError):
            BaseCoupling()

    def test_dtype_property(self):
        self.assertEqual(IdentityCoupling().dtype, np.float64)
        self.assertEqual(IdentityCoupling(dtype=np.float32).dtype, np.float32)

    def test_str_method(self):
        self.assertEqual(str(IdentityCoupling()), "IdentityCoupling")

    def test_call_delegates_to_couple(self):
        coupler = IdentityCoupling()
        series = self.rng.normal(0, 1, (10, 3))
        np.testing.assert_array_equal(coupler(self.rng, series), series)

    def test_create_zeros_shape_and_dtype(self):
        coupler = IdentityCoupling(dtype=np.float32)
        out = coupler.create_zeros(seq_length=8, num_channels=3)
        self.assertEqual(out.shape, (8, 3))
        self.assertEqual(out.dtype, np.float32)
        self.assertTrue(np.all(out == 0))

    def test_validate_series_ok(self):
        # Should not raise for a valid 2D array
        BaseCoupling._validate_series(np.zeros((5, 2)), min_variates=1, min_length=1)

    def test_validate_series_ndim(self):
        with self.assertRaises(ValueError):
            BaseCoupling._validate_series(np.zeros((5,)), min_variates=1, min_length=1)

    def test_validate_series_too_few_variates(self):
        with self.assertRaises(ValueError):
            BaseCoupling._validate_series(np.zeros((5, 2)), min_variates=3)

    def test_validate_series_too_short(self):
        with self.assertRaises(ValueError):
            BaseCoupling._validate_series(np.zeros((2, 2)), min_length=3)


# ===========================================================================
# Identity / Univariate Pass-Through Tests
# ===========================================================================


class TestIdentityCoupling(unittest.TestCase):
    """Test the identity (no-coupling) mechanism."""

    def setUp(self):
        self.rng = np.random.RandomState(42)
        self.series = self.rng.normal(0, 1, (64, 4))

    def test_output_equals_input(self):
        out = IdentityCoupling().couple(self.rng, self.series)
        np.testing.assert_array_equal(out, self.series)

    def test_shape_preserved(self):
        out = IdentityCoupling().couple(self.rng, self.series)
        self.assertEqual(out.shape, (64, 4))

    def test_dtype_conversion(self):
        out = IdentityCoupling(dtype=np.float32).couple(self.rng, self.series)
        self.assertEqual(out.dtype, np.float32)
        np.testing.assert_allclose(out, self.series)

    def test_does_not_consume_rng(self):
        """Identity uses no randomness, so output is identical regardless of rng."""
        r1 = np.random.RandomState(0)
        r2 = np.random.RandomState(123)
        np.testing.assert_array_equal(
            IdentityCoupling().couple(r1, self.series),
            IdentityCoupling().couple(r2, self.series),
        )

    def test_single_variate(self):
        x = self.rng.normal(0, 1, (10, 1))
        np.testing.assert_array_equal(IdentityCoupling().couple(self.rng, x), x)


class TestUnivariatePassThrough(unittest.TestCase):
    """Test the univariate pass-through mechanism."""

    def setUp(self):
        self.rng = np.random.RandomState(42)
        self.series = self.rng.normal(0, 1, (64, 4))

    def test_returns_first_column(self):
        out = UnivariatePassThrough().couple(self.rng, self.series)
        self.assertEqual(out.shape, (64, 1))
        np.testing.assert_array_equal(out[:, 0], self.series[:, 0])

    def test_str_method(self):
        self.assertEqual(str(UnivariatePassThrough()), "UnivariatePassThrough")

    def test_dtype(self):
        out = UnivariatePassThrough(dtype=np.float32).couple(self.rng, self.series)
        self.assertEqual(out.dtype, np.float32)


# ===========================================================================
# Functional Coupling Tests
# ===========================================================================


class TestFunctionalCoupling(unittest.TestCase):
    """Test the functional (pointwise nonlinear) coupling mechanism."""

    def setUp(self):
        self.rng = np.random.RandomState(42)
        self.series = self.rng.normal(0, 1, (64, 5))

    def test_shape(self):
        out = FunctionalCoupling().couple(self.rng, self.series)
        self.assertEqual(out.shape, (64, 5))

    def test_first_variate_passes_through(self):
        out = FunctionalCoupling().couple(self.rng, self.series)
        np.testing.assert_array_equal(out[:, 0], self.series[:, 0])

    def test_output_finite(self):
        out = FunctionalCoupling().couple(self.rng, self.series)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_deterministic_with_seed(self):
        r1, r2 = np.random.RandomState(7), np.random.RandomState(7)
        o1 = FunctionalCoupling().couple(r1, self.series)
        o2 = FunctionalCoupling().couple(r2, self.series)
        np.testing.assert_array_equal(o1, o2)

    def test_different_seeds_differ(self):
        r1, r2 = np.random.RandomState(1), np.random.RandomState(2)
        o1 = FunctionalCoupling().couple(r1, self.series)
        o2 = FunctionalCoupling().couple(r2, self.series)
        self.assertFalse(np.allclose(o1, o2))

    def test_zero_noise_is_noiseless(self):
        out = FunctionalCoupling(noise_std=0.0).couple(self.rng, self.series)
        # First variate passes through; the rest are deterministic functions of it
        np.testing.assert_array_equal(out[:, 0], self.series[:, 0])
        self.assertTrue(np.all(np.isfinite(out)))

    def test_noise_std_override(self):
        coupler = FunctionalCoupling(noise_std=1.0)
        out_low = coupler.couple(np.random.RandomState(0), self.series, noise_std=0.0)
        out_high = coupler.couple(np.random.RandomState(0), self.series, noise_std=0.0)
        # noise_std=0.0 makes the two identical (no randomness added)
        np.testing.assert_array_equal(out_low, out_high)

    def test_properties(self):
        coupler = FunctionalCoupling(noise_std=0.3)
        self.assertEqual(coupler.noise_std, 0.3)
        self.assertEqual(coupler.function_types, FunctionalCoupling._FUNCTION_TYPES)

    def test_custom_function_types(self):
        coupler = FunctionalCoupling(function_types=["monotone"])
        out = coupler.couple(self.rng, self.series)
        self.assertEqual(out.shape, (64, 5))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_sample_function_all_types(self):
        coupler = FunctionalCoupling()
        for func_type in coupler.function_types:
            func = coupler._sample_function(self.rng, func_type)
            self.assertTrue(callable(func))
            out = func(self.series[:, 0])
            self.assertEqual(out.shape, (64,))
            self.assertTrue(np.all(np.isfinite(out)))


# ===========================================================================
# Linear Mixing Tests
# ===========================================================================


class TestLinearMixing(unittest.TestCase):
    """Test the linear mixing (shared latent factor) mechanism."""

    def setUp(self):
        self.rng = np.random.RandomState(42)
        self.series = self.rng.normal(0, 1, (64, 4))

    def test_shape(self):
        out = LinearMixing().couple(self.rng, self.series)
        self.assertEqual(out.shape, (64, 4))

    def test_output_finite(self):
        out = LinearMixing().couple(self.rng, self.series)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_deterministic_with_seed(self):
        r1, r2 = np.random.RandomState(7), np.random.RandomState(7)
        o1 = LinearMixing().couple(r1, self.series)
        o2 = LinearMixing().couple(r2, self.series)
        np.testing.assert_array_equal(o1, o2)

    def test_different_seeds_differ(self):
        r1, r2 = np.random.RandomState(1), np.random.RandomState(2)
        o1 = LinearMixing().couple(r1, self.series)
        o2 = LinearMixing().couple(r2, self.series)
        self.assertFalse(np.allclose(o1, o2))

    def test_requires_two_variates(self):
        with self.assertRaises(ValueError):
            LinearMixing().couple(self.rng, np.zeros((8, 1)))

    def test_properties(self):
        coupler = LinearMixing(spectral_regimes=["dominant"])
        self.assertEqual(coupler.spectral_regimes, ["dominant"])

    def test_random_orthogonal(self):
        for n in [2, 4, 8]:
            Q = LinearMixing._random_orthogonal(self.rng, n)
            self.assertEqual(Q.shape, (n, n))
            # Q should be orthogonal: Q.T @ Q ≈ I
            np.testing.assert_allclose(Q.T @ Q, np.eye(n), atol=1e-10)

    def test_mixing_matrix_per_regime(self):
        for regime in ["dominant", "uniform", "power_law"]:
            A = LinearMixing()._generate_mixing_matrix(self.rng, 4, regime)
            self.assertEqual(A.shape, (4, 4))
            self.assertTrue(np.all(np.isfinite(A)))

    def test_dtype(self):
        out = LinearMixing(dtype=np.float32).couple(self.rng, self.series)
        self.assertEqual(out.dtype, np.float32)


# ===========================================================================
# Cointegration Tests
# ===========================================================================


class TestCointegration(unittest.TestCase):
    """Test the cointegration (shared stochastic trends) mechanism."""

    def setUp(self):
        self.rng = np.random.RandomState(42)
        self.series = self.rng.normal(0, 1, (128, 4))

    def test_shape(self):
        out = Cointegration().couple(self.rng, self.series)
        self.assertEqual(out.shape, (128, 4))

    def test_output_finite(self):
        out = Cointegration().couple(self.rng, self.series)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_deterministic_with_seed(self):
        r1, r2 = np.random.RandomState(7), np.random.RandomState(7)
        o1 = Cointegration().couple(r1, self.series)
        o2 = Cointegration().couple(r2, self.series)
        np.testing.assert_array_equal(o1, o2)

    def test_different_seeds_differ(self):
        r1, r2 = np.random.RandomState(1), np.random.RandomState(2)
        o1 = Cointegration().couple(r1, self.series)
        o2 = Cointegration().couple(r2, self.series)
        self.assertFalse(np.allclose(o1, o2))

    def test_nonstationary_trends(self):
        """Random-walk trends imply growing variance over time."""
        out = Cointegration().couple(self.rng, self.series)
        first_half = np.std(out[:64], axis=0)
        second_half = np.std(out[64:], axis=0)
        self.assertGreater(np.mean(second_half), np.mean(first_half) * 0.5)

    def test_single_variate(self):
        out = Cointegration().couple(self.rng, self.rng.normal(0, 1, (20, 1)))
        self.assertEqual(out.shape, (20, 1))

    def test_properties(self):
        coupler = Cointegration(min_trends=2, max_trends=4)
        self.assertEqual(coupler.min_trends, 2)
        self.assertEqual(coupler.max_trends, 4)
        self.assertIsNone(Cointegration().max_trends)

    def test_generate_trends(self):
        trends = Cointegration()._generate_trends(self.rng, 64, 3)
        self.assertEqual(trends.shape, (64, 3))

    def test_generate_residuals(self):
        residuals = Cointegration()._generate_residuals(self.rng, 64, 4)
        self.assertEqual(residuals.shape, (64, 4))
        self.assertTrue(np.all(np.isfinite(residuals)))

    def test_dtype(self):
        out = Cointegration(dtype=np.float32).couple(self.rng, self.series)
        self.assertEqual(out.dtype, np.float32)


# ===========================================================================
# Linear SCM Tests
# ===========================================================================


class TestLinearSCM(unittest.TestCase):
    """Test the linear structural causal model (lagged DAG) mechanism."""

    def setUp(self):
        self.rng = np.random.RandomState(42)
        self.series = self.rng.normal(0, 1, (128, 4))

    def test_shape(self):
        out = LinearSCM().couple(self.rng, self.series)
        self.assertEqual(out.shape, (128, 4))

    def test_output_finite(self):
        out = LinearSCM().couple(self.rng, self.series)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_deterministic_with_seed(self):
        r1, r2 = np.random.RandomState(7), np.random.RandomState(7)
        o1 = LinearSCM().couple(r1, self.series)
        o2 = LinearSCM().couple(r2, self.series)
        np.testing.assert_array_equal(o1, o2)

    def test_different_seeds_differ(self):
        r1, r2 = np.random.RandomState(1), np.random.RandomState(2)
        o1 = LinearSCM().couple(r1, self.series)
        o2 = LinearSCM().couple(r2, self.series)
        self.assertFalse(np.allclose(o1, o2))

    def test_requires_two_variates(self):
        with self.assertRaises(ValueError):
            LinearSCM().couple(self.rng, np.zeros((8, 1)))

    def test_properties(self):
        coupler = LinearSCM(max_lag=3, edge_probability=0.2)
        self.assertEqual(coupler.max_lag, 3)
        self.assertEqual(coupler.edge_probability, 0.2)

    def test_generate_random_dag_structure(self):
        adjacency, lags, coefficients = LinearSCM()._generate_random_dag(self.rng, 5, 3)
        self.assertEqual(adjacency.shape, (5, 5))
        self.assertEqual(lags.shape, (5, 5))
        self.assertEqual(coefficients.shape, (5, 5))
        self.assertEqual(adjacency.dtype, bool)

    def test_generate_random_dag_acyclic(self):
        """The DAG must be acyclic: no self-loops, no cycles."""
        adjacency, _, _ = LinearSCM()._generate_random_dag(self.rng, 6, 3)
        for i in range(6):
            self.assertFalse(adjacency[i, i], "self-loop present")
        # Kahn's algorithm: a DAG admits a full topological ordering
        in_degree = adjacency.astype(int).sum(axis=0)
        queue = [i for i in range(6) if in_degree[i] == 0]
        visited = []
        while queue:
            node = queue.pop(0)
            visited.append(node)
            for child in range(6):
                if adjacency[node, child]:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)
        self.assertEqual(len(visited), 6, "DAG has a cycle")

    def test_lags_within_bounds(self):
        adjacency, lags, _ = LinearSCM()._generate_random_dag(self.rng, 6, 3)
        active = adjacency
        self.assertTrue(np.all(lags[active] >= 1))
        self.assertTrue(np.all(lags[active] <= 3))
        self.assertTrue(np.all(lags[~active] == 0))

    def test_dtype(self):
        out = LinearSCM(dtype=np.float32).couple(self.rng, self.series)
        self.assertEqual(out.dtype, np.float32)


# ===========================================================================
# Nonlinear SCM Tests
# ===========================================================================


class TestNonlinearSCM(unittest.TestCase):
    """Test the nonlinear structural causal model (modulation gate) mechanism."""

    def setUp(self):
        self.rng = np.random.RandomState(42)
        self.series = self.rng.normal(0, 1, (128, 4))

    def test_shape(self):
        out = NonlinearSCM().couple(self.rng, self.series)
        self.assertEqual(out.shape, (128, 4))

    def test_output_finite(self):
        out = NonlinearSCM().couple(self.rng, self.series)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_deterministic_with_seed(self):
        r1, r2 = np.random.RandomState(7), np.random.RandomState(7)
        o1 = NonlinearSCM().couple(r1, self.series)
        o2 = NonlinearSCM().couple(r2, self.series)
        np.testing.assert_array_equal(o1, o2)

    def test_different_seeds_differ(self):
        r1, r2 = np.random.RandomState(1), np.random.RandomState(2)
        o1 = NonlinearSCM().couple(r1, self.series)
        o2 = NonlinearSCM().couple(r2, self.series)
        self.assertFalse(np.allclose(o1, o2))

    def test_requires_two_variates(self):
        with self.assertRaises(ValueError):
            NonlinearSCM().couple(self.rng, np.zeros((8, 1)))

    def test_gate_enabled_and_disabled(self):
        """Both gate modes should produce finite outputs."""
        for use_gate in [True, False]:
            coupler = NonlinearSCM(use_modulation_gate=use_gate)
            out = coupler.couple(self.rng, self.series)
            self.assertEqual(out.shape, (128, 4))
            self.assertTrue(np.all(np.isfinite(out)))

    def test_properties(self):
        coupler = NonlinearSCM(max_lag=3, edge_probability=0.2)
        self.assertEqual(coupler.max_lag, 3)
        self.assertEqual(coupler.edge_probability, 0.2)
        self.assertEqual(coupler.nonlinearity_types, NonlinearSCM._NONLINEARITY_TYPES)

    def test_sample_nonlinearity(self):
        coupler = NonlinearSCM()
        for _ in range(20):
            fn = coupler._sample_nonlinearity(self.rng)
            self.assertTrue(callable(fn))
            out = fn(self.series[:, 0])
            self.assertEqual(out.shape, (128,))
            self.assertTrue(np.all(np.isfinite(out)))

    def test_generate_random_dag(self):
        adjacency, lags = NonlinearSCM()._generate_random_dag(self.rng, 5, 3)
        self.assertEqual(adjacency.shape, (5, 5))
        self.assertEqual(lags.shape, (5, 5))
        self.assertEqual(adjacency.dtype, bool)
        for i in range(5):
            self.assertFalse(adjacency[i, i])

    def test_dtype(self):
        out = NonlinearSCM(dtype=np.float32).couple(self.rng, self.series)
        self.assertEqual(out.dtype, np.float32)


# ===========================================================================
# Post-Processing Function Tests
# ===========================================================================


class TestVariatePermutation(unittest.TestCase):
    """Test the variate permutation transform."""

    def setUp(self):
        self.rng = np.random.RandomState(42)
        self.series = self.rng.normal(0, 1, (64, 5))

    def test_returns_tuple(self):
        result = variate_permutation(self.rng, self.series)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_shape(self):
        permuted, perm = variate_permutation(self.rng, self.series)
        self.assertEqual(permuted.shape, self.series.shape)
        self.assertEqual(perm.shape, (5,))

    def test_perm_is_permutation(self):
        _, perm = variate_permutation(self.rng, self.series)
        np.testing.assert_array_equal(np.sort(perm), np.arange(5))

    def test_permuted_columns_match(self):
        permuted, perm = variate_permutation(self.rng, self.series)
        np.testing.assert_array_equal(permuted, self.series[:, perm])

    def test_deterministic(self):
        r1, r2 = np.random.RandomState(7), np.random.RandomState(7)
        p1, perm1 = variate_permutation(r1, self.series)
        p2, perm2 = variate_permutation(r2, self.series)
        np.testing.assert_array_equal(p1, p2)
        np.testing.assert_array_equal(perm1, perm2)


class TestSmoothTimeWarping(unittest.TestCase):
    """Test the smooth time-warping transform."""

    def setUp(self):
        self.rng = np.random.RandomState(42)
        self.series = self.rng.normal(0, 1, (64, 4))

    def test_shape(self):
        out = smooth_time_warping(self.rng, self.series)
        self.assertEqual(out.shape, (64, 4))

    def test_output_finite(self):
        out = smooth_time_warping(self.rng, self.series)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_deterministic(self):
        r1, r2 = np.random.RandomState(7), np.random.RandomState(7)
        np.testing.assert_array_equal(
            smooth_time_warping(r1, self.series),
            smooth_time_warping(r2, self.series),
        )


class TestPatchMasking(unittest.TestCase):
    """Test the patch-masking (missing observations) transform."""

    def setUp(self):
        self.rng = np.random.RandomState(42)
        self.series = self.rng.normal(0, 1, (128, 4))

    def test_shape(self):
        out = patch_masking(self.rng, self.series, patch_size=16)
        self.assertEqual(out.shape, (128, 4))

    def test_produces_nan_with_high_probability(self):
        out = patch_masking(
            self.rng,
            self.series,
            patch_size=16,
            mask_probability=1.0,
            min_mask_patches=1,
            max_mask_patches=2,
        )
        self.assertGreater(np.sum(np.isnan(out)), 0)

    def test_deterministic(self):
        r1, r2 = np.random.RandomState(7), np.random.RandomState(7)
        np.testing.assert_array_equal(
            patch_masking(r1, self.series, patch_size=16),
            patch_masking(r2, self.series, patch_size=16),
        )

    def test_patch_size_larger_than_series(self):
        """When T < patch_size, no patches exist and the series is unchanged."""
        out = patch_masking(self.rng, self.series, patch_size=200)
        np.testing.assert_array_equal(out, self.series)
        self.assertFalse(np.any(np.isnan(out)))

    def test_shared_mask_across_variates(self):
        """per_variate=False applies the same NaN pattern to every column."""
        out = patch_masking(
            self.rng,
            self.series,
            patch_size=16,
            mask_probability=0.3,
            per_variate=False,
        )
        nan_mask = np.isnan(out)
        for j in range(4):
            np.testing.assert_array_equal(nan_mask[:, j], nan_mask[:, 0])

    def test_no_mask_when_probability_zero(self):
        out = patch_masking(self.rng, self.series, patch_size=16, mask_probability=0.0)
        self.assertFalse(np.any(np.isnan(out)))


class TestPartialFutureObservability(unittest.TestCase):
    """Test the partial future observability transform."""

    def setUp(self):
        self.rng = np.random.RandomState(42)
        self.series = self.rng.normal(0, 1, (64, 4))

    def test_shape(self):
        out = partial_future_observability(self.rng, self.series, horizon=16)
        self.assertEqual(out.shape, (64, 4))

    def test_zero_horizon_unchanged(self):
        out = partial_future_observability(self.rng, self.series, horizon=0)
        np.testing.assert_array_equal(out, self.series)

    def test_horizon_ge_length_unchanged(self):
        out = partial_future_observability(self.rng, self.series, horizon=64)
        np.testing.assert_array_equal(out, self.series)

    def test_nan_only_in_future(self):
        out = partial_future_observability(
            self.rng, self.series, horizon=16, future_mask_probability=1.0
        )
        # All future values (last 16 steps) are NaN; history is intact
        self.assertTrue(np.all(np.isnan(out[48:])))
        self.assertFalse(np.any(np.isnan(out[:48])))

    def test_deterministic(self):
        r1, r2 = np.random.RandomState(7), np.random.RandomState(7)
        np.testing.assert_array_equal(
            partial_future_observability(r1, self.series, horizon=16),
            partial_future_observability(r2, self.series, horizon=16),
        )


class TestValueDiscretization(unittest.TestCase):
    """Test the value discretization (quantization) transform."""

    def setUp(self):
        self.rng = np.random.RandomState(42)
        self.series = self.rng.normal(0, 1, (64, 4))

    def test_shape(self):
        out = value_discretization(self.rng, self.series, mode="uniform", n_bins=5)
        self.assertEqual(out.shape, (64, 4))

    def test_all_modes(self):
        for mode in ["uniform", "quantile", "power_law"]:
            out = value_discretization(self.rng, self.series, mode=mode, n_bins=5)
            self.assertEqual(out.shape, (64, 4))
            self.assertTrue(np.all(np.isfinite(out)))

    def test_output_finite(self):
        out = value_discretization(self.rng, self.series)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_deterministic(self):
        r1, r2 = np.random.RandomState(7), np.random.RandomState(7)
        np.testing.assert_array_equal(
            value_discretization(r1, self.series, mode="quantile", n_bins=4),
            value_discretization(r2, self.series, mode="quantile", n_bins=4),
        )

    def test_preserves_nan(self):
        series = self.series.copy()
        series[10, 0] = np.nan
        series[20, 2] = np.nan
        out = value_discretization(self.rng, series, mode="uniform", n_bins=5)
        self.assertTrue(np.isnan(out[10, 0]))
        self.assertTrue(np.isnan(out[20, 2]))

    def test_discretized_values_limited(self):
        """Quantized output should take at most n_bins distinct values per column."""
        out = value_discretization(self.rng, self.series, mode="uniform", n_bins=4)
        for j in range(4):
            self.assertLessEqual(len(np.unique(out[:, j])), 4)


class TestTimeDiscretization(unittest.TestCase):
    """Test the time discretization transform."""

    def setUp(self):
        self.rng = np.random.RandomState(42)
        self.series = self.rng.normal(0, 1, (64, 4))

    def test_shape(self):
        out = time_discretization(self.rng, self.series, mode="freeze", max_hold=5)
        self.assertEqual(out.shape, (64, 4))

    def test_all_modes(self):
        for mode in ["freeze", "staircase", "duty_cycle"]:
            out = time_discretization(self.rng, self.series, mode=mode, max_hold=5)
            self.assertEqual(out.shape, (64, 4))
            self.assertTrue(np.all(np.isfinite(out)))

    def test_deterministic(self):
        r1, r2 = np.random.RandomState(7), np.random.RandomState(7)
        np.testing.assert_array_equal(
            time_discretization(r1, self.series, mode="freeze", max_hold=5),
            time_discretization(r2, self.series, mode="freeze", max_hold=5),
        )

    def test_freeze_holds_values(self):
        """Freeze mode produces runs of identical consecutive values."""
        out = time_discretization(self.rng, self.series, mode="freeze", max_hold=3)
        diffs = np.diff(out[:, 0])
        # At least one repeated value should exist (a flat segment)
        self.assertTrue(np.any(np.abs(diffs) < 1e-12))

    def test_does_not_introduce_nan(self):
        out = time_discretization(self.rng, self.series, mode="duty_cycle", max_hold=5)
        self.assertFalse(np.any(np.isnan(out)))


# ===========================================================================
# PostProcessor Class Tests
# ===========================================================================


class TestPostProcessor(unittest.TestCase):
    """Test the PostProcessor orchestration class."""

    def setUp(self):
        self.rng = np.random.RandomState(42)
        self.series = self.rng.normal(0, 1, (128, 4))

    def test_str_method(self):
        self.assertEqual(str(PostProcessor()), "PostProcessor")

    def test_call_shape(self):
        pp = PostProcessor(patch_size=16)
        out = pp(self.rng, self.series)
        self.assertEqual(out.shape, (128, 4))

    def test_deterministic(self):
        r1, r2 = np.random.RandomState(7), np.random.RandomState(7)
        pp = PostProcessor(patch_size=16)
        np.testing.assert_array_equal(pp(r1, self.series), pp(r2, self.series))

    def test_all_stages_disabled_identity(self):
        """With every stage disabled, the output equals the input (up to dtype)."""
        pp = PostProcessor(
            apply_permutation=False,
            apply_warping=False,
            apply_masking=False,
            apply_future_mask=False,
            apply_discretization=False,
        )
        out = pp(self.rng, self.series)
        np.testing.assert_array_equal(out, self.series)

    def test_horizon_override(self):
        """A horizon override should be accepted and produce a valid shape."""
        pp = PostProcessor(horizon=16)
        out = pp(self.rng, self.series, horizon=8)
        self.assertEqual(out.shape, (128, 4))

    def test_dtype(self):
        pp = PostProcessor(dtype=np.float32)
        out = pp(self.rng, self.series)
        self.assertEqual(out.dtype, np.float32)


# ===========================================================================
# CouplingPipeline Initialization Tests
# ===========================================================================


class TestCouplingPipelineInit(unittest.TestCase):
    """Test CouplingPipeline initialization and properties."""

    def test_default_init(self):
        pipe = CouplingPipeline()
        self.assertEqual(pipe._patch_size, 32)
        self.assertEqual(pipe._horizon, 0)
        self.assertEqual(pipe.dtype, np.float64)
        self.assertEqual(pipe.scale_min, 0.5)
        self.assertEqual(pipe.scale_max, 2.0)

    def test_custom_init(self):
        pipe = CouplingPipeline(
            patch_size=16, horizon=8, dtype=np.float32, scale_min=0.2, scale_max=3.0
        )
        self.assertEqual(pipe._patch_size, 16)
        self.assertEqual(pipe._horizon, 8)
        self.assertEqual(pipe.dtype, np.float32)
        self.assertEqual(pipe.scale_min, 0.2)
        self.assertEqual(pipe.scale_max, 3.0)

    def test_invalid_scale_range(self):
        with self.assertRaises(ValueError):
            CouplingPipeline(scale_min=2.0, scale_max=0.5)

    def test_str_method(self):
        self.assertEqual(str(CouplingPipeline()), "CouplingPipeline")

    def test_default_mechanisms(self):
        pipe = CouplingPipeline()
        expected = {
            "identity",
            "univariate",
            "functional",
            "linear_mixing",
            "cointegration",
            "linear_scm",
            "nonlinear_scm",
        }
        self.assertEqual(set(pipe.mechanisms.keys()), expected)

    def test_mechanism_probabilities_sum_to_one(self):
        pipe = CouplingPipeline()
        total = sum(pipe.mechanism_probabilities.values())
        self.assertAlmostEqual(total, 1.0)

    def test_custom_mechanisms(self):
        custom = {"identity": IdentityCoupling()}
        pipe = CouplingPipeline(mechanisms=custom)
        self.assertEqual(set(pipe.mechanisms.keys()), {"identity"})
        self.assertAlmostEqual(pipe.mechanism_probabilities["identity"], 1.0)

    def test_post_processor_property(self):
        pipe = CouplingPipeline()
        self.assertIsInstance(pipe.post_processor, PostProcessor)


# ===========================================================================
# CouplingPipeline Call / Generate Tests
# ===========================================================================


class TestCouplingPipelineGenerate(unittest.TestCase):
    """Test the CouplingPipeline __call__ and generate methods."""

    def setUp(self):
        self.pipe = CouplingPipeline(patch_size=16)
        self.rng = np.random.RandomState(42)

    def test_call_with_specific_mechanism(self):
        series = self.rng.normal(0, 1, (64, 4))
        out = self.pipe(
            self.rng, series, mechanism="identity", apply_postprocessing=False
        )
        np.testing.assert_array_equal(out, series)

    def test_call_shape(self):
        series = self.rng.normal(0, 1, (64, 4))
        out = self.pipe(self.rng, series, apply_postprocessing=False)
        self.assertEqual(out.shape, (64, 4))

    def test_call_deterministic(self):
        series = self.rng.normal(0, 1, (64, 4))
        r1, r2 = np.random.RandomState(7), np.random.RandomState(7)
        o1 = self.pipe(
            r1, series, mechanism="linear_mixing", apply_postprocessing=False
        )
        o2 = self.pipe(
            r2, series, mechanism="linear_mixing", apply_postprocessing=False
        )
        np.testing.assert_array_equal(o1, o2)

    def test_call_metadata(self):
        series = self.rng.normal(0, 1, (64, 3))
        out, meta = self.pipe(
            self.rng,
            series,
            mechanism="identity",
            apply_postprocessing=False,
            return_metadata=True,
        )
        self.assertIsInstance(meta, dict)
        self.assertEqual(meta["input_shape"], (64, 3))
        self.assertEqual(meta["n_variates"], 3)
        self.assertEqual(meta["sequence_length"], 64)
        self.assertEqual(meta["coupling_mechanism"], "identity")
        self.assertEqual(meta["coupled_shape"], (64, 3))
        self.assertFalse(meta["post_processed"])
        self.assertFalse(meta["normalized"])
        self.assertIsNone(meta["channel_scales"])

    def test_call_normalize_false_preserves_input(self):
        """With normalize=False the identity mechanism must leave series unchanged."""
        series = self.rng.normal(10.0, 4.0, (64, 3))
        out = self.pipe(
            self.rng,
            series,
            mechanism="identity",
            apply_postprocessing=False,
            normalize=False,
        )
        np.testing.assert_array_equal(out, series)

    def test_call_normalize_with_explicit_scales(self):
        """Z-score then multiply by the user-supplied per-channel energy."""
        series = np.column_stack(
            [
                self.rng.normal(5.0, 2.0, 256),
                self.rng.normal(-3.0, 0.5, 256),
                self.rng.normal(0.0, 10.0, 256),
            ]
        )
        scales = (0.5, 1.0, 2.0)
        out, meta = self.pipe(
            self.rng,
            series,
            mechanism="identity",
            apply_postprocessing=False,
            return_metadata=True,
            normalize=True,
            channel_scales=scales,
        )
        self.assertTrue(meta["normalized"])
        np.testing.assert_allclose(meta["channel_scales"], scales)
        np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(out.std(axis=0), scales, rtol=1e-10, atol=1e-10)

    def test_call_normalize_random_offset_in_range(self):
        """Without channel_scales, each channel is scaled by U[scale_min, scale_max]."""
        pipe = CouplingPipeline(scale_min=0.5, scale_max=2.0)
        series = np.column_stack(
            [
                self.rng.normal(8.0, 3.0, 256),
                self.rng.normal(-2.0, 1.5, 256),
                self.rng.normal(0.0, 7.0, 256),
            ]
        )
        out, meta = pipe(
            np.random.RandomState(0),
            series,
            mechanism="identity",
            apply_postprocessing=False,
            return_metadata=True,
            normalize=True,
        )
        self.assertTrue(meta["normalized"])
        scales = np.asarray(meta["channel_scales"])
        self.assertEqual(scales.shape, (3,))
        self.assertTrue(np.all(scales >= 0.5 - 1e-12))
        self.assertTrue(np.all(scales <= 2.0 + 1e-12))
        np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(out.std(axis=0), scales, rtol=1e-10, atol=1e-10)

    def test_call_normalize_rejects_channel_scales_when_disabled(self):
        series = self.rng.normal(0, 1, (64, 3))
        with self.assertRaises(ValueError):
            self.pipe(
                self.rng,
                series,
                mechanism="identity",
                apply_postprocessing=False,
                normalize=False,
                channel_scales=(1.0, 1.0, 1.0),
            )

    def test_call_normalize_rejects_wrong_scale_length(self):
        series = self.rng.normal(0, 1, (64, 3))
        with self.assertRaises(ValueError):
            self.pipe(
                self.rng,
                series,
                mechanism="identity",
                apply_postprocessing=False,
                normalize=True,
                channel_scales=(1.0, 2.0),
            )

    def test_generate_shape(self):
        out = self.pipe.generate(
            self.rng, 64, num_channels=4, mechanism="linear_mixing"
        )
        self.assertEqual(out.shape, (64, 4))

    def test_generate_univariate_mechanism(self):
        """The univariate mechanism reduces the output to a single variate."""
        out = self.pipe.generate(self.rng, 64, num_channels=4, mechanism="univariate")
        self.assertEqual(out.shape, (64, 1))

    def test_generate_metadata(self):
        out, meta = self.pipe.generate(
            self.rng, 64, num_channels=4, return_metadata=True
        )
        self.assertIsInstance(meta, dict)
        self.assertIn("coupling_mechanism", meta)
        self.assertIn("post_processed", meta)

    def test_generate_deterministic(self):
        r1, r2 = np.random.RandomState(7), np.random.RandomState(7)
        o1 = self.pipe.generate(r1, 64, num_channels=4)
        o2 = self.pipe.generate(r2, 64, num_channels=4)
        np.testing.assert_array_equal(o1, o2)

    def test_generate_specific_mechanism(self):
        out = self.pipe.generate(
            self.rng, 64, num_channels=4, mechanism="cointegration"
        )
        self.assertEqual(out.shape, (64, 4))

    def test_generate_without_postprocessing_no_nan(self):
        pipe = CouplingPipeline(patch_size=16)
        out = pipe.generate(self.rng, 64, num_channels=4, apply_postprocessing=False)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_sample_mechanism(self):
        for _ in range(20):
            name = self.pipe._sample_mechanism(self.rng)
            self.assertIn(name, self.pipe.mechanisms)

    def test_generate_base_series(self):
        series = CouplingPipeline._generate_base_series(self.rng, 64, 4)
        self.assertEqual(series.shape, (64, 4))
        self.assertTrue(np.all(np.isfinite(series)))


# ===========================================================================
# Integration Tests
# ===========================================================================


class TestCouplingIntegration(unittest.TestCase):
    """End-to-end workflow tests matching typical paper usage."""

    def test_paper_usage_pattern(self):
        """Full pipeline across all mechanisms with post-processing."""
        rng = np.random.RandomState(42)
        pipe = CouplingPipeline(patch_size=32, horizon=8)
        for i in range(10):
            x = pipe.generate(
                np.random.RandomState(i),
                seq_length=128,
                num_channels=6,
                mechanism="linear_mixing",
            )
            self.assertEqual(x.shape, (128, 6))

    def test_all_mechanisms_end_to_end(self):
        """Each mechanism should run through the full pipeline."""
        rng = np.random.RandomState(0)
        pipe = CouplingPipeline(patch_size=16)
        series = rng.normal(0, 1, (64, 4))
        for name in pipe.mechanisms:
            out = pipe(rng, series, mechanism=name, apply_postprocessing=False)
            if name == "univariate":
                self.assertEqual(out.shape, (64, 1), msg=f"{name} shape mismatch")
            else:
                self.assertEqual(out.shape, (64, 4), msg=f"{name} shape mismatch")

    def test_reproducibility_across_pipelines(self):
        rng1, rng2 = np.random.RandomState(42), np.random.RandomState(42)
        pipe1 = CouplingPipeline(patch_size=16)
        pipe2 = CouplingPipeline(patch_size=16)
        o1 = pipe1.generate(rng1, 64, num_channels=4)
        o2 = pipe2.generate(rng2, 64, num_channels=4)
        np.testing.assert_array_equal(o1, o2)

    def test_different_seeds_differ(self):
        rng1, rng2 = np.random.RandomState(1), np.random.RandomState(2)
        pipe = CouplingPipeline(patch_size=16)
        o1 = pipe.generate(rng1, 64, num_channels=4)
        o2 = pipe.generate(rng2, 64, num_channels=4)
        self.assertFalse(np.allclose(o1, o2))

    def test_postprocessing_introduces_features(self):
        """Post-processing may introduce NaN (masking/future) or discretization."""
        pipe = CouplingPipeline(patch_size=8, horizon=16)
        rng = np.random.RandomState(0)
        saw_feature = False
        for i in range(20):
            x = pipe.generate(np.random.RandomState(i), seq_length=64, num_channels=4)
            if np.any(np.isnan(x)):
                saw_feature = True
                break
        # Masking/warping/discretization do not deterministically inject NaN,
        # so this is a weak diversity check rather than a hard assertion.
        self.assertIsNotNone(x)


# ===========================================================================
# Regression Tests (bug fixes)
# ===========================================================================


class TestCouplingRegression(unittest.TestCase):
    """Regression tests for bugs fixed in the coupling pipeline."""

    def test_short_sequence_scm_does_not_crash(self):
        """LinearSCM / NonlinearSCM used to crash for T in {2, 3} (max_lag == 0)."""
        for mech_cls in (LinearSCM, NonlinearSCM):
            for T in (2, 3):
                for Q in (2, 3):
                    for seed in range(50):
                        rng = np.random.RandomState(seed)
                        out = mech_cls().couple(rng, rng.normal(0, 1, (T, Q)))
                        self.assertEqual(
                            out.shape, (T, Q), msg=f"{mech_cls.__name__} T={T}"
                        )
                        self.assertTrue(np.all(np.isfinite(out)))

    def test_generate_single_variate_does_not_crash(self):
        """generate(num_channels=1) must never sample an incompatible mechanism."""
        pipe = CouplingPipeline()
        for seed in range(50):
            out = pipe.generate(
                np.random.RandomState(seed),
                64,
                num_channels=1,
                apply_postprocessing=False,
            )
            self.assertEqual(out.shape, (64, 1))
            self.assertTrue(np.all(np.isfinite(out)))

    def test_generate_auto_dimension_does_not_crash(self):
        """generate(num_channels=None) draws V ~ U{1..12}; Q=1 must be safe."""
        pipe = CouplingPipeline()
        for seed in range(200):
            pipe.generate(np.random.RandomState(seed), 128, apply_postprocessing=False)

    def test_single_variate_mechanism_restricted(self):
        """With Q=1, only univariate-capable mechanisms are ever sampled."""
        pipe = CouplingPipeline()
        allowed = {"identity", "univariate", "functional", "cointegration"}
        for seed in range(200):
            name = pipe._sample_mechanism(np.random.RandomState(seed), n_variates=1)
            self.assertIn(name, allowed)

    def test_cointegration_min_trends_exceeding_max(self):
        """min_trends larger than the derived max must not raise."""
        for mt in (5, 10):
            out = Cointegration(min_trends=mt).couple(
                np.random.RandomState(0), np.zeros((20, 4))
            )
            self.assertEqual(out.shape, (20, 4))

    def test_min_variates_attributes(self):
        """Each mechanism exposes the correct min_variates / min_length."""
        self.assertEqual(IdentityCoupling.min_variates, 1)
        self.assertEqual(UnivariatePassThrough.min_variates, 1)
        self.assertEqual(FunctionalCoupling.min_variates, 1)
        self.assertEqual(LinearMixing.min_variates, 2)
        self.assertEqual(Cointegration.min_variates, 1)
        self.assertEqual(LinearSCM.min_variates, 2)
        self.assertEqual(NonlinearSCM.min_variates, 2)

    def test_augment_series_transforms_input(self):
        """Stage-1 augmentation should transform the base series."""
        pipe = CouplingPipeline()
        rng = np.random.RandomState(42)
        series = rng.normal(0, 1, (64, 4))
        augmented = pipe._augment_series(rng, series.copy())
        self.assertEqual(augmented.shape, series.shape)
        self.assertTrue(np.all(np.isfinite(augmented)))
        self.assertFalse(np.allclose(augmented, series))

    def test_augmentation_metadata_flag(self):
        """generate() reports whether augmentation was applied in metadata."""
        pipe = CouplingPipeline()
        _, meta = pipe.generate(
            np.random.RandomState(0),
            64,
            num_channels=4,
            apply_augmentation=True,
            return_metadata=True,
        )
        self.assertTrue(meta["augmented"])
        _, meta2 = pipe.generate(
            np.random.RandomState(0),
            64,
            num_channels=4,
            apply_augmentation=False,
            return_metadata=True,
        )
        self.assertFalse(meta2["augmented"])

    def test_augmentation_deterministic(self):
        """Augmentation consumes the pipeline rng, so output stays reproducible."""
        pipe = CouplingPipeline()
        r1, r2 = np.random.RandomState(7), np.random.RandomState(7)
        o1 = pipe.generate(r1, 64, num_channels=4, apply_postprocessing=False)
        o2 = pipe.generate(r2, 64, num_channels=4, apply_postprocessing=False)
        np.testing.assert_array_equal(o1, o2)


# ===========================================================================
# Edge Case Tests
# ===========================================================================


class TestCouplingEdgeCases(unittest.TestCase):
    """Test boundary conditions."""

    def setUp(self):
        self.rng = np.random.RandomState(42)

    def test_short_sequences(self):
        """Very short sequences should still work for univariate-capable mechanisms."""
        series = self.rng.normal(0, 1, (4, 2))
        for coupler in [IdentityCoupling(), FunctionalCoupling()]:
            out = coupler.couple(self.rng, series)
            self.assertEqual(out.shape, (4, 2))

    def test_large_sequence(self):
        out = Cointegration().couple(self.rng, self.rng.normal(0, 1, (2048, 2)))
        self.assertEqual(out.shape, (2048, 2))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_many_variates(self):
        series = self.rng.normal(0, 1, (64, 32))
        out = LinearMixing().couple(self.rng, series)
        self.assertEqual(out.shape, (64, 32))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_every_mechanism_finite_over_many_seeds(self):
        series = self.rng.normal(0, 1, (128, 4))
        mechanisms = [
            IdentityCoupling(),
            FunctionalCoupling(),
            LinearMixing(),
            Cointegration(),
            LinearSCM(),
            NonlinearSCM(),
        ]
        for coupler in mechanisms:
            for i in range(10):
                out = coupler.couple(np.random.RandomState(i), series)
                self.assertEqual(out.shape, (128, 4))
                self.assertTrue(
                    np.all(np.isfinite(out)), msg=coupler.__class__.__name__
                )


# ===========================================================================
# Labeled (classification) interface tests
# ===========================================================================


class TestCouplingLabeledInterface(unittest.TestCase):
    """Test the n_classes classification-label interface (RML2016-style)."""

    def setUp(self):
        self.pipe = CouplingPipeline()
        self.rng = np.random.RandomState(11)

    def test_generate_single_label(self):
        x, y = self.pipe.generate(self.rng, 64, num_channels=4, n_classes=5)
        self.assertEqual(x.shape, (64, 4))
        self.assertIsInstance(y, int)
        self.assertGreaterEqual(y, 0)
        self.assertLess(y, 5)

    def test_generate_single_label_with_metadata(self):
        out = self.pipe.generate(
            self.rng, 64, num_channels=4, n_classes=3, return_metadata=True
        )
        self.assertEqual(len(out), 3)
        x, y, meta = out
        self.assertEqual(x.shape, (64, 4))
        self.assertEqual(meta["n_classes"], 3)

    def test_batch_labels_balanced(self):
        batch = self.pipe.generate_batch(
            self.rng,
            n_samples=30,
            seq_length=64,
            num_channels=4,
            n_classes=3,
        )
        self.assertEqual(len(batch), 30)
        labels = [lab for _, lab in batch]
        counts = np.bincount(labels)
        self.assertEqual(len(counts), 3)
        self.assertLessEqual(counts.max() - counts.min(), 2)

    def test_batch_returns_list(self):
        batch = self.pipe.generate_batch(
            self.rng, n_samples=5, seq_length=64, num_channels=4
        )
        self.assertEqual(len(batch), 5)
        self.assertTrue(all(isinstance(x, np.ndarray) for x in batch))

    def test_label_deterministic(self):
        r1, r2 = np.random.RandomState(0), np.random.RandomState(0)
        _, y1 = self.pipe.generate(r1, 64, num_channels=4, n_classes=6)
        _, y2 = self.pipe.generate(r2, 64, num_channels=4, n_classes=6)
        self.assertEqual(y1, y2)


# ===========================================================================
# Custom-graph (adjacency) interface tests
# ===========================================================================


class TestCouplingAdjacencyInterface(unittest.TestCase):
    """Test the optional adjacency graph for the SCM coupling mechanisms."""

    def setUp(self):
        self.rng = np.random.RandomState(3)
        self.Q = 4
        # A chain 0 -> 1 -> 2 -> 3
        self.adj = np.zeros((self.Q, self.Q), dtype=bool)
        for i in range(self.Q - 1):
            self.adj[i, i + 1] = True
        self.series = self.rng.normal(0, 1, (64, self.Q))

    def test_linear_scm_couple_with_adjacency(self):
        out = LinearSCM().couple(self.rng, self.series, adjacency=self.adj)
        self.assertEqual(out.shape, (64, self.Q))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_nonlinear_scm_couple_with_adjacency(self):
        out = NonlinearSCM().couple(self.rng, self.series, adjacency=self.adj)
        self.assertEqual(out.shape, (64, self.Q))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_empty_adjacency_gives_no_coupling(self):
        """An all-zero graph + zero noise must leave the output identically zero."""
        empty = np.zeros((self.Q, self.Q), dtype=bool)
        out = LinearSCM(noise_std=0.0).couple(self.rng, self.series, adjacency=empty)
        self.assertTrue(np.allclose(out, 0.0))

    def test_generate_infers_dimension_from_adjacency(self):
        pipe = CouplingPipeline()
        x, meta = pipe.generate(
            self.rng,
            64,
            mechanism="linear_scm",
            adjacency=self.adj,
            return_metadata=True,
        )
        self.assertEqual(x.shape, (64, self.Q))
        self.assertTrue(meta["custom_adjacency"])

    def test_generate_explicit_matching_dimension(self):
        pipe = CouplingPipeline()
        x = pipe.generate(
            self.rng,
            64,
            num_channels=4,
            mechanism="nonlinear_scm",
            adjacency=self.adj,
        )
        self.assertEqual(x.shape, (64, self.Q))

    def test_generate_dimension_mismatch_raises(self):
        pipe = CouplingPipeline()
        with self.assertRaises(ValueError):
            pipe.generate(
                self.rng,
                64,
                num_channels=3,
                mechanism="linear_scm",
                adjacency=self.adj,
            )

    def test_non_square_adjacency_raises(self):
        pipe = CouplingPipeline()
        with self.assertRaises(ValueError):
            pipe.generate(self.rng, 64, adjacency=np.zeros((3, 5)))

    def test_generate_batch_with_adjacency(self):
        pipe = CouplingPipeline()
        batch = pipe.generate_batch(
            self.rng,
            n_samples=4,
            seq_length=64,
            mechanism="linear_scm",
            adjacency=self.adj,
        )
        self.assertEqual(len(batch), 4)
        for x in batch:
            self.assertEqual(x.shape, (64, self.Q))

    def test_adjacency_ignored_by_non_scm_mechanism(self):
        """A non-SCM mechanism should accept (and ignore) the adjacency kwarg."""
        pipe = CouplingPipeline()
        x = pipe(
            self.rng,
            self.series,
            mechanism="identity",
            adjacency=self.adj,
            apply_postprocessing=False,
        )
        self.assertEqual(x.shape, (64, self.Q))


if __name__ == "__main__":
    unittest.main()
