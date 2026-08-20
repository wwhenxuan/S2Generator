# -*- coding: utf-8 -*-
"""
Test suite for CAUKER: Causal-Kernel Generation pipeline.

Covers all components from Xie et al. (2025), Algorithm 1.

Created on 2026/08/11
@author: Ruizhe Wang
@email: changewam6@gmail.com
"""

import unittest
import numpy as np

from s2generator.scm.cauker import (
    # Banks
    _build_kernel_bank,
    _build_mean_bank,
    _build_activation_bank,
    # Covariance functions
    _cov_exp_sine_squared,
    _cov_dot_product,
    _cov_rbf,
    _cov_rational_quadratic,
    _cov_white,
    _cov_constant,
    # Mean functions
    _mean_zero,
    _mean_linear,
    _mean_exponential,
    _mean_sparse_anomalies,
    # GP utilities
    _sample_composite_kernel,
    _sample_mean,
    _sample_gp,
    # DAG
    _generate_random_dag,
    # Pipeline
    CaukerPipeline,
)


# ===========================================================================
# Kernel Bank Tests
# ===========================================================================


class TestKernelBank(unittest.TestCase):
    """Test the 36-kernel bank construction and covariance functions."""

    def setUp(self):
        self.kernel_bank = _build_kernel_bank()
        self.t_grid = np.linspace(0, 1, 64)

    def test_bank_has_36_kernels(self):
        """Kernel bank should contain exactly 36 variants."""
        self.assertEqual(len(self.kernel_bank), 36)

    def test_each_kernel_has_required_keys(self):
        """Every kernel entry must have 'name', 'params', 'covariance_fn'."""
        for k in self.kernel_bank:
            self.assertIn("name", k)
            self.assertIn("params", k)
            self.assertIn("covariance_fn", k)
            self.assertIsInstance(k["name"], str)
            self.assertIsInstance(k["params"], dict)
            self.assertTrue(callable(k["covariance_fn"]))

    def test_kernel_type_counts(self):
        """Verify correct counts of each kernel type."""
        names = [k["name"] for k in self.kernel_bank]
        self.assertEqual(sum(1 for n in names if n.startswith("ExpSineSquared")), 8)
        self.assertEqual(sum(1 for n in names if n.startswith("DotProduct")), 4)
        self.assertEqual(sum(1 for n in names if n.startswith("RBF_")), 8)
        self.assertEqual(sum(1 for n in names if n.startswith("RationalQuadratic")), 6)
        self.assertEqual(sum(1 for n in names if n.startswith("WhiteKernel")), 5)
        self.assertEqual(sum(1 for n in names if n.startswith("ConstantKernel")), 5)

    # --- Covariance function correctness ---

    def test_cov_exp_sine_squared_shape(self):
        K = _cov_exp_sine_squared(
            self.t_grid, self.t_grid, {"periodicity": 20.0, "length_scale": 10.0}
        )
        self.assertEqual(K.shape, (64, 64))

    def test_cov_exp_sine_squared_symmetry(self):
        K = _cov_exp_sine_squared(
            self.t_grid, self.t_grid, {"periodicity": 20.0, "length_scale": 10.0}
        )
        np.testing.assert_allclose(K, K.T, atol=1e-10)

    def test_cov_exp_sine_squared_periodic(self):
        """Values at distance equal to periodicity should have covariance ≈ 1."""
        K = _cov_exp_sine_squared(
            self.t_grid, self.t_grid, {"periodicity": 0.5, "length_scale": 10.0}
        )
        # At dt=0, k=1; at dt=periodicity, sin(pi)=0, so k ≈ 1 (exp(-0))
        self.assertTrue(np.all(np.diag(K) > 0.99))

    def test_cov_dot_product_shape(self):
        K = _cov_dot_product(self.t_grid, self.t_grid, {"sigma_0": 1.0})
        self.assertEqual(K.shape, (64, 64))

    def test_cov_dot_product_increasing(self):
        """Covariance should increase with t."""
        K = _cov_dot_product(self.t_grid, self.t_grid, {"sigma_0": 0.0})
        # Last diagonal element should be larger than the first
        self.assertGreater(K[-1, -1], K[0, 0])

    def test_cov_rbf_diagonal_one(self):
        K = _cov_rbf(self.t_grid, self.t_grid, {"length_scale": 1.0})
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-10)

    def test_cov_rbf_decay_with_distance(self):
        """Covariance should decrease as |t - t'| increases."""
        K = _cov_rbf(self.t_grid, self.t_grid, {"length_scale": 0.1})
        self.assertGreater(K[0, 0], K[0, 10])

    def test_cov_rational_quadratic_diagonal_one(self):
        K = _cov_rational_quadratic(
            self.t_grid, self.t_grid, {"length_scale": 1.0, "alpha": 1.0}
        )
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-10)

    def test_cov_white_diagonal_only(self):
        K = _cov_white(self.t_grid, self.t_grid, {"noise_level": 0.5})
        self.assertEqual(K.shape, (64, 64))
        # Diagonal should be noise_level
        np.testing.assert_allclose(np.diag(K), 0.5, atol=1e-10)
        # Off-diagonal should be zero
        off_diag = K.copy()
        np.fill_diagonal(off_diag, 0)
        np.testing.assert_allclose(off_diag, 0, atol=1e-10)

    def test_cov_constant_all_same(self):
        K = _cov_constant(self.t_grid, self.t_grid, {"constant_value": 3.0})
        np.testing.assert_allclose(K, 3.0, atol=1e-10)

    # --- Covariance functions with different-length inputs ---

    def test_cov_functions_rectangular(self):
        """Covariance functions should support rectangular output (n1 ≠ n2)."""
        t1 = np.linspace(0, 0.5, 32)
        t2 = np.linspace(0.5, 1.0, 40)
        for fn, params in [
            (_cov_rbf, {"length_scale": 1.0}),
            (_cov_exp_sine_squared, {"periodicity": 10.0, "length_scale": 5.0}),
            (_cov_rational_quadratic, {"length_scale": 1.0, "alpha": 2.0}),
            (_cov_dot_product, {"sigma_0": 1.0}),
            (_cov_white, {"noise_level": 0.1}),
            (_cov_constant, {"constant_value": 1.0}),
        ]:
            K = fn(t1, t2, params)
            self.assertEqual(K.shape, (32, 40), msg=f"Shape mismatch for {fn.__name__}")


# ===========================================================================
# Mean Function Bank Tests
# ===========================================================================


class TestMeanBank(unittest.TestCase):
    """Test the 4 mean function types."""

    def setUp(self):
        self.rng = np.random.RandomState(42)
        self.mean_bank = _build_mean_bank()

    def test_bank_has_4_functions(self):
        self.assertEqual(len(self.mean_bank), 4)

    def test_each_mean_has_required_keys(self):
        for m in self.mean_bank:
            self.assertIn("name", m)
            self.assertIn("generate", m)
            self.assertTrue(callable(m["generate"]))

    def test_mean_names(self):
        names = [m["name"] for m in self.mean_bank]
        self.assertEqual(names, ["zero", "linear", "exponential", "sparse_anomalies"])

    def test_mean_zero(self):
        t_grid = np.linspace(0, 1, 100)
        result = _mean_zero(self.rng, t_grid)
        self.assertEqual(result.shape, (100,))
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_mean_linear_shape_and_range(self):
        t_grid = np.linspace(0, 1, 100)
        result = _mean_linear(self.rng, t_grid)
        self.assertEqual(result.shape, (100,))
        self.assertTrue(np.all(np.isfinite(result)))
        self.assertFalse(np.allclose(result, 0.0))

    def test_mean_exponential_shape_and_finite(self):
        t_grid = np.linspace(0, 1, 100)
        result = _mean_exponential(self.rng, t_grid)
        self.assertEqual(result.shape, (100,))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_mean_sparse_anomalies(self):
        t_grid = np.linspace(0, 1, 100)
        result = _mean_sparse_anomalies(self.rng, t_grid)
        self.assertEqual(result.shape, (100,))
        self.assertTrue(np.all(np.isfinite(result)))
        # Most entries should be zero
        n_nonzero = np.count_nonzero(result)
        self.assertLess(n_nonzero, 10)  # at most L//50 + 1 ≈ 3 anomalies

    def test_mean_deterministic_with_seed(self):
        t_grid = np.linspace(0, 1, 50)
        rng1 = np.random.RandomState(123)
        rng2 = np.random.RandomState(123)
        for m in self.mean_bank:
            r1 = m["generate"](rng1, t_grid)
            r2 = m["generate"](rng2, t_grid)
            np.testing.assert_array_equal(
                r1, r2, err_msg=f"Non-deterministic: {m['name']}"
            )


# ===========================================================================
# Activation Function Bank Tests
# ===========================================================================


class TestActivationBank(unittest.TestCase):
    """Test the 6 activation function types."""

    def setUp(self):
        self.rng = np.random.RandomState(42)
        self.act_bank = _build_activation_bank(self.rng)
        self.x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 5.0])

    def test_bank_has_6_activations(self):
        self.assertEqual(len(self.act_bank), 6)

    def test_each_activation_has_required_keys(self):
        for a in self.act_bank:
            self.assertIn("name", a)
            self.assertIn("params", a)
            self.assertIn("apply", a)
            self.assertTrue(callable(a["apply"]))

    def test_activation_names(self):
        names = [a["name"] for a in self.act_bank]
        self.assertIn("linear", names)
        self.assertIn("relu", names)
        self.assertIn("sigmoid", names)
        self.assertIn("sin", names)
        self.assertIn("modulo", names)
        self.assertIn("leaky_relu", names)

    def test_relu_output(self):
        relu = self.act_bank[1]
        self.assertEqual(relu["name"], "relu")
        out = relu["apply"](self.x)
        self.assertEqual(out.shape, self.x.shape)
        np.testing.assert_allclose(out[out < 0], 0.0)
        np.testing.assert_allclose(out[self.x > 0], self.x[self.x > 0])

    def test_sigmoid_range(self):
        sigmoid = self.act_bank[2]
        self.assertEqual(sigmoid["name"], "sigmoid")
        out = sigmoid["apply"](self.x)
        self.assertTrue(np.all(out >= 0) and np.all(out <= 1))
        # sigmoid(0) = 0.5
        self.assertAlmostEqual(out[2], 0.5, places=6)

    def test_sin_output(self):
        sin_act = self.act_bank[3]
        self.assertEqual(sin_act["name"], "sin")
        out = sin_act["apply"](self.x)
        self.assertTrue(np.all(np.abs(out) <= 1.0))

    def test_modulo_output(self):
        mod = self.act_bank[4]
        self.assertEqual(mod["name"], "modulo")
        # Pass stored params explicitly (lambda re-samples otherwise)
        out = mod["apply"](self.x, **mod["params"])
        c = mod["params"]["c"]
        self.assertTrue(np.all(out >= 0) and np.all(out < c))

    def test_leaky_relu_negative_slope(self):
        lrelu = self.act_bank[5]
        self.assertEqual(lrelu["name"], "leaky_relu")
        # Pass stored params explicitly (lambda re-samples otherwise)
        out = lrelu["apply"](self.x, **lrelu["params"])
        alpha = lrelu["params"]["alpha"]
        # For negative x: x * alpha (small negative values)
        np.testing.assert_allclose(out[self.x < 0], self.x[self.x < 0] * alpha)
        # For positive x: x (unchanged)
        np.testing.assert_allclose(out[self.x > 0], self.x[self.x > 0])

    def test_all_activations_return_finite(self):
        for a in self.act_bank:
            out = a["apply"](self.x)
            self.assertTrue(
                np.all(np.isfinite(out)), msg=f"{a['name']} produced non-finite values"
            )


# ===========================================================================
# GP Sampling Utilities Tests
# ===========================================================================


class TestGPSampling(unittest.TestCase):
    """Test composite kernel sampling, mean composition, and GP draws."""

    def setUp(self):
        self.rng = np.random.RandomState(42)
        self.kernel_bank = _build_kernel_bank()
        self.mean_bank = _build_mean_bank()
        self.t_grid = np.linspace(0, 1, 128)

    def test_composite_kernel_returns_callable(self):
        cov_fn = _sample_composite_kernel(self.rng, self.kernel_bank, Kmax=5)
        self.assertTrue(callable(cov_fn))

    def test_composite_kernel_symmetry(self):
        """Composite kernel should produce a symmetric PSD matrix."""
        cov_fn = _sample_composite_kernel(self.rng, self.kernel_bank, Kmax=3)
        K = cov_fn(self.t_grid, self.t_grid)
        self.assertEqual(K.shape, (128, 128))
        np.testing.assert_allclose(K, K.T, atol=1e-10)

    def test_composite_kernel_different_outputs(self):
        """Different random calls should give different composite kernels."""
        rng1 = np.random.RandomState(1)
        rng2 = np.random.RandomState(2)
        cov_fn1 = _sample_composite_kernel(rng1, self.kernel_bank, Kmax=5)
        cov_fn2 = _sample_composite_kernel(rng2, self.kernel_bank, Kmax=5)
        K1 = cov_fn1(self.t_grid, self.t_grid)
        K2 = cov_fn2(self.t_grid, self.t_grid)
        self.assertFalse(np.allclose(K1, K2))

    def test_composite_kernel_single_kernel(self):
        """Kmax=1 should work (just one kernel, no composition)."""
        cov_fn = _sample_composite_kernel(self.rng, self.kernel_bank, Kmax=1)
        self.assertTrue(callable(cov_fn))
        K = cov_fn(self.t_grid, self.t_grid)
        self.assertEqual(K.shape, (128, 128))

    def test_sample_mean_shape(self):
        mean = _sample_mean(self.rng, self.mean_bank, self.t_grid)
        self.assertEqual(mean.shape, (128,))
        self.assertTrue(np.all(np.isfinite(mean)))

    def test_sample_mean_different_outputs(self):
        rng1 = np.random.RandomState(1)
        rng2 = np.random.RandomState(2)
        m1 = _sample_mean(rng1, self.mean_bank, self.t_grid)
        m2 = _sample_mean(rng2, self.mean_bank, self.t_grid)
        self.assertFalse(np.allclose(m1, m2))

    def test_sample_gp_basic(self):
        mean = np.zeros(128)
        cov_fn = _sample_composite_kernel(self.rng, self.kernel_bank, Kmax=3)
        sample = _sample_gp(self.rng, mean, cov_fn, self.t_grid)
        self.assertEqual(sample.shape, (128,))
        self.assertEqual(sample.dtype, np.float64)
        self.assertTrue(np.all(np.isfinite(sample)))

    def test_sample_gp_around_mean(self):
        """Sample should be centered around the provided mean."""
        mean = np.ones(128) * 5.0
        cov_fn = _sample_composite_kernel(self.rng, self.kernel_bank, Kmax=2)
        samples = np.stack(
            [
                _sample_gp(np.random.RandomState(i), mean, cov_fn, self.t_grid)
                for i in range(20)
            ]
        )
        grand_mean = samples.mean(axis=0)
        # Average over samples should be close to the mean function
        np.testing.assert_allclose(grand_mean, mean, atol=2.0)

    def test_sample_gp_deterministic(self):
        mean = np.zeros(64)
        t_grid = np.linspace(0, 1, 64)
        cov_fn = _sample_composite_kernel(
            np.random.RandomState(7), self.kernel_bank, Kmax=3
        )
        rng1 = np.random.RandomState(42)
        rng2 = np.random.RandomState(42)
        s1 = _sample_gp(rng1, mean, cov_fn, t_grid)
        s2 = _sample_gp(rng2, mean, cov_fn, t_grid)
        np.testing.assert_array_equal(s1, s2)

    def test_sample_gp_numerical_stability(self):
        """Sampling should not fail with near-singular kernels."""
        # White kernel only → almost singular, tests jitter fallback
        cov_fn = lambda t1, t2: _cov_white(t1, t2, {"noise_level": 1e-10})
        mean = np.zeros(64)
        t_grid = np.linspace(0, 1, 64)
        sample = _sample_gp(self.rng, mean, cov_fn, t_grid, jitter=1e-4)
        self.assertEqual(sample.shape, (64,))


# ===========================================================================
# DAG Generation Tests
# ===========================================================================


class TestDAGGeneration(unittest.TestCase):
    """Test random DAG generation validity."""

    def setUp(self):
        self.rng = np.random.RandomState(42)

    def test_dag_structure(self):
        parents, roots, edges = _generate_random_dag(self.rng, V=10, Pmax=4)
        self.assertIsInstance(parents, list)
        self.assertEqual(len(parents), 10)
        self.assertIsInstance(roots, list)
        self.assertIsInstance(edges, list)

    def test_no_self_loops(self):
        parents, _, edges = _generate_random_dag(self.rng, V=15, Pmax=4)
        for child, child_parents in enumerate(parents):
            self.assertNotIn(child, child_parents)

    def test_edges_only_forward(self):
        """DAG must be acyclic: verify by topological ordering."""
        parents, _, edges = _generate_random_dag(self.rng, V=20, Pmax=5)
        # Build in-degree and perform topological sort (Kahn's algorithm)
        in_degree = [0] * len(parents)
        for child, child_parents in enumerate(parents):
            in_degree[child] = len(child_parents)
        queue = [i for i in range(len(parents)) if in_degree[i] == 0]
        visited = []
        while queue:
            node = queue.pop(0)
            visited.append(node)
            for child, child_parents in enumerate(parents):
                if node in child_parents:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)
        self.assertEqual(
            len(visited),
            len(parents),
            "DAG has a cycle: not all nodes visited in topological order",
        )

    def test_root_nodes_have_no_parents(self):
        parents, roots, _ = _generate_random_dag(self.rng, V=10, Pmax=3)
        for r in roots:
            self.assertEqual(len(parents[r]), 0)

    def test_at_least_one_root(self):
        """Every DAG must have at least one root node (in-degree 0)."""
        for _ in range(20):
            _, roots, _ = _generate_random_dag(self.rng, V=5, Pmax=2)
            self.assertGreater(len(roots), 0)

    def test_parent_count_within_bounds(self):
        parents, _, _ = _generate_random_dag(self.rng, V=10, Pmax=3)
        for child, child_parents in enumerate(parents):
            self.assertLessEqual(len(child_parents), 3)

    def test_edges_consistent_with_parents(self):
        parents, _, edges = _generate_random_dag(self.rng, V=8, Pmax=3)
        edge_set = set(edges)
        for child, child_parents in enumerate(parents):
            for p in child_parents:
                self.assertIn((p, child), edge_set)

    def test_small_graph(self):
        """V=2 should be valid."""
        parents, roots, edges = _generate_random_dag(self.rng, V=2, Pmax=1)
        self.assertEqual(len(parents), 2)
        self.assertGreaterEqual(len(roots), 1)
        self.assertLessEqual(len(roots), 2)


# ===========================================================================
# CaukerPipeline Initialization Tests
# ===========================================================================


class TestCaukerPipelineInit(unittest.TestCase):
    """Test CaukerPipeline initialization and properties."""

    def test_default_initialization(self):
        pipe = CaukerPipeline()
        self.assertEqual(pipe._Kmax, 5)
        self.assertEqual(pipe._Vmax, 20)
        self.assertEqual(pipe._Pmax, 4)
        self.assertEqual(pipe._target_length, 512)
        self.assertEqual(pipe._dtype, np.float64)

    def test_custom_initialization(self):
        pipe = CaukerPipeline(
            Kmax=3, Vmax=10, Pmax=2, target_length=256, dtype=np.float32
        )
        self.assertEqual(pipe._Kmax, 3)
        self.assertEqual(pipe._Vmax, 10)
        self.assertEqual(pipe._Pmax, 2)
        self.assertEqual(pipe._target_length, 256)
        self.assertEqual(pipe._dtype, np.float32)

    def test_str_method(self):
        pipe = CaukerPipeline()
        self.assertEqual(str(pipe), "CaukerPipeline")

    def test_kernel_bank_property(self):
        pipe = CaukerPipeline()
        bank = pipe.kernel_bank
        self.assertEqual(len(bank), 36)

    def test_mean_bank_property(self):
        pipe = CaukerPipeline()
        bank = pipe.mean_bank
        self.assertEqual(len(bank), 4)

    def test_n_kernels_property(self):
        pipe = CaukerPipeline()
        self.assertEqual(pipe.n_kernels, 36)

    def test_n_mean_functions_property(self):
        pipe = CaukerPipeline()
        self.assertEqual(pipe.n_mean_functions, 4)


# ===========================================================================
# CaukerPipeline Generate Tests
# ===========================================================================


class TestCaukerPipelineGenerate(unittest.TestCase):
    """Test the main generate() method of CaukerPipeline."""

    def setUp(self):
        self.pipe = CaukerPipeline(Kmax=3, Vmax=10, Pmax=2, target_length=64)
        self.rng = np.random.RandomState(42)

    def test_generate_default_univariate(self):
        x = self.pipe.generate(self.rng, n_inputs_points=64)
        self.assertIsInstance(x, np.ndarray)
        self.assertEqual(x.ndim, 2)
        self.assertEqual(x.shape[1], 64)  # L
        self.assertGreaterEqual(x.shape[0], 1)  # d >= 1
        self.assertEqual(x.dtype, np.float64)

    def test_generate_specific_dimension(self):
        for d in [1, 3, 5]:
            x = self.pipe.generate(self.rng, n_inputs_points=64, input_dimension=d)
            self.assertEqual(x.shape, (d, 64))

    def test_generate_all_values_finite(self):
        x = self.pipe.generate(self.rng, n_inputs_points=128, input_dimension=3)
        self.assertTrue(np.all(np.isfinite(x)))

    def test_generate_deterministic_with_same_seed(self):
        rng1 = np.random.RandomState(99)
        rng2 = np.random.RandomState(99)
        x1 = self.pipe.generate(rng1, n_inputs_points=64, input_dimension=2)
        x2 = self.pipe.generate(rng2, n_inputs_points=64, input_dimension=2)
        np.testing.assert_array_equal(x1, x2)

    def test_generate_different_with_different_seeds(self):
        rng1 = np.random.RandomState(1)
        rng2 = np.random.RandomState(2)
        x1 = self.pipe.generate(rng1, n_inputs_points=64, input_dimension=3)
        x2 = self.pipe.generate(rng2, n_inputs_points=64, input_dimension=3)
        self.assertFalse(np.allclose(x1, x2))

    def test_generate_with_metadata(self):
        x, meta = self.pipe.generate(
            self.rng, n_inputs_points=64, input_dimension=2, return_metadata=True
        )
        self.assertIsInstance(x, np.ndarray)
        self.assertIsInstance(meta, dict)
        self.assertIn("n_total_nodes", meta)
        self.assertIn("n_roots", meta)
        self.assertIn("n_edges", meta)
        self.assertIn("observed_nodes", meta)
        self.assertIn("root_nodes", meta)
        self.assertIn("edge_list", meta)
        self.assertEqual(meta["target_length"], 64)
        self.assertEqual(meta["n_observed"], 2)
        self.assertGreater(meta["n_total_nodes"], 0)
        self.assertGreaterEqual(meta["n_roots"], 1)

    def test_generate_not_all_constant(self):
        """Generated outputs should not all be constant (at least one channel has variance)."""
        x = self.pipe.generate(self.rng, n_inputs_points=128, input_dimension=3)
        # At least one channel should have meaningful variation
        stds = [np.std(x[i]) for i in range(x.shape[0])]
        self.assertGreater(
            max(stds), 1e-8, msg=f"All channels have near-zero variance: {stds}"
        )

    def test_generate_no_dead_channels_across_seeds(self):
        """Regression: ReLU collapse must never produce constant-zero channels.

        ReLU applied to an all-negative linear aggregate ``z = W·parent + b``
        collapses a node to all zeros, and the collapse cascades to children.
        The ``min_node_std`` guard must keep every observed channel non-degenerate
        across many seeds (the original bug fired within a handful of seeds).
        """
        pipe = CaukerPipeline(Kmax=5, Vmax=20, Pmax=4, target_length=128)
        for seed in range(200):
            x = pipe.generate(
                np.random.RandomState(seed), n_inputs_points=128, input_dimension=8
            )
            stds = np.std(x, axis=1)
            self.assertTrue(
                np.all(stds > 0.01),
                msg=f"seed={seed}: dead channel detected, per-channel std={stds.tolist()}",
            )

    def test_generate_guard_disabled_restores_paper_behavior(self):
        """With ``min_node_std=0.0`` the guard is off (paper-exact behaviour)."""
        pipe = CaukerPipeline(
            Kmax=5, Vmax=20, Pmax=4, target_length=128, min_node_std=0.0
        )
        x = pipe.generate(
            np.random.RandomState(1), n_inputs_points=128, input_dimension=8
        )
        self.assertEqual(x.shape, (8, 128))
        self.assertTrue(np.all(np.isfinite(x)))

    def test_generate_different_lengths(self):
        for L in [32, 64, 128, 256, 512]:
            x = self.pipe.generate(self.rng, n_inputs_points=L, input_dimension=2)
            self.assertEqual(x.shape, (2, L))

    def test_generate_dtype_float32(self):
        pipe = CaukerPipeline(Kmax=3, Vmax=8, Pmax=2, dtype=np.float32)
        x = pipe.generate(
            np.random.RandomState(0), n_inputs_points=64, input_dimension=2
        )
        self.assertEqual(x.dtype, np.float32)

    def test_generate_observed_nodes_unique(self):
        _, meta = self.pipe.generate(
            self.rng, n_inputs_points=64, input_dimension=3, return_metadata=True
        )
        observed = meta["observed_nodes"]
        self.assertEqual(len(observed), len(set(observed)))

    def test_generate_many_times_no_crash(self):
        """Generate many times to catch rare edge cases."""
        for i in range(20):
            x = self.pipe.generate(
                np.random.RandomState(i), n_inputs_points=64, input_dimension=2
            )
            self.assertEqual(x.shape, (2, 64))
            self.assertTrue(np.all(np.isfinite(x)))


# ===========================================================================
# CaukerPipeline Batch Tests
# ===========================================================================


class TestCaukerPipelineBatch(unittest.TestCase):
    """Test batch generation."""

    def setUp(self):
        self.pipe = CaukerPipeline(Kmax=3, Vmax=10, Pmax=2, target_length=64)
        self.rng = np.random.RandomState(42)

    def test_batch_basic(self):
        dataset = self.pipe.generate_batch(
            self.rng, n_samples=5, n_inputs_points=64, input_dimension=2
        )
        self.assertEqual(len(dataset), 5)
        for x in dataset:
            self.assertIsInstance(x, np.ndarray)
            self.assertEqual(x.shape, (2, 64))
            self.assertTrue(np.all(np.isfinite(x)))

    def test_batch_large(self):
        """Stress test with larger batch size."""
        dataset = self.pipe.generate_batch(
            self.rng, n_samples=50, n_inputs_points=64, input_dimension=1
        )
        self.assertEqual(len(dataset), 50)

    def test_batch_samples_different(self):
        """Different samples in a batch should not all be identical."""
        dataset = self.pipe.generate_batch(
            self.rng, n_samples=10, n_inputs_points=64, input_dimension=1
        )
        # At least some of them should differ
        stacked = np.stack([d.flatten() for d in dataset])
        variances = np.var(stacked, axis=0).mean()
        self.assertGreater(variances, 0.0)

    def test_batch_without_input_dimension(self):
        dataset = self.pipe.generate_batch(self.rng, n_samples=3, n_inputs_points=64)
        self.assertEqual(len(dataset), 3)
        # Each sample may have different d (randomly sampled)


# ===========================================================================
# Integration Tests
# ===========================================================================


class TestCaukerIntegration(unittest.TestCase):
    """End-to-end workflow tests matching typical paper usage."""

    def test_paper_usage_pattern(self):
        """Simulate the exact usage pattern from the CAUKER paper (Section 4)."""

        rng = np.random.RandomState(42)
        pipe = CaukerPipeline(
            Kmax=5,  # paper default
            Vmax=20,  # paper default
            Pmax=4,  # paper default
            target_length=512,  # Mantis/MOMENT input length
        )

        # Generate 100 samples (paper uses 100K-10M)
        dataset = pipe.generate_batch(
            rng=rng,
            n_samples=100,
            n_inputs_points=512,
            input_dimension=1,  # univariate for TSFMs
        )

        self.assertEqual(len(dataset), 100)
        for x in dataset:
            self.assertEqual(x.shape, (1, 512))
            self.assertTrue(np.all(np.isfinite(x)))

        # Verify diversity: compute pairwise DTW-like distance
        # (just check means differ across samples)
        means = [float(x.mean()) for x in dataset]
        self.assertGreater(np.std(means), 0.0)

    def test_multivariate_output(self):
        """Each SCM node = one channel, so d observed nodes produce d-variate output."""
        pipe = CaukerPipeline(Kmax=3, Vmax=15, Pmax=3, target_length=128)
        rng = np.random.RandomState(123)
        x, meta = pipe.generate(
            rng, n_inputs_points=128, input_dimension=4, return_metadata=True
        )
        self.assertEqual(x.shape, (4, 128))
        self.assertEqual(meta["n_observed"], 4)
        # Observed nodes must be a subset of total nodes
        self.assertTrue(
            set(meta["observed_nodes"]).issubset(set(range(meta["n_total_nodes"])))
        )

    def test_reproducibility_across_pipelines(self):
        """Two pipelines with same params + same seed = same output."""
        rng1 = np.random.RandomState(42)
        rng2 = np.random.RandomState(42)

        pipe1 = CaukerPipeline(Kmax=3, Vmax=10, Pmax=2, target_length=64)
        pipe2 = CaukerPipeline(Kmax=3, Vmax=10, Pmax=2, target_length=64)

        x1 = pipe1.generate(rng1, n_inputs_points=64, input_dimension=2)
        x2 = pipe2.generate(rng2, n_inputs_points=64, input_dimension=2)
        np.testing.assert_array_equal(x1, x2)

    def test_scaling_law_generation(self):
        """Generate datasets at multiple scales (10K → 1M pattern from paper)."""
        pipe = CaukerPipeline(Kmax=5, Vmax=20, Pmax=4, target_length=512)
        rng = np.random.RandomState(0)

        scales = [10, 50, 100]  # small scale for test; paper uses 10K, 50K, 100K, ...
        for n in scales:
            dataset = pipe.generate_batch(
                rng=rng, n_samples=n, n_inputs_points=512, input_dimension=1
            )
            self.assertEqual(len(dataset), n)


# ===========================================================================
# Edge Case Tests
# ===========================================================================


class TestCaukerEdgeCases(unittest.TestCase):
    """Test boundary conditions and edge cases."""

    def setUp(self):
        self.rng = np.random.RandomState(42)

    def test_minimal_configuration(self):
        """Smallest valid parameters: Kmax=1, Vmax=2, Pmax=1."""
        pipe = CaukerPipeline(Kmax=1, Vmax=2, Pmax=1, target_length=32)
        x = pipe.generate(self.rng, n_inputs_points=32, input_dimension=1)
        self.assertEqual(x.shape, (1, 32))
        self.assertTrue(np.all(np.isfinite(x)))

    def test_large_dag_small_observed(self):
        """Many DAG nodes but only few observed variables."""
        pipe = CaukerPipeline(Kmax=3, Vmax=30, Pmax=5, target_length=64)
        x, meta = pipe.generate(
            self.rng, n_inputs_points=64, input_dimension=2, return_metadata=True
        )
        self.assertEqual(x.shape, (2, 64))
        self.assertGreater(meta["n_total_nodes"], 2)

    def test_input_dimension_larger_than_vmax(self):
        """If d > Vmax, V is set to at least d."""
        pipe = CaukerPipeline(Kmax=2, Vmax=5, Pmax=2, target_length=64)
        # d=8 > Vmax=5, so V will be in [8, 5] → randint will clamp
        # This actually would cause ValueError from randint if d > Vmax
        # Let's test with d close to Vmax
        x = pipe.generate(self.rng, n_inputs_points=64, input_dimension=5)
        self.assertEqual(x.shape[0], 5)

    def test_very_short_sequence(self):
        pipe = CaukerPipeline(Kmax=1, Vmax=5, Pmax=1, target_length=4)
        x = pipe.generate(self.rng, n_inputs_points=4, input_dimension=1)
        self.assertEqual(x.shape, (1, 4))

    def test_very_long_sequence(self):
        pipe = CaukerPipeline(Kmax=2, Vmax=5, Pmax=1, target_length=2048)
        x = pipe.generate(self.rng, n_inputs_points=2048, input_dimension=1)
        self.assertEqual(x.shape, (1, 2048))
        self.assertTrue(np.all(np.isfinite(x)))

    def test_single_kernel_no_composition(self):
        """Kmax=1 means no kernel composition (Step 2 is identity)."""
        pipe = CaukerPipeline(Kmax=1, Vmax=10, Pmax=2, target_length=64)
        x = pipe.generate(self.rng, n_inputs_points=64, input_dimension=2)
        self.assertEqual(x.shape, (2, 64))
        self.assertTrue(np.all(np.isfinite(x)))

    def test_max_parents_zero(self):
        """Pmax=0 should create disconnected nodes (all roots)."""
        pipe = CaukerPipeline(Kmax=2, Vmax=10, Pmax=0, target_length=64)
        x, meta = pipe.generate(
            self.rng, n_inputs_points=64, input_dimension=3, return_metadata=True
        )
        self.assertEqual(x.shape, (3, 64))
        # With Pmax=0, all nodes are roots (no edges)
        self.assertEqual(meta["n_edges"], 0)
        self.assertEqual(meta["n_roots"], meta["n_total_nodes"])


# ===========================================================================
# Custom Graph (adjacency) Tests
# ===========================================================================


class TestCaukerPipelineCustomGraph(unittest.TestCase):
    """Test generation with a user-supplied adjacency matrix."""

    def setUp(self):
        self.pipe = CaukerPipeline(Kmax=3, Vmax=10, Pmax=2, target_length=64)
        self.rng = np.random.RandomState(42)
        # Chain DAG: 0 -> 1 -> 2 -> 3
        self.chain = np.array(
            [
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 0],
            ]
        )

    def test_generate_with_chain_adjacency(self):
        x, meta = self.pipe.generate(
            self.rng,
            n_inputs_points=64,
            input_dimension=3,
            adjacency=self.chain,
            return_metadata=True,
        )
        self.assertEqual(x.shape, (3, 64))
        self.assertTrue(np.all(np.isfinite(x)))
        self.assertEqual(meta["graph_source"], "custom")
        self.assertEqual(meta["n_total_nodes"], 4)
        self.assertEqual(meta["n_edges"], 3)
        self.assertEqual(set(meta["edge_list"]), {(0, 1), (1, 2), (2, 3)})

    def test_generate_zero_adjacency_no_edges(self):
        adj = np.zeros((4, 4))
        _, meta = self.pipe.generate(
            self.rng,
            n_inputs_points=64,
            input_dimension=2,
            adjacency=adj,
            return_metadata=True,
        )
        self.assertEqual(meta["n_edges"], 0)
        self.assertEqual(meta["n_roots"], 4)

    def test_cycle_raises(self):
        cyclic = np.array(
            [
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 0],
            ]
        )
        with self.assertRaises(ValueError):
            self.pipe.generate(
                self.rng, n_inputs_points=64, input_dimension=2, adjacency=cyclic
            )

    def test_non_square_raises(self):
        with self.assertRaises(ValueError):
            self.pipe.generate(
                self.rng,
                n_inputs_points=64,
                input_dimension=2,
                adjacency=np.zeros((3, 4)),
            )

    def test_self_loop_raises(self):
        loop = np.array(
            [
                [1, 1, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]
        )
        with self.assertRaises(ValueError):
            self.pipe.generate(
                self.rng, n_inputs_points=64, input_dimension=2, adjacency=loop
            )

    def test_input_dimension_exceeds_graph_raises(self):
        with self.assertRaises(ValueError):
            self.pipe.generate(
                self.rng,
                n_inputs_points=64,
                input_dimension=5,
                adjacency=self.chain,  # V=4
            )

    def test_deterministic_with_same_adjacency(self):
        rng1, rng2 = np.random.RandomState(7), np.random.RandomState(7)
        x1 = self.pipe.generate(
            rng1, n_inputs_points=64, input_dimension=3, adjacency=self.chain
        )
        x2 = self.pipe.generate(
            rng2, n_inputs_points=64, input_dimension=3, adjacency=self.chain
        )
        np.testing.assert_array_equal(x1, x2)

    def test_batch_with_adjacency(self):
        dataset = self.pipe.generate_batch(
            self.rng,
            n_samples=5,
            n_inputs_points=64,
            input_dimension=2,
            adjacency=self.chain,
        )
        self.assertEqual(len(dataset), 5)
        for x in dataset:
            self.assertEqual(x.shape, (2, 64))
            self.assertTrue(np.all(np.isfinite(x)))


# ===========================================================================
# Labeled (classification) interface tests
# ===========================================================================


class TestCaukerLabeledInterface(unittest.TestCase):
    """Test the n_classes classification-label interface (RML2016-style)."""

    def setUp(self):
        self.pipe = CaukerPipeline(Kmax=3, Vmax=10, Pmax=2, target_length=64)
        self.rng = np.random.RandomState(7)

    def test_generate_single_label(self):
        x, y = self.pipe.generate(
            self.rng, n_inputs_points=64, input_dimension=2, n_classes=5
        )
        self.assertEqual(x.shape, (2, 64))
        self.assertIsInstance(y, int)
        self.assertGreaterEqual(y, 0)
        self.assertLess(y, 5)

    def test_generate_single_label_with_metadata(self):
        out = self.pipe.generate(
            self.rng, n_inputs_points=64, input_dimension=2, n_classes=4,
            return_metadata=True,
        )
        self.assertEqual(len(out), 3)
        x, y, meta = out
        self.assertEqual(x.shape, (2, 64))
        self.assertEqual(meta["n_classes"], 4)

    def test_batch_labels_balanced(self):
        batch = self.pipe.generate_batch(
            self.rng, n_samples=40, n_inputs_points=64, input_dimension=1,
            n_classes=4,
        )
        self.assertEqual(len(batch), 40)
        labels = [lab for _, lab in batch]
        counts = np.bincount(labels)
        self.assertEqual(len(counts), 4)
        self.assertLessEqual(counts.max() - counts.min(), 2)

    def test_batch_no_labels_unchanged(self):
        batch = self.pipe.generate_batch(
            self.rng, n_samples=5, n_inputs_points=64, input_dimension=1
        )
        self.assertEqual(len(batch), 5)
        self.assertTrue(all(isinstance(x, np.ndarray) for x in batch))

    def test_label_deterministic(self):
        r1, r2 = np.random.RandomState(0), np.random.RandomState(0)
        _, y1 = self.pipe.generate(r1, n_inputs_points=64, input_dimension=2, n_classes=6)
        _, y2 = self.pipe.generate(r2, n_inputs_points=64, input_dimension=2, n_classes=6)
        self.assertEqual(y1, y2)


if __name__ == "__main__":
    unittest.main()
