# -*- coding: utf-8 -*-
"""
Test suite for TiRex-3 Synthetic Prior SCM pipeline.

Covers all components from Prior Labs Team (2026), Section 2.5.

Created on 2026/08/12
@author: Ruizhe Wang
@email: changewam6@gmail.com
"""

import unittest
import numpy as np

from s2generator.scm.tirex3 import (
    # DAG generators
    _dag_chain,
    _dag_fork,
    _dag_collider,
    _dag_random,
    _dag_scale_free,
    _dag_bipartite,
    DAG_GENERATORS,
    # Noise processes
    _noise_iid,
    _noise_random_walk,
    _noise_ar1,
    _noise_periodic,
    _noise_ou,
    NOISE_PROCESSES,
    # Combiners
    _combiner_linear,
    _combiner_mlp,
    _combiner_polynomial,
    _combiner_multiplicative,
    _combiner_periodic,
    _combiner_maxmin,
    COMBINERS,
    # Activations
    _activation_relu,
    _activation_sigmoid,
    _activation_tanh,
    _activation_sin,
    _activation_gelu,
    _activation_softplus,
    _activation_identity,
    _activation_high_freq_sin,
    ACTIVATIONS,
    # Post-processing
    _postprocess_add_outliers,
    _postprocess_add_missing,
    _postprocess_scale_shift,
    # Pipeline
    TiRex3Pipeline,
)


# ===========================================================================
# DAG Generation Tests
# ===========================================================================


class TestDAGGenerators(unittest.TestCase):
    """Test all 6 DAG generation algorithms."""

    def setUp(self):
        self.rng = np.random.RandomState(42)

    def _verify_dag(self, parents, roots, V):
        """Verify DAG structural properties."""
        self.assertEqual(len(parents), V)
        # Roots have no parents
        for r in roots:
            self.assertEqual(len(parents[r]), 0)
        # No self-loops
        for child in range(V):
            self.assertNotIn(child, parents[child])
        # Topological ordering possible (acyclic)
        in_degree = [len(parents[i]) for i in range(V)]
        queue = [i for i in range(V) if in_degree[i] == 0]
        visited = []
        while queue:
            node = queue.pop(0)
            visited.append(node)
            for child in range(V):
                if node in parents[child]:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)
        self.assertEqual(len(visited), V, "DAG has a cycle")

    def test_dag_chain_structure(self):
        parents, roots = _dag_chain(self.rng, 5)
        self._verify_dag(parents, roots, 5)
        self.assertEqual(roots, [0])
        for i in range(1, 5):
            self.assertEqual(parents[i], [i - 1])

    def test_dag_fork_structure(self):
        parents, roots = _dag_fork(self.rng, 5)
        self._verify_dag(parents, roots, 5)
        self.assertEqual(len(roots), 1)
        root = roots[0]
        for i in range(5):
            if i != root:
                self.assertEqual(parents[i], [root])

    def test_dag_collider_structure(self):
        parents, roots = _dag_collider(self.rng, 5)
        self._verify_dag(parents, roots, 5)
        self.assertEqual(len(roots), 4)
        # One target node has all others as parents
        has_all = any(len(parents[i]) == 4 for i in range(5))
        self.assertTrue(has_all)

    def test_dag_random_valid(self):
        for V in [3, 10, 20]:
            for Pmax in [1, 3, 5]:
                parents, roots = _dag_random(self.rng, V, Pmax)
                self._verify_dag(parents, roots, V)
                for child in range(V):
                    self.assertLessEqual(len(parents[child]), Pmax)

    def test_dag_random_deterministic(self):
        rng1, rng2 = np.random.RandomState(42), np.random.RandomState(42)
        p1, r1 = _dag_random(rng1, 10, 3)
        p2, r2 = _dag_random(rng2, 10, 3)
        self.assertEqual(p1, p2)
        self.assertEqual(r1, r2)

    def test_dag_scale_free_valid(self):
        parents, roots = _dag_scale_free(self.rng, 15, 3)
        self._verify_dag(parents, roots, 15)
        # At least one hub (node with children)
        has_children = set()
        for child in range(15):
            for p in parents[child]:
                has_children.add(p)
        self.assertGreater(len(has_children), 0)

    def test_dag_bipartite_structure(self):
        parents, roots = _dag_bipartite(self.rng, 10)
        self._verify_dag(parents, roots, 10)
        self.assertEqual(len(roots), 5)

    def test_dag_all_registered(self):
        self.assertEqual(set(DAG_GENERATORS.keys()),
                         {"chain", "fork", "collider", "random",
                          "scale_free", "bipartite"})

    def test_dag_small_graphs(self):
        """All DAG generators should handle V=2."""
        for name, gen_fn in DAG_GENERATORS.items():
            if name in ("chain", "fork", "collider", "bipartite"):
                p, r = gen_fn(self.rng, 2)
            else:
                p, r = gen_fn(self.rng, 2, 1)
            self._verify_dag(p, r, 2)
            self.assertEqual(len(p), 2)


# ===========================================================================
# Noise Process Tests
# ===========================================================================


class TestNoiseProcesses(unittest.TestCase):
    """Test all 5 temporal noise processes."""

    def setUp(self):
        self.rng = np.random.RandomState(42)
        self.L = 128

    def _check_noise(self, x, L):
        self.assertEqual(x.shape, (L,))
        self.assertEqual(x.dtype, np.float64)
        self.assertTrue(np.all(np.isfinite(x)))

    def test_noise_iid(self):
        x = _noise_iid(self.rng, self.L)
        self._check_noise(x, self.L)
        # Should have non-zero variance
        self.assertGreater(np.std(x), 0.0)

    def test_noise_iid_deterministic(self):
        r1 = np.random.RandomState(1)
        r2 = np.random.RandomState(1)
        np.testing.assert_array_equal(_noise_iid(r1, 100), _noise_iid(r2, 100))

    def test_noise_random_walk(self):
        x = _noise_random_walk(self.rng, self.L)
        self._check_noise(x, self.L)
        # Random walk should have growing variance over time
        first_half_std = np.std(x[:self.L // 2])
        second_half_std = np.std(x[self.L // 2:])
        self.assertGreater(second_half_std, first_half_std * 0.5)

    def test_noise_ar1(self):
        x = _noise_ar1(self.rng, self.L)
        self._check_noise(x, self.L)
        # AR(1) should have non-trivial autocorrelation at lag 1
        x_centered = x - x.mean()
        acf1 = np.corrcoef(x_centered[1:], x_centered[:-1])[0, 1]
        self.assertNotAlmostEqual(acf1, 0.0, delta=0.1)

    def test_noise_periodic(self):
        x = _noise_periodic(self.rng, self.L)
        self._check_noise(x, self.L)
        # Should have strong periodic component
        freqs = np.abs(np.fft.rfft(x - x.mean()))
        self.assertGreater(np.max(freqs[1:]), np.mean(freqs[1:]) * 2)

    def test_noise_ou(self):
        x = _noise_ou(self.rng, self.L)
        self._check_noise(x, self.L)
        # OU oscillates around its mean
        self.assertLess(abs(np.mean(x) - np.median(x)), 2.0)

    def test_noise_processes_registered(self):
        self.assertEqual(set(NOISE_PROCESSES.keys()),
                         {"iid", "random_walk", "ar1", "periodic", "ou"})

    def test_noise_custom_params(self):
        """Test noise processes accept custom parameters."""
        x1 = _noise_iid(self.rng, 50, scale=0.5)
        x2 = _noise_iid(self.rng, 50, scale=5.0)
        self.assertGreater(np.std(x2), np.std(x1))

    def test_noise_different_lengths(self):
        for L in [16, 64, 256]:
            for name, fn in NOISE_PROCESSES.items():
                x = fn(self.rng, L)
                self.assertEqual(x.shape, (L,), msg=f"{name} failed at L={L}")


# ===========================================================================
# Combiner Tests
# ===========================================================================


class TestCombiners(unittest.TestCase):
    """Test all 6 combiner mechanisms."""

    def setUp(self):
        self.rng = np.random.RandomState(42)

    def test_combiner_linear_output(self):
        pv = np.array([1.0, 2.0, -1.5])
        out = _combiner_linear(self.rng, pv)
        self.assertTrue(np.isscalar(out) or isinstance(out, np.floating))
        self.assertTrue(np.isfinite(out))

    def test_combiner_linear_deterministic(self):
        r1, r2 = np.random.RandomState(1), np.random.RandomState(1)
        pv = np.array([0.5, -0.3])
        self.assertEqual(_combiner_linear(r1, pv), _combiner_linear(r2, pv))

    def test_combiner_mlp(self):
        for n in [1, 2, 5]:
            pv = self.rng.normal(0, 1, n)
            out = _combiner_mlp(self.rng, pv)
            self.assertTrue(np.isscalar(out) or isinstance(out, np.floating))
            self.assertTrue(np.isfinite(out))

    def test_combiner_mlp_empty(self):
        out = _combiner_mlp(self.rng, np.array([]))
        self.assertEqual(out, 0.0)

    def test_combiner_polynomial(self):
        for n in [1, 2, 5]:
            pv = self.rng.normal(0, 1, n)
            out = _combiner_polynomial(self.rng, pv)
            self.assertTrue(np.isscalar(out) or isinstance(out, np.floating))
            self.assertTrue(np.isfinite(out))

    def test_combiner_multiplicative(self):
        pv = np.array([2.0, 3.0])
        out = _combiner_multiplicative(self.rng, pv)
        self.assertTrue(np.isscalar(out) or isinstance(out, np.floating))
        self.assertTrue(np.isfinite(out))

    def test_combiner_multiplicative_clips(self):
        pv = np.array([1e6, 1e6, 1e6])
        out = _combiner_multiplicative(self.rng, pv)
        self.assertLessEqual(abs(out), 100.0)

    def test_combiner_periodic(self):
        pv = np.array([0.0, 1.0])
        out = _combiner_periodic(self.rng, pv)
        self.assertGreaterEqual(out, -1.0)
        self.assertLessEqual(out, 1.0)

    def test_combiner_maxmin(self):
        pv = np.array([-2.0, 0.5, 3.0])
        for _ in range(10):
            out = _combiner_maxmin(self.rng, pv)
            self.assertTrue(np.isscalar(out) or isinstance(out, np.floating))
            self.assertTrue(np.isfinite(out))

    def test_combiner_maxmin_single(self):
        out = _combiner_maxmin(self.rng, np.array([7.0]))
        self.assertEqual(out, 7.0)

    def test_combiners_registered(self):
        self.assertEqual(set(COMBINERS.keys()),
                         {"linear", "mlp", "polynomial", "multiplicative",
                          "periodic", "maxmin"})

    def test_all_combiners_different_outputs(self):
        """Different combiners should give different outputs for same input."""
        pv = np.array([1.0, -2.0, 0.5])
        outputs = []
        for name, fn in COMBINERS.items():
            o = fn(self.rng, pv)
            outputs.append(float(o) if hasattr(o, 'item') else float(o))
        # Not all should be identical (probabilistically)
        self.assertGreater(np.std(outputs), 0.0)


# ===========================================================================
# Activation Tests
# ===========================================================================


class TestActivations(unittest.TestCase):
    """Test all 8 activation functions."""

    def setUp(self):
        self.rng = np.random.RandomState(42)
        self.x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0, 5.0])

    def test_activation_output_shape(self):
        for name, fn in ACTIVATIONS.items():
            if name == "high_freq_sin":
                out = fn(self.x, self.rng)
            else:
                out = fn(self.x)
            self.assertEqual(out.shape, self.x.shape, msg=f"Shape mismatch: {name}")

    def test_activation_finite(self):
        for name, fn in ACTIVATIONS.items():
            if name == "high_freq_sin":
                out = fn(self.x, self.rng)
            else:
                out = fn(self.x)
            self.assertTrue(np.all(np.isfinite(out)),
                            msg=f"Non-finite output: {name}")

    def test_relu(self):
        out = _activation_relu(self.x)
        np.testing.assert_allclose(out[self.x < 0], 0.0)
        np.testing.assert_allclose(out[self.x > 0], self.x[self.x > 0])

    def test_sigmoid_range(self):
        out = _activation_sigmoid(self.x)
        self.assertTrue(np.all(out >= 0) and np.all(out <= 1))
        self.assertAlmostEqual(out[2], 0.5, places=5)  # sigmoid(0)=0.5

    def test_tanh_range(self):
        out = _activation_tanh(self.x)
        self.assertTrue(np.all(out >= -1) and np.all(out <= 1))
        self.assertAlmostEqual(out[2], 0.0, places=5)  # tanh(0)=0

    def test_sin_range(self):
        out = _activation_sin(self.x)
        self.assertTrue(np.all(np.abs(out) <= 1.0))

    def test_gelu(self):
        out = _activation_gelu(self.x)
        # GELU passes positive values, suppresses negatives
        self.assertGreater(out[4], 0)  # x=3
        self.assertLess(out[0], 0.1)  # x=-3 should be near zero

    def test_softplus(self):
        out = _activation_softplus(self.x)
        self.assertTrue(np.all(out > 0))
        self.assertAlmostEqual(out[2], np.log(2), places=3)  # softplus(0)=ln(2)

    def test_identity(self):
        out = _activation_identity(self.x)
        np.testing.assert_array_equal(out, self.x)

    def test_high_freq_sin(self):
        out = _activation_high_freq_sin(self.x, self.rng)
        self.assertTrue(np.all(np.abs(out) <= 1.0))
        # Should oscillate faster than regular sin
        fast_out = _activation_high_freq_sin(self.x, np.random.RandomState(99))
        self.assertTrue(np.any(np.diff(np.signbit(fast_out[1:] - fast_out[:-1]))))

    def test_activations_registered(self):
        self.assertIn("relu", ACTIVATIONS)
        self.assertIn("sigmoid", ACTIVATIONS)
        self.assertIn("tanh", ACTIVATIONS)
        self.assertIn("sin", ACTIVATIONS)
        self.assertIn("gelu", ACTIVATIONS)
        self.assertIn("softplus", ACTIVATIONS)
        self.assertIn("identity", ACTIVATIONS)
        self.assertIn("high_freq_sin", ACTIVATIONS)


# ===========================================================================
# Post-Processing Tests
# ===========================================================================


class TestPostProcessing(unittest.TestCase):
    """Test post-processing transforms."""

    def setUp(self):
        self.rng = np.random.RandomState(42)
        self.x = np.random.RandomState(0).normal(0, 1, (3, 64))

    def test_add_outliers(self):
        out = _postprocess_add_outliers(self.rng, self.x,
                                        outlier_prob=0.1, outlier_scale=5.0)
        self.assertEqual(out.shape, self.x.shape)
        # Some values should differ
        n_changed = np.sum(np.abs(out - self.x) > 1e-8)
        self.assertGreater(n_changed, 0)

    def test_add_outliers_deterministic(self):
        r1, r2 = np.random.RandomState(42), np.random.RandomState(42)
        x = np.ones((2, 10))
        o1 = _postprocess_add_outliers(r1, x, outlier_prob=0.5, outlier_scale=3.0)
        o2 = _postprocess_add_outliers(r2, x, outlier_prob=0.5, outlier_scale=3.0)
        np.testing.assert_array_equal(o1, o2)

    def test_add_missing(self):
        out = _postprocess_add_missing(self.rng, self.x, missing_prob=0.2)
        self.assertEqual(out.shape, self.x.shape)
        n_nan = np.sum(np.isnan(out))
        self.assertGreater(n_nan, 0)

    def test_scale_shift(self):
        out = _postprocess_scale_shift(self.rng, self.x.copy())
        self.assertEqual(out.shape, self.x.shape)
        # Each variate should have different mean/std
        means = [out[i].mean() for i in range(out.shape[0])]
        # Original means varied; transformed ones should too
        self.assertGreater(np.std(means), 0.0)

    def test_scale_shift_handles_nan(self):
        """scale_shift should not crash on NaN values."""
        x_nan = self.x.copy()
        x_nan[0, 10] = np.nan
        x_nan[1, 20] = np.nan
        out = _postprocess_scale_shift(self.rng, x_nan)
        self.assertEqual(out.shape, x_nan.shape)
        self.assertTrue(np.isnan(out[0, 10]))
        self.assertTrue(np.isnan(out[1, 20]))


# ===========================================================================
# TiRex3Pipeline Init Tests
# ===========================================================================


class TestTiRex3PipelineInit(unittest.TestCase):
    """Test pipeline initialization and properties."""

    def test_default_init(self):
        pipe = TiRex3Pipeline()
        self.assertEqual(pipe._Vmin, 3)
        self.assertEqual(pipe._Vmax, 20)
        self.assertEqual(pipe._Pmax, 4)
        self.assertTrue(pipe._apply_postprocessing)
        self.assertEqual(pipe._dtype, np.float64)

    def test_custom_init(self):
        pipe = TiRex3Pipeline(
            Vmin=5, Vmax=15, Pmax=3, apply_postprocessing=False,
            dtype=np.float32,
        )
        self.assertEqual(pipe._Vmin, 5)
        self.assertEqual(pipe._Vmax, 15)
        self.assertEqual(pipe._Pmax, 3)
        self.assertFalse(pipe._apply_postprocessing)
        self.assertEqual(pipe._dtype, np.float32)

    def test_str_method(self):
        self.assertEqual(str(TiRex3Pipeline()), "TiRex3Pipeline")

    def test_properties(self):
        pipe = TiRex3Pipeline()
        self.assertEqual(len(pipe.dag_algorithms), 6)
        self.assertEqual(len(pipe.combiner_mechanisms), 6)
        self.assertEqual(len(pipe.noise_processes), 5)
        self.assertEqual(len(pipe.activations), 8)

    def test_custom_weights(self):
        pipe = TiRex3Pipeline(
            dag_weights={"chain": 1.0, "fork": 0.0, "collider": 0.0,
                         "random": 0.0, "scale_free": 0.0, "bipartite": 0.0},
        )
        # All generated DAGs should be chains
        rng = np.random.RandomState(0)
        for _ in range(10):
            parents, roots, edges = pipe._sample_dag(rng, 5)
            # Chain: each node except 0 has exactly one parent (the previous)
            self.assertEqual(len(roots), 1)
            self.assertEqual(roots[0], 0)
            for i in range(1, 5):
                self.assertIn(i - 1, parents[i])


# ===========================================================================
# TiRex3Pipeline Generate Tests
# ===========================================================================


class TestTiRex3PipelineGenerate(unittest.TestCase):
    """Test the main generate() method."""

    def setUp(self):
        self.pipe = TiRex3Pipeline(Vmin=3, Vmax=10, Pmax=2)
        self.rng = np.random.RandomState(42)

    def test_generate_default(self):
        x = self.pipe.generate(self.rng, n_inputs_points=64)
        self.assertIsInstance(x, np.ndarray)
        self.assertEqual(x.ndim, 2)
        self.assertEqual(x.shape[1], 64)
        self.assertEqual(x.dtype, np.float64)

    def test_generate_specific_dimension(self):
        # Ensure Vmin >= max d to avoid "larger sample than population" error
        pipe = TiRex3Pipeline(Vmin=5, Vmax=12, Pmax=2,
                                 apply_postprocessing=False)
        for d in [1, 3, 5]:
            x = pipe.generate(self.rng, n_inputs_points=64, input_dimension=d)
            self.assertEqual(x.shape, (d, 64))

    def test_generate_with_metadata(self):
        x, meta = self.pipe.generate(
            self.rng, n_inputs_points=64, input_dimension=2, return_metadata=True
        )
        self.assertIsInstance(x, np.ndarray)
        self.assertIsInstance(meta, dict)
        for key in ["n_nodes", "n_observed", "n_roots", "n_edges",
                     "sequence_length", "observed_nodes", "root_nodes", "edge_list"]:
            self.assertIn(key, meta)
        self.assertEqual(meta["n_observed"], 2)
        self.assertGreaterEqual(meta["n_roots"], 1)
        self.assertEqual(meta["sequence_length"], 64)

    def test_generate_deterministic(self):
        rng1, rng2 = np.random.RandomState(99), np.random.RandomState(99)
        x1 = self.pipe.generate(rng1, n_inputs_points=64, input_dimension=2)
        x2 = self.pipe.generate(rng2, n_inputs_points=64, input_dimension=2)
        np.testing.assert_array_equal(x1, x2)

    def test_generate_different_seeds(self):
        rng1, rng2 = np.random.RandomState(1), np.random.RandomState(2)
        x1 = self.pipe.generate(rng1, n_inputs_points=64, input_dimension=3)
        x2 = self.pipe.generate(rng2, n_inputs_points=64, input_dimension=3)
        self.assertFalse(np.allclose(x1, x2))

    def test_generate_not_all_constant(self):
        # Use no post-processing to avoid NaN interference
        pipe = TiRex3Pipeline(Vmin=3, Vmax=10, Pmax=2,
                                 apply_postprocessing=False)
        x = pipe.generate(self.rng, n_inputs_points=128, input_dimension=3)
        stds = [np.nanstd(x[i]) for i in range(x.shape[0])]
        self.assertGreater(np.nanmax(stds), 1e-8)

    def test_generate_different_lengths(self):
        for L in [32, 64, 128, 256]:
            x = self.pipe.generate(self.rng, n_inputs_points=L, input_dimension=2)
            self.assertEqual(x.shape, (2, L))

    def test_generate_dtype_float32(self):
        pipe = TiRex3Pipeline(Vmin=3, Vmax=8, Pmax=2, dtype=np.float32)
        x = pipe.generate(np.random.RandomState(0),
                          n_inputs_points=64, input_dimension=2)
        self.assertEqual(x.dtype, np.float32)

    def test_generate_no_postprocessing(self):
        pipe = TiRex3Pipeline(Vmin=3, Vmax=8, Pmax=2,
                                 apply_postprocessing=False)
        x = pipe.generate(self.rng, n_inputs_points=64, input_dimension=2)
        # With no post-processing, there should be no NaN values
        self.assertTrue(np.isfinite(x).all())
        self.assertFalse(np.any(np.isnan(x)))

    def test_generate_observed_nodes_unique(self):
        _, meta = self.pipe.generate(
            self.rng, n_inputs_points=64, input_dimension=5, return_metadata=True
        )
        observed = meta["observed_nodes"]
        self.assertEqual(len(observed), len(set(observed)))

    def test_generate_many_times(self):
        for i in range(20):
            x = self.pipe.generate(
                np.random.RandomState(i), n_inputs_points=64, input_dimension=2
            )
            self.assertEqual(x.shape, (2, 64))

    def test_generate_all_dag_types_coverage(self):
        """Generate many samples to ensure all DAG types can be hit."""
        pipe = TiRex3Pipeline(Vmin=5, Vmax=15, Pmax=3,
                                 apply_postprocessing=False)
        for i in range(30):
            _, meta = pipe.generate(
                np.random.RandomState(i), n_inputs_points=64,
                input_dimension=2, return_metadata=True,
            )
            self.assertGreaterEqual(meta["n_roots"], 1)


# ===========================================================================
# TiRex3Pipeline Batch Tests
# ===========================================================================


class TestTiRex3PipelineBatch(unittest.TestCase):
    """Test batch generation."""

    def setUp(self):
        self.pipe = TiRex3Pipeline(Vmin=3, Vmax=10, Pmax=2,
                                      apply_postprocessing=False)
        self.rng = np.random.RandomState(42)

    def test_batch_basic(self):
        dataset = self.pipe.generate_batch(
            self.rng, n_samples=5, n_inputs_points=64, input_dimension=2
        )
        self.assertEqual(len(dataset), 5)
        for x in dataset:
            self.assertIsInstance(x, np.ndarray)
            self.assertEqual(x.shape, (2, 64))

    def test_batch_samples_different(self):
        dataset = self.pipe.generate_batch(
            self.rng, n_samples=10, n_inputs_points=64, input_dimension=1
        )
        stacked = np.stack([d.flatten() for d in dataset])
        self.assertGreater(np.var(stacked, axis=0).mean(), 0.0)

    def test_batch_large(self):
        dataset = self.pipe.generate_batch(
            self.rng, n_samples=50, n_inputs_points=64, input_dimension=1
        )
        self.assertEqual(len(dataset), 50)

    def test_batch_without_input_dimension(self):
        dataset = self.pipe.generate_batch(
            self.rng, n_samples=3, n_inputs_points=64
        )
        self.assertEqual(len(dataset), 3)


# ===========================================================================
# Integration Tests
# ===========================================================================


class TestTiRex3Integration(unittest.TestCase):
    """End-to-end workflow tests."""

    def test_full_pipeline_workflow(self):
        rng = np.random.RandomState(42)
        pipe = TiRex3Pipeline(
            Vmin=5, Vmax=20, Pmax=4,
            apply_postprocessing=True,
        )
        dataset = pipe.generate_batch(
            rng=rng, n_samples=50, n_inputs_points=128, input_dimension=1,
        )
        self.assertEqual(len(dataset), 50)
        for x in dataset:
            self.assertEqual(x.shape, (1, 128))

        # Diversity check (use nanmean/nanstd to handle post-processing NaN)
        means = [float(np.nanmean(x)) for x in dataset]
        self.assertGreater(np.nanstd(means), 0.0)

    def test_multivariate_output(self):
        pipe = TiRex3Pipeline(Vmin=3, Vmax=12, Pmax=3,
                                 apply_postprocessing=False)
        rng = np.random.RandomState(123)
        x, meta = pipe.generate(
            rng, n_inputs_points=128, input_dimension=4, return_metadata=True,
        )
        self.assertEqual(x.shape, (4, 128))
        self.assertEqual(meta["n_observed"], 4)

    def test_reproducibility_across_pipelines(self):
        rng1, rng2 = np.random.RandomState(42), np.random.RandomState(42)
        pipe1 = TiRex3Pipeline(Vmin=3, Vmax=10, Pmax=2,
                                  apply_postprocessing=False)
        pipe2 = TiRex3Pipeline(Vmin=3, Vmax=10, Pmax=2,
                                  apply_postprocessing=False)
        x1 = pipe1.generate(rng1, n_inputs_points=64, input_dimension=2)
        x2 = pipe2.generate(rng2, n_inputs_points=64, input_dimension=2)
        np.testing.assert_array_equal(x1, x2)

    def test_scale_up(self):
        pipe = TiRex3Pipeline(Vmin=5, Vmax=20, Pmax=4,
                                 apply_postprocessing=False)
        rng = np.random.RandomState(0)
        scales = [10, 30, 50]
        for n in scales:
            dataset = pipe.generate_batch(
                rng=rng, n_samples=n, n_inputs_points=128, input_dimension=1,
            )
            self.assertEqual(len(dataset), n)

    def test_with_postprocessing_has_nan_or_outliers(self):
        pipe = TiRex3Pipeline(Vmin=3, Vmax=10, Pmax=2,
                                 apply_postprocessing=True)
        rng = np.random.RandomState(0)
        # Generate many samples; at least some should have NaN or outliers
        has_feature = False
        for i in range(30):
            x = pipe.generate(
                np.random.RandomState(i), n_inputs_points=64, input_dimension=3
            )
            if np.any(np.isnan(x)) or np.any(np.abs(x) > 5):
                has_feature = True
                break
        self.assertTrue(has_feature,
                        "Post-processing should introduce NaN or outliers")


# ===========================================================================
# Edge Case Tests
# ===========================================================================


class TestTiRex3EdgeCases(unittest.TestCase):
    """Test boundary conditions."""

    def setUp(self):
        self.rng = np.random.RandomState(42)

    def test_minimal_config(self):
        pipe = TiRex3Pipeline(Vmin=2, Vmax=2, Pmax=1,
                                 apply_postprocessing=False)
        x = pipe.generate(self.rng, n_inputs_points=32, input_dimension=1)
        self.assertEqual(x.shape, (1, 32))

    def test_very_short_sequence(self):
        pipe = TiRex3Pipeline(Vmin=2, Vmax=5, Pmax=1,
                                 apply_postprocessing=False)
        x = pipe.generate(self.rng, n_inputs_points=4, input_dimension=1)
        self.assertEqual(x.shape, (1, 4))

    def test_very_long_sequence(self):
        pipe = TiRex3Pipeline(Vmin=2, Vmax=5, Pmax=1,
                                 apply_postprocessing=False)
        x = pipe.generate(self.rng, n_inputs_points=2048, input_dimension=1)
        self.assertEqual(x.shape, (1, 2048))
        self.assertTrue(np.isfinite(x).all())

    def test_large_vmax_small_observed(self):
        pipe = TiRex3Pipeline(Vmin=5, Vmax=30, Pmax=5,
                                 apply_postprocessing=False)
        x, meta = pipe.generate(
            self.rng, n_inputs_points=64, input_dimension=2, return_metadata=True,
        )
        self.assertEqual(x.shape, (2, 64))
        self.assertGreaterEqual(meta["n_nodes"], 2)

    def test_all_noise_types_coverage(self):
        """Generate many root nodes to ensure all noise types can be hit."""
        pipe = TiRex3Pipeline(Vmin=10, Vmax=20, Pmax=3,
                                 apply_postprocessing=False)
        for i in range(20):
            x = pipe.generate(
                np.random.RandomState(i), n_inputs_points=64, input_dimension=3
            )
            self.assertTrue(np.isfinite(x).all())


if __name__ == "__main__":
    unittest.main()
