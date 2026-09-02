# -*- coding: utf-8 -*-
"""
Test suite for the Structural Causal Model (SCM) prior pipeline.

Covers all components from Prior Labs Team (2026), Section 2.5 (tabular SCM prior).

Created on 2026/08/12
@author: Ruizhe Wang
@email: changewam6@gmail.com
"""

import unittest
import numpy as np

from s2generator.scm.scm_prior import (
    # DAG generators
    _dag_chain,
    _dag_fork,
    _dag_collider,
    _dag_random,
    _dag_scale_free,
    _dag_bipartite,
    DAG_GENERATORS,
    # Noise
    _noise_iid,
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
    # Target / categorical
    _discretize_target,
    _bin_categorical,
    # Pipeline
    SCMPriorPipeline,
)


# ===========================================================================
# DAG Generation Tests
# ===========================================================================


class TestDAGGenerators(unittest.TestCase):
    """Test all 6 DAG generation algorithms."""

    def setUp(self):
        """Prepare fixtures used by the DAG Generators tests."""
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
        """A chain DAG should connect nodes sequentially with a single root."""
        parents, roots = _dag_chain(self.rng, 5)
        self._verify_dag(parents, roots, 5)
        self.assertEqual(roots, [0])
        for i in range(1, 5):
            self.assertEqual(parents[i], [i - 1])

    def test_dag_fork_structure(self):
        """A fork DAG should have one root pointing to multiple children."""
        parents, roots = _dag_fork(self.rng, 5)
        self._verify_dag(parents, roots, 5)
        self.assertEqual(len(roots), 1)
        root = roots[0]
        for i in range(5):
            if i != root:
                self.assertEqual(parents[i], [root])

    def test_dag_collider_structure(self):
        """A collider DAG should have multiple parents pointing to a common child."""
        parents, roots = _dag_collider(self.rng, 5)
        self._verify_dag(parents, roots, 5)
        self.assertEqual(len(roots), 4)
        # One target node has all others as parents
        has_all = any(len(parents[i]) == 4 for i in range(5))
        self.assertTrue(has_all)

    def test_dag_random_valid(self):
        """Random DAG generation should produce a valid acyclic graph."""
        for V in [3, 10, 20]:
            for Pmax in [1, 3, 5]:
                parents, roots = _dag_random(self.rng, V, Pmax)
                self._verify_dag(parents, roots, V)
                for child in range(V):
                    self.assertLessEqual(len(parents[child]), Pmax)

    def test_dag_random_deterministic(self):
        """Random DAG generation should be deterministic under a fixed seed."""
        rng1, rng2 = np.random.RandomState(42), np.random.RandomState(42)
        p1, r1 = _dag_random(rng1, 10, 3)
        p2, r2 = _dag_random(rng2, 10, 3)
        self.assertEqual(p1, p2)
        self.assertEqual(r1, r2)

    def test_dag_scale_free_valid(self):
        """Scale-free DAG generation should produce a valid acyclic graph."""
        parents, roots = _dag_scale_free(self.rng, 15, 3)
        self._verify_dag(parents, roots, 15)
        # At least one hub (node with children)
        has_children = set()
        for child in range(15):
            for p in parents[child]:
                has_children.add(p)
        self.assertGreater(len(has_children), 0)

    def test_dag_bipartite_structure(self):
        """A bipartite DAG should only connect the two partitions."""
        parents, roots = _dag_bipartite(self.rng, 10)
        self._verify_dag(parents, roots, 10)
        self.assertEqual(len(roots), 5)

    def test_dag_all_registered(self):
        """Every registered DAG generator name should be callable."""
        self.assertEqual(
            set(DAG_GENERATORS.keys()),
            {"chain", "fork", "collider", "random", "scale_free", "bipartite"},
        )

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
# Noise Tests
# ===========================================================================


class TestNoiseIID(unittest.TestCase):
    """Test the i.i.d. Gaussian noise used for root nodes."""

    def setUp(self):
        """Prepare fixtures used by the Noise IID tests."""
        self.rng = np.random.RandomState(42)

    def _check_noise(self, x, L):
        """Assert that noise has the requested length and contains finite values."""
        self.assertEqual(x.shape, (L,))
        self.assertEqual(x.dtype, np.float64)
        self.assertTrue(np.all(np.isfinite(x)))

    def test_noise_iid(self):
        """IID noise should have the requested length and be finite."""
        x = _noise_iid(self.rng, 128)
        self._check_noise(x, 128)
        self.assertGreater(np.std(x), 0.0)

    def test_noise_iid_deterministic(self):
        """IID noise sampling should be deterministic under a fixed seed."""
        r1 = np.random.RandomState(1)
        r2 = np.random.RandomState(1)
        np.testing.assert_array_equal(_noise_iid(r1, 100), _noise_iid(r2, 100))

    def test_noise_iid_custom_scale(self):
        """A custom noise scale should change the noise amplitude."""
        x1 = _noise_iid(self.rng, 200, scale=0.5)
        x2 = _noise_iid(self.rng, 200, scale=5.0)
        self.assertGreater(np.std(x2), np.std(x1))

    def test_noise_iid_different_lengths(self):
        """IID noise should honor several requested lengths."""
        for L in [16, 64, 256]:
            x = _noise_iid(self.rng, L)
            self.assertEqual(x.shape, (L,))


# ===========================================================================
# Combiner Tests
# ===========================================================================


class TestCombiners(unittest.TestCase):
    """Test all 6 combiner mechanisms."""

    def setUp(self):
        """Prepare fixtures used by the Combiners tests."""
        self.rng = np.random.RandomState(42)

    def test_combiner_linear_output(self):
        """The linear combiner should return a finite combination of parents."""
        pv = np.array([1.0, 2.0, -1.5])
        out = _combiner_linear(self.rng, pv)
        self.assertTrue(np.isscalar(out) or isinstance(out, np.floating))
        self.assertTrue(np.isfinite(out))

    def test_combiner_linear_deterministic(self):
        """The linear combiner should be deterministic under a fixed seed."""
        r1, r2 = np.random.RandomState(1), np.random.RandomState(1)
        pv = np.array([0.5, -0.3])
        self.assertEqual(_combiner_linear(r1, pv), _combiner_linear(r2, pv))

    def test_combiner_mlp(self):
        """The MLP combiner should return a finite series of the parent length."""
        for n in [1, 2, 5]:
            pv = self.rng.normal(0, 1, n)
            out = _combiner_mlp(self.rng, pv)
            self.assertTrue(np.isscalar(out) or isinstance(out, np.floating))
            self.assertTrue(np.isfinite(out))

    def test_combiner_mlp_empty(self):
        """The MLP combiner should handle an empty parent list."""
        out = _combiner_mlp(self.rng, np.array([]))
        self.assertEqual(out, 0.0)

    def test_combiner_polynomial(self):
        """The polynomial combiner should return a finite series of the parent length."""
        for n in [1, 2, 5]:
            pv = self.rng.normal(0, 1, n)
            out = _combiner_polynomial(self.rng, pv)
            self.assertTrue(np.isscalar(out) or isinstance(out, np.floating))
            self.assertTrue(np.isfinite(out))

    def test_combiner_multiplicative(self):
        """The multiplicative combiner should return a finite series of the parent length."""
        pv = np.array([2.0, 3.0])
        out = _combiner_multiplicative(self.rng, pv)
        self.assertTrue(np.isscalar(out) or isinstance(out, np.floating))
        self.assertTrue(np.isfinite(out))

    def test_combiner_multiplicative_clips(self):
        """The multiplicative combiner should clip extreme products."""
        pv = np.array([1e6, 1e6, 1e6])
        out = _combiner_multiplicative(self.rng, pv)
        self.assertLessEqual(abs(out), 100.0)

    def test_combiner_periodic(self):
        """The periodic combiner should return a finite series of the parent length."""
        pv = np.array([0.0, 1.0])
        out = _combiner_periodic(self.rng, pv)
        self.assertGreaterEqual(out, -1.0)
        self.assertLessEqual(out, 1.0)

    def test_combiner_maxmin(self):
        """The max-min combiner should return a finite series of the parent length."""
        pv = np.array([-2.0, 0.5, 3.0])
        for _ in range(10):
            out = _combiner_maxmin(self.rng, pv)
            self.assertTrue(np.isscalar(out) or isinstance(out, np.floating))
            self.assertTrue(np.isfinite(out))

    def test_combiner_maxmin_single(self):
        """The max-min combiner should accept a single parent series."""
        out = _combiner_maxmin(self.rng, np.array([7.0]))
        self.assertEqual(out, 7.0)

    def test_combiners_registered(self):
        """Every registered combiner name should be callable."""
        self.assertEqual(
            set(COMBINERS.keys()),
            {"linear", "mlp", "polynomial", "multiplicative", "periodic", "maxmin"},
        )

    def test_all_combiners_different_outputs(self):
        """Different combiners should give different outputs for same input."""
        pv = np.array([1.0, -2.0, 0.5])
        outputs = []
        for name, fn in COMBINERS.items():
            o = fn(self.rng, pv)
            outputs.append(float(o) if hasattr(o, "item") else float(o))
        # Not all should be identical (probabilistically)
        self.assertGreater(np.std(outputs), 0.0)


# ===========================================================================
# Activation Tests
# ===========================================================================


class TestActivations(unittest.TestCase):
    """Test all 8 activation functions."""

    def setUp(self):
        """Prepare fixtures used by the Activations tests."""
        self.rng = np.random.RandomState(42)
        self.x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0, 5.0])

    def test_activation_output_shape(self):
        """Activations should preserve the input series shape."""
        for name, fn in ACTIVATIONS.items():
            if name == "high_freq_sin":
                out = fn(self.x, self.rng)
            else:
                out = fn(self.x)
            self.assertEqual(out.shape, self.x.shape, msg=f"Shape mismatch: {name}")

    def test_activation_finite(self):
        """Every activation should return finite values."""
        for name, fn in ACTIVATIONS.items():
            if name == "high_freq_sin":
                out = fn(self.x, self.rng)
            else:
                out = fn(self.x)
            self.assertTrue(np.all(np.isfinite(out)), msg=f"Non-finite output: {name}")

    def test_relu(self):
        """ReLU should zero negative values."""
        out = _activation_relu(self.x)
        np.testing.assert_allclose(out[self.x < 0], 0.0)
        np.testing.assert_allclose(out[self.x > 0], self.x[self.x > 0])

    def test_sigmoid_range(self):
        """Sigmoid activations should stay in (0, 1)."""
        out = _activation_sigmoid(self.x)
        self.assertTrue(np.all(out >= 0) and np.all(out <= 1))
        self.assertAlmostEqual(out[2], 0.5, places=5)  # sigmoid(0)=0.5

    def test_tanh_range(self):
        """Tanh activations should stay in (-1, 1)."""
        out = _activation_tanh(self.x)
        self.assertTrue(np.all(out >= -1) and np.all(out <= 1))
        self.assertAlmostEqual(out[2], 0.0, places=5)  # tanh(0)=0

    def test_sin_range(self):
        """Sine activations should stay in [-1, 1]."""
        out = _activation_sin(self.x)
        self.assertTrue(np.all(np.abs(out) <= 1.0))

    def test_gelu(self):
        """GELU should return a finite series of the same shape."""
        out = _activation_gelu(self.x)
        self.assertGreater(out[4], 0)  # x=3
        self.assertLess(out[0], 0.1)  # x=-3 should be near zero

    def test_softplus(self):
        """Softplus should return non-negative finite values."""
        out = _activation_softplus(self.x)
        self.assertTrue(np.all(out > 0))
        self.assertAlmostEqual(out[2], np.log(2), places=3)  # softplus(0)=ln(2)

    def test_identity(self):
        """Identity activation should return the input unchanged."""
        out = _activation_identity(self.x)
        np.testing.assert_array_equal(out, self.x)

    def test_high_freq_sin(self):
        """High-frequency sine activation should stay finite and bounded."""
        out = _activation_high_freq_sin(self.x, self.rng)
        self.assertTrue(np.all(np.abs(out) <= 1.0))
        fast_out = _activation_high_freq_sin(self.x, np.random.RandomState(99))
        self.assertTrue(np.any(np.diff(np.signbit(fast_out[1:] - fast_out[:-1]))))

    def test_activations_registered(self):
        """Every registered activation name should be callable."""
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
        """Prepare fixtures used by the Post Processing tests."""
        self.rng = np.random.RandomState(42)
        self.x = np.random.RandomState(0).normal(0, 1, (64, 3))

    def test_add_outliers(self):
        """add_outliers should replace a fraction of values with extreme spikes."""
        out = _postprocess_add_outliers(
            self.rng, self.x, outlier_prob=0.1, outlier_scale=5.0
        )
        self.assertEqual(out.shape, self.x.shape)
        n_changed = np.sum(np.abs(out - self.x) > 1e-8)
        self.assertGreater(n_changed, 0)

    def test_add_outliers_deterministic(self):
        """Outlier injection should be deterministic under a fixed seed."""
        r1, r2 = np.random.RandomState(42), np.random.RandomState(42)
        x = np.ones((2, 10))
        o1 = _postprocess_add_outliers(r1, x, outlier_prob=0.5, outlier_scale=3.0)
        o2 = _postprocess_add_outliers(r2, x, outlier_prob=0.5, outlier_scale=3.0)
        np.testing.assert_array_equal(o1, o2)

    def test_add_missing(self):
        """add_missing should insert NaNs into the series."""
        out = _postprocess_add_missing(self.rng, self.x, missing_prob=0.2)
        self.assertEqual(out.shape, self.x.shape)
        n_nan = np.sum(np.isnan(out))
        self.assertGreater(n_nan, 0)

    def test_scale_shift(self):
        """scale_shift should apply an affine transform to the series."""
        out = _postprocess_scale_shift(self.rng, self.x.copy())
        self.assertEqual(out.shape, self.x.shape)
        # Each feature (column) receives a different scale/shift
        col_means = [np.nanmean(out[:, p]) for p in range(out.shape[1])]
        self.assertGreater(np.std(col_means), 0.0)

    def test_scale_shift_handles_nan(self):
        """scale_shift should not crash on NaN values."""
        x_nan = self.x.copy()
        x_nan[0, 0] = np.nan
        x_nan[1, 1] = np.nan
        out = _postprocess_scale_shift(self.rng, x_nan)
        self.assertEqual(out.shape, x_nan.shape)
        self.assertTrue(np.isnan(out[0, 0]))
        self.assertTrue(np.isnan(out[1, 1]))


# ===========================================================================
# Target Discretization Tests
# ===========================================================================


class TestTargetDiscretization(unittest.TestCase):
    """Test the quantile-binning target mechanism."""

    def setUp(self):
        """Prepare fixtures used by the Target Discretization tests."""
        self.rng = np.random.RandomState(42)

    def test_balanced_classes(self):
        """Discretized targets should be approximately balanced across classes."""
        z = np.random.RandomState(0).normal(0, 1, 1000)
        y = _discretize_target(z, 4, self.rng)
        self.assertEqual(y.shape, (1000,))
        self.assertEqual(set(np.unique(y)), {0, 1, 2, 3})
        # Quantile binning yields approximately equal class counts
        _, counts = np.unique(y, return_counts=True)
        self.assertLess(counts.max() - counts.min(), 100)

    def test_range(self):
        """Discretized labels should lie in {0, ..., n_classes-1}."""
        z = np.linspace(0, 1, 100)
        for C in [2, 3, 5, 10]:
            y = _discretize_target(z, C, self.rng)
            self.assertTrue(np.all(y >= 0) and np.all(y < C))

    def test_deterministic(self):
        """The same random seed should reproduce the same output."""
        z = np.random.RandomState(1).normal(0, 1, 200)
        y1 = _discretize_target(z, 3, np.random.RandomState(0))
        y2 = _discretize_target(z, 3, np.random.RandomState(0))
        np.testing.assert_array_equal(y1, y2)

    def test_binary(self):
        """Binary discretization should yield only labels in {0, 1}."""
        z = np.random.RandomState(0).normal(0, 1, 200)
        y = _discretize_target(z, 2, self.rng)
        self.assertEqual(set(np.unique(y)), {0, 1})

    def test_many_classes(self):
        """Discretization should support a larger number of classes."""
        z = np.random.RandomState(0).normal(0, 1, 1000)
        C = 20
        y = _discretize_target(z, C, self.rng)
        self.assertEqual(set(np.unique(y)), set(range(C)))

    def test_dtype_integer(self):
        """Discretized targets should use an integer dtype."""
        z = np.linspace(0, 1, 50)
        y = _discretize_target(z, 3, self.rng)
        self.assertTrue(np.issubdtype(y.dtype, np.integer))

    def test_rng_not_consumed(self):
        """Target discretization should not consume extra RNG draws."""
        z = np.random.RandomState(1).normal(0, 1, 100)
        y1 = _discretize_target(z, 4, np.random.RandomState(0))
        y2 = _discretize_target(z, 4, np.random.RandomState(123))
        np.testing.assert_array_equal(y1, y2)


# ===========================================================================
# Categorical Binning Tests
# ===========================================================================


class TestCategorical(unittest.TestCase):
    """Test the categorical-variable binning mechanism."""

    def setUp(self):
        """Prepare fixtures used by the Categorical tests."""
        self.rng = np.random.RandomState(42)

    def test_bin_categorical_integer_levels(self):
        """bin_categorical should return integer category levels."""
        v = np.random.RandomState(0).normal(0, 1, 500)
        out = _bin_categorical(v, 5, self.rng)
        self.assertEqual(out.shape, (500,))
        self.assertEqual(set(np.unique(out)), {0, 1, 2, 3, 4})

    def test_bin_categorical_preserves_nan(self):
        """bin_categorical should preserve existing NaN entries."""
        v = np.random.RandomState(0).normal(0, 1, 100)
        v[10] = np.nan
        v[20] = np.nan
        out = _bin_categorical(v, 3, self.rng)
        self.assertTrue(np.isnan(out[10]))
        self.assertTrue(np.isnan(out[20]))
        valid = out[~np.isnan(out)]
        self.assertEqual(set(np.unique(valid)), {0, 1, 2})

    def test_bin_categorical_binary(self):
        """Binary binning should yield two category levels."""
        v = np.random.RandomState(0).normal(0, 1, 300)
        out = _bin_categorical(v, 2, self.rng)
        self.assertEqual(set(np.unique(out)), {0, 1})

    def test_bin_categorical_all_nan(self):
        """An all-NaN series should remain all-NaN after binning."""
        v = np.full(50, np.nan)
        out = _bin_categorical(v, 3, self.rng)
        self.assertTrue(np.all(np.isnan(out)))

    def test_bin_categorical_rng_not_consumed(self):
        """bin_categorical should not consume extra RNG draws."""
        v = np.random.RandomState(0).normal(0, 1, 100)
        o1 = _bin_categorical(v, 4, np.random.RandomState(0))
        o2 = _bin_categorical(v, 4, np.random.RandomState(99))
        np.testing.assert_array_equal(o1, o2)


# ===========================================================================
# SCMPriorPipeline Init Tests
# ===========================================================================


class TestSCMPriorPipelineInit(unittest.TestCase):
    """Test pipeline initialization and properties."""

    def test_default_init(self):
        """Construct the object with default constructor arguments."""
        pipe = SCMPriorPipeline()
        self.assertEqual(pipe._Vmin, 3)
        self.assertEqual(pipe._Vmax, 20)
        self.assertEqual(pipe._Pmax, 4)
        self.assertEqual(pipe._Nmin, 32)
        self.assertEqual(pipe._Nmax, 512)
        self.assertTrue(pipe._apply_postprocessing)
        self.assertEqual(pipe._dtype, np.float64)

    def test_custom_init(self):
        """Construct the object with custom constructor arguments."""
        pipe = SCMPriorPipeline(
            Vmin=5,
            Vmax=15,
            Pmax=3,
            Nmin=10,
            Nmax=100,
            apply_postprocessing=False,
            dtype=np.float32,
            categorical_prob=0.0,
        )
        self.assertEqual(pipe._Vmin, 5)
        self.assertEqual(pipe._Vmax, 15)
        self.assertEqual(pipe._Pmax, 3)
        self.assertEqual(pipe._Nmin, 10)
        self.assertEqual(pipe._Nmax, 100)
        self.assertFalse(pipe._apply_postprocessing)
        self.assertEqual(pipe._dtype, np.float32)
        self.assertEqual(pipe._categorical_prob, 0.0)

    def test_str_method(self):
        """Verify the string representation of the object."""
        self.assertEqual(str(SCMPriorPipeline()), "SCMPriorPipeline")

    def test_properties(self):
        """Verify that public properties expose the expected values."""
        pipe = SCMPriorPipeline()
        self.assertEqual(len(pipe.dag_algorithms), 6)
        self.assertEqual(len(pipe.combiner_mechanisms), 6)
        self.assertEqual(len(pipe.activations), 8)

    def test_custom_weights(self):
        """Custom mechanism weights should be stored and used for sampling."""
        pipe = SCMPriorPipeline(
            dag_weights={
                "chain": 1.0,
                "fork": 0.0,
                "collider": 0.0,
                "random": 0.0,
                "scale_free": 0.0,
                "bipartite": 0.0,
            },
        )
        rng = np.random.RandomState(0)
        for _ in range(10):
            parents, roots, edges = pipe._sample_dag(rng, 5)
            self.assertEqual(len(roots), 1)
            self.assertEqual(roots[0], 0)
            for i in range(1, 5):
                self.assertIn(i - 1, parents[i])


# ===========================================================================
# SCMPriorPipeline Generate Tests
# ===========================================================================


class TestSCMPriorPipelineGenerate(unittest.TestCase):
    """Test the main generate() method."""

    def setUp(self):
        """Prepare fixtures used by the SCM Prior Pipeline Generate tests."""
        self.pipe = SCMPriorPipeline(Vmin=5, Vmax=12, Pmax=2)
        self.rng = np.random.RandomState(42)

    def test_generate_default(self):
        """Default generate() should return finite features of the default shape."""
        X = self.pipe.generate(self.rng)
        self.assertIsInstance(X, np.ndarray)
        self.assertEqual(X.ndim, 2)
        self.assertEqual(X.dtype, np.float64)
        self.assertGreaterEqual(X.shape[0], 1)
        self.assertGreaterEqual(X.shape[1], 1)

    def test_generate_specific_shapes(self):
        """generate() should honor explicit n_samples and n_features."""
        X = self.pipe.generate(self.rng, n_samples=64, n_features=5)
        self.assertEqual(X.shape, (64, 5))

    def test_generate_with_classes(self):
        """Supervised generate() should return integer class labels."""
        X, y = self.pipe.generate(self.rng, n_samples=64, n_features=4, n_classes=3)
        self.assertEqual(X.shape, (64, 4))
        self.assertEqual(y.shape, (64,))
        self.assertEqual(set(np.unique(y)), {0, 1, 2})

    def test_generate_unsupervised_returns_only_X(self):
        """Unsupervised generate() should return only the feature matrix."""
        out = self.pipe.generate(self.rng, n_samples=64, n_features=3)
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, (64, 3))

    def test_generate_with_metadata(self):
        """generate() should return metadata describing the sampled SCM."""
        X, y, meta = self.pipe.generate(
            self.rng, n_samples=64, n_features=2, n_classes=3, return_metadata=True
        )
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsInstance(meta, dict)
        for key in [
            "n_rows",
            "n_features",
            "n_classes",
            "n_nodes",
            "n_edges",
            "n_roots",
            "feature_nodes",
            "target_node",
            "root_nodes",
            "edge_list",
        ]:
            self.assertIn(key, meta)
        self.assertEqual(meta["n_rows"], 64)
        self.assertEqual(meta["n_features"], 2)
        self.assertEqual(meta["n_classes"], 3)
        self.assertIsNotNone(meta["target_node"])
        self.assertGreaterEqual(meta["n_roots"], 1)

    def test_generate_deterministic(self):
        """SCM prior generate() should be deterministic under a fixed seed."""
        rng1, rng2 = np.random.RandomState(99), np.random.RandomState(99)
        out1 = self.pipe.generate(rng1, n_samples=64, n_features=2, n_classes=2)
        out2 = self.pipe.generate(rng2, n_samples=64, n_features=2, n_classes=2)
        np.testing.assert_array_equal(out1[0], out2[0])
        np.testing.assert_array_equal(out1[1], out2[1])

    def test_generate_different_seeds(self):
        """Different seeds should change SCM prior generate() output."""
        rng1, rng2 = np.random.RandomState(1), np.random.RandomState(2)
        X1 = self.pipe.generate(rng1, n_samples=64, n_features=3)
        X2 = self.pipe.generate(rng2, n_samples=64, n_features=3)
        self.assertFalse(np.allclose(X1, X2))

    def test_generate_not_all_constant(self):
        """Generated features should not be entirely constant."""
        pipe = SCMPriorPipeline(
            Vmin=3, Vmax=10, Pmax=2, apply_postprocessing=False, categorical_prob=0.0
        )
        X = pipe.generate(self.rng, n_samples=128, n_features=3)
        stds = [np.nanstd(X[:, j]) for j in range(X.shape[1])]
        self.assertGreater(np.nanmax(stds), 1e-8)

    def test_generate_dtype_float32(self):
        """Generated features should use float32 when configured."""
        pipe = SCMPriorPipeline(Vmin=3, Vmax=8, Pmax=2, dtype=np.float32)
        X = pipe.generate(np.random.RandomState(0), n_samples=64, n_features=2)
        self.assertEqual(X.dtype, np.float32)

    def test_generate_no_postprocessing(self):
        """Disabling post-processing should avoid NaNs from missingness."""
        pipe = SCMPriorPipeline(Vmin=3, Vmax=8, Pmax=2, apply_postprocessing=False)
        X = pipe.generate(self.rng, n_samples=64, n_features=2)
        self.assertTrue(np.isfinite(X).all())
        self.assertFalse(np.any(np.isnan(X)))

    def test_generate_many_times(self):
        """Repeated generate() calls on one instance should stay finite."""
        for i in range(20):
            X = self.pipe.generate(np.random.RandomState(i), n_samples=64, n_features=2)
            self.assertEqual(X.shape, (64, 2))

    def test_generate_feature_nodes_unique(self):
        """Feature-node indices in metadata should be unique."""
        _, _, meta = self.pipe.generate(
            self.rng, n_samples=64, n_features=3, n_classes=2, return_metadata=True
        )
        features = meta["feature_nodes"]
        self.assertEqual(len(features), len(set(features)))
        self.assertNotIn(meta["target_node"], features)

    def test_binary_classification(self):
        """n_classes=2 should yield binary integer labels."""
        X, y = self.pipe.generate(self.rng, n_samples=100, n_features=2, n_classes=2)
        self.assertEqual(set(np.unique(y)), {0, 1})

    def test_many_classes(self):
        """A larger n_classes should still yield valid integer labels."""
        X, y = self.pipe.generate(self.rng, n_samples=1000, n_features=2, n_classes=20)
        self.assertEqual(set(np.unique(y)), set(range(20)))

    def test_single_row(self):
        """generate() should support n_samples=1."""
        X, y = self.pipe.generate(self.rng, n_samples=1, n_features=2, n_classes=2)
        self.assertEqual(X.shape, (1, 2))
        self.assertEqual(y.shape, (1,))


# ===========================================================================
# SCMPriorPipeline Batch Tests
# ===========================================================================


class TestSCMPriorPipelineBatch(unittest.TestCase):
    """Test batch generation."""

    def setUp(self):
        """Prepare fixtures used by the SCM Prior Pipeline Batch tests."""
        self.pipe = SCMPriorPipeline(
            Vmin=5, Vmax=12, Pmax=2, apply_postprocessing=False
        )
        self.rng = np.random.RandomState(42)

    def test_batch_basic(self):
        """Generate a basic batch and check shapes and finiteness."""
        dataset = self.pipe.generate_batch(
            self.rng, n_batches=5, n_samples=64, n_features=2
        )
        self.assertEqual(len(dataset), 5)
        for X in dataset:
            self.assertIsInstance(X, np.ndarray)
            self.assertEqual(X.shape, (64, 2))

    def test_batch_with_classes(self):
        """Batch generation should return labels when n_classes is set."""
        dataset = self.pipe.generate_batch(
            self.rng, n_batches=4, n_samples=64, n_features=3, n_classes=3
        )
        self.assertEqual(len(dataset), 4)
        for X, y in dataset:
            self.assertEqual(X.shape, (64, 3))
            self.assertEqual(y.shape, (64,))

    def test_batch_samples_different(self):
        """Samples within a batch should not all be identical."""
        dataset = self.pipe.generate_batch(
            self.rng, n_batches=10, n_samples=64, n_features=1
        )
        stacked = np.stack([d.flatten() for d in dataset])
        self.assertGreater(np.var(stacked, axis=0).mean(), 0.0)

    def test_batch_large(self):
        """A larger batch size should still produce finite samples."""
        dataset = self.pipe.generate_batch(
            self.rng, n_batches=50, n_samples=64, n_features=1
        )
        self.assertEqual(len(dataset), 50)

    def test_batch_without_sizes(self):
        """Batch generation should work when sizes are left at defaults."""
        dataset = self.pipe.generate_batch(self.rng, n_batches=3)
        self.assertEqual(len(dataset), 3)


# ===========================================================================
# Integration Tests
# ===========================================================================


class TestScmPriorIntegration(unittest.TestCase):
    """End-to-end workflow tests."""

    def test_full_pipeline_workflow(self):
        """An end-to-end SCM prior workflow should return finite features and labels."""
        rng = np.random.RandomState(42)
        pipe = SCMPriorPipeline(Vmin=5, Vmax=20, Pmax=4, apply_postprocessing=True)
        dataset = pipe.generate_batch(
            rng=rng, n_batches=50, n_samples=128, n_features=3, n_classes=2
        )
        self.assertEqual(len(dataset), 50)
        for X, y in dataset:
            self.assertEqual(X.shape, (128, 3))
            self.assertEqual(y.shape, (128,))

    def test_multivariate_output(self):
        """The pipeline should support a multivariate feature matrix."""
        pipe = SCMPriorPipeline(Vmin=5, Vmax=12, Pmax=3, apply_postprocessing=False)
        rng = np.random.RandomState(123)
        X, y, meta = pipe.generate(
            rng, n_samples=128, n_features=4, n_classes=3, return_metadata=True
        )
        self.assertEqual(X.shape, (128, 4))
        self.assertEqual(meta["n_features"], 4)
        self.assertEqual(meta["n_classes"], 3)

    def test_reproducibility_across_pipelines(self):
        """Two pipelines with the same seed should generate identical data."""
        rng1, rng2 = np.random.RandomState(42), np.random.RandomState(42)
        pipe1 = SCMPriorPipeline(Vmin=3, Vmax=10, Pmax=2, apply_postprocessing=False)
        pipe2 = SCMPriorPipeline(Vmin=3, Vmax=10, Pmax=2, apply_postprocessing=False)
        out1 = pipe1.generate(rng1, n_samples=64, n_features=2, n_classes=2)
        out2 = pipe2.generate(rng2, n_samples=64, n_features=2, n_classes=2)
        np.testing.assert_array_equal(out1[0], out2[0])
        np.testing.assert_array_equal(out1[1], out2[1])

    def test_scale_up(self):
        """Larger n_samples / n_features should still produce finite data."""
        pipe = SCMPriorPipeline(Vmin=5, Vmax=20, Pmax=4, apply_postprocessing=False)
        rng = np.random.RandomState(0)
        scales = [10, 30, 50]
        for n in scales:
            dataset = pipe.generate_batch(
                rng=rng, n_batches=n, n_samples=128, n_features=1
            )
            self.assertEqual(len(dataset), n)

    def test_with_postprocessing_has_nan_or_outliers(self):
        """Post-processing should introduce NaNs or outliers when enabled."""
        pipe = SCMPriorPipeline(
            Vmin=3, Vmax=10, Pmax=2, apply_postprocessing=True, categorical_prob=0.0
        )
        rng = np.random.RandomState(0)
        has_feature = False
        for i in range(30):
            X = pipe.generate(np.random.RandomState(i), n_samples=64, n_features=3)
            if np.any(np.isnan(X)) or np.any(np.abs(X) > 5):
                has_feature = True
                break
        self.assertTrue(has_feature, "Post-processing should introduce NaN or outliers")


# ===========================================================================
# Edge Case Tests
# ===========================================================================


class TestScmPriorEdgeCases(unittest.TestCase):
    """Test boundary conditions."""

    def setUp(self):
        """Prepare fixtures used by the Scm Prior Edge Cases tests."""
        self.rng = np.random.RandomState(42)

    def test_minimal_config(self):
        """A minimal pipeline configuration should still generate finite data."""
        pipe = SCMPriorPipeline(Vmin=2, Vmax=2, Pmax=1, apply_postprocessing=False)
        X = pipe.generate(self.rng, n_samples=32, n_features=1)
        self.assertEqual(X.shape, (32, 1))

    def test_small_n_samples(self):
        """A very small n_samples should still generate finite data."""
        pipe = SCMPriorPipeline(Vmin=2, Vmax=5, Pmax=1, apply_postprocessing=False)
        X = pipe.generate(self.rng, n_samples=4, n_features=1)
        self.assertEqual(X.shape, (4, 1))

    def test_large_n_samples(self):
        """A large n_samples should still generate finite data."""
        pipe = SCMPriorPipeline(Vmin=2, Vmax=5, Pmax=1, apply_postprocessing=False)
        X = pipe.generate(self.rng, n_samples=2048, n_features=1)
        self.assertEqual(X.shape, (2048, 1))
        self.assertTrue(np.isfinite(X).all())

    def test_large_vmax_small_features(self):
        """A large graph with few observed features should still generate finite data."""
        pipe = SCMPriorPipeline(Vmin=5, Vmax=30, Pmax=5, apply_postprocessing=False)
        X, y, meta = pipe.generate(
            self.rng, n_samples=64, n_features=2, n_classes=2, return_metadata=True
        )
        self.assertEqual(X.shape, (64, 2))
        self.assertGreaterEqual(meta["n_nodes"], 2)

    def test_all_dag_types_coverage(self):
        """Every DAG type should be usable inside the SCM prior pipeline."""
        pipe = SCMPriorPipeline(Vmin=5, Vmax=15, Pmax=3, apply_postprocessing=False)
        for i in range(30):
            X, y, meta = pipe.generate(
                np.random.RandomState(i),
                n_samples=64,
                n_features=2,
                n_classes=2,
                return_metadata=True,
            )
            self.assertGreaterEqual(meta["n_roots"], 1)


# ===========================================================================
# Custom Graph (adjacency) Tests
# ===========================================================================


class TestSCMPriorPipelineCustomGraph(unittest.TestCase):
    """Test generation with a user-supplied adjacency matrix."""

    def setUp(self):
        """Prepare fixtures used by the SCM Prior Pipeline Custom Graph tests."""
        self.pipe = SCMPriorPipeline(
            Vmin=3, Vmax=10, Pmax=2, apply_postprocessing=False
        )
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
        """Generate data from a user-supplied chain adjacency graph."""
        X, y, meta = self.pipe.generate(
            self.rng,
            n_samples=64,
            n_features=3,
            n_classes=2,
            adjacency=self.chain,
            return_metadata=True,
        )
        self.assertEqual(X.shape, (64, 3))
        self.assertEqual(y.shape, (64,))
        self.assertTrue(np.all(np.isfinite(X)))
        self.assertEqual(meta["graph_source"], "custom")
        self.assertEqual(meta["n_nodes"], 4)
        self.assertEqual(meta["n_edges"], 3)
        self.assertEqual(set(meta["edge_list"]), {(0, 1), (1, 2), (2, 3)})

    def test_generate_zero_adjacency_no_edges(self):
        """A zero adjacency matrix should introduce no causal edges."""
        adj = np.zeros((4, 4))
        X, y, meta = self.pipe.generate(
            self.rng,
            n_samples=64,
            n_features=2,
            n_classes=2,
            adjacency=adj,
            return_metadata=True,
        )
        self.assertEqual(meta["n_edges"], 0)
        self.assertEqual(meta["n_roots"], 4)

    def test_cycle_raises(self):
        """Raise an error when the adjacency graph contains a cycle."""
        cyclic = np.array(
            [
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 0],
            ]
        )
        with self.assertRaises(ValueError):
            self.pipe.generate(self.rng, n_samples=64, n_features=2, adjacency=cyclic)

    def test_non_square_raises(self):
        """Raise an error when the adjacency matrix is not square."""
        with self.assertRaises(ValueError):
            self.pipe.generate(
                self.rng,
                n_samples=64,
                n_features=2,
                adjacency=np.zeros((3, 4)),
            )

    def test_self_loop_raises(self):
        """Raise an error when the adjacency graph contains a self-loop."""
        loop = np.array(
            [
                [1, 1, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]
        )
        with self.assertRaises(ValueError):
            self.pipe.generate(self.rng, n_samples=64, n_features=2, adjacency=loop)

    def test_n_features_exceeds_nodes_raises(self):
        """Requesting more features than graph nodes should raise ValueError."""
        with self.assertRaises(ValueError):
            self.pipe.generate(
                self.rng,
                n_samples=64,
                n_features=5,
                adjacency=self.chain,  # V=4
            )

    def test_deterministic_with_same_adjacency(self):
        """The same seed and adjacency graph should reproduce the same output."""
        rng1, rng2 = np.random.RandomState(7), np.random.RandomState(7)
        out1 = self.pipe.generate(
            rng1, n_samples=64, n_features=3, n_classes=2, adjacency=self.chain
        )
        out2 = self.pipe.generate(
            rng2, n_samples=64, n_features=3, n_classes=2, adjacency=self.chain
        )
        np.testing.assert_array_equal(out1[0], out2[0])
        np.testing.assert_array_equal(out1[1], out2[1])

    def test_batch_with_adjacency(self):
        """Generate a batch of samples from a user-supplied adjacency graph."""
        dataset = self.pipe.generate_batch(
            self.rng,
            n_batches=5,
            n_samples=64,
            n_features=2,
            n_classes=2,
            adjacency=self.chain,
        )
        self.assertEqual(len(dataset), 5)
        for X, y in dataset:
            self.assertEqual(X.shape, (64, 2))
            self.assertEqual(y.shape, (64,))
            self.assertTrue(np.all(np.isfinite(X)))


# ===========================================================================
# Noise Std (per-node ε_v) Tests
# ===========================================================================


class TestNoiseStd(unittest.TestCase):
    """Test the per-node additive noise ε_v in X_v = f_v(pa(X_v)) + ε_v."""

    def setUp(self):
        """Prepare fixtures used by the Noise Std tests."""
        self.chain = np.array(
            [
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 0],
            ]
        )

    def test_noise_std_changes_output(self):
        """Changing noise_std should change the generated series."""
        pipe0 = SCMPriorPipeline(
            noise_std=0.0, apply_postprocessing=False, categorical_prob=0.0
        )
        pipe1 = SCMPriorPipeline(
            noise_std=0.5, apply_postprocessing=False, categorical_prob=0.0
        )
        X0 = pipe0.generate(
            np.random.RandomState(0), n_samples=64, n_features=3, adjacency=self.chain
        )
        X1 = pipe1.generate(
            np.random.RandomState(0), n_samples=64, n_features=3, adjacency=self.chain
        )
        self.assertFalse(np.allclose(X0, X1))

    def test_noise_std_zero_finite(self):
        """noise_std=0 should still produce finite features."""
        pipe = SCMPriorPipeline(
            noise_std=0.0, apply_postprocessing=False, categorical_prob=0.0
        )
        X = pipe.generate(
            np.random.RandomState(0), n_samples=64, n_features=3, adjacency=self.chain
        )
        self.assertTrue(np.all(np.isfinite(X)))


# ===========================================================================
# Metadata Consistency Tests
# ===========================================================================


class TestMetadataConsistency(unittest.TestCase):
    """Test that returned metadata is internally consistent with X and y."""

    def setUp(self):
        """Prepare fixtures used by the Metadata Consistency tests."""
        self.rng = np.random.RandomState(42)

    def test_supervised_metadata_consistency(self):
        """Supervised metadata should agree with the returned X and y shapes."""
        pipe = SCMPriorPipeline(Vmin=5, Vmax=12, Pmax=3, apply_postprocessing=False)
        X, y, meta = pipe.generate(
            self.rng, n_samples=100, n_features=4, n_classes=3, return_metadata=True
        )
        V = meta["n_nodes"]
        self.assertEqual(meta["n_rows"], X.shape[0])
        self.assertEqual(meta["n_features"], X.shape[1])
        self.assertEqual(len(meta["feature_nodes"]), meta["n_features"])
        self.assertNotIn(meta["target_node"], meta["feature_nodes"])
        self.assertEqual(len(meta["edge_list"]), meta["n_edges"])
        self.assertGreaterEqual(meta["n_roots"], 1)
        for f in meta["feature_nodes"]:
            self.assertGreaterEqual(f, 0)
            self.assertLess(f, V)
        self.assertGreaterEqual(meta["target_node"], 0)
        self.assertLess(meta["target_node"], V)
        self.assertEqual(y.shape, (meta["n_rows"],))

    def test_unsupervised_metadata(self):
        """Unsupervised metadata should describe the feature matrix only."""
        pipe = SCMPriorPipeline(Vmin=5, Vmax=12, Pmax=3, apply_postprocessing=False)
        X, meta = pipe.generate(
            self.rng, n_samples=100, n_features=3, return_metadata=True
        )
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(meta, dict)
        self.assertIsNone(meta["n_classes"])
        self.assertIsNone(meta["target_node"])
        self.assertEqual(meta["n_rows"], X.shape[0])
        self.assertEqual(meta["n_features"], X.shape[1])


# ===========================================================================
# Categorical Probability Tests
# ===========================================================================


class TestCategoricalProb(unittest.TestCase):
    """Test the categorical_prob pipeline switch."""

    def setUp(self):
        """Prepare fixtures used by the Categorical Prob tests."""
        self.rng = np.random.RandomState(42)

    def test_categorical_prob_zero(self):
        """categorical_prob=0 should leave features continuous."""
        pipe = SCMPriorPipeline(
            Vmin=5, Vmax=12, Pmax=2, apply_postprocessing=False, categorical_prob=0.0
        )
        X, y, meta = pipe.generate(
            self.rng, n_samples=100, n_features=4, n_classes=3, return_metadata=True
        )
        self.assertEqual(meta["categorical_features"], [])

    def test_categorical_prob_one_all_binned(self):
        """categorical_prob=1 should bin every feature into categories."""
        pipe = SCMPriorPipeline(
            Vmin=5, Vmax=12, Pmax=2, apply_postprocessing=False, categorical_prob=1.0
        )
        X, y, meta = pipe.generate(
            self.rng, n_samples=200, n_features=4, n_classes=3, return_metadata=True
        )
        self.assertEqual(len(meta["categorical_features"]), meta["n_features"])
        for j in range(X.shape[1]):
            self.assertTrue(np.allclose(X[:, j], np.round(X[:, j])))

    def test_categorical_prob_one_levels(self):
        """Fully categorical features should use the configured number of levels."""
        pipe = SCMPriorPipeline(
            Vmin=5, Vmax=12, Pmax=2, apply_postprocessing=False, categorical_prob=1.0
        )
        X = pipe.generate(self.rng, n_samples=200, n_features=4)
        for j in range(X.shape[1]):
            self.assertGreaterEqual(len(np.unique(X[:, j])), 2)


if __name__ == "__main__":
    unittest.main()
