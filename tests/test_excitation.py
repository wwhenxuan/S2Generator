# -*- coding: utf-8 -*-
"""
Unit tests for the unified Excitation interface.
"""

import unittest
import numpy as np

from s2generator.excitation import (
    Excitation,
    MixedDistribution,
    AutoregressiveMovingAverage,
    ForecastPFN,
    KernelSynth,
    IntrinsicModeFunction,
)
from s2generator.symbol.params import SeriesParams


class TestExcitationInterface(unittest.TestCase):
    """Testing the unified Excitation facade in excitation/_interface.py"""

    rng = np.random.RandomState(42)

    def test_package_exports(self) -> None:
        """All public excitation symbols should be importable"""
        for cls in (
            Excitation,
            MixedDistribution,
            AutoregressiveMovingAverage,
            ForecastPFN,
            KernelSynth,
            IntrinsicModeFunction,
        ):
            self.assertTrue(callable(cls))

    def test_init_default_and_custom_series_params(self) -> None:
        """Default SeriesParams is created; custom params are stored"""
        default_exc = Excitation()
        self.assertIsInstance(default_exc.series_params, SeriesParams)

        custom = SeriesParams(p_min=1, p_max=2, upper_bound=128.0)
        custom_exc = Excitation(series_params=custom)
        self.assertIs(custom_exc.series_params, custom)
        self.assertEqual(custom_exc.series_params.upper_bound, 128.0)

    def test_str_and_call(self) -> None:
        """__str__ and __call__ should work as documented"""
        exc = Excitation()
        self.assertEqual(str(exc), "Excitation")
        out = exc(self.rng, n_inputs_points=64, input_dimension=1)
        self.assertEqual(out.shape, (64, 1))

    def test_sampling_dict_keys_and_types(self) -> None:
        """sampling_dict should contain the five generation mechanisms"""
        exc = Excitation()
        expected = {
            "mixed_distribution": MixedDistribution,
            "autoregressive_moving_average": AutoregressiveMovingAverage,
            "forecast_pfn": ForecastPFN,
            "kernel_synth": KernelSynth,
            "intrinsic_mode_function": IntrinsicModeFunction,
        }
        self.assertEqual(set(exc.sampling_dict.keys()), set(expected.keys()))
        for name, cls in expected.items():
            self.assertIsInstance(exc.sampling_dict[name], cls)
        self.assertEqual(len(exc.sampling_object), 5)
        self.assertEqual(exc.sampling_methods, list(expected.keys()))

    def test_create_factories_wire_params(self) -> None:
        """Factory helpers should forward SeriesParams fields"""
        params = SeriesParams(
            p_min=1,
            p_max=2,
            q_min=2,
            q_max=4,
            upper_bound=64.0,
            min_kernels=1,
            max_kernels=2,
            min_base_imfs=1,
            max_base_imfs=2,
            min_choice_imfs=1,
            max_choice_imfs=2,
            is_sub_day=False,
            transition=False,
            random_walk=False,
        )
        exc = Excitation(series_params=params)

        arma = exc.create_autoregressive_moving_average()
        self.assertEqual(arma.p_min, 1)
        self.assertEqual(arma.p_max, 2)
        self.assertEqual(arma.q_min, 2)
        self.assertEqual(arma.q_max, 4)
        self.assertEqual(arma.upper_bound, 64.0)

        pfn = exc.create_forecast_pfn()
        self.assertFalse(pfn.is_sub_day)
        self.assertFalse(pfn.transition)

        ks = exc.create_kernel_synth()
        self.assertEqual(ks.min_kernels, 1)
        self.assertEqual(ks.max_kernels, 2)

        imf = exc.create_intrinsic_mode_function()
        self.assertEqual(imf.min_base_imfs, 1)
        self.assertEqual(imf.max_base_imfs, 2)

        mixed = exc.create_mixed_distribution()
        self.assertIsInstance(mixed, MixedDistribution)

    def test_choice_respects_dimension_and_methods(self) -> None:
        """choice() returns one method name per dimension"""
        exc = Excitation()
        for dim in [1, 2, 4]:
            choices = exc.choice(rng=self.rng, input_dimension=dim)
            self.assertEqual(len(choices), dim)
            for name in choices:
                self.assertIn(name, exc.sampling_methods)

    def test_prob_array_sums_to_one(self) -> None:
        """Probability array should be normalized"""
        exc = Excitation()
        self.assertAlmostEqual(float(np.sum(exc.prob_array)), 1.0, places=8)
        self.assertEqual(len(exc.prob_array), len(exc.sampling_methods))

    def test_forced_single_method_via_series_params(self) -> None:
        """One-hot method probabilities force a single sampler"""
        params = SeriesParams(
            mixed_distribution=0.0,
            autoregressive_moving_average=1.0,
            forecast_pfn=0.0,
            kernel_synth=0.0,
            intrinsic_mode_function=0.0,
        )
        exc = Excitation(series_params=params)
        choices = exc.choice(rng=np.random.RandomState(0), input_dimension=8)
        self.assertTrue(np.all(choices == "autoregressive_moving_average"))

    def test_generate_shape_and_dtype(self) -> None:
        """generate() returns the requested shape and dtype"""
        params = SeriesParams(dtype=np.float64)
        exc = Excitation(series_params=params)
        for length, dim in [(32, 1), (64, 2), (128, 3)]:
            series = exc.generate(
                rng=self.rng, n_inputs_points=length, input_dimension=dim
            )
            self.assertEqual(series.shape, (length, dim))
            self.assertEqual(series.dtype, np.float64)
            self.assertTrue(np.all(np.isfinite(series)))

    def test_generate_return_choice(self) -> None:
        """return_choice=True should also return selected method names"""
        exc = Excitation()
        series, choices = exc.generate(
            rng=self.rng,
            n_inputs_points=48,
            input_dimension=3,
            return_choice=True,
        )
        self.assertEqual(series.shape, (48, 3))
        self.assertEqual(len(choices), 3)
        for name in choices:
            self.assertIn(name, exc.sampling_methods)

    def test_normalization_zscore(self) -> None:
        """z-score normalization should approximately center each channel"""
        # Prefer ARMA for more stable non-constant channels.
        params = SeriesParams(
            mixed_distribution=0.0,
            autoregressive_moving_average=1.0,
            forecast_pfn=0.0,
            kernel_synth=0.0,
            intrinsic_mode_function=0.0,
        )
        exc = Excitation(series_params=params)
        series = exc.generate(
            rng=np.random.RandomState(1),
            n_inputs_points=256,
            input_dimension=2,
            normalization="z-score",
        )
        for dim in range(series.shape[1]):
            self.assertAlmostEqual(float(np.mean(series[:, dim])), 0.0, places=5)
            self.assertAlmostEqual(float(np.std(series[:, dim])), 1.0, places=5)

    def test_normalization_maxmin(self) -> None:
        """max-min normalization should map each channel into ~[0, 1]"""
        params = SeriesParams(
            mixed_distribution=0.0,
            autoregressive_moving_average=1.0,
            forecast_pfn=0.0,
            kernel_synth=0.0,
            intrinsic_mode_function=0.0,
        )
        exc = Excitation(series_params=params)
        series = exc.generate(
            rng=np.random.RandomState(2),
            n_inputs_points=256,
            input_dimension=2,
            normalization="max-min",
        )
        for dim in range(series.shape[1]):
            self.assertGreaterEqual(float(np.min(series[:, dim])), -1e-8)
            self.assertLessEqual(float(np.max(series[:, dim])), 1.0 + 1e-8)

    def test_normalization_invalid_raises(self) -> None:
        """Invalid normalization option should raise ValueError"""
        exc = Excitation()
        with self.assertRaises(ValueError):
            exc.generate(
                rng=self.rng,
                n_inputs_points=32,
                input_dimension=1,
                normalization="unknown",
            )


if __name__ == "__main__":
    unittest.main()
