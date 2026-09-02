# -*- coding: utf-8 -*-
"""Tests for bundled real time-series loaders."""

import unittest

import numpy as np
import pandas as pd

from s2generator.utils.data import (
    AVAILABLE_MULTIVARIATE_DATASETS,
    AVAILABLE_UNIVARIATE_DATASETS,
    list_datasets,
    load_multivariate,
    load_univariate,
)


class TestDataLoader(unittest.TestCase):
    """Load the packaged 4096-step benchmark slices."""

    SEQ_LENGTH = 4096

    def test_list_datasets(self) -> None:
        """list_datasets should return the packaged univariate and multivariate names."""
        self.assertEqual(
            list_datasets("univariate"), list(AVAILABLE_UNIVARIATE_DATASETS)
        )
        self.assertEqual(
            list_datasets("multivariate"), list(AVAILABLE_MULTIVARIATE_DATASETS)
        )
        self.assertEqual(list_datasets("all"), list(AVAILABLE_UNIVARIATE_DATASETS))
        with self.assertRaises(ValueError):
            list_datasets("unknown")

    def test_load_univariate_csv_ot(self) -> None:
        """Univariate loaders should return a finite 4096-step OT slice."""
        for name in AVAILABLE_MULTIVARIATE_DATASETS:
            series = load_univariate(name)
            self.assertIsInstance(series, np.ndarray)
            self.assertEqual(series.shape, (self.SEQ_LENGTH,))
            self.assertEqual(series.dtype, np.float64)
            self.assertTrue(np.isfinite(series).all(), msg=name)

    def test_load_univariate_electricity(self) -> None:
        """Electricity should load as a finite univariate series and accept aliases."""
        series = load_univariate("electricity")
        self.assertEqual(series.shape, (self.SEQ_LENGTH,))
        self.assertEqual(series.dtype, np.float64)
        self.assertTrue(np.isfinite(series).all())
        np.testing.assert_array_equal(series, load_univariate("Electricity"))

    def test_load_univariate_aliases(self) -> None:
        """Dataset name aliases should resolve to the same univariate series."""
        np.testing.assert_array_equal(
            load_univariate("exchange"), load_univariate("exchange_rate")
        )
        np.testing.assert_array_equal(
            load_univariate("Weather"), load_univariate("weather")
        )
        np.testing.assert_array_equal(
            load_univariate("ETTh1"), load_univariate("etth1")
        )

    def test_load_multivariate_csv(self) -> None:
        """Multivariate loaders should return a DataFrame with a finite OT column."""
        for name in AVAILABLE_MULTIVARIATE_DATASETS:
            frame = load_multivariate(name)
            self.assertIsInstance(frame, pd.DataFrame)
            self.assertEqual(len(frame), self.SEQ_LENGTH, msg=name)
            self.assertIn("OT", frame.columns)
            self.assertTrue(np.isfinite(frame["OT"].to_numpy(dtype=float)).all())

    def test_multivariate_matches_univariate_ot(self) -> None:
        """The multivariate OT column should match the univariate loader."""
        for name in AVAILABLE_MULTIVARIATE_DATASETS:
            ot = load_multivariate(name)["OT"].to_numpy(dtype=np.float64)
            np.testing.assert_array_equal(ot, load_univariate(name))

    def test_electricity_has_no_multivariate(self) -> None:
        """Electricity is univariate-only and should reject multivariate loading."""
        with self.assertRaises(ValueError):
            load_multivariate("electricity")

    def test_unknown_dataset(self) -> None:
        """Unknown dataset names should raise ValueError."""
        with self.assertRaises(ValueError):
            load_univariate("not-a-dataset")
        with self.assertRaises(ValueError):
            load_multivariate("")


if __name__ == "__main__":
    unittest.main()
