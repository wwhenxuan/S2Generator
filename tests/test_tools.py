# -*- coding: utf-8 -*-
"""
Created on 2025/08/23 17:17:23
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/S2Generator
"""

import unittest
from importlib.util import find_spec
from os import path

import numpy as np

from s2generator.utils.tools import (
    get_time_now,
    ensure_directory_exists,
    save_s2data,
    save_npy,
    save_npz,
    load_s2data,
    load_npy,
    load_npz,
    is_all_zeros,
    z_score_normalization,
    max_min_normalization,
    save_table,
)


class TestTools(unittest.TestCase):
    """
    Unit tests for utility functions in the tools module.

    This test suite validates the functionality of various utility functions
    including normalization methods, file operations, and data validation.
    """

    # Create test time series data
    time_series = np.random.uniform(low=-10, high=10, size=(256, 10))

    # Create all-zero time series for edge case testing
    zero_series = np.zeros((256, 10))

    # Create time series with infinite values for edge case testing
    inf_series = np.random.uniform(low=-10, high=10, size=(256, 10))
    inf_series[0, 0] = np.inf
    inf_series[-1, -1] = -np.inf

    # Create time series with NaN values for edge case testing
    nan_series = np.random.uniform(low=-10, high=10, size=(256, 10))
    nan_series[0, 0] = np.nan
    nan_series[-1, -1] = np.nan

    # Define file paths for testing
    npy_path = "./tests/data/data.npy"
    npz_path = "./tests/data/data.npz"
    s2_npy_path = "./tests/data/s2data.npy"
    s2_npz_path = "./tests/data/s2data.npz"

    # Create sample data for file operation tests
    data = {"symbol": "hello", "excitation": 256, "response": 256}

    def test_z_score_normalization(self) -> None:
        """
        Tests the z-score normalization function with valid input data.

        Validates that normalized data has:
        - Mean close to 0
        - Standard deviation close to 1
        """
        # Apply z-score normalization
        normalized = z_score_normalization(self.time_series)

        # Validate mean and standard deviation
        self.assertAlmostEqual(
            first=np.mean(normalized),
            second=0,
            delta=0.01,
            msg="Z-score normalization failed mean validation!",
        )
        self.assertAlmostEqual(
            first=np.mean(np.std(normalized, axis=0, keepdims=True)),
            second=1,
            delta=0.01,
            msg="Z-score normalization failed standard deviation validation!",
        )

    def test_z_score_normalization_zero(self) -> None:
        """
        Tests z-score normalization with all-zero input series.

        Validates that the function returns None for all-zero input.
        """
        normalized = z_score_normalization(self.zero_series)
        self.assertEqual(
            first=normalized,
            second=None,
            msg="Function should return None for all-zero input series!",
        )

    def test_z_score_normalization_inf(self) -> None:
        """
        Tests z-score normalization with input containing infinite values.

        Validates that the function returns None for input with infinite values.
        """
        normalized = z_score_normalization(self.inf_series)
        self.assertEqual(
            first=normalized,
            second=None,
            msg="Function should return None for input with infinite values!",
        )

    def test_z_score_normalization_nan(self) -> None:
        """
        Tests z-score normalization with input containing NaN values.

        Validates that the function returns None for input with NaN values.
        """
        normalized = z_score_normalization(self.nan_series)
        self.assertEqual(
            first=normalized,
            second=None,
            msg="Function should return None for input with NaN values!",
        )

    def test_max_min_normalization(self) -> None:
        """
        Tests max-min normalization function with valid input data.

        Validates that normalized data has:
        - Maximum values close to 1
        - Minimum values close to 0
        """
        # Apply max-min normalization
        normalized = max_min_normalization(self.time_series)

        # Validate maximum and minimum values for each dimension
        for i in range(normalized.shape[1]):
            time_dim = normalized[:, i]
            self.assertAlmostEqual(
                first=np.max(time_dim),
                second=1,
                delta=1e-3,
                msg="Max-min normalization failed maximum value validation!",
            )
            self.assertAlmostEqual(
                first=np.min(time_dim),
                second=0,
                delta=1e-3,
                msg="Max-min normalization failed minimum value validation!",
            )

    def test_max_min_normalization_zero(self) -> None:
        """
        Tests max-min normalization with all-zero input series.

        Validates that the function returns None for all-zero input.
        """
        normalized = max_min_normalization(self.zero_series)
        self.assertEqual(
            first=normalized,
            second=None,
            msg="Function should return None for all-zero input series!",
        )

    def test_max_min_normalization_inf(self) -> None:
        """
        Tests max-min normalization with input containing infinite values.

        Validates that the function returns None for input with infinite values.
        """
        normalized = max_min_normalization(self.inf_series)
        self.assertEqual(
            first=normalized,
            second=None,
            msg="Function should return None for input with infinite values!",
        )

    def test_max_min_normalization_nan(self) -> None:
        """
        Tests max-min normalization with input containing NaN values.

        Validates that the function returns None for input with NaN values.
        """
        normalized = max_min_normalization(self.nan_series)
        self.assertEqual(
            first=normalized,
            second=None,
            msg="Function should return None for input with NaN values!",
        )

    def test_is_all_zeros(self) -> None:
        """
        Tests the function that detects all-zero time series.

        Validates that:
        - All-zero series are correctly identified
        - Non-zero series are correctly identified as not all-zero
        """
        # Test all-zero series
        self.assertTrue(
            expr=is_all_zeros(self.zero_series),
            msg="All-zero series should return True!",
        )

        # Test non-zero series
        self.assertFalse(
            expr=is_all_zeros(self.time_series),
            msg="Non-zero series should return False!",
        )

    def test_get_time_now(self) -> None:
        """
        Tests the function that returns current timestamp.

        Validates that:
        - Return value is a string
        - Format follows expected pattern (YYYYMMDDHHMMSS)
        """
        current_time = get_time_now()

        # Validate return type
        self.assertIsInstance(obj=current_time, cls=str)

        # Validate format (extract date part and check length)
        date_part = current_time.split(" ")[0]
        self.assertEqual(
            first=len(date_part), second=10, msg="Date format should be YYYYMMDD!"
        )

    def test_ensure_directory_exists(self) -> None:
        """
        Tests the directory creation utility function.

        Validates that:
        - Directories are created when they don't exist
        - Correct file paths are returned for both directory and file inputs
        """
        test_dir = "./data"

        if not path.exists(test_dir):
            # Test directory creation
            ensure_directory_exists(test_dir)
            self.assertTrue(
                expr=path.exists(path=test_dir),
                msg="Directory should be created when it doesn't exist!",
            )
        else:
            # Test path return for existing directory
            return_path = ensure_directory_exists(test_dir)
            expected_path = path.join(test_dir, "s2data.npz")
            self.assertEqual(
                first=return_path,
                second=expected_path,
                msg="Should return path with default filename for directory input!",
            )

    # def test_save_npy(self) -> None:
    #     """
    #     Tests the function that saves data to NPY format.
    #
    #     Validates that the save operation returns success status.
    #     """
    #     status = save_npy(data=self.data, save_path=self.npy_path)
    #     self.assertTrue(
    #         expr=status, msg="NPY save operation should return True on success!"
    #     )

    # def test_load_npy(self) -> None:
    #     """
    #     Tests the function that loads data from NPY format.
    #
    #     Validates that loaded data matches the original saved data.
    #     """
    #     loaded_data = load_npy(data_path=self.npy_path)
    #
    #     # Validate all key-value pairs match
    #     for key in self.data.keys():
    #         self.assertEqual(
    #             first=loaded_data[key],
    #             second=self.data[key],
    #             msg="Loaded data should match original data!",
    #         )

    # def test_save_npz(self) -> None:
    #     """
    #     Tests the function that saves data to NPZ format.
    #
    #     Validates that the save operation returns success status.
    #     """
    #     status = save_npz(data=self.data, save_path=self.npz_path)
    #     self.assertTrue(
    #         expr=status, msg="NPZ save operation should return True on success!"
    #     )

    # def test_load_npz(self) -> None:
    #     """
    #     Tests the function that loads data from NPZ format.
    #
    #     Validates that loaded data matches the original saved data.
    #     """
    #     loaded_data = load_npz(data_path=self.npz_path)
    #
    #     # Validate all key-value pairs match
    #     for key in self.data.keys():
    #         self.assertEqual(
    #             first=loaded_data[key],
    #             second=self.data[key],
    #             msg="Loaded data should match original data!",
    #         )

    # def test_save_s2data(self) -> None:
    #     """
    #     Tests the function that saves S2 data in multiple formats.

    #     Validates that save operations return success status for both NPY and NPZ formats.
    #     """
    #     # Test NPY format
    #     status = save_s2data(
    #         save_path=self.s2_npy_path,
    #         symbol=self.data["symbol"],
    #         excitation=self.data["excitation"],
    #         response=self.data["response"],
    #     )
    #     self.assertTrue(
    #         expr=status, msg="S2 data NPY save should return True on success!"
    #     )

    #     # Test NPZ format
    #     status = save_s2data(
    #         save_path=self.s2_npz_path,
    #         symbol=self.data["symbol"],
    #         excitation=self.data["excitation"],
    #         response=self.data["response"],
    #     )
    #     self.assertTrue(
    #         expr=status, msg="S2 data NPZ save should return True on success!"
    #     )

    # def test_load_s2data(self) -> None:
    #     """
    #     Tests the function that loads S2 data from file.

    #     Validates that loaded S2 data matches the original saved data.
    #     """
    #     symbol, excitation, response = load_s2data(data_path=self.s2_npy_path)

    #     # Validate all components match
    #     self.assertEqual(
    #         first=symbol,
    #         second=self.data["symbol"],
    #         msg="Loaded symbol should match original!",
    #     )
    #     self.assertEqual(
    #         first=excitation,
    #         second=self.data["excitation"],
    #         msg="Loaded excitation should match original!",
    #     )
    #     self.assertEqual(
    #         first=response,
    #         second=self.data["response"],
    #         msg="Loaded response should match original!",
    #     )


class TestSaveTable(unittest.TestCase):
    """Tests for exporting (N, P) tables to CSV / Excel."""

    def setUp(self) -> None:
        """Prepare fixtures used by the Save Table tests."""
        self.X = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            dtype=np.float64,
        )
        self.y = np.array([0, 1, 0], dtype=np.int64)

    def test_save_csv_features_only(self) -> None:
        """Save a feature-only table to CSV and reload the original values."""
        import tempfile

        import pandas as pd

        with tempfile.TemporaryDirectory() as tmp:
            path_csv = path.join(tmp, "table.csv")
            save_table(self.X, path_csv)
            loaded = pd.read_csv(path_csv)
            self.assertEqual(list(loaded.columns), ["x0", "x1", "x2"])
            np.testing.assert_allclose(loaded.to_numpy(), self.X)

    def test_save_csv_with_target_tuple(self) -> None:
        """Save (X, y) to CSV with an appended target column."""
        import tempfile

        import pandas as pd

        with tempfile.TemporaryDirectory() as tmp:
            path_csv = path.join(tmp, "supervised.csv")
            save_table((self.X, self.y), path_csv)
            loaded = pd.read_csv(path_csv)
            self.assertEqual(list(loaded.columns), ["x0", "x1", "x2", "target"])
            np.testing.assert_array_equal(loaded["target"].to_numpy(), self.y)

    @unittest.skipUnless(
        find_spec("openpyxl") is not None,
        "openpyxl is required to write .xlsx files",
    )
    def test_save_xlsx_roundtrip(self) -> None:
        """Save a named Excel table and reload columns, features, and labels."""
        import tempfile

        import pandas as pd

        with tempfile.TemporaryDirectory() as tmp:
            path_xlsx = path.join(tmp, "nested", "table.xlsx")
            save_table(
                self.X,
                path_xlsx,
                y=self.y,
                columns=["a", "b", "c"],
                target_name="label",
            )
            loaded = pd.read_excel(path_xlsx, engine="openpyxl")
            self.assertEqual(list(loaded.columns), ["a", "b", "c", "label"])
            np.testing.assert_allclose(loaded[["a", "b", "c"]].to_numpy(), self.X)
            np.testing.assert_array_equal(loaded["label"].to_numpy(), self.y)

    def test_rejects_unknown_extension(self) -> None:
        """Raise TypeError when the output path has an unsupported extension."""
        with self.assertRaises(TypeError):
            save_table(self.X, "table.parquet")


if __name__ == "__main__":
    unittest.main()
