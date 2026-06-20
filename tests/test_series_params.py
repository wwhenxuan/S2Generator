# -*- coding: utf-8 -*-
"""
Created on 2025/08/22 18:00:34
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/S2Generator
"""

import unittest
import numpy as np

from s2generator import SeriesParams


class TestSeriesParams(unittest.TestCase):
    """Test parameter object used to generate stimulus time series data"""

    def test_create(self) -> None:
        """Test the creation of verification class"""
        series_params = SeriesParams()
        # Verify that the data in the class is correct
        self.assertIsInstance(obj=series_params, cls=SeriesParams)

    def test_sampling_methods(self) -> None:
        """Test and verify that the sampling method in the class attributes is correct"""
        # Create a class instance object
        series_params = SeriesParams()
        # Iterate through different sampling methods
        for class_method, list_method in zip(
            series_params.sampling_methods,
            [
                "mixed_distribution",
                "autoregressive_moving_average",
                "forecast_pfn",
                "kernel_synth",
                "intrinsic_mode_function",
            ],
        ):
            self.assertIsInstance(
                obj=class_method, cls=str, msg="Data Type Error in the sampling type!"
            )
            self.assertEqual(
                first=class_method,
                second=list_method,
                msg="Data Error in the sampling method!",
            )

    def test_prob_array(self) -> None:
        """Test and verify that the probability array is correct"""
        # First verify the default parameters
        series_params = SeriesParams()
        prob_array = series_params.prob_array
        # print(prob_array)
        self.assertIsInstance(
            obj=prob_array, cls=np.ndarray, msg="Data Type Error in the prob_array!"
        )
        self.assertEqual(
            first=np.sum(prob_array), second=1.0, msg="Data Value in the prob_array!"
        )


if __name__ == "__main__":
    unittest.main()
