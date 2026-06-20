# -*- coding: utf-8 -*-
"""
Created on 2025/08/13 23:47:51
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""

import unittest
import numpy as np

from s2generator.excitation import AutoregressiveMovingAverage


class TestARMA(unittest.TestCase):
    """Testing the ARMA module for generating stimulus time series data"""

    # Random number generator for testing
    rng = np.random.RandomState(42)

    # Instance object for testing
    arma = AutoregressiveMovingAverage()

    def test_setup(self) -> None:
        """Test module creation process"""
        for p_max in [2, 3, 4, 5]:
            for q_max in [2, 3, 4, 5]:
                for upper_bound in [100, 200, 300, 400]:
                    # Building an excitation time series generator
                    arma = AutoregressiveMovingAverage(
                        p_max, q_max, upper_bound=upper_bound
                    )
                    self.assertIsInstance(
                        arma,
                        cls=AutoregressiveMovingAverage,
                        msg="Wrong ARMA type in `test_setup` method",
                    )

    def test_create_autoregressive_params(self) -> None:
        """Test whether the parameters of the autoregressive process can be generated normally"""
        for p_order in [1, 2, 3, 4]:
            # Traverse different orders to generate parameters
            p_params = self.arma.create_autoregressive_params(
                rng=self.rng, p_order=p_order
            )

            # Check the length of the parameter
            self.assertEqual(
                len(p_params),
                p_order,
                msg="Wrong parameter length for autoregressive process!",
            )
            self.assertIsInstance(
                p_params,
                np.ndarray,
                msg="Wrong parameter type for autoregressive process!",
            )

            # Checks whether the parameter range of the autoregressive process meets the constraints
            self.assertTrue(
                np.sum(p_params) < 1,
                msg="The sum of the parameters of the autoregressive process is not less than 1!",
            )
            self.assertTrue(
                np.abs(p_params[-1]) < 1,
                msg="The absolute value of the last parameter of the autoregressive process is not less than 1!",
            )

    def test_create_moving_average_params(self) -> None:
        """Test whether the parameters of the sliding average process can be generated normally"""
        for q_order in [1, 2, 3, 4, 5]:
            # Traverse different orders to generate parameters
            q_params = self.arma.create_autoregressive_params(
                rng=self.rng, p_order=q_order
            )

            # Check the length of the parameter
            self.assertEqual(
                len(q_params),
                q_order,
                msg="Wrong parameter length for sliding average process!",
            )
            self.assertIsInstance(
                q_params,
                np.ndarray,
                msg="The parameter type of the sliding average process is incorrect!",
            )

    def test_create_params(self) -> None:
        """Test whether the parameters of the ARAM model can be generated normally"""

        # Execute the method to create parameters
        self.arma.create_params(rng=self.rng)

        # Verify by model order and parameter array size
        p_order = self.arma.p_order
        q_order = self.arma.q_order

        self.assertEqual(
            first=p_order,
            second=len(self.arma.p_params),
            msg="The order of the autoregressive process does not match the generated parameters!",
        )
        self.assertEqual(
            first=q_order,
            second=len(self.arma.q_params),
            msg="The order of the moving average process does not match the generated parameters!",
        )

    def test_order(self) -> None:
        """Test the function that attempts to obtain the model order"""

        # Execute the method to create parameters
        self.arma.create_params(rng=self.rng)

        # Get the order of the model
        order_dict = self.arma.order

        # Test dictionary data type
        self.assertIsInstance(
            obj=order_dict,
            cls=dict,
            msg="The function that tests the order returns the wrong data type.!",
        )

        # 遍历字典测试数据类型
        for key, value in order_dict.items():
            self.assertIsInstance(obj=key, cls=str, msg="Return content error!")
            self.assertIsInstance(obj=value, cls=int, msg="Return content error!")

    def test_params(self) -> None:
        """Test the function that tries to get the model parameters"""
        # Execute the method to create parameters
        self.arma.create_params(rng=self.rng)

        # Get the parameters of the model
        params_dict = self.arma.params

        # Test dictionary data type
        self.assertIsInstance(
            obj=params_dict,
            cls=dict,
            msg="The function that tests the parameters returned an incorrect data type.!",
        )

        # Traversing the dictionary to test data types
        for key, value in params_dict.items():
            self.assertIsInstance(obj=key, cls=str, msg="Return content error!")
            self.assertIsInstance(
                obj=value, cls=np.ndarray, msg="Return content error!"
            )

    def test_generate(self) -> None:
        """Test whether the stimulus time series data can be generated correctly"""
        for p_max in [2, 3, 4]:
            # Ergodic autoregressive order
            for q_max in [2, 3, 4, 5]:
                # Ergodic moving average process order
                arma = AutoregressiveMovingAverage(p_max=p_max, q_max=q_max)

                # Execute the data generation algorithm
                for length in [32, 128, 256]:
                    # Iterate over different input lengths
                    for dim in [1, 3, 5]:
                        # Iterate over different input dimensions
                        time_series = arma.generate(
                            rng=self.rng, n_inputs_points=length, input_dimension=dim
                        )

                        # Test output data type
                        self.assertIsInstance(time_series, np.ndarray)

                        # Test output length and dimensions
                        self.assertEqual(first=time_series.shape, second=(length, dim))

    def test_call(self) -> None:
        """Test data generation class response"""
        time_series = self.arma(rng=self.rng, n_inputs_points=256, input_dimension=1)

        # Test input data type and dimension
        self.assertIsInstance(
            obj=time_series, cls=np.ndarray, msg="Wrong class response for ARMA!"
        )
        self.assertEqual(
            time_series.shape, (256, 1), msg="Wrong data dimension for ARMA!"
        )

    def test_str(self) -> None:
        """Test the magic method to get string description"""
        # Test data types and return contents
        self.assertIsInstance(
            obj=str(self.arma),
            cls=str,
            msg="The __str__ method gets the wrong data type!",
        )
        self.assertEqual(
            str(self.arma), "ARMA", msg="The __str__ method returns the wrong content!"
        )
