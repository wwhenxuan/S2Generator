# -*- coding: utf-8 -*-
"""
Created on 2025/08/25
@author:Yifan Wu
@email: wy3370868155@outlook.com
"""

import unittest
import numpy as np

from s2generator.excitation.base_excitation import BaseExcitation


class ConcreteExcitation(BaseExcitation):
    """Concrete implementation of BaseExcitation for testing purposes"""

    def generate(
        self, rng: np.random.RandomState, seq_length: int = 512, num_channels=1
    ) -> np.ndarray:
        """Generate random time series data for testing"""
        return rng.random(size=(seq_length, num_channels)).astype(self.data_type)


class TestBaseExcitation(unittest.TestCase):
    """Testing the BaseExcitation module for generating stimulus time series data"""

    # Random number generator for testing
    rng = np.random.RandomState(42)

    def setUp(self) -> None:
        """Set up test fixtures before each test method"""
        self.excitation = ConcreteExcitation()

    def test_init_default_dtype(self) -> None:
        """Test module creation with default data type"""
        excitation = ConcreteExcitation()
        self.assertEqual(
            excitation.data_type,
            np.float64,
            msg="Default data type should be np.float64",
        )
        self.assertIsInstance(
            excitation,
            BaseExcitation,
            msg="ConcreteExcitation should be instance of BaseExcitation",
        )

    def test_init_custom_dtype(self) -> None:
        """Test module creation with custom data types"""
        for dtype in [np.float32, np.float64, np.int32, np.int64]:
            excitation = ConcreteExcitation(dtype=dtype)
            self.assertEqual(
                excitation.data_type, dtype, msg=f"Data type should be {dtype}"
            )
            self.assertIsInstance(
                excitation,
                BaseExcitation,
                msg="ConcreteExcitation should be instance of BaseExcitation",
            )

    def test_str_method(self) -> None:
        """Test the magic method to get string description"""
        # Test data types and return contents
        self.assertIsInstance(
            obj=str(self.excitation),
            cls=str,
            msg="The __str__ method gets the wrong data type!",
        )
        self.assertEqual(
            str(self.excitation),
            "ConcreteExcitation",
            msg="The __str__ method returns the wrong content!",
        )

    def test_dtype_property(self) -> None:
        """Test the dtype property getter"""
        for dtype in [np.float32, np.float64, np.int32, np.int64]:
            excitation = ConcreteExcitation(dtype=dtype)
            self.assertEqual(
                excitation.dtype, dtype, msg=f"dtype property should return {dtype}"
            )
            self.assertIsInstance(
                excitation.dtype,
                type(dtype),
                msg="dtype property should return numpy dtype",
            )

    def test_create_zeros_default_params(self) -> None:
        """Test create_zeros method with default parameters"""
        zeros_array = self.excitation.create_zeros()

        # Test output data type
        self.assertIsInstance(
            zeros_array, np.ndarray, msg="create_zeros should return numpy array"
        )

        # Test output shape
        self.assertEqual(
            zeros_array.shape, (512, 1), msg="Default shape should be (512, 1)"
        )

        # Test data type
        self.assertEqual(
            zeros_array.dtype,
            self.excitation.data_type,
            msg="Array dtype should match excitation data_type",
        )

        # Test that all values are zeros
        self.assertTrue(np.all(zeros_array == 0), msg="All values should be zero")

    def test_create_zeros_custom_params(self) -> None:
        """Test create_zeros method with custom parameters"""
        for n_points in [64, 128, 256, 1024]:
            for dimension in [1, 2, 3, 5, 10]:
                zeros_array = self.excitation.create_zeros(
                    seq_length=n_points, num_channels=dimension
                )

                # Test output shape
                self.assertEqual(
                    zeros_array.shape,
                    (n_points, dimension),
                    msg=f"Shape should be ({n_points}, {dimension})",
                )

                # Test data type
                self.assertEqual(
                    zeros_array.dtype,
                    self.excitation.data_type,
                    msg="Array dtype should match excitation data_type",
                )

                # Test that all values are zeros
                self.assertTrue(
                    np.all(zeros_array == 0), msg="All values should be zero"
                )

    def test_create_zeros_different_dtypes(self) -> None:
        """Test create_zeros method with different data types"""
        for dtype in [np.float32, np.float64, np.int32, np.int64]:
            excitation = ConcreteExcitation(dtype=dtype)
            zeros_array = excitation.create_zeros(seq_length=100, num_channels=2)

            # Test data type
            self.assertEqual(
                zeros_array.dtype, dtype, msg=f"Array dtype should be {dtype}"
            )

            # Test that all values are zeros
            self.assertTrue(np.all(zeros_array == 0), msg="All values should be zero")

    def test_generate_method(self) -> None:
        """Test the concrete implementation of generate method"""
        for n_points in [32, 128, 256, 512]:
            for dimension in [1, 3, 5]:
                time_series = self.excitation.generate(
                    rng=self.rng, seq_length=n_points, num_channels=dimension
                )

                # Test output data type
                self.assertIsInstance(
                    time_series, np.ndarray, msg="generate should return numpy array"
                )

                # Test output shape
                self.assertEqual(
                    time_series.shape,
                    (n_points, dimension),
                    msg=f"Shape should be ({n_points}, {dimension})",
                )

                # Test data type
                self.assertEqual(
                    time_series.dtype,
                    self.excitation.data_type,
                    msg="Array dtype should match excitation data_type",
                )

    def test_abstract_class_instantiation(self) -> None:
        """Test that BaseExcitation cannot be instantiated directly"""
        with self.assertRaises(TypeError):
            BaseExcitation()

    def test_inheritance(self) -> None:
        """Test that ConcreteExcitation properly inherits from BaseExcitation"""
        self.assertTrue(
            issubclass(ConcreteExcitation, BaseExcitation),
            msg="ConcreteExcitation should be subclass of BaseExcitation",
        )

        self.assertIsInstance(
            self.excitation,
            BaseExcitation,
            msg="ConcreteExcitation instance should be instance of BaseExcitation",
        )


if __name__ == "__main__":
    unittest.main()
