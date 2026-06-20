# -*- coding: utf-8 -*-
"""
Created on 2025/02/15 16:18:33
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""

import sys
import unittest

import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    # Creating a test case loader
    test_suite = unittest.defaultTestLoader.discover(".", "*test_*.py")
    # Test case runner
    test_runner = unittest.TextTestRunner(
        resultclass=unittest.TextTestResult, verbosity=2
    )
    # Execute all test cases in the current directory
    result = test_runner.run(test_suite)  # The run method will return the test results
    sys.exit(not result.wasSuccessful())
