#!/usr/bin/python3.4

import unittest
from signal_model import UnitVector
import numpy as np


class UnitVectorTestCase(unittest.TestCase):
    """
    Unit tests for the vector test case.
    """
    def test_theta(self):
        """Test what the vector is for easy cases.
        :returns: TODO

        """
        vec = UnitVector(0, np.pi / 2)
        self.assertEqual(vec.Omega, np.array([0, 0, 1]))
        self.assertEqual(vec.m, np.array([-1, 0, 0]))
        self.assertEqual(vec.n, np.array([0, 1, 0]))

if __name__ == "__main__":
    unittest.main()
