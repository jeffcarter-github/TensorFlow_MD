import numpy as np
import tensorflow as tf
import unittest

from TensorFlow_MD.Potentials.LJ import LJ


class TestPotentials(unittest.TestCase):

    def test_LJ(self):
        potential_func = LJ('H2')
        self.assertAlmostEquals(potential_func(3), -0.0337152553557)
        self.assertAlmostEquals(potential_func(4), -0.0384127876163)
        self.assertAlmostEquals(potential_func(5), -0.0114270710625)


if __name__ == '__main__':
    unittest.main()
