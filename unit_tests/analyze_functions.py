import numpy as np
import tensorflow as tf
import unittest

from TensorFlow_MD.Analyze import energy


class TestEneryFunctions(unittest.TestCase):

    def test_calc_kinetic(self):
        np.random.seed(0)
        shape = (10, 3)
        v_0 = np.random.randn(10, 3)

        v_t = tf.placeholder_with_default(input=v_0, shape=shape)
        ke_t = energy.calc_kinetic(v_t)

        with tf.Session() as session:
            ke = session.run(ke_t)
        self.assertAlmostEqual(ke, 0.045180565447062344)

    def test_calc_potential(self):
        np.random.seed(0)
        shape = (10, 10)
        u_ij_0 = np.random.randn(10, 10)

        u_ij_t = tf.placeholder_with_default(input=u_ij_0, shape=shape)
        u_t = energy.calc_potential(u_ij_t)

        with tf.Session() as session:
            u = session.run(u_t)
        self.assertAlmostEqual(u, 40.485290778709988)

    def test_calc_temperature(self):
        np.random.seed(0)
        shape = (10, 3)
        v_0 = np.random.randn(10, 3)

        v_t = tf.placeholder_with_default(input=v_0, shape=shape)
        temperature_t = energy.calc_temperature(v_t)

        with tf.Session() as session:
            temperature = session.run(temperature_t)
        self.assertAlmostEqual(temperature, 1.51572631744)


if __name__ == '__main__':
    unittest.main()