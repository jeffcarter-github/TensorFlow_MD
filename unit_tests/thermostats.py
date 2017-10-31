import numpy as np
import tensorflow as tf
import unittest

from TensorFlow_MD.Controls.Thermostats import BaseThermostat, Anderson


class TestThermostat(unittest.TestCase):

    def test_BaseThermostat(self):

        thermostat = BaseThermostat()
        self.assertIsNone(thermostat.temperature)

        test_temperature = 298.0
        thermostat.set_temperature(test_temperature)
        self.assertEqual(thermostat.temperature, test_temperature)

    def test_Anderson_1(self):
        '''
        test temperature setter works...
        '''
        test_temperature = 298.0
        dt, freq = 0.1, 5.0
        thermostat = Anderson(test_temperature, dt, freq)
        self.assertEqual(thermostat.temperature, test_temperature)
        self.assertEqual(thermostat.threshold, dt * freq)

    def test_Anderson_2(self):
        '''
        test that with zero collision frequency with heat bath
        there is no velocity rescaling...
        '''
        n, nd = 2, 3
        shape = (n, nd)
        m_0 = np.random.rand(n)
        m = tf.placeholder_with_default(input=m_0, shape=(n,))

        v_0 = np.random.randn(n, nd)
        v_i = tf.placeholder_with_default(input=v_0, shape=shape)

        test_temperature = 298.0
        dt, freq = 0.1, 0.0
        thermostat = Anderson(test_temperature, dt, freq)

        v_j = thermostat.scale_velocities(v_i, m)

        with tf.Session() as session:
            new_velocities = session.run(v_j)
        self.assertTrue(np.all(new_velocities == v_0))

    def test_Anderson_3(self):
        '''
        test that with definate rescaling, i.e. one collision
        per time step, all the velocities are rescaled...
        '''

        n, nd = 2, 3
        shape = (n, nd)
        m_0 = np.random.rand(n)
        m = tf.placeholder_with_default(input=m_0, shape=(n,))

        v_0 = np.random.randn(n, nd)
        v_i = tf.placeholder_with_default(input=v_0, shape=shape)

        test_temperature = 298.0
        dt, freq = 0.1, 10.0
        thermostat = Anderson(test_temperature, dt, freq)

        v_j = thermostat.scale_velocities(v_i, m)

        with tf.Session() as session:
            new_velocities = session.run(v_j)
        self.assertTrue(np.all(new_velocities != v_0))

if __name__ == '__main__':
    unittest.main()
