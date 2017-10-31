import tensorflow as tf

from TensorFlow_MD.utilities import constants


class BaseThermostat(object):
    def __init__(self, temperature=None):
        self.temperature = temperature

    def set_temperature(self, temperature):
        '''
        Adjust desired simulation temperature

        Args:
            temperature (float): target temperature in K
        '''
        self.temperature = temperature


class Anderson(BaseThermostat):
    '''
    Monte Carlo thermostat that simulates stochastic
    molecular collisions with an invisible heat bath...
    The coupling strength is defined by the rate of
    collisions with the heat bath... Successive collisions
    are assumed to be uncorrelated and modeled by the
    Poisson Distribution... The velocity distribution
    is the 3D Maxwell-Boltmann (Gaussian) probability distribution:

    f(v) = (alpha / pi)^(3/2) exp(- alpha * v^2;
        where alpha = m / (2 * k_b * T) and thus
              sigma = 1 / sqrt(2 * alpha)

    Args:
        temperature (float): target temperature in K
        dt (float): time step in ps
        freq (float): collision frequency in ps-1 typical(1 to 50 ps-1)
    '''
    def __init__(self, temperature, dt, freq=5.0):
        super(Anderson, self).__init__(temperature)
        self.threshold = self._calculate_threshold(freq, dt)

    def set_collision_frequency(self, collision_freq):
        '''
        Adjust collision frequency with heat bath.

        Args:
            freq (float): collision frequency in ps-1
        '''
        self.temperature = temperature

    def scale_velocities(self, velocities, mass):
        '''
        Monte Carlo algorithm for scaling particle velocities...

        Args:
            velocities (tf.array, shape=(n, 3)): particle Velocities
            mass (tf.array, shape=(n,1)): particle masses

        Returns:
            scaled velocities (tf.array, shape=(n, 3)): particle Velocities
        '''
        # L2 Norm of the Velocities... explicit reshape for broadcasting...
        v_L2 = tf.reshape(tf.norm(velocities, ord=2, axis=1), (-1, 1))
        # mask on which particles to resample Velocites...
        random_numbers = tf.random_uniform(shape=tf.shape(v_L2))
        mask = tf.cast(random_numbers < self.threshold, tf.float64)
        # vectorized Maxwell Boltzmann (Gaussian Distribution) Sigma values...
        sigma = tf.divide(constants.k_boltzman * self.temperature, mass)
        # Generate L2 norm for each uniquely sampled velocity distribution...
        new_v_L2 = tf.square(
            tf.distributions.Normal(
                loc=tf.zeros_like(sigma), scale=sigma).sample())
        # scalar ratio of L2 Norms to maintain vector, modify speed...
        scalar =\
            tf.divide(tf.reshape(new_v_L2, (-1,)), tf.reshape(v_L2, (-1,)))
        # scaled velocities... for each particle...
        new_velocities = tf.multiply(tf.reshape(scalar, (-1, 1)), velocities)
        # return only Monte Carlo Sampling...
        return tf.add(
            tf.multiply(mask, new_velocities),
            tf.multiply(1.0 - mask, velocities))

    def _calculate_threshold(self, freq, dt):
        '''
        Stochastic collisions can be sampled by selecting
        random numbers from a uniform distribution on the
        range [0,1) and comparing them to a threshold value.
        If the random number is greater than the threshold,
        then the particle velocity will be resampled...

        Returns:
            threshold (float): resampling probability
                spanning [0,1] in collisions per time step
        '''

        return min(1, freq * dt)
