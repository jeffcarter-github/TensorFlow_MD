import tensorflow as tf

from TensorFlow_MD.utilities import constants


def calc_kinetic(velocity_components, mass):
    '''
        Args:
            velocity_components (tf.array, shape (n, 3)): v_i (x, y, z)
            mass (tf.array, shape (n, 1)): mass in g / mol
        Returns:
            instantaneous total kinetic energy (np.float) in kcal / mol
    '''

    # in units of g / mol * square(Angstroms / picosecond) to kcal / mol
    conversion = 2.39e-3

    ke = tf.reduce_sum(
        tf.multiply(mass, tf.norm(velocity_components, ord=2, axis=1)))

    return conversion * ke


def calc_potential(potential_ij):
    '''
    Sums over all pairwise interactions...
        Args:
            potential_ij (tf.array): U_ij i.e. pairwise potentials
        Returns:
            instantaneous total potential energy (np.float) in kcal / mol
    '''
    return tf.divide(tf.norm(potential_ij, ord=1), 2.0)


def calc_temperature(velocity_components, mass):
    '''
    Calculates instantaneous thermodynamic temperature...

        Args:
            velocity_components (tf.array): v_i (x, y, z)
            mass (tf.array): mass in g / mol

        Returns:
            temperature (np.float): T = 2 / (3 k_b) * <KE>
    '''
    # in units of g / mol * square(Angstroms / picosecond) to kcal / mol
    conversion = 2.39e-3
    scalar = 2.0 / (3.0 * constants.k_boltzman)
    mean_ke = tf.reduce_mean(tf.multiply(mass, tf.norm(velocity_components, ord=2, axis=1)))

    return scalar * conversion * mean_ke
