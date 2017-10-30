import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial.distance import pdist
from TensorFlow_MD.utilities.constants import n_avogadro


class Liquid(object):
    '''
    Args:
        density (float): density of fluid in g/ml
        mass (float): mass of atom or particle in g/mol
        boundary (list or tuple): (x_length, y_length, z_length), lengths in Angstroms
    '''
    def __init__(self, density, mass, boundary, seed=None):
        self.boundary = boundary
        self.coordinates = None
        self.assignments = None
        self.n_particles = None
        np.random.seed(seed)

        self.calc_n_particles(density, mass)
        self.calc_boundary(density, mass)
        self.generate_coordinates()

    def generate_coordinates(self):
        self.coordinates = np.random.rand(self.n_particles, 3)
        self.coordinates[:, ] = self.boundary * self.coordinates[:, ]

    def calc_n_particles(self, density, mass):
        '''calculates the number of particles within nominal boundary lengths'''
        volume = self.calc_volume()
        self.n_particles = int(density * 1e-24 / mass * n_avogadro * volume)

    def calc_boundary(self, density, mass):
        '''scale nominal boundary values to ensure the correct density'''
        volume = self.n_particles * mass / (density * 1e-24 * n_avogadro)
        scalar = np.cbrt(volume / self.calc_volume())
        self.boundary = [scalar * boundary for boundary in self.boundary]

    def calc_pairwise_distance(self):
        return pdist(self.coordinates)

    def calc_volume(self):
        '''calculates the volume given a list of boundary lengths'''
        volume = 1.0
        for limit in self.boundary:
            volume = volume * limit
        return volume

    def plot_liqud(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.coordinates[:, 0],
                   self.coordinates[:, 1],
                   self.coordinates[:, 2])
        plt.show()


if __name__ == '__main__':

    water = Liquid(1.0, 18.0, (20.0, 20.0, 20.0), 1)
    print water.n_particles
    water.plot_liqud()
    print 'done'
