import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist
import numpy as np

import build_lattice as bl


class Lattice(object):
    def __init__(self):
        self.coordinates = None
        self.assignments = None

    def generate_cubic(self, length_a, n_cells=(1, 1, 1), orientation=None):
        self.coordinates, self.assignments =\
            bl.build_cubic(length_a, n_cells, orientation)
        self.boundary = [np.max(self.coordinates[:, i]) for i in range(self.coordinates.shape[1])]

    def generate_bcc(self, length_a, n_cells=(1, 1, 1), orientation=None):
        self.coordinates, self.assignments =\
            bl.build_bcc(length_a, n_cells, orientation)
        self.boundary = [np.max(self.coordinates[:, i]) for i in range(self.coordinates.shape[1])]

    def generate_fcc(self, length_a, n_cells=(1, 1, 1), orientation=None):
        self.coordinates, self.assignments =\
            bl.build_fcc(length_a, n_cells, orientation)
        self.boundary = [np.max(self.coordinates[:, i]) for i in range(self.coordinates.shape[1])]

    def plot_lattice(self):
        color_map = {'corner': 'b', 'face': 'g', 'center': 'r'}
        colors = [color_map[a] for a in self.assignments]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.coordinates[:, 0],
                   self.coordinates[:, 1],
                   self.coordinates[:, 2],
                   c=colors)
        plt.show()


if __name__ == '__main__':

    lattice = Lattice()
    lattice.generate_fcc(4, (1, 1, 1))
    lattice.plot_lattice()
