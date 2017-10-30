import numpy as np

DEGREES_90 = 90.0 / 180.0 * np.pi


def parse_cubic(length_a, n_cells):
    l = {'a': length_a, 'b': length_a, 'c': length_a}
    a = {'alpha': DEGREES_90, 'beta': DEGREES_90, 'gamma': DEGREES_90}
    n_cells = {k: v for k, v in zip(['n_a', 'n_b', 'n_c'], n_cells)}
    return l, a, n_cells


def build_cubic(length_a, n_cells, orientation):
    l, a, n_cells = parse_cubic(length_a, n_cells)
    return build_triclinic(bravais_vectors(l, a), n_cells)


def build_bcc(length_a, n_cells, orientation):
    l, a, n_cells = parse_cubic(length_a, n_cells)
    cubic_atoms, assignment_0 = build_triclinic(bravais_vectors(l, a), n_cells)
    center_atoms, assignment_1 = build_center(bravais_vectors(l, a), n_cells)
    atoms = np.concatenate([cubic_atoms, center_atoms], axis=0)
    assignment = assignment_0 + assignment_1
    return atoms, assignment


def build_fcc(length_a, n_cells, orientation):
    l, a, n_cells = parse_cubic(length_a, n_cells)
    cubic_atoms, assignment_0 = build_triclinic(bravais_vectors(l, a), n_cells)
    face_atoms, assignment_1 = build_face(bravais_vectors(l, a), n_cells)
    atoms = np.concatenate([cubic_atoms, face_atoms], axis=0)
    assignment = assignment_0 + assignment_1
    return atoms, assignment


def bravais_vectors(length, angle):
    '''
    Builds Bravais Vectors...

    Args:
        length (dictionary): (a, b, c) in Angstroms
        angle (dictionary): (alpha, beta, gamma) in radians

    Returns:
        vectors (tuple of np.array): (a, b, c)
    '''

    # cartesian unit vectors...
    x = np.array([1.0, 0.0, 0.0])
    y = np.array([0.0, 1.0, 0.0])
    z = np.array([0.0, 0.0, 1.0])

    # projections onto cartesian unit vectors...
    a_x = x
    b_x = np.cos(angle['gamma'])
    b_y = np.sin(angle['gamma'])

    c_x = np.cos(angle['beta'])
    c_y = (np.cos(angle['alpha']) - np.cos(angle['gamma']) * np.cos(angle['beta'])) / np.sin(angle['gamma'])
    c_z = np.sqrt(1.0 - c_x * c_x - c_y * c_y)

    # Bravais Lattice Vectors...
    r_a = length['a'] * a_x * x
    r_b = length['b'] * (b_x * x + b_y * y)
    r_c = length['c'] * (c_x * x + c_y * y + c_z * z)

    return r_a, r_b, r_c


def build_triclinic(bravais_tuple, n_cells):

    r_a, r_b, r_c = bravais_tuple

    pts = [r_a * i + r_b * j + r_c * k
           for i in range(n_cells['n_a'] + 1)
           for j in range(n_cells['n_b'] + 1)
           for k in range(n_cells['n_c'] + 1)]

    return np.array(pts), ['corner' for _ in pts]


def build_center(bravais_tuple, n_cells):
    r_a, r_b, r_c = bravais_tuple
    r_center = r_a / 2.0 + r_b / 2.0 + r_c / 2.0

    pts = [r_center + r_a * i + r_b * j + r_c * k
           for i in range(n_cells['n_a'])
           for j in range(n_cells['n_b'])
           for k in range(n_cells['n_c'])]

    return np.array(pts), ['center' for _ in pts]


def build_face(bravais_tuple, n_cells):
    r_a, r_b, r_c = bravais_tuple
    r_f_ab = r_a / 2.0 + r_b / 2.0
    r_f_ac = r_a / 2.0 + r_c / 2.0
    r_f_bc = r_b / 2.0 + r_c / 2.0

    pts_ab = [r_f_ab + r_a * i + r_b * j + r_c * k
              for i in range(n_cells['n_a'])
              for j in range(n_cells['n_b'])
              for k in range(n_cells['n_c'] + 1)]

    pts_ac = [r_f_ac + r_a * i + r_b * j + r_c * k
              for i in range(n_cells['n_a'])
              for j in range(n_cells['n_b'] + 1)
              for k in range(n_cells['n_c'])]

    pts_bc = [r_f_bc + r_a * i + r_b * j + r_c * k
              for i in range(n_cells['n_a'] + 1)
              for j in range(n_cells['n_b'])
              for k in range(n_cells['n_c'])]

    pts = pts_ab + pts_ac + pts_bc

    return np.array(pts), ['face' for _ in pts]
