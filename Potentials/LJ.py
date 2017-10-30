import json

from TensorFlow_MD.utilities import constants


def LJ(atom_id):
    '''
        LJ Potential...

        Args:
            atom_id (str): atom_id, i.e. 'Xe' or H2' or 'CH4'
        Returns:
            lambda expression with r_ij (Angstroms) as input...
            Potential Energy is kcal/mol...

    '''

    with open('../Potentials/LJ_parameters.json', 'r') as fp:
        lj_data = json.load(fp)

    if atom_id in lj_data.keys():
        data = lj_data[atom_id]
    else:
        pass

    epsilon, sigma = data['epsilon'], data['sigma']
    epsilon = epsilon * constants.k_boltzman

    return lambda r: 4.0 * epsilon * ((sigma / r)**12 - (sigma / r)**6)
