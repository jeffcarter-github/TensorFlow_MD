import numpy as np
import constants


class BaseEnsemble(object):
    def __init__(self, dt=None):
        self.dt = dt
        self.atom_coords = None
        self.md_integrator = None

    def set_timestep(self, dt):
        ''''''
        self.dt = dt

    def create_atoms(self, )


class NVT(BaseEnsemble):
    '''Canonical Ensemble (NVT) Simulation Class

    Attributes:
        dt (float): time step for EOM integration in picoseconds
        atom_coords: (tf.array) ...
        md_integrator: (object): alogrithm for EOM integration
        temperature (float): temperature in Kelvin
        thermostat (object): thermostat instance

    '''
    def __init__(self):
        self.temperature = None

    def set_temperature(self, temperature):
        '''Set the system temperature

        Args:
            temperature (float): in Kelvin'''
        self.temperature = temperature

    def set_thermostat(self, thermostat):
        '''Set the alorithm for temperature control

        Args:
            thermostat (object): instance of thermostat class'''
        self.thermostat = thermostat


class NVE(BaseEnsemble):
    def __init__(self):
        pass


class NPT(BaseEnsemble):
    def _init__(self):
        pass
