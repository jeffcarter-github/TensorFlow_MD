class BaseThermostat(object):
    def __init__(self, temperature):
        self.temperature = temperature

    def adjust_temperature(self):
        raise Exception('')


class NoseHoover(BaseThermostat):
    ''' balh '''
    def __init(self, temperature, Q_mass=None):
        if Q_mass:
            self.Q_mass = Q_mass
        else:
            self.Q_mass = '' # 6nkbT
