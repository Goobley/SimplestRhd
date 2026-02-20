import numpy as np
import lightweaver as lw
from lightweaver.rh_atoms import H_6_atom

from h_collisional_rates import Johnson_CE, Johnson_CI

def H_6_cooling_collisions():
    H6 = H_6_atom()
    new_temperature_grid = np.array([1e3, 2e3, 3e3, 4e3, 5e3, 6e3, 7e3, 8e3, 9e3, 10e3, 20e3, 30e3, 50e3, 100e3, 500e3, 1e6, 2e6])
    for coll in H6.collisions:
        trans_energy = coll.jLevel.E_SI - coll.iLevel.E_SI
        n_i = np.sqrt(coll.iLevel.g / 2)
        coll.temperature = new_temperature_grid
        if isinstance(coll, lw.collisional_rates.CE):
            n_j = np.sqrt(coll.jLevel.g / 2)
            rate = Johnson_CE(n_i, n_j, trans_energy, new_temperature_grid)
        elif isinstance(coll, lw.collisional_rates.CI):
            rate = Johnson_CI(n_i, trans_energy, new_temperature_grid)
        else:
            raise ValueError("Unexpected rate on H model")
        coll.rates = rate
    lw.reconfigure_atom(H6)
    return H6