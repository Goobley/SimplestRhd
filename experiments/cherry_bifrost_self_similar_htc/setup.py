import numpy as np
from simplestrhd import (
    prim_to_cons,
    IRHO,
    IVEL,
    IPRE,
    IIONE,
    NUM_GHOST,
    USER_BC,
    IMOM,
    IENE,
    PERIODIC_BC,
)

import astropy.constants as const
M_P = const.m_p.value
K_B = const.k_B.value

DENSITY_SCALE = 1e-12
LENGTH_SCALE = 1e6
T0 = 9.9e5
Tinf = 1e4

# Configuration for this experiment
config = {
    "max_time": 3.2,
    "output_cadence": 0.4,
    "max_cfl": 0.5,
    "gamma": 5/3,
    "num_grid_points": 250,
    "x_min": -LENGTH_SCALE,
    "x_max": LENGTH_SCALE,
    "kappa0": 1e-11,
    "h_mass": M_P,
    "k_B": K_B,
    "y": np.ones(256),
    "saturate_flux": True,
    # "htc_hyperdiff": 3e-2,
    "htc_hyperdiff": 0.0,
    "htc_despike": 0.3,
    "htc_order": 1,
    "htc_use_riemann_flux": True,
}


def conduction_ics(x, gamma):
    temperature = T0 * np.exp(-x**2 / 0.1e6**2) + Tinf
    rho = np.ones_like(x) * DENSITY_SCALE
    v = np.zeros_like(x)
    p = rho / M_P * (1.0 + config["y"]) * K_B * temperature
    w = np.stack([
        rho,
        v,
        p,
        np.zeros_like(x),
    ])
    return prim_to_cons(w, gamma=gamma)


def conduction_bcs():
    return [PERIODIC_BC, PERIODIC_BC]


