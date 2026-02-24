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
)

import astropy.constants as const
# M_P = const.m_p.value
# K_B = const.k_B.value

M_P = 1.0
K_B = 1.0

# Configuration for this experiment
config = {
    "max_time": 1.0,
    "output_cadence": 0.125,
    "max_cfl": 0.8,
    "gamma": 5/3,
    "num_grid_points": 250,
    "x_min": 0.0,
    "x_max": 1.0,
    "kappa0": 1.0,
    "h_mass": M_P,
    "k_B": K_B,
    "y": np.zeros(256),
}


def conduction_ics(x, gamma):
    temperature = 0.1 + 0.9*x**5
    rho = np.ones_like(x) * M_P
    v = np.zeros_like(x)
    p = rho / ((1.0 + config["y"]) * M_P) * K_B * temperature
    w = np.stack([
        rho,
        v,
        p,
        np.zeros_like(x),
    ])
    return prim_to_cons(w, gamma=gamma)


def conduction_bcs():
    return [USER_BC, USER_BC]

def conduction_left_bc(Q, dt, gamma):
    Q[IRHO, :NUM_GHOST] = 1.0
    Q[IMOM, :NUM_GHOST] = 0.0
    p = Q[IRHO, :NUM_GHOST] / ((1.0 + config["y"][:NUM_GHOST]) * M_P) * K_B * 0.1
    Q[IENE, :NUM_GHOST] = p / (gamma - 1.0)
    Q[IIONE, :NUM_GHOST] = 0.0

def conduction_right_bc(Q, dt, gamma):
    Q[IRHO, -NUM_GHOST:] = 1.0
    Q[IMOM, -NUM_GHOST:] = 0.0
    p = Q[IRHO, -NUM_GHOST:] / ((1.0 + config["y"][-NUM_GHOST:]) * M_P) * K_B * 1.0
    Q[IENE, -NUM_GHOST:] = p / (gamma - 1.0)
    Q[IIONE, -NUM_GHOST:] = 0.0

