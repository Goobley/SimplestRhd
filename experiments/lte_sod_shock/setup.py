import numpy as np
from simplestrhd import (
    prim_to_cons,
    cons_to_prim,
    IRHO,
    IVEL,
    IPRE,
    IIONE,
    NUM_GHOST,
    SYMMETRIC_BC,
    IMOM,
    IENE,
    y_from_nhtot,
    y_from_ntot,
    lte_eos,
)
import astropy.constants as const

# L0 = 1e6
# rho0 = 1e5 * const.m_p.value
# v0 = 5.25e3 # 5.25 km/s
# t0 = L0 / v0

L0 = 1e6
rho0 = 1e10 * const.m_p.value
v0 = 7.8e3
t0 = L0 / v0

# L0 = 1e6
# rho0 = 1e15 * const.m_p.value
# v0 = 7.8e4
# t0 = L0 / v0

# Configuration for this experiment
config = {
    "max_time": 0.2 * t0,
    "output_cadence": 0.1 * t0,
    "max_cfl": 0.8,
    "gamma": 5/3,
    "num_grid_points": 256,
    "x_min": 0.0,
    "x_max": 1.0 * L0,
    "include_ion_e": True,
}


def lte_sod_ics(x, gamma):
    """Sod shock tube initial conditions.

    Args:
        x: Grid positions
        gamma: Adiabatic index

    Returns:
        Q: Conserved variables
    """
    w = np.stack([
        np.where(x < 0.5 * L0, 1.0, 0.125) * rho0,
        np.zeros_like(x),
        np.where(x < 0.5 * L0, 1.0, 0.1) * rho0 * v0**2,
        np.zeros_like(x),
    ])
    Q = prim_to_cons(w, gamma=gamma)
    state = dict(xcc=x, Q=Q, gamma=gamma)
    # NOTE(cmo): Fill the specific ionisation energy for this setup
    lte_eos(state, {}, include_ion_e=config["include_ion_e"], find_initial_ion_e=True, verbose=False, temp_err_bound=1e-7)
    return Q


def lte_sod_bcs():
    """Boundary conditions for Sod shock tube.

    Returns:
        bc_modes: Boundary condition types
    """
    return [SYMMETRIC_BC, SYMMETRIC_BC]
