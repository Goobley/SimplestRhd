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
import astropy.units as u

# L0 = 1e6
# rho0 = 1e5 * const.m_p.value
# v0 = 5.25e3 # 5.25 km/s
# t0 = L0 / v0

# L0 = 1e6
# rho0 = 1e10 * const.m_p.value
# v0 = 7.8e3
# t0 = L0 / v0
# t_max = 0.2

L0 = 1e6
rho0 = 1e15 * const.m_p.value
v0 = 7.8e3
t0 = L0 / v0
t_max = 0.2

# T_ion = 157888.0
# nq = (2.0 * np.pi * const.m_e.value * const.k_B.value * T_ion / const.h.value**2)**1.5
# L0 = nq**(-1/3)
# rho0 = nq * const.m_p.value
# v0 = np.sqrt(const.k_B.value * T_ion / const.m_p.value)
# t0 = L0 / v0
# P0 = nq * const.k_B.value * T_ion
# t_max = 0.25


L0 = 1e6
rho0 = 3.8497e-11 * u.Unit("g/cm3").to("kg/m3")
rho0_p = 1e20 * const.m_p.value
v0 = 7.8e3
t0 = L0 / v0
t_max = 0.09945629

# Configuration for this experiment
config = {
    "max_time": t_max * t0,
    "output_cadence": 0.1 * t0 * t_max,
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
        np.where(x < 0.5 * L0, 1.0, 0.1 * 0.422265) * rho0_p * v0**2,
        np.zeros_like(x),
    ])
    # y0 = 0.095
    # temperature = y0 * T_ion
    # w = np.stack([
    #     np.where(x < 0.5 * L0, 6.0, 8.0) * 1e-5 * rho0,
    #     np.where(x < 0.5 * L0, -0.5, 0.9) * v0,
    #     # np.where(x < 0.5 * L0, 6.3, 8.3) * 1e-6 * P0,
    #     np.zeros_like(x),
    #     np.zeros_like(x),
    # ])
    # w[IPRE] = w[IRHO] / const.m_p.value * (1.0 + y0) * const.k_B.value * temperature
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
