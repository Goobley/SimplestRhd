"""
Sod Shock Tube Test Case

Initial conditions and helper functions for the Sod shock tube problem.
"""
import numpy as np
from simplestrhd import (
    prim_to_cons,
    IRHO,
    IVEL,
    IPRE,
    IIONE,
    NUM_GHOST,
    SYMMETRIC_BC,
    IMOM,
    IENE,
)

# Configuration for this experiment
config = {
    "max_time": 0.2,
    "output_cadence": 0.1,
    "max_cfl": 0.1,
    "gamma": 1.4,
    "num_grid_points": 64,
    "x_min": 0.0,
    "x_max": 1.0,
}


def sod_ics(x, gamma):
    """Sod shock tube initial conditions.

    Args:
        x: Grid positions
        gamma: Adiabatic index

    Returns:
        Q: Conserved variables
    """
    w = np.stack([
        np.where(x < 0.5, 1.0, 0.125),
        np.zeros_like(x),
        np.where(x < 0.5, 1.0, 0.1),
        np.zeros_like(x),
    ])
    return prim_to_cons(w, gamma=gamma)


def sod_bcs():
    """Boundary conditions for Sod shock tube.

    Returns:
        bc_modes: Boundary condition types
    """
    return [SYMMETRIC_BC, SYMMETRIC_BC]
