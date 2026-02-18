"""SimplestRhd: A minimal 1D hydrodynamics solver"""

__version__ = "0.1.0"

from .eos import *
from .solver import *
from .reconstruction import *
from .conduction import *
from .simulator import *
from .indices import *

__all__ = [
    # EOS
    "cons_to_prim",
    "prim_to_cons",
    "prim_to_flux",
    "sound_speed",
    "temperature_si",
    "DEFAULT_GAMMA",
    # Solver
    "rusanov_flux",
    "hll_flux",
    # Reconstruction
    "reconstruct_fog",
    "reconstruct_plm",
    "reconstruct_ppm",
    # Conduction
    "implicit_thermal_conduction",
    # Simulator
    "run_sim",
    "run_step",
    "compute_dt",
    "set_bcs",
    "TimestepInfo",
    # Indices
    "IRHO",
    "IMOM",
    "IENE",
    "IION",
    "IVEL",
    "IPRE",
    "NUM_EQ",
    "SYMMETRIC_BC",
    "REFLECTING_BC",
    "FIXED_BC",
    "USER_BC",
    "k_B",
]
