"""SimplestRhd: A minimal 1D hydrodynamics solver"""

__version__ = "0.1.0"

from .eos import *
from .riemann_flux import *
from .reconstruction import *
from .conduction import *
from .simulation import *
from .indices import *
from .custom_eos import *
from .thin_loss import *

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
    "IIONE",
    "IVEL",
    "IPRE",
    "NUM_EQ",
    "SYMMETRIC_BC",
    "REFLECTING_BC",
    "FIXED_BC",
    "USER_BC",
    "k_B",
    "y_from_ntot",
    "y_from_nhtot",
    "lte_eos",
    "rad_loss_dm",
    "logt_DM",
    "lambda_DM",
    "TownsendThinLoss",
]
