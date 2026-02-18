"""
Experiment configuration template

Copy this file to your experiment directory and modify as needed.
Each experiment should have its own configuration.
"""
import astropy.constants as const

# ============================================================================
# Simulation Parameters
# ============================================================================

# Time integration
MAX_TIME = 0.2
OUTPUT_CADENCE = 0.1
MAX_STEPS = 10_000_000
MAX_CFL = 0.1

# Grid
NUM_GRID_POINTS = 100
X_MIN = 0.0
X_MAX = 1.0

# ============================================================================
# Physics Parameters
# ============================================================================

# Equation of state
DEFAULT_GAMMA = 1.4

# Conduction
USE_CONDUCTION = False
CONDUCTION_ONLY = False
SATURATE_HEAT_FLUX = True
SPITZER_CONDUCTIVITY = True
HTC_HYPERDIFFUSION = 5e-2
KAPPA0 = 8e-12  # Hyperbolic conduction coefficient [W m-1 K-7/2]

# ============================================================================
# Numerical Parameters
# ============================================================================

NUM_GHOST = 2

# ============================================================================
# Physical Constants
# ============================================================================

# Base particle mass [kg]
P_MASS = 1.6737830080950003e-27

# Mean molecular mass (Pure H plasma = 0.5)
MEAN_MOLECULAR_MASS = 0.5

# Boltzmann constant [J/K]
k_B = const.k_B.value

# ============================================================================
# Boundary Condition Types
# ============================================================================

SYMMETRIC_BC = 0
REFLECTING_BC = 1
FIXED_BC = 2
USER_BC = 3
