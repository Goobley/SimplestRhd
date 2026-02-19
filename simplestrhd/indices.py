"""
Index definitions and constants for SimplestRhd
"""
import astropy.constants as const

# Boundary condition indices
SYMMETRIC_BC = 0
REFLECTING_BC = 1
FIXED_BC = 2
USER_BC = 3

# Variable indices - conserved variables (Q)
IRHO = 0
IMOM = 1
IENE = 2
IIONE = 3

# Variable indices - primitive variables (W)
IVEL = 1
IPRE = 2

# Number of equations (must match the number of variables -- excluding tracers)
NUM_EQ = 4

# Number of ghost cells (should always be sufficient)
NUM_GHOST = 3

# Physical constants
k_B = const.k_B.value
