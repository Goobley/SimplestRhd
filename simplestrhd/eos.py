"""
Equation of state and thermodynamic functions
"""
import numpy as np
from .indices import IRHO, IMOM, IENE, IIONE, IVEL, IPRE, k_B

Array = np.ndarray


def cons_to_prim(Q: Array, gamma: float) -> Array:
    """Convert conserved variables to primitive variables.

    Args:
        Q: Conserved variables [rho, mom, E, spec_e_ion]
        gamma: Adiabatic index

    Returns:
        W: Primitive variables [rho, v, p, spec_e_ion]
    """
    W = np.empty_like(Q)

    rho = Q[IRHO, :]
    mom = Q[IMOM, :]
    E = Q[IENE, :]
    spec_e_ion = Q[IIONE, :]

    v = mom / rho
    kinetic = 0.5 * rho * v**2
    e = E - kinetic - rho * spec_e_ion
    # e = E - kinetic - spec_e_ion
    p = (gamma - 1.0) * e

    W[IRHO] = rho
    W[IVEL] = v
    W[IPRE] = p
    W[IIONE] = spec_e_ion

    return W


def prim_to_cons(W: Array, gamma: float) -> Array:
    """Convert primitive variables to conserved variables.

    Args:
        W: Primitive variables [rho, v, p, spec_e_ion]
        gamma: Adiabatic index

    Returns:
        Q: Conserved variables [rho, mom, E, spec_e_ion]
    """
    Q = np.empty_like(W)

    rho = W[IRHO, :]
    v = W[IVEL, :]
    p = W[IPRE, :]
    spec_e_ion = W[IIONE, :]

    mom = rho * v
    energy = p / (gamma - 1.0) + 0.5 * mom**2 / rho + rho * spec_e_ion
    # energy = p / (gamma - 1.0) + 0.5 * mom**2 / rho + spec_e_ion

    Q[IRHO] = rho
    Q[IMOM] = mom
    Q[IENE] = energy
    Q[IIONE] = spec_e_ion

    return Q


def prim_to_flux(W: Array, gamma: float) -> Array:
    """Compute flux from primitive variables.

    Args:
        W: Primitive variables [rho, v, p, spec_e_ion]
        gamma: Adiabatic index

    Returns:
        flux: Fluxes of conserved variables
    """
    flux = np.empty_like(W)
    rho = W[IRHO, :]
    v = W[IVEL, :]
    p = W[IPRE, :]
    spec_e_ion = W[IIONE, :]

    mass_flux = rho * v
    mom_flux = mass_flux * v + p

    e_kin = 0.5 * rho * v**2
    e_tot = p / (gamma - 1.0) + e_kin + rho * spec_e_ion
    # e_tot = p / (gamma - 1.0) + e_kin + spec_e_ion
    ene_flux = (e_tot + p) * v

    flux[IRHO] = mass_flux
    flux[IMOM] = mom_flux
    flux[IENE] = ene_flux
    flux[IIONE] = 0.0

    return flux


def sound_speed(W: Array, gamma: float) -> Array:
    """Compute sound speed from primitive variables.

    Args:
        W: Primitive variables
        gamma: Adiabatic index

    Returns:
        cs: Sound speed
    """
    rho = W[IRHO]
    p = W[IPRE]
    return np.sqrt(gamma * p / rho)


def temperature_si(pressure, n_baryon, y=1.0):
    """Compute temperature in SI units.

    Args:
        pressure: Pressure in Pa
        n_baryon: Baryon number density in m^-3
        y: Ionization parameter (default 1.0)

    Returns:
        T: Temperature in K
    """
    return pressure / (n_baryon * (1.0 + y) * k_B)
