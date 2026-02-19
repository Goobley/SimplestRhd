"""
Riemann solvers and flux computation
"""
import numpy as np
from .eos import cons_to_prim, prim_to_cons, prim_to_flux, sound_speed
from .indices import IVEL

Array = np.ndarray


def rusanov_flux(WL: Array, WR: Array, gamma: float) -> Array:
    """Rusanov (local Lax-Friedrichs) Riemann solver.

    Args:
        WL: Left primitive state
        WR: Right primitive state
        gamma: Adiabatic index

    Returns:
        flux: Numerical flux
    """
    vL = WL[IVEL]
    vR = WR[IVEL]

    qL = prim_to_cons(WL, gamma)
    qR = prim_to_cons(WR, gamma)

    fL = prim_to_flux(WL, gamma)
    fR = prim_to_flux(WR, gamma)

    csL = sound_speed(WL, gamma)
    csR = sound_speed(WR, gamma)

    max_c = 0.5 * (csL + np.abs(vL) + csR + np.abs(vR))

    flux = 0.5 * (fL + fR - max_c * (qR - qL))
    return flux


def hll_flux(WL: Array, WR: Array, gamma: float) -> Array:
    """HLL (Harten-Lax-van Leer) Riemann solver.

    Uses Einfeldt wave speed estimates.

    Args:
        WL: Left primitive state
        WR: Right primitive state
        gamma: Adiabatic index

    Returns:
        flux: Numerical flux
    """
    csL = sound_speed(WL, gamma)
    csR = sound_speed(WR, gamma)

    tiny = 1.0e-30
    # NOTE(cmo): Einfeldt description
    sqrt_rho_L = np.sqrt(WL[0])
    sqrt_rho_R = np.sqrt(WR[0])
    ubar = (sqrt_rho_L * WL[IVEL] + sqrt_rho_R * WR[IVEL]) / (sqrt_rho_L + sqrt_rho_R)
    eta2 = 0.5 * (sqrt_rho_L * sqrt_rho_R) / (sqrt_rho_L + sqrt_rho_R) ** 2
    dbar2 = (
        (sqrt_rho_L * csL**2 + sqrt_rho_R * csR**2) / (sqrt_rho_L + sqrt_rho_R)
        + eta2 * (WR[IVEL] - WL[IVEL]) ** 2
    )
    dbar = np.sqrt(dbar2)

    sL = np.minimum(-tiny, ubar - dbar)
    sR = np.maximum(tiny, ubar + dbar)
    sM = 1.0 / (sR - sL)

    qL = prim_to_cons(WL, gamma)
    qR = prim_to_cons(WR, gamma)

    fL = prim_to_flux(WL, gamma)
    fR = prim_to_flux(WR, gamma)

    flux = sM * (sR * fL - sL * fR + sR * sL * (qR - qL))
    return flux
