"""
Riemann solvers and flux computation
"""
import numpy as np
from .eos import cons_to_prim, prim_to_cons, prim_to_flux, sound_speed
from .indices import IVEL, IENE
from .utils import all_not_none

from numba import njit

Array = np.ndarray

USE_NUMBA = True

@njit(cache=True, parallel=True)
def rusanov_flux_jit(
    WL: Array,
    WR: Array,
    gamma: float,
) -> Array:
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

@njit(cache=True, parallel=True)
def rusanov_flux_jit_heatf(
    WL: Array,
    WR: Array,
    gamma: float,
    heatf_l: Array,
    heatf_r: Array,
    heatf_flux: Array,
) -> Array:
    vL = WL[IVEL]
    vR = WR[IVEL]

    qL = prim_to_cons(WL, gamma)
    qR = prim_to_cons(WR, gamma)

    fL = prim_to_flux(WL, gamma)
    fR = prim_to_flux(WR, gamma)

    csL = sound_speed(WL, gamma)
    csR = sound_speed(WR, gamma)

    max_c = 0.5 * (csL + np.abs(vL) + csR + np.abs(vR))

    # NOTE(cmo): The flux of heatf is 0 -- this is essentially a diffusion
    # term. The true update happens in the source term.
    heatf_flux[1:] = -0.5 * max_c * (heatf_r - heatf_l)
    fL[IENE] += heatf_l
    fR[IENE] += heatf_r

    flux = 0.5 * (fL + fR - max_c * (qR - qL))

    return flux

def rusanov_flux(
    WL: Array,
    WR: Array,
    gamma: float,
    heatf_l: Array | None = None,
    heatf_r: Array | None = None,
    heatf_flux: Array | None = None,
) -> Array:
    """Rusanov (local Lax-Friedrichs) Riemann solver.

    Args:
        WL: Left primitive state
        WR: Right primitive state
        gamma: Adiabatic index
        heatf_l: Reconstructed htc q
        heatf_r: Reconstructed htc q
        heatf_flux: Output flux in heatf (length should be number of interfaces
        including ghosts)

    Returns:
        flux: Numerical flux
    """
    if USE_NUMBA:
        if all_not_none(heatf_l, heatf_r, heatf_flux):
            return rusanov_flux_jit_heatf(WL, WR, gamma, heatf_l, heatf_r, heatf_flux)
        else:
            return rusanov_flux_jit(WL, WR, gamma)
    else:
        vL = WL[IVEL]
        vR = WR[IVEL]

        qL = prim_to_cons(WL, gamma)
        qR = prim_to_cons(WR, gamma)

        fL = prim_to_flux(WL, gamma)
        fR = prim_to_flux(WR, gamma)

        csL = sound_speed(WL, gamma)
        csR = sound_speed(WR, gamma)

        max_c = 0.5 * (csL + np.abs(vL) + csR + np.abs(vR))

        if all_not_none(heatf_l, heatf_r, heatf_flux):
            # NOTE(cmo): The flux of heatf is 0 -- this is essentially a diffusion
            # term. The true update happens in the source term.
            heatf_flux[1:] = -0.5 * max_c * (heatf_r - heatf_l)
            fL[IENE] += heatf_l
            fR[IENE] += heatf_r

        flux = 0.5 * (fL + fR - max_c * (qR - qL))

        return flux

@njit(cache=True, parallel=True)
def hll_flux_jit(
        WL: Array,
        WR: Array,
        gamma: float,
    ) -> Array:
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

@njit(cache=True, parallel=True)
def hll_flux_jit_heatf(
        WL: Array,
        WR: Array,
        gamma: float,
        heatf_l: Array | None = None,
        heatf_r: Array | None = None,
        heatf_flux: Array | None = None,
    ) -> Array:
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

    # NOTE(cmo): The flux of heatf is 0 -- this is essentially a diffusion
    # term. The true update happens in the source term.
    heatf_flux[1:] = sM * sR * sL * (heatf_r - heatf_l)
    fL[IENE] += heatf_l
    fR[IENE] += heatf_r

    flux = sM * (sR * fL - sL * fR + sR * sL * (qR - qL))

    return flux


def hll_flux(
        WL: Array,
        WR: Array,
        gamma: float,
        heatf_l: Array | None = None,
        heatf_r: Array | None = None,
        heatf_flux: Array | None = None,
    ) -> Array:
    """HLL (Harten-Lax-van Leer) Riemann solver.

    Uses Einfeldt wave speed estimates.

    Args:
        WL: Left primitive state
        WR: Right primitive state
        gamma: Adiabatic index
        heatf_l: Reconstructed htc q
        heatf_r: Reconstructed htc q
        heatf_flux: Output flux in heatf (length should be number of interfaces
        including ghosts)

    Returns:
        flux: Numerical flux
    """
    if USE_NUMBA:
        if all_not_none(heatf_l, heatf_r, heatf_flux):
            return hll_flux_jit_heatf(WL, WR, gamma, heatf_l, heatf_r, heatf_flux)
        else:
            return hll_flux_jit(WL, WR, gamma)
    else:
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

        if all_not_none(heatf_l, heatf_r, heatf_flux):
            # NOTE(cmo): The flux of heatf is 0 -- this is essentially a diffusion
            # term. The true update happens in the source term.
            heatf_flux[1:] = sM * sR * sL * (heatf_r - heatf_l)
            fL[IENE] += heatf_l
            fR[IENE] += heatf_r

        flux = sM * (sR * fL - sL * fR + sR * sL * (qR - qL))

        return flux
