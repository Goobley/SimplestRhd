import numpy as np
Array = np.ndarray
from config import DEFAULT_GAMMA, IRHO, IMOM, IENE, IVEL, IPRE, USE_CONDUCTION, CONDUCTION_ONLY, k_B

def cons_to_prim(Q: Array, gamma: float=DEFAULT_GAMMA) -> Array:
    W = np.empty_like(Q)

    rho = Q[IRHO, :]
    mom = Q[IMOM, :]
    E = Q[IENE, :]

    v = mom / rho
    kinetic = 0.5 * rho * v**2
    e = E - kinetic
    p = (gamma - 1.0) * e

    W[IRHO] = rho
    W[IVEL] = v
    W[IPRE] = p

    return W

def prim_to_cons(W: Array, gamma: float=DEFAULT_GAMMA) -> Array:
    Q = np.empty_like(W)

    rho = W[IRHO, :]
    v = W[IVEL, :]
    p = W[IPRE, :]

    mom = rho * v
    energy = p / (gamma - 1.0) + 0.5 * mom**2 / rho

    Q[IRHO] = rho
    Q[IMOM] = mom
    Q[IENE] = energy

    return Q

def prim_to_flux(W: Array, gamma: float=DEFAULT_GAMMA) -> Array:
    flux = np.empty_like(W)
    rho = W[IRHO, :]
    v = W[IVEL, :]
    p = W[IPRE, :]

    mass_flux = rho * v
    mom_flux = mass_flux * v + p

    e_kin = 0.5 * rho * v**2
    e_tot = p / (gamma - 1.0) + e_kin
    ene_flux = (e_tot + p) * v

    if USE_CONDUCTION and CONDUCTION_ONLY:
        mass_flux = 0.0
        mom_flux = 0.0
        ene_flux = 0.0

    flux[IRHO] = mass_flux
    flux[IMOM] = mom_flux
    flux[IENE] = ene_flux

    return flux

def sound_speed(W: Array, gamma: float=DEFAULT_GAMMA) -> Array:
    rho = W[IRHO]
    p = W[IPRE]
    return np.sqrt(gamma * p / rho)

def rusanov_flux(WL: Array, WR: Array, gamma: float=DEFAULT_GAMMA) -> Array:
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

def hll_flux(WL: Array, WR: Array, gamma: float=DEFAULT_GAMMA) -> Array:
    csL = sound_speed(WL, gamma)
    csR = sound_speed(WR, gamma)

    tiny = 1.0e-30
    # NOTE(cmo): Einfeldt description
    sqrt_rho_L = np.sqrt(WL[IRHO])
    sqrt_rho_R = np.sqrt(WR[IRHO])
    ubar = (sqrt_rho_L * WL[IVEL] + sqrt_rho_R * WR[IVEL]) / (sqrt_rho_L + sqrt_rho_R)
    eta2 = 0.5 * (sqrt_rho_L * sqrt_rho_R) / (sqrt_rho_L + sqrt_rho_R)**2
    dbar2 = (sqrt_rho_L * csL**2 + sqrt_rho_R * csR**2) / (sqrt_rho_L + sqrt_rho_R) + eta2 * (WR[IVEL] - WL[IVEL])**2
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


def temperature_si(pressure, n_baryon, y=1.0):
    return pressure / (n_baryon * (1.0 + y) * k_B)