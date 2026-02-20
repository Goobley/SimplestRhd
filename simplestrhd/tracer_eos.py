import astropy.constants as const
import astropy.units as u
import lightweaver as lw

from .indices import IRHO, IMOM, IENE, IIONE, k_B
from .eos import temperature_si

M_P = const.m_p.value
CHI_H = const.Ryd.to(u.J, equivalencies=u.spectral()).value

def tracer_eos(state, sim_config, verbose=False, total_abund=1.0):
    """
    Updates the total and ionisation energy to be consistent with the tracer array.
    Assumes tracer array is [ne, n_H, ...]
    """
    gamma = state["gamma"]
    mass_per_h = sim_config.get("avg_mass", 1.0)
    h_mass = sim_config.get("h_mass", M_P)
    chi_H = sim_config.get("chi_H", CHI_H)
    if total_abund is None:
        total_abund = lw.DefaultAtomicAbundance.totalAbundance

    Q = state["Q"]
    ne = state["tracers"][0]

    e_kinetic = Q[IMOM]**2 / Q[IRHO]

    rho = Q[IRHO]
    rho_to_nh_tot = 1.0 / (h_mass * mass_per_h)
    nh = rho * rho_to_nh_tot
    y = nh / ne
    # NOTE(cmo): Freeze temperature over EOS step
    # TODO(cmo): Use a fixed temperature in the state if present.
    pressure = (Q[IENE] - Q[IIONE] * rho - e_kinetic) * (gamma - 1.0)
    temperature = temperature_si(pressure, nh, y, total_abund=total_abund)
    spec_ion_e = y * rho_to_nh_tot * chi_H
    etot = (
        1.0 / (gamma - 1.0) * (total_abund + y) * nh * k_B * temperature
        + spec_ion_e * rho
        + e_kinetic
    )

    Q[IENE, :] = etot
    Q[IIONE, :] = spec_ion_e
    state["y"] = y
