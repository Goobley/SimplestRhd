import astropy.constants as const
import astropy.units as u
import lightweaver as lw
import numpy as np

from .indices import IRHO, IMOM, IENE, IIONE
from .eos import temperature_si

M_P = const.m_p.value
CHI_H = const.Ryd.to(u.J, equivalencies=u.spectral()).value
K_B = const.k_B.value

def tracer_eos(state, sim_config, verbose=False):
    """
    Updates the total and ionisation energy to be consistent with the tracer array.
    Assumes tracer array is [ne, n_H, ...]
    Uses tracer_energy as the excitation/ionisation energy per tracer row if
    present in state, otherwise n_e * chi_H
    """
    gamma = state["gamma"]
    mass_per_h = sim_config.get("avg_mass", 1.0)
    h_mass = sim_config.get("h_mass", M_P)
    chi_H = sim_config.get("chi_H", CHI_H)
    k_B = sim_config.get("k_B", K_B)
    total_abund = sim_config.get("total_abund", 1.0)
    if total_abund is None:
        total_abund = lw.DefaultAtomicAbundance.totalAbundance
    tracer_energy = state.get("tracer_energy", None)
    tracer_is_h = state.get("tracer_is_h", None)
    tracer_charge = state.get("tracer_charge", None)
    min_temperature = sim_config.get("min_temperature", None)

    Q = state["Q"]
    rho = Q[IRHO]
    rho_to_nh_tot = 1.0 / (h_mass * mass_per_h)
    nh = rho * rho_to_nh_tot
    tracers = state["tracers"]

    if tracer_is_h is not None:
        nh_from_tracer = tracers[tracer_is_h, :].sum(axis=0)
        tracer_err = nh / nh_from_tracer
        tracers *= tracer_err[None, :]

    if tracer_charge is not None:
        ne = np.sum(tracers * tracer_charge[:, None], axis=0)
        tracers[0, :] = ne
        ion_e = np.sum(tracers * tracer_energy[:, None], axis=0)
        spec_ion_e = ion_e / rho
    else:
        ne = tracers[0]
        spec_ion_e = Q[IIONE]
        ion_e = spec_ion_e * rho
    e_kinetic = 0.5 * Q[IMOM]**2 / Q[IRHO]

    y = ne / nh
    # NOTE(cmo): Freeze temperature over EOS step
    # TODO(cmo): Use a fixed temperature in the state if present.
    pressure = (Q[IENE] - ion_e - e_kinetic) * (gamma - 1.0)
    temperature = temperature_si(
        pressure,
        nh,
        y,
        total_abund=total_abund,
        k_B=k_B,
    )
    # NOTE(cmo): Temperature sometimes seems to drop below this
    # TODO(cmo): Check how this all behaves relative to the energy update due to changes in ionisation!
    if min_temperature is not None:
        temperature = np.maximum(temperature, min_temperature)
    etot = (
        1.0 / (gamma - 1.0) * (total_abund + y) * nh * k_B * temperature
        + spec_ion_e * rho
        + e_kinetic
    )

    Q[IENE, :] = etot
    Q[IIONE, :] = spec_ion_e
    state["y"] = y
