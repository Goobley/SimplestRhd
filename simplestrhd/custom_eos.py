from .indices import (
    IRHO,
    IMOM,
    IENE,
    IIONE,
    k_B
)

import numpy as np
import astropy.constants as const
import astropy.units as u

M_P = const.m_p.value
CHI_H = const.Ryd.to(u.J, equivalencies=u.spectral()).value

def saha_rhs_H(temperature):
    # NOTE(cmo): From sympy
    return 2.4146830395719654e+21 * temperature**1.5 * np.exp(-157763.42386247337 / temperature)

def y_from_nhtot(nh_tot, temperature):
    x = saha_rhs_H(temperature)
    # return 0.5 * (-x + np.sqrt(x**2 + 4.0 * nh_tot * x)) / nh_tot

    x /= nh_tot
    # return 0.5 * x * (np.sqrt(1.0 + 4.0 / x) - 1.0)
    return 0.5 * np.sqrt(x) * (4.0 / (np.sqrt(x + 4) + np.sqrt(x)))

    # y = 0.5
    # for i in range(100):
    #     fy = y**2 / (1.0 - y) - x
    #     fpy = (2 * y * (1.0 - y) + y**2) / (1.0 - y)**2
    #     prev_y = y
    #     y = y - 0.1 * (fy / fpy)
    #     y = np.clip(y, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))
    #     # print(y)
    #     # print(fy)
    #     # print('----')
    #     if np.max(np.abs((y - prev_y) / prev_y)) < 1e-8:
    #         break
    # return y



def y_from_ntot(n_tot, temperature):
    x = saha_rhs_H(temperature)
    ne = 0.5 * (-2 * x + np.sqrt((2 * x)**2 + 4.0 * n_tot * x))
    return ne / (n_tot - ne)

def lte_eos(state, sim_config, include_ion_e=True, temp_err_bound=1e-3, find_initial_ion_e=False, verbose=False):
    """
    Updates the total and ionisation energies in state to be consistent with LTE.
    """

    gamma = state["gamma"]
    mass_per_h = sim_config.get("avg_mass", 1.0)
    h_mass = sim_config.get("h_mass", M_P)
    chi_H = sim_config.get("chi_H", CHI_H)
    Q = state["Q"]
    rho = Q[IRHO]
    mom = Q[IMOM]
    e_tot = Q[IENE]
    # NOTE(cmo): Ignore this term if we're iterating for the initial ion_e to be
    # consistent with a pressure (following a prim_to_cons call).
    #### THIS DOESN'T WORK. NEED A FUNCTION TO FIND INITIAL ION_E
    inc_ion_e = 1.0 if include_ion_e and not find_initial_ion_e else 0.0

    e_kinetic = mom**2 / rho
    e_int_ion = e_tot - e_kinetic

    rho_to_nh_tot = 1.0 / (h_mass * mass_per_h)
    nh_tot = rho * rho_to_nh_tot
    e_to_T = (gamma - 1.0) / (nh_tot * k_B)
    def temperature_from_y(y):
        return e_to_T / (1.0 + y) * (e_int_ion - inc_ion_e * y * nh_tot * chi_H)

    min_temperature = 100.0
    temp_bounds = [
        np.maximum(temperature_from_y(1.0) * 0.98, min_temperature),
        temperature_from_y(0.0) * 1.02,
    ]

    if np.any(temp_bounds[0] > temp_bounds[1]):
        raise ValueError("Temperature bounds flipped!")

    temp_step = temp_bounds[1] - temp_bounds[0]
    temperature = temp_bounds[0]
    for _ in range(100):
        temp_step *= 0.5
        test_temp = temperature + temp_step
        y = y_from_nhtot(nh_tot, test_temp)
        temp_err = test_temp - temperature_from_y(y)

        temperature = np.where(temp_err <= 0.0, test_temp, temperature)

        if (np.max(np.abs(temp_step)) < temp_err_bound) or (np.max(np.abs(temp_err)) < temp_err_bound):
            break

    if verbose:
        print(f"Setting temperature (iters: {_})")
        print(temp_bounds)
        print(temperature)
        print(y)
        print(temp_err)

    new_spec_ion_e = y * rho_to_nh_tot * chi_H if include_ion_e else 0.0
    # new_spec_ion_e = y * rho * rho_to_nh_tot * chi_H if include_ion_e else 0.0
    new_etot = (
        1.0 / (gamma - 1.0) * (1.0 + y) * rho * rho_to_nh_tot * k_B * temperature
        + new_spec_ion_e * rho
        # + new_spec_ion_e
        + e_kinetic
    )
    Q[IENE, :] = new_etot
    Q[IIONE, :] = new_spec_ion_e
    state['y'] = y

