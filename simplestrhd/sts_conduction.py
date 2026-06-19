from .indices import IRHO, IENE, IMOM, IIONE, NUM_GHOST
import numpy as np

import astropy.constants as const
M_P = const.m_p.value
M_E = const.m_e.value
K_B = const.k_B.value

def compute_kappa(temperature, kappa0, alpha, beta):
    kappa = alpha * kappa0 * temperature**beta
    return kappa

def compute_explicit_flux(T, kappa, dx):
    dT = T[1:] - T[:-1]
    kappa_face = 0.5 * (kappa[1:] + kappa[:-1])
    q = -kappa_face * dT / dx
    return q

def compute_explicit_fdiv(T, kappa, dx):
    q = compute_explicit_flux(T, kappa, dx)
    div_q = np.zeros_like(T)
    div_q[1:-1] = (q[:-1] - q[1:]) / dx
    div_q[:NUM_GHOST] = 0
    div_q[-NUM_GHOST:] = 0
    return div_q

def compute_temperature(eint, rho, cv):
    alpha = 1.0 / (rho * cv)
    temperature = eint * alpha
    return temperature

# def conduction_rhs(state, sim_config, cv):
#     Q = state["Q"]
#     dx = state["dx"]
#     kappa0 = sim_config.get("kappa0", 8e-12)
#     temperature = compute_temperature(Q, cv)
#     kappa = compute_kappa(temperature, kappa0, alpha=1.0, beta=2.5)
#     q = compute_explicit_flux(temperature, kappa, dx)
#     div_q = np.zeros_like(Q[IENE])
#     div_q[1:-1] = (q[:-1] - q[1:]) / dx
#     return div_q

def explicit_thermal_conduction(
    state,
    sim_config,
    sources,
    ts
):
    Q = state["Q"]
    dx = state["dx"]
    gamma = state["gamma"]
    y = state.get("y", 1.0)
    h_mass = sim_config.get("h_mass", M_P)
    kappa0 = sim_config.get("kappa0", 8e-12)
    k_B = sim_config.get("k_B", K_B)
    mass_per_h = sim_config.get("avg_mass", 1.0)
    total_abund = sim_config.get("total_abund", 1.0)

    cv = 1.0 / (gamma - 1.0) * k_B / (h_mass * mass_per_h) * (total_abund + y)
    eint = Q[IENE] - (0.5 * Q[IMOM]**2 / Q[IRHO]) - (Q[IRHO] * Q[IIONE])
    temperature = compute_temperature(eint, Q[IRHO], cv)
    kappa = compute_kappa(temperature, kappa0, alpha=1.0, beta=2.5)

    flux = compute_explicit_fdiv(temperature, kappa, dx)
    # sources[IENE, NUM_GHOST:-NUM_GHOST] += flux[NUM_GHOST:-NUM_GHOST] * (Q[IRHO] * cv)
    sources[IENE, NUM_GHOST:-NUM_GHOST] += flux[NUM_GHOST:-NUM_GHOST]



def sts_thermal_conduction(
    state,
    sim_config,
    sources,
    ts
):
    Q = state["Q"]
    dx = state["dx"]
    gamma = state["gamma"]
    y = state.get("y", 1.0)
    h_mass = sim_config.get("h_mass", M_P)
    kappa0 = sim_config.get("kappa0", 8e-12)
    k_B = sim_config.get("k_B", K_B)
    mass_per_h = sim_config.get("avg_mass", 1.0)
    total_abund = sim_config.get("total_abund", 1.0)
    Tc = sim_config.get("conduction_suppression_Tc", None)
    Tlow = sim_config.get("conduction_suppression_Tlow", None)
    min_temperature = sim_config.get("min_temperature", 0.0)

    saturate_flux = sim_config.get("saturate_conductive_flux", False)
    ne = None
    if saturate_flux:
        nh = Q[IRHO] / (h_mass * mass_per_h)
        ne = y * nh

    dt = ts.dt_sub

    cv = 1.0 / (gamma - 1.0) * k_B / (h_mass * mass_per_h) * (total_abund + y)
    eint = Q[IENE] - (0.5 * Q[IMOM]**2 / Q[IRHO]) - (Q[IRHO] * Q[IIONE])
    temperature = compute_temperature(eint, Q[IRHO], cv)
    # NOTE(cmo): First determine the number of stages
    # Explicit Timestep
    kappa = compute_kappa(temperature, kappa0, alpha=1.0, beta=2.5)
    dt_diff = ts.cfl * 0.5 * np.min(((Q[IRHO] * cv * dx**2) / (kappa))[NUM_GHOST:-NUM_GHOST])
    # dt_diff = ts.cfl * 0.5 * np.min((dx**2 / kappa)[NUM_GHOST:-NUM_GHOST])
    stages = 0.5 * np.sqrt(9.0 + 16.0 * (dt / dt_diff)) - 1.0
    n_stages = int(np.ceil(stages))
    if n_stages % 2 == 0:
        n_stages += 1

    if stages <= 1:
        return explicit_thermal_conduction(state, sim_config, sources, ts)

    if n_stages > 120:
        print(f"Lots of stages: {n_stages}")

    Y = np.zeros((4, Q.shape[1]))
    a = np.zeros(n_stages+1)
    b = np.zeros(n_stages+1)
    mu_tilde = np.zeros(n_stages+1)
    mu = np.zeros(n_stages+1)
    nu = np.zeros(n_stages+1)
    gamma_tilde = np.zeros(n_stages+1)

    omega_1 = 4.0 / (n_stages**2 + n_stages - 2.0)
    b[:3] = 1.0 / 3.0
    for j in range(3, n_stages+1):
        b[j] = (j**2 + j - 2.0) / (2.0 * j * (j + 1.0))
    a[:] = 1.0 - b
    mu_tilde[1] = omega_1 / 3.0

    for j in range(2, n_stages+1):
        fac = (2.0 * j - 1.0) / j
        mu_tilde[j] = fac * omega_1 * (b[j] / b[j-1])
        mu[j] = fac * (b[j] / b[j-1])
        nu[j] = (1.0 - j) / j * (b[j] / b[j-2])
        gamma_tilde[j] = - a[j-1] * mu_tilde[j]

    Y[...] = eint[None, :]


    # First stage
    flux = compute_explicit_fdiv(temperature, kappa, dx)

    # Lc_Y0 = flux * (Q[IRHO] * cv)
    Lc_Y0 = flux
    c0 = mu_tilde[1] * dt * Lc_Y0
    Y[1, :] = Y[0, :] + c0

    Y[2, NUM_GHOST:-NUM_GHOST] = Y[1, NUM_GHOST:-NUM_GHOST]
    Y[1, NUM_GHOST:-NUM_GHOST] = Y[0, NUM_GHOST:-NUM_GHOST]

    freeze_kappa = True
    for j in range(2, n_stages+1):
        # breakpoint()
        temperature = compute_temperature(Y[2], Q[IRHO], cv)
        if not freeze_kappa:
            kappa = compute_kappa(temperature, kappa0, alpha=1.0, beta=2.5)

        flux = compute_explicit_fdiv(temperature, kappa, dx)

        c0 = gamma_tilde[j] * dt * Lc_Y0
        # Lc_Yj_1 = flux * (Q[IRHO] * cv)
        Lc_Yj_1 = flux
        c1 = mu_tilde[j] * dt * Lc_Yj_1

        Y[3, ...] = mu[j] * Y[2] + nu[j] * Y[1] + (1.0 - mu[j] - nu[j]) * Y[0]
        Y[3, ...] += c1 + c0

        if j < n_stages:
            Y[1, ...] = Y[2, ...]
            Y[2, ...] = Y[3, ...]
            Y[3, NUM_GHOST:-NUM_GHOST] = 0.0
    delta_E = (Y[3] - eint)
    sources[IENE, NUM_GHOST:-NUM_GHOST] += delta_E[NUM_GHOST:-NUM_GHOST] / dt

