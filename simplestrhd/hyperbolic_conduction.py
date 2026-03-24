from .indices import (
    IRHO,
    IMOM,
    IENE,
    IIONE,
    IPRE,
    IVEL,
    NUM_GHOST
)
import numpy as np
import astropy.constants as const

M_P = const.m_p.value
M_E = const.m_e.value
K_B = const.k_B.value

def compute_kappa(temperature, kappa0, alpha, beta, Tc, Tlow):
    T_kappa = temperature
    if Tc is not None:
        T_kappa = np.where((T_kappa <= Tc) & (T_kappa > Tlow), Tc, T_kappa)
    kappa = alpha * kappa0 * T_kappa**beta
    return kappa

def despike_filter(q, filter_strength=0.1):
    """Filter should be the same as MANCHA"""
    q_filt = np.copy(q)
    delta6 = (q[0:-6] - 6*q[1:-5] + 15*q[2:-4] - 20*q[3:-3] + 15*q[4:-2] - 6*q[5:-1] + q[6:]) / 64.0
    q_filt[3:-3] += filter_strength * delta6
    return q_filt

def compute_heatf_source(state, sim_config, S, temperature, ne, kappa, ts):

    dx = state['dx']
    W = state['W']
    heatf = state.get('heatf', np.zeros(W.shape[1]))
    gamma = state['gamma']
    cfl = sim_config["max_cfl"]
    saturate_flux = sim_config.get("saturate_conductive_flux", False)
    htc_hyperdiffusion = sim_config.get("htc_hyperdiffusion", 0.0)
    htc_despike_strength = sim_config.get("htc_despike_strength", 0.0)
    htc_derivative_order = sim_config.get("htc_order", 1)
    if NUM_GHOST < htc_derivative_order:
        raise ValueError(f"Requires {htc_derivative_order} ghost cells for hyperbolic tc with htc_order={htc_derivative_order}")
    if htc_derivative_order < 1 or htc_derivative_order > 3:
        raise ValueError("This is a silly derivative order")
    inv_dx = 1.0 / dx

    w1 = 8.0 / 12.0
    w2 = 1.0 / 12.0
    w1_6 = 3.0 / 4.0
    w2_6 = -3.0 / 20.0
    w3_6 = 1.0 / 60.0

    B_gradT = np.zeros_like(temperature)
    if htc_derivative_order == 1:
        B_gradT[NUM_GHOST:-NUM_GHOST] = 0.5 * inv_dx * (
            temperature[NUM_GHOST+1:-NUM_GHOST+1] - temperature[NUM_GHOST-1:-NUM_GHOST-1]
        )
    elif htc_derivative_order == 2:
        B_gradT[NUM_GHOST:-NUM_GHOST] = inv_dx * (
            w1 * (temperature[NUM_GHOST+1:-NUM_GHOST+1] - temperature[NUM_GHOST-1:-NUM_GHOST-1]) -
            w2 * (temperature[NUM_GHOST+2:(-NUM_GHOST+2 if NUM_GHOST > 2 else None)] - temperature[NUM_GHOST-2:-NUM_GHOST-2])
        )
    else:
        B_gradT[NUM_GHOST:-NUM_GHOST] = inv_dx * (
            w1_6 * (temperature[NUM_GHOST+1:-NUM_GHOST+1] - temperature[NUM_GHOST-1:-NUM_GHOST-1])
            + w2_6 * (temperature[NUM_GHOST+2:-NUM_GHOST+2] - temperature[NUM_GHOST-2:-NUM_GHOST-2])
            + w3_6 * (temperature[NUM_GHOST+3:(-NUM_GHOST+2 if NUM_GHOST > 3 else None)] - temperature[NUM_GHOST-3:-NUM_GHOST-3])
        )
    sigma_T_72 = kappa * temperature
    f_sat = 1.0
    if saturate_flux:
        # NOTE(cmo): 1/6 Free stream
        f_sat = (0.25 * K_B**1.5) / np.sqrt(M_E) * ne * temperature**1.5
        f_sat = 1.0 / (1.0 + np.abs(kappa * B_gradT) / f_sat)

    tau = np.maximum(
        2.0 * ts.dt_sub,
        f_sat * sigma_T_72 * (gamma - 1.0) / (W[IPRE] * (cfl * dx / ts.dt_sub - np.abs(W[IVEL]))**2)
    )
    heatf_source = (f_sat * kappa * B_gradT + heatf) / tau

    ene_res = np.zeros_like(temperature)
    if htc_derivative_order == 1:
        ene_res[NUM_GHOST:-NUM_GHOST] += 0.5 * inv_dx * (
            heatf[NUM_GHOST+1:-NUM_GHOST+1] - heatf[NUM_GHOST-1:-NUM_GHOST-1]
        )
    elif htc_derivative_order == 2:
        ene_res[NUM_GHOST:-NUM_GHOST] = inv_dx * (
            w1 * (heatf[NUM_GHOST+1:-NUM_GHOST+1] - heatf[NUM_GHOST-1:-NUM_GHOST-1]) -
            w2 * (heatf[NUM_GHOST+2:(-NUM_GHOST+2 if NUM_GHOST > 2 else None)] - heatf[NUM_GHOST-2:-NUM_GHOST-2])
        )
    else:
        ene_res[NUM_GHOST:-NUM_GHOST] = inv_dx * (
            w1_6 * (heatf[NUM_GHOST+1:-NUM_GHOST+1] - heatf[NUM_GHOST-1:-NUM_GHOST-1])
            + w2_6 * (heatf[NUM_GHOST+2:-NUM_GHOST+2] - heatf[NUM_GHOST-2:-NUM_GHOST-2])
            + w3_6 * (heatf[NUM_GHOST+3:(-NUM_GHOST+2 if NUM_GHOST > 3 else None)] - heatf[NUM_GHOST-3:-NUM_GHOST-3])
        )

    if htc_despike_strength > 0.0:
        ene_res = despike_filter(ene_res, htc_despike_strength)
    S[IENE, NUM_GHOST:-NUM_GHOST] -= ene_res[NUM_GHOST:-NUM_GHOST]

    if htc_hyperdiffusion > 0.0:
        hyp = htc_hyperdiffusion / ts.dt
        heatf_source[NUM_GHOST:-NUM_GHOST] += hyp * (
            (heatf[NUM_GHOST+2:(-NUM_GHOST+2 if NUM_GHOST > 2 else None)] + heatf[NUM_GHOST-2:-NUM_GHOST-2])
            - 4.0 * (heatf[NUM_GHOST+1:-NUM_GHOST+1] + heatf[NUM_GHOST-1:-NUM_GHOST-1])
            + 6.0 * heatf[NUM_GHOST:-NUM_GHOST]
        )
    if htc_despike_strength > 0.0:
        heatf_source = despike_filter(heatf_source, htc_despike_strength)


    heatf[NUM_GHOST:-NUM_GHOST] -= heatf_source[NUM_GHOST:-NUM_GHOST] * ts.dt_sub
    state['heatf'] = heatf



def hyperbolic_thermal_conduction(
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

    cv = 1.0 / (gamma - 1.0) * k_B / (h_mass * mass_per_h) * (total_abund + y)
    alpha = 1.0 / (Q[IRHO] * cv)
    eint = Q[IENE] - (0.5 * Q[IMOM]**2 / Q[IRHO]) - (Q[IRHO] * Q[IIONE])
    temperature = eint * alpha

    kappa = compute_kappa(temperature, kappa0, alpha=1.0, beta=2.5, Tc=Tc, Tlow=Tlow)

    compute_heatf_source(state, sim_config, sources, temperature, ne, kappa, ts)