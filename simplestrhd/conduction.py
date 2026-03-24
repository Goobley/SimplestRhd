from functools import partial

import lineax
import jax
import jax.numpy as jnp
import optimistix as optx
from .indices import (
    IRHO,
    IMOM,
    IENE,
    IIONE,
    NUM_GHOST
)
from .eos import temperature_si
import astropy.constants as const

jax.config.update("jax_enable_x64", True)
M_P = const.m_p.value
M_E = const.m_e.value
K_B = const.k_B.value

# NOTE(cmo): Doesn't match the dimensionalised self-similar test from Cherry et
# al with this set to True. Decide whether to prioritise sharpness or diffusion,
# I guess. Could try other limiters.
USE_HARMONIC_MEAN = False

def harmonic_mean(a, b):
    if USE_HARMONIC_MEAN:
        return 2.0 * a * b / (a + b + 1e-30)
    return 0.5 * (a + b)

def compute_kappa(temperature, kappa0, alpha, beta, Tc, Tlow):
    T_kappa = temperature
    if Tc is not None:
        T_kappa = jnp.where((T_kappa <= Tc) & (T_kappa > Tlow), Tc, T_kappa)
    kappa = alpha * kappa0 * T_kappa**beta
    return kappa


def residual(temperature, dx, kappa0, alpha=1.0, beta=2.5, ne=None, Tc=None, Tlow=None):
    flux = jnp.zeros_like(temperature)
    kappa = compute_kappa(temperature=temperature, kappa0=kappa0, alpha=1.0, beta=beta, Tc=Tc, Tlow=Tlow)
    kappa_if = jnp.zeros(flux.shape[0] + 1)
    kappa_if = kappa_if.at[1:-1].set(harmonic_mean(kappa[:-1], kappa[1:]))

    # NOTE(cmo): Uniform grid
    flux_p = 1.0 / dx * (kappa_if[NUM_GHOST+1:-NUM_GHOST] * (temperature[NUM_GHOST+1:-NUM_GHOST+1] - temperature[NUM_GHOST:-NUM_GHOST]))
    flux_m = 1.0 / dx * (kappa_if[NUM_GHOST:-NUM_GHOST-1] * (temperature[NUM_GHOST:-NUM_GHOST] - temperature[NUM_GHOST-1:-NUM_GHOST-1]))

    if ne is not None:
        # NOTE(cmo): 1/6 free-streaming limit
        f_sat_p = (
            (0.25 * jnp.sign(flux_p) * K_B**1.5) / jnp.sqrt(M_E)
            * 0.5 * (ne[NUM_GHOST:-NUM_GHOST] + ne[NUM_GHOST+1:-NUM_GHOST+1])
            * (0.5 * (temperature[NUM_GHOST:-NUM_GHOST] + temperature[NUM_GHOST+1:-NUM_GHOST+1]))**1.5
        )
        # f_sat_p = f_sat_p * 0.5 * (alpha[NUM_GHOST:-NUM_GHOST] + alpha[NUM_GHOST+1:-NUM_GHOST+1])
        flux_p = flux_p / (1.0 + flux_p / (f_sat_p + 1e-10))
        f_sat_m = (
            (0.25 * jnp.sign(flux_m) * K_B**1.5) / jnp.sqrt(M_E)
            * 0.5 * (ne[NUM_GHOST-1:-NUM_GHOST-1] + ne[NUM_GHOST:-NUM_GHOST])
            * (0.5 * (temperature[NUM_GHOST-1:-NUM_GHOST-1] + temperature[NUM_GHOST:-NUM_GHOST]))**1.5
        )
        # f_sat_m = f_sat_m * 0.5 * (alpha[NUM_GHOST-1:-NUM_GHOST-1] + alpha[NUM_GHOST:-NUM_GHOST])
        flux_m = flux_m / (1.0 + flux_m / (f_sat_m + 1e-10))


    flux = flux.at[NUM_GHOST:-NUM_GHOST].set(
        1.0 / dx * (flux_p - flux_m) * alpha[NUM_GHOST:-NUM_GHOST]
    )

    return flux

def residual_fixed_kappa(temperature, dx, kappa, alpha, ne=None):
    flux = jnp.zeros_like(temperature)
    kappa_if = jnp.zeros(flux.shape[0] + 1)
    kappa_if = kappa_if.at[1:-1].set(harmonic_mean(kappa[:-1], kappa[1:]))

    # NOTE(cmo): Uniform grid
    flux_p = 1.0 / dx * (kappa_if[NUM_GHOST+1:-NUM_GHOST] * (temperature[NUM_GHOST+1:-NUM_GHOST+1] - temperature[NUM_GHOST:-NUM_GHOST]))
    flux_m = 1.0 / dx * (kappa_if[NUM_GHOST:-NUM_GHOST-1] * (temperature[NUM_GHOST:-NUM_GHOST] - temperature[NUM_GHOST-1:-NUM_GHOST-1]))

    if ne is not None:
        f_sat_p = (
            (0.25 * jnp.sign(flux_p) * K_B**1.5) / jnp.sqrt(M_E)
            * 0.5 * (ne[NUM_GHOST:-NUM_GHOST] + ne[NUM_GHOST+1:-NUM_GHOST+1])
            * (0.5 * (temperature[NUM_GHOST:-NUM_GHOST] + temperature[NUM_GHOST+1:-NUM_GHOST+1]))**1.5
        )
        # f_sat_p = f_sat_p * 0.5 * (alpha[NUM_GHOST:-NUM_GHOST] + alpha[NUM_GHOST+1:-NUM_GHOST+1])
        flux_p = flux_p / (1.0 + flux_p / (f_sat_p + 1e-5))
        f_sat_m = (
            (0.25 * jnp.sign(flux_m) * K_B**1.5) / jnp.sqrt(M_E)
            * 0.5 * (ne[NUM_GHOST-1:-NUM_GHOST-1] + ne[NUM_GHOST:-NUM_GHOST])
            * (0.5 * (temperature[NUM_GHOST-1:-NUM_GHOST-1] + temperature[NUM_GHOST:-NUM_GHOST]))**1.5
        )
        # f_sat_m = f_sat_m * 0.5 * (alpha[NUM_GHOST-1:-NUM_GHOST-1] + alpha[NUM_GHOST:-NUM_GHOST])
        flux_m = flux_m / (1.0 + flux_m / (f_sat_m + 1e-5))


    flux = flux.at[NUM_GHOST:-NUM_GHOST].set(
        1.0 / dx * (flux_p - flux_m) * alpha[NUM_GHOST:-NUM_GHOST]
    )

    return flux

def implicit_residual(
        temperature,
        prev_temperature,
        dx,
        dt,
        kappa0,
        alpha=1.0,
        beta=2.5,
        theta=0.55,
        ne=None,
        Tc=None,
        Tlow=None,
    ):
    r_new = residual(
        temperature,
        dx,
        kappa0=kappa0,
        alpha=alpha,
        beta=beta,
        ne=ne,
        Tc=Tc,
        Tlow=Tlow,
    )
    r_old = residual(
        prev_temperature,
        dx,
        kappa0=kappa0,
        alpha=alpha,
        beta=beta,
        ne=ne,
        Tc=Tc,
        Tlow=Tlow,
    )
    return temperature - prev_temperature - dt * (theta * r_new + (1.0 - theta) * r_old)

def implicit_residual_fixed(
        temperature,
        prev_temperature,
        dx,
        dt,
        kappa,
        alpha=1.0,
        theta=0.8,
        ne=None,
    ):
    r_new = residual_fixed_kappa(
        temperature,
        dx,
        kappa=kappa,
        alpha=alpha,
        ne=ne,
    )
    r_old = residual_fixed_kappa(
        prev_temperature,
        dx,
        kappa,
        alpha=alpha,
        ne=ne,
    )
    return temperature - prev_temperature - dt * (theta * r_new + (1.0 - theta) * r_old)

@partial(jax.jit, static_argnames=["max_steps"])
def single_nonlinear_solve(
    temperature,
    dx,
    curr_dt,
    kappa0,
    alpha,
    beta,
    ne=None,
    Tc=None,
    Tlow=None,
    min_temperature=0.0,
    max_steps=10
):
    def to_opt(y, args):
        running_temperature, dt_inner = args
        return implicit_residual(
            y,
            running_temperature,
            dx,
            dt_inner,
            kappa0=kappa0,
            alpha=alpha,
            beta=beta,
            ne=ne,
            Tc=Tc,
            Tlow=Tlow,
        )

    sol = optx.root_find(
        to_opt,
        solver=optx.Newton(
            rtol=1e-4,
            atol=1e-2,
            linear_solver=lineax.GMRES(rtol=1e-3, atol=1e-2),
        ),
        y0=jnp.asarray(temperature),
        args=(temperature, curr_dt),
        options=dict(lower=min_temperature),
        max_steps=max_steps,
        throw=False,
    )
    return sol

def solve_step(
        prev_temperature,
        dx,
        dt,
        kappa0,
        alpha,
        beta,
        ne=None,
        Tc=None,
        Tlow=None,
        min_temperature=0.0,
        max_steps=10
):
    curr_dt = dt
    dt_complete = 0.0
    start_temperature = prev_temperature
    num_substeps = 0
    # TODO(cmo): This could be a jax loop, but I don't imagine it really matters.
    while dt_complete < dt:
        sol = single_nonlinear_solve(
            start_temperature,
            dx,
            curr_dt,
            kappa0,
            alpha,
            beta,
            ne=ne,
            Tc=Tc,
            Tlow=Tlow,
            min_temperature=min_temperature,
            max_steps=max_steps,
        )
        if sol.stats['num_steps'] == max_steps:
            curr_dt /= 2
            if dt / curr_dt >= 512:
                raise ValueError("Conductive dt too small!")
            continue

        dt_complete += curr_dt
        start_temperature = sol.value
        if sol.stats['num_steps'] <= 4:
            curr_dt *= 1.5
        if dt_complete + curr_dt > dt:
            curr_dt = dt - dt_complete
            while dt_complete + curr_dt < dt:
                curr_dt = jnp.nextafter(curr_dt, jnp.inf)
        num_substeps += 1
    if num_substeps >= 32:
        print(num_substeps)

    return sol.value

def solve_step_fixed(prev_temperature, dx, dt, kappa0, alpha, beta, ne=None, Tc=None, Tlow=None, min_temperature=0.0, max_steps=3):
    kappa = compute_kappa(prev_temperature, kappa0=kappa0, alpha=1.0, beta=beta, Tc=Tc, Tlow=Tlow)
    def to_opt(y, args):
        running_temperature, dt_inner = args
        return implicit_residual_fixed(
            y,
            running_temperature,
            dx,
            dt_inner,
            kappa=kappa,
            alpha=alpha,
            ne=ne,
        )

    # @jax.jit
    def single_sol(temperature, curr_dt):
        # res = residual_fixed_kappa(temperature, dx, kappa, alpha, ne)
        # jac = jax.jacrev(residual_fixed_kappa)(temperature, dx, kappa, alpha, ne)
        # breakpoint()
        # dtemp = lineax.linear_solve(
        #     lineax.MatrixLinearOperator(jac),
        #     -res,
        # )
        # implicit_temperature_flux = temperature + dtemp

        # temperature = prev_temperature + dt * implicit_temperature_flux

        sol = optx.root_find(
            residual_fixed_kappa,
            solver=optx.Newton(
                rtol=1e-4,
                atol=1e-2,
                linear_solver=lineax.GMRES(rtol=1e-3, atol=1e-2),
            ),
            y0=jnp.asarray(temperature),
            args=(temperature, dt),
            options=dict(lower=min_temperature),
            max_steps=max_steps,
            throw=False,
        )
        breakpoint()
        return temperature

    # curr_dt = dt
    # dt_complete = 0.0
    # start_temperature = prev_temperature
    # while dt_complete < dt:
    #     sol = single_sol(start_temperature, curr_dt)
    #     if sol.stats['num_steps'] == max_steps:
    #         curr_dt /= 2
    #         if dt / curr_dt >= 256:
    #             raise ValueError("Conductive dt too small!")
    #         continue

    #     dt_complete += curr_dt
    #     start_temperature = sol.value
    #     if sol.stats['num_steps'] <= 2:
    #         curr_dt *= 1.5
    #     if dt_complete + curr_dt > dt:
    #         curr_dt = dt - dt_complete
    #         while dt_complete + curr_dt < dt:
    #             curr_dt = jnp.nextafter(curr_dt, jnp.inf)

    return single_sol(prev_temperature, dt)

def implicit_thermal_conduction(
    state,
    sim_config,
    dt
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

    use_fixed = False
    if use_fixed:
        new_temperature = solve_step_fixed(
            temperature,
            dx,
            dt,
            kappa0=kappa0,
            alpha=alpha,
            beta=2.5,
            ne=ne,
            Tc=Tc,
            Tlow=Tlow,
            min_temperature=min_temperature,
        )
    else:
        new_temperature = solve_step(
            temperature,
            dx,
            dt,
            kappa0=kappa0,
            alpha=alpha,
            beta=2.5,
            ne=ne,
            Tc=Tc,
            Tlow=Tlow,
            min_temperature=min_temperature,
        )
    delta_E = (new_temperature - temperature) / alpha
    state['temperature'] = new_temperature
    Q[IENE] += delta_E