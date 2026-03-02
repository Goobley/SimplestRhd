import lineax
import jax
import jax.numpy as jnp
import optimistix as optx
from .indices import (
    IRHO,
    IENE,
    NUM_EQ,
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

def residual(temperature, dx, kappa0, alpha=1.0, beta=2.5, ne=0.0):
    flux = jnp.zeros_like(temperature)
    kappa = alpha * kappa0 * temperature**beta
    kappa_if = jnp.zeros(flux.shape[0] + 1)
    kappa_if = kappa_if.at[1:-1].set(harmonic_mean(kappa[:-1], kappa[1:]))

    # NOTE(cmo): Uniform grid
    flux_p = 1.0 / dx * (kappa_if[NUM_GHOST+1:-NUM_GHOST] * (temperature[NUM_GHOST+1:-NUM_GHOST+1] - temperature[NUM_GHOST:-NUM_GHOST]))
    flux_m = 1.0 / dx * (kappa_if[NUM_GHOST:-NUM_GHOST-1] * (temperature[NUM_GHOST:-NUM_GHOST] - temperature[NUM_GHOST-1:-NUM_GHOST-1]))

    if jnp.atleast_1d(ne).shape[0] > 1:
        f_sat_p = (
            (0.25 * jnp.sign(flux_p) * K_B**1.5) / jnp.sqrt(M_E)
            * 0.5 * (ne[NUM_GHOST:-NUM_GHOST] + ne[NUM_GHOST+1:-NUM_GHOST+1])
            * (0.5 * (temperature[NUM_GHOST:-NUM_GHOST] + temperature[NUM_GHOST+1:-NUM_GHOST+1]))**1.5
        )
        f_sat_p = f_sat_p * 0.5 * (alpha[NUM_GHOST:-NUM_GHOST] + alpha[NUM_GHOST+1:-NUM_GHOST+1])
        flux_p = flux_p / (1.0 + flux_p / (f_sat_p + 1e-5))
        f_sat_m = (
            (0.25 * jnp.sign(flux_m) * K_B**1.5) / jnp.sqrt(M_E)
            * 0.5 * (ne[NUM_GHOST-1:-NUM_GHOST-1] + ne[NUM_GHOST:-NUM_GHOST])
            * (0.5 * (temperature[NUM_GHOST-1:-NUM_GHOST-1] + temperature[NUM_GHOST:-NUM_GHOST]))**1.5
        )
        f_sat_m = f_sat_m * 0.5 * (alpha[NUM_GHOST-1:-NUM_GHOST-1] + alpha[NUM_GHOST:-NUM_GHOST])
        flux_m = flux_m / (1.0 + flux_m / (f_sat_m + 1e-5))


    flux = flux.at[NUM_GHOST:-NUM_GHOST].set(
        1.0 / dx * (flux_p - flux_m)
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
        ne=0.0,
    ):
    r_new = residual(
        temperature,
        dx,
        kappa0=kappa0,
        alpha=alpha,
        beta=beta,
        ne=ne,
    )
    r_old = residual(
        prev_temperature,
        dx,
        kappa0=kappa0,
        alpha=alpha,
        beta=beta,
        ne=ne,
    )
    return temperature - prev_temperature - dt * (theta * r_new + (1.0 - theta) * r_old)



@jax.jit
def solve_step(prev_temperature, dx, dt, kappa0, alpha, beta, ne=0.0):
    def to_opt(y, args):
        return implicit_residual(
            y,
            prev_temperature,
            dx,
            dt,
            kappa0=kappa0,
            alpha=alpha,
            beta=beta,
            ne=ne,
        )

    sol = optx.root_find(
        to_opt,
        solver=optx.Newton(
            rtol=1e-5,
            atol=1e-5,
            linear_solver=lineax.GMRES(rtol=1e-5, atol=1e-5)),
        y0=prev_temperature,
    )
    return sol.value

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

    saturate_flux = sim_config.get("saturate_conductive_flux", False)
    ne = 0.0
    if saturate_flux:
        nh = Q[IRHO] / (h_mass * mass_per_h)
        ne = y * nh

    cv = 1.0 / (gamma - 1.0) * k_B / (h_mass * mass_per_h) * (total_abund + y)
    alpha = 1.0 / (Q[IRHO] * cv)
    temperature = Q[IENE] * alpha

    new_temperature = solve_step(
        temperature,
        dx,
        dt,
        kappa0=kappa0,
        alpha=alpha,
        beta=2.5,
        ne=ne,
    )
    delta_E = (new_temperature - temperature) / alpha
    Q[IENE] += delta_E