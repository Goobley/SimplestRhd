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
import astropy.constants as const

jax.config.update("jax_enable_x64", True)
M_P = const.m_p.value
K_B = const.k_B.value

def harmonic_mean(a, b):
    return 2.0 * a * b / (a + b + 1e-30)

def residual(temperature, dx, kappa0, alpha=1.0, beta=2.5):
    resid = jnp.zeros_like(temperature)
    kappa = alpha * kappa0 * temperature**beta
    kappa = kappa0 * temperature**beta
    kappa_if = jnp.zeros(resid.shape[0] + 1)
    kappa_if = kappa_if.at[1:-1].set(harmonic_mean(kappa[:-1], kappa[1:]))

    # NOTE(cmo): Uniform grid
    resid = resid.at[NUM_GHOST:-NUM_GHOST].set(
        1.0 / dx**2 * (
            kappa_if[NUM_GHOST+1:-NUM_GHOST] * (temperature[NUM_GHOST+1:-NUM_GHOST+1] - temperature[NUM_GHOST:-NUM_GHOST])
            - kappa_if[NUM_GHOST:-NUM_GHOST-1] * (temperature[NUM_GHOST:-NUM_GHOST] - temperature[NUM_GHOST-1:-NUM_GHOST-1])
        )
    )

    return resid

def implicit_residual(temperature, prev_temperature, dx, dt, kappa0, alpha=1.0, beta=2.5, theta=0.55):
    r_new = residual(temperature, dx, kappa0=kappa0, alpha=alpha, beta=beta)
    r_old = residual(prev_temperature, dx, kappa0=kappa0, alpha=alpha, beta=beta)
    return temperature - prev_temperature - dt * (theta * r_new + (1.0 - theta) * r_old)

@jax.jit
def solve_step(prev_temperature, dx, dt, kappa0, alpha, beta):
    def to_opt(y, args):
        return implicit_residual(y, prev_temperature, dx, dt, kappa0=kappa0, alpha=alpha, beta=beta)

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

    # TODO(cmo): This implicitly assumes fully ionised only H
    # TODO(cmo): Infer alpha directly, using the temperature function passed
    # cv = 1.0 / (gamma - 1.0) * k_B / (P_MASS * MEAN_MOLECULAR_MASS)
    # TODO(cmo): Mass per h term
    cv = 1.0 / (gamma - 1.0) * k_B / (h_mass) / (1.0 + y)
    alpha = 1.0 / (Q[IRHO] * cv)
    temperature = Q[IENE] * alpha
    # breakpoint()

    new_temperature = solve_step(temperature, dx, dt, kappa0=kappa0, alpha=alpha, beta=2.5)
    delta_E = (new_temperature - temperature) / alpha
    Q[IENE] += delta_E