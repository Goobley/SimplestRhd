"""
Main simulation runner and time integration
"""
from dataclasses import dataclass
from matplotlib.pylab import gamma
import numpy as np
from .eos import cons_to_prim, prim_to_cons
from .riemann_flux import hll_flux, rusanov_flux
from .reconstruction import reconstruct_plm, reconstruct_ppm
from .indices import (
    NUM_GHOST,
    IRHO,
    IMOM,
    IENE,
    IIONE,
    IVEL,
    IPRE,
    SYMMETRIC_BC,
    REFLECTING_BC,
    FIXED_BC,
    USER_BC,
)

Array = np.ndarray


@dataclass
class TimestepInfo:
    t: float
    """current time"""
    dt: float
    """timestep"""
    dt_sub: float
    """substep"""
    cfl: float
    """associated cfl"""


def numeric_flux_with_padding(wL, wR, gamma: float, flux_fn=hll_flux):
    """Apply Rusanov flux with padding for ghost cells.

    Args:
        wL: Left reconstructed states
        wR: Right reconstructed states
        gamma: Adiabatic index
        flux_fn: Riemann solver function

    Returns:
        full_flux: Flux array with padding for ghost cells
    """
    unpadded_flux = flux_fn(
        wR,
        np.roll(wL, -1, axis=1),
        gamma=gamma
    )
    full_flux = np.empty((wL.shape[0], wL.shape[1] + 1))
    full_flux[:, 1:] = unpadded_flux
    full_flux[:, :NUM_GHOST] = 0.0
    full_flux[:, -NUM_GHOST:] = 0.0
    return full_flux


def set_bcs(state, dt):
    """Set boundary conditions.

    Args:
        state: State dictionary containing Q, bc_modes, fixed_bcs, user_bcs, gamma
        dt: Timestep
    """
    Q = state["Q"]
    bc_modes = state["bc_modes"]
    fixed_bc = state["fixed_bcs"]
    user_bcs = state["user_bcs"]
    gamma = state["gamma"]

    if bc_modes[0] == SYMMETRIC_BC:
        Q[:, :NUM_GHOST] = Q[:, NUM_GHOST : 2 * NUM_GHOST][:, ::-1]
    elif bc_modes[0] == REFLECTING_BC:
        Q[:, :NUM_GHOST] = Q[:, NUM_GHOST : 2 * NUM_GHOST][:, ::-1]
        Q[1, :NUM_GHOST] = -Q[1, NUM_GHOST : 2 * NUM_GHOST][::-1]
    elif bc_modes[0] == FIXED_BC:
        Q[:, :NUM_GHOST] = fixed_bc[0][:, None]
    elif bc_modes[0] == USER_BC:
        user_bcs[0](Q, dt, gamma=gamma)

    if bc_modes[1] == SYMMETRIC_BC:
        Q[:, -NUM_GHOST:] = Q[:, -2 * NUM_GHOST : -NUM_GHOST][:, ::-1]
    elif bc_modes[1] == REFLECTING_BC:
        Q[:, -NUM_GHOST:] = Q[:, -2 * NUM_GHOST : -NUM_GHOST][:, ::-1]
        Q[1, -NUM_GHOST:] = -Q[1, -2 * NUM_GHOST : -NUM_GHOST][::-1]
    elif bc_modes[1] == FIXED_BC:
        Q[:, -NUM_GHOST:] = fixed_bc[1][:, None]
    elif bc_modes[1] == USER_BC:
        user_bcs[1](Q, dt, gamma=gamma)


def run_step(state, sim_config, ts: TimestepInfo, source_terms):
    """Perform one simulation step.

    Args:
        state: State dictionary containing Q, bc_modes, fixed_bcs, user_bcs, gamma, dx, xcc
        sim_config: Simulation configuration dict with reconstruction_fn, flux_fn
        ts: Timestep information
        source_terms: List of source term functions
    """
    Q = state["Q"]
    xcc = state["xcc"]
    dx = state["dx"]
    gamma = state["gamma"]

    reconstruction_fn = sim_config["reconstruction_fn"]
    flux_fn = sim_config["flux_fn"]
    stepper = sim_config.get("timestepper", "ssprk3")
    custom_eos = sim_config.get("eos")
    use_custom_eos = custom_eos is not None

    Q_old = Q.copy()
    sources = np.zeros_like(Q)

    dt = ts.dt
    if stepper == "rk2":
        dt_scheme = [dt, 0.5 * dt]
    elif stepper == "ssprk3":
        dt_scheme = [dt, 0.25 * dt, (2.0 / 3.0) * dt]
    elif stepper == "rk4":
        dt_scheme = [0.5 * dt, 0.5 * dt, (1.0 / 6.0) * dt, 0.5 * dt]

    for substep, dt_sub in enumerate(dt_scheme):
        ts.dt_sub = dt_sub
        fluxes = []
        set_bcs(state, dt_sub)

        sources[...] = 0.0
        w = cons_to_prim(Q, gamma=gamma)
        state['W'] = w
        wL, wR = reconstruction_fn(w)
        fluxes = numeric_flux_with_padding(wL, wR, gamma=gamma, flux_fn=flux_fn)

        for s in source_terms:
            s(state, sim_config, sources, ts)

        flux_div = fluxes[:, NUM_GHOST + 1 : -NUM_GHOST] - fluxes[:, NUM_GHOST : -(NUM_GHOST + 1)]
        flux_update = -dt / dx * flux_div + sources[:, NUM_GHOST : -NUM_GHOST] * dt
        if stepper == "rk2":
            if substep == 0:
                Q[:, NUM_GHOST : -NUM_GHOST] += flux_update
            else:
                Q[:, NUM_GHOST : -NUM_GHOST] = 0.5 * (
                    Q_old[:, NUM_GHOST : -NUM_GHOST] + Q[:, NUM_GHOST : -NUM_GHOST] + flux_update
                )
        elif stepper == "ssprk3":
            if substep == 0:
                Q[:, NUM_GHOST : -NUM_GHOST] += flux_update
            elif substep == 1:
                Q[:, NUM_GHOST : -NUM_GHOST] = (
                    0.75 * Q_old[:, NUM_GHOST : -NUM_GHOST]
                    + 0.25 * (Q[:, NUM_GHOST : -NUM_GHOST] + flux_update)
                )
            else:
                Q[:, NUM_GHOST : -NUM_GHOST] = (
                    (1.0 / 3.0) * Q_old[:, NUM_GHOST : -NUM_GHOST]
                    + (2.0 / 3.0) * (Q[:, NUM_GHOST : -NUM_GHOST] + flux_update)
                )
        elif stepper == "rk4":
            if substep != 2:
                Q[:, NUM_GHOST : -NUM_GHOST] += 0.5 * flux_update
            else:
                Q[:, NUM_GHOST : -NUM_GHOST] = (
                    (2.0 / 3.0) * Q_old[:, NUM_GHOST : -NUM_GHOST]
                    + (1.0 / 3.0) * Q[:, NUM_GHOST : -NUM_GHOST]
                    + (1.0 / 6.0) * flux_update
                )

        if use_custom_eos:
            custom_eos(state, sim_config)


def compute_dt(state, max_cfl):
    """Compute timestep for CFL condition.

    Args:
        state: State dictionary containing Q, gamma, dx
        max_cfl: Maximum CFL number

    Returns:
        dt: Timestep
    """
    from .eos import sound_speed

    gamma = state["gamma"]
    w = cons_to_prim(state["Q"], gamma=gamma)
    cs = sound_speed(w, gamma=gamma)
    fast_speed = np.abs(w[IVEL]) + cs

    dt_local = max_cfl * state["dx"] / fast_speed
    return np.min(dt_local)


def run_sim(state, sim_config, max_time, max_cfl=0.5, max_steps=10_000_000,
            output_cadence=0.25):
    """Run the simulation.

    Args:
        state: State dictionary with initial conditions, bc_modes, fixed_bcs, user_bcs, gamma
        sim_config: Simulation config dict with reconstruction_fn, flux_fn, conduction_fn
        max_time: Maximum simulation time
        max_cfl: Maximum CFL number
        max_steps: Maximum number of steps
        output_cadence: Time between outputs

    Returns:
        snaps: List of (time, state) tuples
    """
    conduction_fn = sim_config.get("conduction_fn")
    use_conduction = conduction_fn is not None

    current_time = 0.0
    snaps = []
    next_output = min(current_time + output_cadence, max_time)
    snaps.append((current_time, state["Q"].copy()))

    dt = compute_dt(state, max_cfl=max_cfl)
    if current_time + dt > next_output:
        dt = next_output - current_time
        while current_time + dt < next_output:
            dt = np.nextafter(dt, np.inf)

    for i in range(max_steps):
        timestep_info = TimestepInfo(current_time, dt, dt, max_cfl)
        run_step(state, sim_config, timestep_info, state["sources"])

        if use_conduction:
            conduction_fn(state, dt)
            set_bcs(state, dt)

        current_time += dt
        if current_time >= next_output:
            snaps.append((current_time, state["Q"].copy()))
            next_output = current_time + output_cadence

        if i % 50 == 0 or current_time >= max_time:
            print(f"t: {current_time:.4f} s, dt: {dt:.2e} s, iter: {i:9d}")

        if current_time >= max_time:
            break

        dt = compute_dt(state, max_cfl=max_cfl)

        if current_time + dt > next_output:
            dt = next_output - current_time
            while current_time + dt < next_output:
                dt = np.nextafter(dt, np.inf)

        if np.isnan(dt):
            break

    return snaps
