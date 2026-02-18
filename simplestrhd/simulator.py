"""
Main simulation runner and time integration
"""
from dataclasses import dataclass
from matplotlib.pylab import gamma
import numpy as np
from .eos import cons_to_prim, prim_to_cons
from .solver import rusanov_flux
from .reconstruction import reconstruct_ppm
from .indices import (
    NUM_GHOST,
    IRHO,
    IMOM,
    IENE,
    IION,
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
    cfl: float
    """associated cfl"""


def rusanov_flux_with_padding(wL, wR, gamma: float, flux_fn=rusanov_flux):
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


def set_bcs(state, dt, bc_modes, fixed_bc, user_bcs, gamma: float):
    """Set boundary conditions.

    Args:
        state: State dictionary
        dt: Timestep
        bc_modes: Boundary condition modes (left, right)
        fixed_bc: Fixed boundary condition values
        user_bcs: User-defined boundary condition functions
        gamma: Adiabatic index
    """
    Q = state["Q"]
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


def run_step(state, ts: TimestepInfo, bc_modes, fixed_bcs, source_terms, gamma: float):
    """Perform one simulation step.

    Args:
        state: State dictionary
        ts: Timestep information
        bc_modes: Boundary condition modes
        fixed_bcs: Fixed boundary condition values
        source_terms: List of source term functions
        gamma: Adiabatic index
    """
    Q = state["Q"]
    xcc = state["xcc"]
    dx = state["dx"]

    Q_old = Q.copy()
    sources = np.zeros_like(Q)

    dt = ts.dt
    stepper = "rk4"
    if stepper == "rk2":
        dt_scheme = [dt, 0.5 * dt]
    elif stepper == "ssprk3":
        dt_scheme = [dt, 0.25 * dt, (2.0 / 3.0) * dt]
    elif stepper == "rk4":
        dt_scheme = [0.5 * dt, 0.5 * dt, (1.0 / 6.0) * dt, 0.5 * dt]

    for substep, dt_sub in enumerate(dt_scheme):
        fluxes = []
        set_bcs(state, dt_sub, bc_modes, fixed_bcs, state["user_bcs"], gamma=gamma)

        sources[...] = 0.0
        w = cons_to_prim(Q, gamma=gamma)
        wL, wR = reconstruct_ppm(w)
        fluxes = rusanov_flux_with_padding(wL, wR, gamma=gamma)

        for s in source_terms:
            s(xcc, Q, w, sources, ts.t)

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


def compute_dt(state, max_cfl, gamma: float):
    """Compute timestep for CFL condition.

    Args:
        state: State dictionary
        max_cfl: Maximum CFL number
        gamma: Adiabatic index

    Returns:
        dt: Timestep
    """
    from .eos import sound_speed

    w = cons_to_prim(state["Q"], gamma=gamma)
    cs = sound_speed(w, gamma=gamma)
    fast_speed = np.abs(w[IVEL]) + cs

    dt_local = max_cfl * state["dx"] / fast_speed
    return np.min(dt_local)


def run_sim(state, bc_modes, max_time, max_cfl=0.5, max_steps=10_000_000,
            output_cadence=0.25, conduction_fn=None):
    """Run the simulation.

    Args:
        state: State dictionary with initial conditions
        bc_modes: Boundary condition modes
        max_time: Maximum simulation time
        max_cfl: Maximum CFL number
        max_steps: Maximum number of steps
        output_cadence: Time between outputs
        gamma: Adiabatic index
        conduction_fn: Optional conduction function

    Returns:
        snaps: List of (time, state) tuples
    """
    USE_CONDUCTION = conduction_fn is not None

    current_time = 0.0
    snaps = []
    next_output = current_time + output_cadence
    snaps.append((current_time, state["Q"].copy()))
    dt = compute_dt(state, max_cfl=max_cfl)

    for i in range(max_steps):
        timestep_info = TimestepInfo(current_time, dt, max_cfl)
        run_step(state, timestep_info, bc_modes, state["fixed_bcs"], state["sources"])

        if USE_CONDUCTION:
            conduction_fn(state, dt)
            set_bcs(state, dt, bc_modes, state["fixed_bcs"], state["user_bcs"])

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
