"""
Main simulation runner and time integration
"""
from dataclasses import dataclass
from matplotlib.pylab import gamma
import numpy as np
from pathlib import Path
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
from .tracers import normalise_tracers, tracer_flux
from .io import save_snapshot

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


def numeric_flux_with_padding(wL, wR, gamma: float, flux_fn=hll_flux, tracersL=None, tracersR=None):
    """Apply numeric flux with padding for ghost cells.

    Args:
        wL: Left reconstructed states for each cell
        wR: Right reconstructed states
        gamma: Adiabatic index
        flux_fn: Riemann solver function
        tracersL: Left reconstructed states for normalised tracers
        tracersR: Right reconstructed states for normalised tracers

    Returns:
        full_flux: Flux array with padding for ghost cells
        tr_flux: Flux array for tracers (equivalently padded). None if tracersL or tracersR are None.
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

    tr_flux = None
    if tracersL is not None and tracersR is not None:
        unpadded_tracer_flux = tracer_flux(
            tracersR,
            np.roll(tracersL, -1, axis=1),
            unpadded_flux[IRHO],
        )
        tr_flux = np.empty((tracersL.shape[0], tracersL.shape[1] + 1))
        tr_flux[:, 1:] = unpadded_tracer_flux
        tr_flux[:, :NUM_GHOST] = 0.0
        tr_flux[:, -NUM_GHOST:] = 0.0

    return full_flux, tr_flux


def set_bcs(state, sim_config, dt):
    """Set boundary conditions.

    Args:
        state: State dictionary containing Q, gamma
        sim_config: Simulation config dict containing bc_modes, fixed_bcs, user_bcs
        dt: Timestep
    """
    Q = state["Q"]
    bc_modes = sim_config["bc_modes"]
    fixed_bc = sim_config["fixed_bcs"]
    user_bcs = sim_config["user_bcs"]
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
        state: State dictionary containing Q, gamma, dx, xcc
        sim_config: Simulation configuration dict with reconstruction_fn, flux_fn, bc_modes, fixed_bcs, user_bcs
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
    run_hydro = sim_config.get("run_hydro", True)
    custom_eos = sim_config.get("eos")
    use_custom_eos = custom_eos is not None

    Q_old = Q.copy()
    sources = np.zeros_like(Q)
    tracers_old = state["tracers"].copy() if "tracers" in state else None

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
        set_bcs(state, sim_config, dt_sub)

        sources[...] = 0.0
        w = cons_to_prim(Q, gamma=gamma)
        state['W'] = w
        wL, wR = reconstruction_fn(w)

        n_tracers_L, n_tracers_R = None, None
        if tracers_old is not None:
            tracers = state["tracers"]
            norm_tracers = normalise_tracers(tracers, Q[IRHO])
            n_tracers_L, n_tracers_R = reconstruction_fn(norm_tracers)

        fluxes, tracer_flux = numeric_flux_with_padding(
            wL,
            wR,
            gamma=gamma,
            flux_fn=flux_fn,
            tracersL=n_tracers_L,
            tracersR=n_tracers_R
        )
        if not run_hydro:
            fluxes[...] = 0.0
            if tracer_flux is not None:
                tracer_flux[...] = 0.0

        for s in source_terms:
            s(state, sim_config, sources, ts)

        flux_div = fluxes[:, NUM_GHOST + 1 : -NUM_GHOST] - fluxes[:, NUM_GHOST : -(NUM_GHOST + 1)]
        flux_update = -dt / dx * flux_div + sources[:, NUM_GHOST : -NUM_GHOST] * dt
        if tracers_old is not None:
            t_flux_update = -dt / dx * (
                tracer_flux[:, NUM_GHOST + 1: -NUM_GHOST] - tracer_flux[:, NUM_GHOST : -(NUM_GHOST + 1)]
                )
        if stepper == "rk2":
            if substep == 0:
                Q[:, NUM_GHOST : -NUM_GHOST] += flux_update
                if tracers_old is not None:
                    tracers[:, NUM_GHOST : -NUM_GHOST] += t_flux_update
            else:
                Q[:, NUM_GHOST : -NUM_GHOST] = 0.5 * (
                    Q_old[:, NUM_GHOST : -NUM_GHOST] + Q[:, NUM_GHOST : -NUM_GHOST] + flux_update
                )
                if tracers_old is not None:
                    tracers[:, NUM_GHOST : -NUM_GHOST] = 0.5 * (
                        tracers_old[:, NUM_GHOST : -NUM_GHOST] + tracers[:, NUM_GHOST : -NUM_GHOST] + t_flux_update
                    )
        elif stepper == "ssprk3":
            if substep == 0:
                Q[:, NUM_GHOST : -NUM_GHOST] += flux_update
                if tracers_old is not None:
                    tracers[:, NUM_GHOST : -NUM_GHOST] += t_flux_update
            elif substep == 1:
                Q[:, NUM_GHOST : -NUM_GHOST] = (
                    0.75 * Q_old[:, NUM_GHOST : -NUM_GHOST]
                    + 0.25 * (Q[:, NUM_GHOST : -NUM_GHOST] + flux_update)
                )
                if tracers_old is not None:
                    tracers[:, NUM_GHOST : -NUM_GHOST] = (
                        0.75 * tracers_old[:, NUM_GHOST : -NUM_GHOST]
                        + 0.25 * (tracers[:, NUM_GHOST : -NUM_GHOST] + t_flux_update)
                    )
            else:
                Q[:, NUM_GHOST : -NUM_GHOST] = (
                    (1.0 / 3.0) * Q_old[:, NUM_GHOST : -NUM_GHOST]
                    + (2.0 / 3.0) * (Q[:, NUM_GHOST : -NUM_GHOST] + flux_update)
                )
                if tracers_old is not None:
                    tracers[:, NUM_GHOST : -NUM_GHOST] = (
                        (1.0 / 3.0) * tracers_old[:, NUM_GHOST : -NUM_GHOST]
                        +  (2.0 / 3.0) * (tracers[:, NUM_GHOST : -NUM_GHOST] + t_flux_update)
                    )
        elif stepper == "rk4":
            if substep != 2:
                Q[:, NUM_GHOST : -NUM_GHOST] += 0.5 * flux_update
                if tracers_old is not None:
                    tracers[:, NUM_GHOST : -NUM_GHOST] += 0.5 * t_flux_update
            else:
                Q[:, NUM_GHOST : -NUM_GHOST] = (
                    (2.0 / 3.0) * Q_old[:, NUM_GHOST : -NUM_GHOST]
                    + (1.0 / 3.0) * Q[:, NUM_GHOST : -NUM_GHOST]
                    + (1.0 / 6.0) * flux_update
                )
                if tracers_old is not None:
                    tracers[:, NUM_GHOST : -NUM_GHOST] = (
                        (2.0 / 3.0) * tracers_old[:, NUM_GHOST : -NUM_GHOST]
                        +  (1.0 / 3.0) * tracers[:, NUM_GHOST : -NUM_GHOST]
                        + (1.0 / 6.0) * t_flux_update
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
            output_cadence=0.25, snapshot_dir="snapshots"):
    """Run the simulation.

    Args:
        state: State dictionary with initial conditions, gamma, sources, unsplit_sources, time
        sim_config: Simulation config dict with reconstruction_fn, flux_fn, conduction_fn,
                   bc_modes, fixed_bcs, user_bcs
        max_time: Maximum simulation time
        max_cfl: Maximum CFL number
        max_steps: Maximum number of steps
        output_cadence: Time between outputs (snapshots are written at these intervals)
        snapshot_dir: Directory to save snapshots to. Snapshots are automatically saved at
                     output_cadence intervals with filenames snap_NNNNN.nc. Default is "snapshots".
                     Use None to disable snapshot writing.

    Returns:
        n_iterations: Number of iterations completed
    """
    conduction_fn = sim_config.get("conduction_fn")
    use_conduction = conduction_fn is not None

    # Initialize snapshot directory if provided
    if snapshot_dir is not None:
        snapshot_dir = Path(snapshot_dir)
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        # Initialize snap_num if not already in state
        if "snap_num" not in state:
            state["snap_num"] = 0

    current_time = state.get("time", 0.0)
    state["time"] = current_time
    next_output = min(current_time + output_cadence, max_time)

    # Save initial snapshot if snapshot_dir is provided
    if snapshot_dir is not None:
        save_snapshot(state, str(snapshot_dir))

    dt = compute_dt(state, max_cfl=max_cfl)
    if current_time + dt > next_output:
        dt = next_output - current_time
        while current_time + dt < next_output:
            dt = np.nextafter(dt, np.inf)

    for i in range(max_steps):
        timestep_info = TimestepInfo(current_time, dt, dt, max_cfl)
        run_step(state, sim_config, timestep_info, state["sources"])

        # Apply unsplit source terms once per full timestep
        unsplit_sources = state.get("unsplit_sources", [])
        if unsplit_sources:
            unsplit_state_update = np.zeros_like(state["Q"])
            for s in unsplit_sources:
                s(state, sim_config, unsplit_state_update, timestep_info)
            state["Q"][:, NUM_GHOST : -NUM_GHOST] += unsplit_state_update[:, NUM_GHOST : -NUM_GHOST] * timestep_info.dt

        if use_conduction:
            conduction_fn(state, sim_config, dt)
            set_bcs(state, sim_config, dt)

        current_time += dt
        state["time"] = current_time
        if current_time >= next_output:
            # Save snapshot if snapshot_dir is provided
            if snapshot_dir is not None:
                save_snapshot(state, str(snapshot_dir))
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

    return i + 1
