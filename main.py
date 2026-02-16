from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt

from implicit_thermal_conduction import implicit_thermal_conduction
try:
    get_ipython().run_line_magic("matplotlib", "")
except:
    plt.ion()

import numpy as np
from euler import DEFAULT_GAMMA, cons_to_prim, prim_to_cons, prim_to_flux, rusanov_flux, hll_flux, sound_speed, temperature_si
from reconstruct import reconstruct_fog, reconstruct_plm, reconstruct_ppm

from config import *

def rusanov_flux_with_padding(wL, wR, gamma: float=DEFAULT_GAMMA, flux_fn=rusanov_flux):
    # N.B. This takes the reconstructed faces in the cell frame, and returns a padded array with length (M+2*NUM_GHOST+1), i.e the flux at each interface in the block
    unpadded_flux = flux_fn(
        wR,
        np.roll(wL, -1, axis=1),
        gamma=gamma
    )
    full_flux = np.empty((wL.shape[0], wL.shape[1]+1))
    full_flux[:, 1:] = unpadded_flux
    full_flux[:, :NUM_GHOST] = 0.0
    full_flux[:, -NUM_GHOST:] = 0.0
    return full_flux

def set_bcs(state, dt, bc_modes, fixed_bc, user_bcs, gamma=DEFAULT_GAMMA):
    Q = state["Q"]
    if bc_modes[0] == SYMMETRIC_BC:
        Q[:, :NUM_GHOST] = Q[:, NUM_GHOST:2*NUM_GHOST][:, ::-1]
    elif bc_modes[0] == REFLECTING_BC:
        Q[:, :NUM_GHOST] = Q[:, NUM_GHOST:2*NUM_GHOST][:, ::-1]
        Q[1, :NUM_GHOST] = -Q[1, NUM_GHOST:2*NUM_GHOST][::-1]
    elif bc_modes[0] == FIXED_BC:
        Q[:, :NUM_GHOST] = fixed_bc[0][:, None]
    elif bc_modes[0] == USER_BC:
        user_bcs[0](Q, dt, gamma=gamma)

    if bc_modes[1] == SYMMETRIC_BC:
        Q[:, -NUM_GHOST:] = Q[:, -2*NUM_GHOST:-NUM_GHOST][:, ::-1]
    elif bc_modes[1] == REFLECTING_BC:
        Q[:, -NUM_GHOST:] = Q[:, -2*NUM_GHOST:-NUM_GHOST][:, ::-1]
        Q[1, -NUM_GHOST:] = -Q[1, -2*NUM_GHOST:-NUM_GHOST][::-1]
    elif bc_modes[1] == FIXED_BC:
        Q[:, -NUM_GHOST:] = fixed_bc[1][:, None]
    elif bc_modes[1] == USER_BC:
        user_bcs[1](Q, dt, gamma=gamma)

@dataclass
class TimestepInfo:
    t: float
    """current time"""
    dt: float
    """timestep"""
    cfl: float
    """associated cfl"""

def run_step(state, ts: TimestepInfo, bc_modes, fixed_bcs, source_terms, gamma: float=DEFAULT_GAMMA):
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
        # wL, wR = reconstruct_fog(w)
        fluxes = rusanov_flux_with_padding(wL, wR, gamma=gamma)

        for s in source_terms:
            s(xcc, Q, w, sources, ts.t)

        flux_div = fluxes[:, NUM_GHOST+1:-NUM_GHOST] - fluxes[:, NUM_GHOST:-(NUM_GHOST+1)]
        flux_update = - dt / dx * flux_div + sources[:, NUM_GHOST:-NUM_GHOST] * dt
        if stepper == "rk2":
            if substep == 0:
                Q[:, NUM_GHOST:-NUM_GHOST] += flux_update
            else:
                Q[:, NUM_GHOST:-NUM_GHOST] = 0.5 * (
                    Q_old[:, NUM_GHOST:-NUM_GHOST] + Q[:, NUM_GHOST:-NUM_GHOST]
                ) + flux_update
        elif stepper == "ssprk3":
            if substep == 0:
                Q[:, NUM_GHOST:-NUM_GHOST] += flux_update
            elif substep == 1:
                Q[:, NUM_GHOST:-NUM_GHOST] = (
                    0.75 * Q_old[:, NUM_GHOST:-NUM_GHOST]
                    + 0.25 * (Q[:, NUM_GHOST:-NUM_GHOST] + flux_update)
                )
            else:
                Q[:, NUM_GHOST:-NUM_GHOST] = (
                    (1.0 / 3.0) * Q_old[:, NUM_GHOST:-NUM_GHOST]
                    + (2.0 / 3.0) * (Q[:, NUM_GHOST:-NUM_GHOST] + flux_update)
                )
        elif stepper == "rk4":
            if substep != 2:
                Q[:, NUM_GHOST:-NUM_GHOST] += 0.5 * flux_update
            else:
                Q[:, NUM_GHOST:-NUM_GHOST] = (
                    (2.0 / 3.0) * Q_old[:, NUM_GHOST:-NUM_GHOST]
                    + (1.0 / 3.0) * Q[:, NUM_GHOST:-NUM_GHOST]
                    + (1.0 / 6.0) * flux_update
                )

def compute_dt(state, max_cfl):
    """
    Computes the min dt for the model
    """
    w = cons_to_prim(state["Q"])
    cs = sound_speed(w, gamma=gamma)
    fast_speed = np.abs(w[1]) + cs

    dt_local = max_cfl * state["dx"] / fast_speed
    return np.min(dt_local)

def sod_ics(x, gamma=DEFAULT_GAMMA):
    w = np.stack([
        np.where(x < 0.5, 1.0, 0.125),
        np.zeros_like(x),
        np.where(x < 0.5, 1.0, 0.1),
    ])
    return prim_to_cons(w, gamma=gamma)

def sod_bcs():
    return [SYMMETRIC_BC, SYMMETRIC_BC]

def big_sod_ics(x, gamma=DEFAULT_GAMMA):
    w = np.stack([
        np.where(x < 0.5, 1.0, 0.125),
        np.zeros_like(x),
        np.where(x < 0.5, 10.0, 0.1),
    ])
    return prim_to_cons(w, gamma=gamma)

def big_sod_bcs():
    return [SYMMETRIC_BC, SYMMETRIC_BC]

def woodward_collela_ics(x, gamma=DEFAULT_GAMMA):
    w = np.stack([
        np.ones_like(x),
        np.zeros_like(x),
        np.where(x < 0.1, 1e3, np.where(x < 0.9, 0.01, 100.0)),
    ])
    return prim_to_cons(w, gamma=gamma)

def woodward_collela_bcs():
    return [REFLECTING_BC, REFLECTING_BC]

def navarro_hypertc_test_ics(x, gamma=DEFAULT_GAMMA):
    temperature = 0.1 + 0.9*x**5
    rho = np.ones_like(x)
    v = np.zeros_like(x)
    avg_mass = 1.0
    p = 1.0 * rho / (avg_mass * P_MASS) * k_B * temperature

    if not USE_CONDUCTION:
        raise ValueError("Need conduction")
    if NUM_GHOST < 2:
        raise ValueError("Need 2 ghost cells for conduction")

    w = np.empty((NUM_EQ, x.shape[0]))
    w[IRHO] = rho
    w[IVEL] = v
    w[IPRE] = p
    return prim_to_cons(w, gamma=gamma)

def navarro_hypertc_test_bcs():
    return [USER_BC, USER_BC]

def navarro_hypertc_test_left_bc(block, dt, gamma=DEFAULT_GAMMA):
    block.Q[IRHO, :NUM_GHOST] = 1.0
    block.Q[IMOM, :NUM_GHOST] = 0.0
    p = 1.0 / P_MASS * k_B * 0.1
    block.Q[IENE, :NUM_GHOST] = p / (gamma - 1.0)

def navarro_hypertc_test_right_bc(block, dt, gamma=DEFAULT_GAMMA):
    block.Q[IRHO, -NUM_GHOST:] = 1.0
    block.Q[IMOM, -NUM_GHOST:] = 0.0
    p = 1.0 / P_MASS * k_B * 1.0
    block.Q[IENE, -NUM_GHOST:] = p / (gamma - 1.0)

def uniform_coronal_loop_ics(x, gamma=DEFAULT_GAMMA):
    temperature = 1e6
    rho = np.ones_like(x) * 1e-12
    v = np.zeros_like(x)
    p = rho / (P_MASS * MEAN_MOLECULAR_MASS) * k_B * temperature
    w = np.empty((NUM_EQ, x.shape[0]))
    w[IRHO] = rho
    w[IVEL] = v
    w[IPRE] = p

def uniform_coronal_loop_bcs():
    # return [USER_BC, USER_BC]
    return [REFLECTING_BC, REFLECTING_BC]

def uniform_coronal_loop_left_bc(block, dt, gamma=DEFAULT_GAMMA):
    temperature = 1e6
    rho = 1e-12
    v = 0.0
    p = rho / (P_MASS * MEAN_MOLECULAR_MASS) * k_B * temperature
    w = np.empty((NUM_EQ, 1))
    w[IRHO] = rho
    w[IVEL] = v
    w[IPRE] = p
    q = prim_to_cons(w, gamma=gamma)
    block.Q[:, :NUM_GHOST] = q
    block.Q[IMOM, :NUM_GHOST] = np.maximum(-block.Q[IMOM, NUM_GHOST], 0.0)
    # block.Q[ENE, :NUM_GHOST] = block.Q[ENE, NUM_GHOST]

def uniform_coronal_loop_right_bc(block, dt, gamma=DEFAULT_GAMMA):
    temperature = 1e6
    rho = 1e-12
    v = 0.0
    p = rho / (P_MASS * MEAN_MOLECULAR_MASS) * k_B * temperature
    w = np.empty((NUM_EQ, 1))
    w[IRHO] = rho
    w[IVEL] = v
    w[IPRE] = p
    q = prim_to_cons(w, gamma=gamma)
    block.Q[:, -NUM_GHOST:] = q
    block.Q[IMOM, -NUM_GHOST:] = np.minimum(-block.Q[IMOM, -NUM_GHOST-1], 0.0)
    # block.Q[ENE, -NUM_GHOST:] = block.Q[ENE, -NUM_GHOST-1]


def run_sim(state, bc_modes, max_time, max_cfl=0.5, max_steps=10_000_000, output_cadence=0.25, gamma=DEFAULT_GAMMA):
    current_time = 0.0
    snaps = []
    next_output = current_time + output_cadence
    snaps.append((current_time, state["Q"].copy()))
    dt = compute_dt(state, max_cfl=max_cfl)
    for i in range(max_steps):
        timestep_info = TimestepInfo(current_time, dt, max_cfl)
        run_step(state, timestep_info, bc_modes, state["fixed_bcs"], state["sources"], gamma=gamma)
        if USE_CONDUCTION:
            implicit_thermal_conduction(state, dt, gamma=gamma)
            set_bcs(state, dt, bc_modes, state["fixed_bcs"], state["user_bcs"], gamma=gamma)

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

def cons_to_temperature(Q, gamma=DEFAULT_GAMMA):
    w = cons_to_prim(Q, gamma=gamma)
    nh_tot = w[IRHO] / (P_MASS)
    return temperature_si(w[IPRE], nh_tot, 1.0)

def rad_loss_thin(temperature):
    lambda_cgs = np.where(
        temperature < 29.5e3,
        0.0,
        np.where(
            temperature < 30e3,
            10**(-25.518) * (temperature - 29.5e3),
            np.where(
                temperature < 100e3,
                10**(-36.25) * temperature**3,
                10**(-18.75) * temperature**(-0.5)
            )
        )
    )
    # u.Unit('erg cm3 s-1').to('J m3 s-1')
    # Out[6]: 1.0000000000000003e-13
    lambda_si = lambda_cgs * 1e-13
    return lambda_si

def radiative_loss_source(x, Q, W, S, t):
    y = 1.0
    nh_tot = W[IRHO] / P_MASS
    temperature = temperature_si(W[IPRE], nh_tot, y)

    lambda_si = rad_loss_thin(temperature)
    # NOTE(cmo): Fully ionised pure H for now
    loss = nh_tot**2 * lambda_si
    S[IENE, NUM_GHOST:-NUM_GHOST] -= loss[NUM_GHOST:-NUM_GHOST]

def background_heating(x, Q, W, S, t):
    initial_nh_tot = 1e-12 / P_MASS
    lambda_si = rad_loss_thin(1e6)
    heating = initial_nh_tot**2 * lambda_si
    S[IENE, NUM_GHOST:-NUM_GHOST] += heating

def stratified_heating(x, Q, W, S, t):
    if t < 10.0:
        return

    x0 = 1e6
    heating_width = 2e6
    x1 = 19e6
    heating_rate = 1e-4

    S[IENE, NUM_GHOST:-NUM_GHOST] += np.exp(-np.abs(x[NUM_GHOST:-NUM_GHOST] - x0) / heating_width) * heating_rate
    S[IENE, NUM_GHOST:-NUM_GHOST] += np.exp(-np.abs(x1 - x[NUM_GHOST:-NUM_GHOST]) / heating_width) * heating_rate * 1.5

def construct_x_grid(x0, x1, num_grid):
    dx = (x1 - x0) / (num_grid)
    return (x0 - NUM_GHOST * dx) + (np.arange(num_grid + 2 * NUM_GHOST) + 0.5) * dx

if __name__ == '__main__':
    fixed_bcs = None
    user_bcs = None
    sources = []

    ics = sod_ics
    bcs = sod_bcs
    max_time = 0.2
    output_cadence = 0.1
    gamma = 1.4

    # ics = big_sod_ics
    # bcs = big_sod_bcs
    # max_time = 0.1
    # output_cadence = max_time

    # ics = woodward_collela_ics
    # bcs = woodward_collela_bcs
    # max_time = 0.038
    # output_cadence = max_time

    # k_B = 1.0
    # P_MASS = 1.0
    # P_MASS = k_B
    # ics = navarro_hypertc_test_ics
    # bcs = navarro_hypertc_test_bcs
    # user_bcs = (navarro_hypertc_test_left_bc, navarro_hypertc_test_right_bc)
    # max_time = 2.0
    # output_cadence = 0.25

    # ics = uniform_coronal_loop_ics
    # bcs = uniform_coronal_loop_bcs
    # user_bcs = (uniform_coronal_loop_left_bc, uniform_coronal_loop_right_bc)
    # sources = [
    #     radiative_loss_source,
    #     background_heating,
    #     stratified_heating
    # ]
    # max_time = 3000.0
    # output_cadence = 5.0

    grid = construct_x_grid(0.0, 1.0, 100)
    state = dict(
        xcc=grid,
        dx=grid[1]-grid[0],
        gamma=gamma,
        Q=ics(grid, gamma=gamma),
        fixed_bcs=fixed_bcs,
        user_bcs=user_bcs,
        sources=sources,
    )
    bc_modes = bcs()

    states = run_sim(state, bc_modes, max_time=max_time, output_cadence=output_cadence, max_cfl=0.1)

