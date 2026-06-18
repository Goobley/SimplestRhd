import sys
from pathlib import Path

import astropy.constants as const
import numpy as np
import matplotlib.pyplot as plt
try:
    get_ipython().run_line_magic("matplotlib", "")
except:
    plt.ion()

# Add parent directory to path so we can import simplestrhd
sys.path.insert(0, str(Path(__file__).absolute().parent.parent.parent))

# TODO(cmo): Look at setting up a finite volume like pievewise constant solver from the cell centre values, or interpolating somehow.

from simplestrhd import (
    run_sim,
    cons_to_prim,
    prim_to_cons,
    IRHO,
    IVEL,
    IPRE,
    IIONE,
    NUM_GHOST,
    reconstruct_plm,
    reconstruct_ppm,
    rusanov_flux,
    hll_flux,
    hllc_flux,
    SYMMETRIC_BC,
    USER_BC
)

PURE_CONTACT = False

def construct_x_grid(x0, x1, num_grid):
    """Construct grid with ghost cells.

    Args:
        x0: Left boundary
        x1: Right boundary
        num_grid: Number of interior grid points

    Returns:
        x: Grid positions (including ghost cells)
    """
    dx = (x1 - x0) / num_grid
    return (x0 - NUM_GHOST * dx) + (np.arange(num_grid + 2 * NUM_GHOST) + 0.5) * dx

def reflect_left(Q, state, sim_config, ts):
    Q[:, :NUM_GHOST] = Q[:, NUM_GHOST:2 * NUM_GHOST][:, ::-1]
    Q[1, :NUM_GHOST] = -Q[1, NUM_GHOST:2 * NUM_GHOST][::-1]
    if "tracers" in state:
        tracers = state["tracers"]
        tracers[:, :NUM_GHOST] = tracers[:, NUM_GHOST:2 * NUM_GHOST][:, ::-1]

def reflect_right(Q, state, sim_config, ts):
    Q[:, -NUM_GHOST:] =  Q[:, -2 * NUM_GHOST:-NUM_GHOST][:, ::-1]
    Q[1, -NUM_GHOST:] = -Q[1, -2 * NUM_GHOST:-NUM_GHOST][::-1]
    if "tracers" in state:
        tracers = state["tracers"]
        tracers[:, -NUM_GHOST:] = tracers[:, -2*NUM_GHOST:-NUM_GHOST][:, ::-1]

if __name__ == "__main__":
    grid = construct_x_grid(
        0.0,
        1.0,
        50,
    )

    sim_config = dict(
        gamma=1.4,
        reconstruction_fn=reconstruct_ppm,
        flux_fn=hll_flux,
        timestepper="ssprk3",
        bc_modes=[USER_BC, USER_BC],
        fixed_bcs=None,
        user_bcs=[reflect_left, reflect_right],
    )

    def setup_ics(x, gamma, use_cma=False):
        # mach = 2.0
        # rho_l = (gamma + 1.0) * mach**2 / (2.0 + (gamma - 1.0) * mach**2)
        # vx_l = -mach / rho_l
        # rpres = 1.0 + gamma * mach**2 * (1.0 - 1.0 / rho_l)
        # pres_l = rpres / gamma

        # rho_r = 1.0
        # vx_r = -mach
        # pres_r = 1.0 / gamma

        rho_l = 1.0
        vx_l = 0.0
        pres_l = 1.0

        rho_r = 0.125
        vx_r = 0.0
        pres_r = 0.1

        if PURE_CONTACT:
            rho_l = 1.0
            vx_l = 0.0
            pres_l = 1.0

            rho_r = 0.125
            vx_r = vx_l
            pres_r = pres_l

        w = np.stack([
            np.where(x <= 0.5, rho_l, rho_r),
            np.where(x <= 0.5, vx_l, vx_r),
            np.where(x <= 0.5, pres_l, pres_r),
            np.zeros_like(x)
        ])
        q = prim_to_cons(w, gamma=gamma)
        tracers = np.stack([
            # np.where(x < 0.0, 2.0 * rho_l * (1.0 - 1e-6), 2.0 * rho_r * 1e-5),
            # np.where(x < 0.0, 2.0 * rho_l * 1e-6, 2.0 * rho_r * (1.0 - 1e-5)),
            np.where(
                x <= 0.5,
                0.7,
                np.where(x <= 0.75, 0.3, 0.1)
            ),
            0.3 * np.sin(20.0 * np.pi * x)**2,
            np.zeros_like(x),
        ])
        tracers[2, :] = 1.0 - tracers[0] - tracers[1]
        density_scale = 1e0
        tracers[:, :] *= density_scale * q[0]
        state = dict(
            xcc=x,
            dx=x[1]-x[0],
            Q=q,
            gamma=gamma,
            time=0.0,
            tracers=tracers,
        )
        extra_config = {}
        if use_cma:
            extra_config["use_tracer_cma"] = True
            extra_config['tracer_cma_start_idx'] = [0]
            extra_config['tracer_cma_end_idx'] = [tracers.shape[0]]
            # extra_config['tracer_cma_inv_density'] = [2.0]
            extra_config['tracer_cma_inv_sum'] = [density_scale]

        return state, extra_config

    state, extra_config = setup_ics(grid, sim_config["gamma"])
    state = state | {
        "sources": []
    }
    base_config = sim_config | extra_config

    num_iter = run_sim(
        state,
        base_config,
        max_time=1.0,
        output_cadence=0.1,
        max_cfl=0.6,
    )

    state_cma, extra_config_cma = setup_ics(grid, sim_config["gamma"], use_cma=True)
    state_cma = state_cma | {
        "sources": []
    }
    sim_config_cma = sim_config | extra_config_cma | dict(flux_fn=hllc_flux)

    num_iter = run_sim(
        state_cma,
        sim_config_cma,
        max_time=1.0,
        output_cadence=0.1,
        max_cfl=0.6,
        snapshot_dir='cma'
    )

    state_cma_flat, extra_config_cma_flat = setup_ics(grid, sim_config["gamma"], use_cma=True)
    state_cma_flat = state_cma_flat | {
        "sources": []
    }
    flat = sim_config | extra_config_cma_flat | dict(tracer_cma_flatten=True, flux_fn=hllc_flux)

    num_iter = run_sim(
        state_cma_flat,
        flat,
        max_time=1.0,
        output_cadence=0.1,
        max_cfl=0.6,
        snapshot_dir='cma_flatten'
    )


