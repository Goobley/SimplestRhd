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

from simplestrhd import (
    run_sim,
    cons_to_prim,
    prim_to_cons,
    IRHO,
    IVEL,
    IPRE,
    IIONE,
    NUM_GHOST,
    reconstruct_ppm,
    rusanov_flux,
    hll_flux,
    lte_eos,
    SYMMETRIC_BC,
    rad_loss_dm,
    logt_DM,
    lambda_DM,
    temperature_si,
    IENE,
    TownsendThinLoss,
)

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

config = dict(
    x_min = -2e6,
    x_max = 2e6,
    num_grid_points = 1000,
    gamma = 5/3,
    max_time = 100.0,
    output_cadence = 0.5,
    max_cfl = 0.9,
    base_pressure = 0.023,
    base_density = 1e-12,
    blob_density = 5e-11,
    blob_delta = 0.5e6,
)

if __name__ == "__main__":
    # Construct grid
    grid = construct_x_grid(
        config["x_min"],
        config["x_max"],
        config["num_grid_points"],
    )

    gamma = config["gamma"]

    # Create simulation config
    sim_config = {
        "reconstruction_fn": reconstruct_ppm,
        "flux_fn": hll_flux,
        "timestepper": "ssprk3",
        "conduction_fn": None,
        "bc_modes": [SYMMETRIC_BC, SYMMETRIC_BC],
        "fixed_bcs": None,
        "user_bcs": None,
    }

    def condensation_ics(x, gamma):
        base_pressure = config["base_pressure"]
        base_density = config["base_density"]
        blob_density = config["blob_density"]
        blob_delta = config["blob_delta"]
        w = np.stack([
            np.ones_like(x) * base_density,
            np.zeros_like(x),
            np.ones_like(x) * base_pressure,
            np.zeros_like(x),
        ])
        w[IRHO, :] += blob_density * np.exp(-x**2 / blob_delta**2)
        return prim_to_cons(w, gamma=gamma)

    def background_heating(state, sim_config, sources, time):
        m_p = const.m_p.value
        initial_nh_tot = config["base_density"] / m_p
        temperature = temperature_si(config["base_pressure"], initial_nh_tot, y=1)
        lambda_si = 10**np.interp(np.log10(temperature), logt_DM, lambda_DM)
        heating = initial_nh_tot**2 * lambda_si
        sources[IENE, NUM_GHOST:-NUM_GHOST] += heating

    # Create state dictionary
    state = {
        "xcc": grid,
        "dx": grid[1] - grid[0],
        "Q": condensation_ics(grid, gamma=gamma),
        "sources": [
            TownsendThinLoss("simple"),
            # background_heating,
        ],
        "gamma": gamma,
        "time": 0.0,
    }

    # Run simulation
    num_iter = run_sim(
        state,
        sim_config,
        max_time=config["max_time"],
        output_cadence=config["output_cadence"],
        max_cfl=config["max_cfl"],
    )


    # Convert to primitive variables
    w = cons_to_prim(state["Q"], gamma=gamma)
    final_time = state["time"]

    # Extract interior points (excluding ghost cells)
    interior_slice = slice(NUM_GHOST, -NUM_GHOST)
    x_plot = grid[interior_slice]
    rho = w[IRHO, interior_slice]
    v = w[IVEL, interior_slice]
    p = w[IPRE, interior_slice]

    # Create plots
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].plot(x_plot, rho)
    axes[0].set_ylabel("Density")
    axes[0].set_xlabel("x")
    axes[0].set_title(f"t = {final_time:.3f}")
    axes[0].grid(True)

    axes[1].plot(x_plot, v)
    axes[1].set_ylabel("Velocity")
    axes[1].set_xlabel("x")
    axes[1].grid(True)

    axes[2].plot(x_plot, p)
    axes[2].set_ylabel("Pressure")
    axes[2].set_xlabel("x")
    axes[2].grid(True)

