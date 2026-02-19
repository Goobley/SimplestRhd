#!/usr/bin/env python3
"""
Run the Sod shock tube test case.

This script demonstrates how to run an experiment with the SimplestRhd package.
Each experiment is self-contained in its own directory with its own setup and configuration.
"""
from functools import partial
import sys
from pathlib import Path

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
    IRHO,
    IVEL,
    IPRE,
    IIONE,
    NUM_GHOST,
    reconstruct_ppm,
    rusanov_flux,
    hll_flux,
    lte_eos,
)
from setup import lte_sod_ics, lte_sod_bcs, config, L0, rho0, v0, t0


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
        "eos": partial(lte_eos, include_ion_e=config["include_ion_e"]),
    }

    # Create state dictionary
    state = {
        "xcc": grid,
        "dx": grid[1] - grid[0],
        "Q": lte_sod_ics(grid, gamma=gamma),
        "fixed_bcs": None,
        "user_bcs": None,
        "sources": [],
        "gamma": gamma,
        "bc_modes": lte_sod_bcs(),
    }

    # Run simulation
    print("Running LTE Sod shock tube test case...")
    snaps = run_sim(
        state,
        sim_config,
        max_time=config["max_time"],
        output_cadence=config["output_cadence"],
        max_cfl=config["max_cfl"],
    )

    print(f"Simulation complete. Generated {len(snaps)} snapshots.")

    # Plot final state
    times, states = zip(*snaps)
    final_time = times[-1]
    final_state = states[-1]

    # Convert to primitive variables
    w = cons_to_prim(final_state, gamma=gamma)

    # Extract interior points (excluding ghost cells)
    interior_slice = slice(NUM_GHOST, -NUM_GHOST)
    x_plot = grid[interior_slice] / L0
    rho = w[IRHO, interior_slice] / rho0
    v = w[IVEL, interior_slice] / v0
    p = w[IPRE, interior_slice] / (rho0 * v0**2)

    # Create plots
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].plot(x_plot, rho)
    axes[0].set_ylabel("Density")
    axes[0].set_xlabel("x")
    axes[0].set_title(f"t = {final_time / t0:.3f}")
    axes[0].grid(True)

    axes[1].plot(x_plot, v)
    axes[1].set_ylabel("Velocity")
    axes[1].set_xlabel("x")
    axes[1].grid(True)

    axes[2].plot(x_plot, p)
    axes[2].set_ylabel("Pressure")
    axes[2].set_xlabel("x")
    axes[2].grid(True)

    plt.tight_layout()
    # plt.savefig("sod_results.png", dpi=150)
    # print("Results saved to sod_results.png")

    try:
        plt.show()
    except:
        pass
