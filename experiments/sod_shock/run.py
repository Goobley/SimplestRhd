#!/usr/bin/env python3
"""
Run the Sod shock tube test case.

This script demonstrates how to run an experiment with the SimplestRhd package.
Each experiment is self-contained in its own directory with its own setup and configuration.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path so we can import simplestrhd
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from simplestrhd import (
    run_sim,
    cons_to_prim,
    IRHO,
    IVEL,
    IPRE,
    IION,
    NUM_GHOST,
)
from setup import sod_ics, sod_bcs, config


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

    # Create state dictionary
    state = {
        "xcc": grid,
        "dx": grid[1] - grid[0],
        "Q": sod_ics(grid, gamma=gamma),
        "fixed_bcs": None,
        "user_bcs": None,
        "sources": [],
    }

    bc_modes = sod_bcs()

    # Run simulation
    print("Running Sod shock tube test case...")
    snaps = run_sim(
        state,
        bc_modes,
        max_time=config["max_time"],
        output_cadence=config["output_cadence"],
        max_cfl=config["max_cfl"],
        gamma=gamma,
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

    plt.tight_layout()
    plt.savefig("sod_results.png", dpi=150)
    print("Results saved to sod_results.png")

    try:
        plt.show()
    except:
        pass
