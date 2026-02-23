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
    load_latest_snapshot,
    load_snapshot,
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

    # Create simulation config
    sim_config = {
        "reconstruction_fn": reconstruct_ppm,
        "flux_fn": hll_flux,
        "timestepper": "rk2",
        "conduction_fn": None,
        "bc_modes": sod_bcs(),
        "fixed_bcs": None,
        "user_bcs": None,
    }

    # Create state dictionary
    state = {
        "xcc": grid,
        "dx": grid[1] - grid[0],
        "Q": sod_ics(grid, gamma=gamma),
        "sources": [],
        "gamma": gamma,
        "time": 0.0,
        "snap_num": 0,
    }

    # Run simulation
    print("Running Sod shock tube test case...")
    snapshot_dir = Path(__file__).parent / "snapshots"
    n_iterations = run_sim(
        state,
        sim_config,
        max_time=config["max_time"],
        output_cadence=config["output_cadence"],
        max_cfl=config["max_cfl"],
    )

    print(f"Simulation complete. {n_iterations} iterations, snapshots saved to {snapshot_dir}")

    # Load the final snapshot from the first run before continuing
    print("\nTesting snapshot loading...")
    state_template = {
        "sources": [],
        "bc_modes": sod_bcs(),
        "fixed_bcs": None,
        "user_bcs": None,
    }
    try:
        final_state_1 = load_latest_snapshot(str(snapshot_dir), state_template=state_template)
        loaded_time = final_state_1["time"]
        loaded_snap_num = final_state_1.get("snap_num", -1)
        print(f"Successfully loaded snapshot at t={loaded_time:.3f}, snap_num={loaded_snap_num}")
    except FileNotFoundError as e:
        print(f"Failed to load snapshot: {e}")
        final_state_1 = state.copy()

    restart_state = load_snapshot(str(snapshot_dir / "snap_00001.nc"), state_template=state_template, decrement_snap_num=True)
    # Continue simulation from loaded snapshot
    print(f"\nContinuing simulation from t={restart_state['time']:.3f}...")
    n_iterations_2 = run_sim(
        restart_state,
        sim_config,
        max_time=config["max_time"],
        output_cadence=config["output_cadence"],
        max_cfl=config["max_cfl"],
    )

    print(f"Continued simulation complete. {n_iterations_2} iterations, snapshots saved to snapshots")

    # Load final snapshot from the continued run
    print("\nLoading final snapshot from continued run...")
    final_state_2 = load_latest_snapshot(str(snapshot_dir), state_template=state_template)

    final_time_1 = final_state_1["time"]
    final_time_2 = final_state_2["time"]

    # Convert to primitive variables
    w1 = cons_to_prim(final_state_1["Q"], gamma=gamma)
    w2 = cons_to_prim(final_state_2["Q"], gamma=gamma)

    # Extract interior points (excluding ghost cells)
    interior_slice = slice(NUM_GHOST, -NUM_GHOST)
    x_plot = grid[interior_slice]
    rho1 = w1[IRHO, interior_slice]
    v1 = w1[IVEL, interior_slice]
    p1 = w1[IPRE, interior_slice]

    rho2 = w2[IRHO, interior_slice]
    v2 = w2[IVEL, interior_slice]
    p2 = w2[IPRE, interior_slice]

    # Create plots comparing both runs
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].plot(x_plot, rho1, label=f"Run 1 (t={final_time_1:.3f})")
    axes[0].plot(x_plot, rho2, '--', label=f"Run 2 (t={final_time_2:.3f})")
    axes[0].set_ylabel("Density")
    axes[0].set_xlabel("x")
    axes[0].set_title("Density Comparison")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(x_plot, v1, label=f"Run 1 (t={final_time_1:.3f})")
    axes[1].plot(x_plot, v2, '--', label=f"Run 2 (t={final_time_2:.3f})")
    axes[1].set_ylabel("Velocity")
    axes[1].set_xlabel("x")
    axes[1].set_title("Velocity Comparison")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(x_plot, p1, label=f"Run 1 (t={final_time_1:.3f})")
    axes[2].plot(x_plot, p2, '--', label=f"Run 2 (t={final_time_2:.3f})")
    axes[2].set_ylabel("Pressure")
    axes[2].set_xlabel("x")
    axes[2].set_title("Pressure Comparison")
    axes[2].grid(True)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("sod_results.png", dpi=150)
    print("Results saved to sod_results.png")

    try:
        plt.show()
    except:
        pass

