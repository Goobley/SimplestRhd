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
    implicit_thermal_conduction,
)
from setup import *


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
        "conduction_fn": implicit_thermal_conduction,
        "bc_modes": conduction_bcs(),
        "fixed_bcs": None,
        "user_bcs": [conduction_left_bc, conduction_right_bc],
        "run_hydro": False,
        "kappa0": config["kappa0"],
        "h_mass": config["h_mass"],
        "k_B": config["k_B"],
    }

    # Create state dictionary
    state = {
        "xcc": grid,
        "dx": grid[1] - grid[0],
        "Q": conduction_ics(grid, gamma=gamma),
        "sources": [],
        "gamma": gamma,
        "time": 0.0,
        "snap_num": 0,
        "y": config["y"],
    }

    # Run simulation
    snapshot_dir = Path(__file__).parent / "snapshots"
    n_iterations = run_sim(
        state,
        sim_config,
        max_time=config["max_time"],
        output_cadence=config["output_cadence"],
        max_cfl=config["max_cfl"],
    )
