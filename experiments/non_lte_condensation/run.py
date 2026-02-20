from functools import partial
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
    PwInterface,
    tracer_eos,
)
from h_model import H_6_cooling_collisions
import lightweaver as lw
import promweaver as pw
from lightweaver import rh_atoms

atomic_models_boundary = [
    H_6_cooling_collisions(),
    rh_atoms.C_atom(),
    rh_atoms.O_atom(),
    rh_atoms.Si_atom(),
    rh_atoms.Al_atom(),
    rh_atoms.CaII_atom(),
    rh_atoms.Fe_atom(),
    rh_atoms.He_atom(),
    rh_atoms.MgII_atom(),
    rh_atoms.N_atom(),
    rh_atoms.NaI_fine_atom(),
    rh_atoms.S_atom()
]
# active_atoms_boundary = ["H", "Mg", "Ca"]
active_atoms_boundary = ["H"]
atomic_models_condensation = [
    H_6_cooling_collisions()
]
active_atoms_condensation = ["H"]

config = dict(
    x_min = -2e6,
    x_max = 2e6,
    num_grid_points = 1000,
    gamma = 5/3,
    max_time = 0.1,
    output_cadence = 0.5,
    max_cfl = 0.9,
    base_pressure = 0.023,
    base_density = 1e-12,
    blob_density = 5e-11,
    blob_delta = 0.5e6,
)

PrdBoundary = False
LymanContTembri = True
BackgroundParams = dict(
    temperature=1e6,
    vlos=0.0,
    vturb=2e3,
    pressure=config["base_pressure"],
    nh_tot=config["base_density"] / (const.m_p.value * lw.DefaultAtomicAbundance.totalAbundance),
    ne=config["base_density"] / (const.m_p.value * lw.DefaultAtomicAbundance.totalAbundance),
)
ThresholdTemperature = 200e3

def construct_bc_table():
    print("Constructing BC Table...")
    boundary_ctx = pw.compute_falc_bc_ctx(
        active_atoms=active_atoms_boundary,
        atomic_models=atomic_models_boundary,
        prd=PrdBoundary,
        Nthreads=4,
        quiet=True,
    )
    bc_table = pw.tabulate_bc(boundary_ctx)
    if LymanContTembri:
        boundary_wavelengths = boundary_ctx.spect.wavelength
        mask = boundary_wavelengths < 91.2
        waves_to_compute = boundary_wavelengths[mask]
        lyman_rad = np.zeros_like(waves_to_compute)
        tembri = np.genfromtxt("tembri.dat")
        tembri_waves = tembri[:, 0] * 1e3
        brightness_temps = np.ascontiguousarray(tembri[:, 1])
        for i, w in enumerate(waves_to_compute):
            lyman_rad[i] = lw.planck(np.interp(w, tembri_waves, brightness_temps), w)
        # Is the continuum limb brightened? No
        bc_table["I"][mask, :]  = lyman_rad[:, None]
    print("Done Constructing BC Table")
    return bc_table

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
    bc_table = construct_bc_table()

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
        "eos": partial(tracer_eos, total_abund=None),
        # TODO(cmo): This is a bit of a hack
        "h_mass": const.m_p.value * lw.DefaultAtomicAbundance.totalAbundance
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
        q = prim_to_cons(w, gamma=gamma)
        state = dict(xcc=x, Q=q, gamma=gamma)
        pw_interface = PwInterface(
            state,
            sim_config,
            prom_bc=pw.TabulatedPromBcProvider(**bc_table),
            active_atoms=active_atoms_condensation,
            atomic_models=atomic_models_condensation,
            background_params=BackgroundParams,
            threshold_temperature=ThresholdTemperature,
            stat_eq=True,
            total_abund=None,
            num_rays=3,
            bc_type=pw.UniformJPromBc,
        )
        pw_interface.update_initial_density_profile(state, sim_config)
        pw_interface.set_initial_tracers(state, sim_config)
        pw_interface.update_tracers(state, sim_config)

        return q, state.get("tracers"), pw_interface

    q, tracers, pw_interface = condensation_ics(grid, gamma)

    # Create state dictionary
    state = {
        "xcc": grid,
        "dx": grid[1] - grid[0],
        "Q": q,
        "tracers": tracers,
        "fixed_bcs": None,
        "user_bcs": None,
        "sources": [
            pw_interface,
        ],
        "gamma": gamma,
        "bc_modes": [SYMMETRIC_BC, SYMMETRIC_BC],
    }

    # # Run simulation
    snaps = run_sim(
        state,
        sim_config,
        max_time=config["max_time"],
        output_cadence=config["output_cadence"],
        max_cfl=config["max_cfl"],
    )


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

