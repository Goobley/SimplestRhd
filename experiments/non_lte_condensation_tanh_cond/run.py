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
    IMOM,
    IVEL,
    IPRE,
    IIONE,
    NUM_GHOST,
    reconstruct_plm,
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
    SpongeLayer,
    implicit_thermal_conduction,
    hyperbolic_thermal_conduction,
    sts_thermal_conduction,
    load_latest_snapshot,
    load_snapshot,
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
active_atoms_boundary = ["H", "Mg", "Ca"]
# active_atoms_boundary = ["H"]
atomic_models_condensation = [
    H_6_cooling_collisions(),
    rh_atoms.CaII_atom(),
    rh_atoms.MgII_atom(),
]
# active_atoms_condensation = ["H", "Ca", "Mg"]
active_atoms_condensation = ["H"]

config = dict(
    # x_min = -2.2e6,
    # x_max = 2.2e6,
    # num_grid_points = 1500,
    x_min = -5e6,
    x_max = 5e6,
    num_grid_points = 2_500,
    gamma = 5/3,
    max_time = 2000.0,
    output_cadence = 0.5,
    max_cfl = 0.3,
    base_pressure = 0.023,
    # base_density = 1.7e-12,
    base_temperature = 1e6,
    base_density = 3.4e-12,
    # blob_density = 4e-11,
    # blob_delta = 0.7e6,
    condensation_temperature = 100e3,
    condensation_width = 3e6,
    transition_region_width = 1.5e4,
    stat_eq = False,
    fdiv = True,
    use_cma = True,
    suppress_conduction = True
)

PrdBoundary = False
LymanContTembri = True
BackgroundParams = dict(
    temperature=2e6,
    vlos=0.0,
    vturb=2e3,
    pressure=config["base_pressure"],
    nh_tot=config["base_density"] / (const.m_p.value * lw.DefaultAtomicAbundance.massPerH) * 0.3,
    ne=config["base_density"] / (const.m_p.value * lw.DefaultAtomicAbundance.massPerH) * 0.3,
)
ThresholdTemperature = 120e3

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
        # "conduction_fn": implicit_thermal_conduction,
        "conduction_fn": None,
        "strang_split_conduction": False,
        "saturate_conductive_flux": True,
        "eos": tracer_eos,
        # "eos": lte_eos,
        "h_mass": const.m_p.value,
        "avg_mass": lw.DefaultAtomicAbundance.massPerH,
        "total_abund": lw.DefaultAtomicAbundance.totalAbundance,
        "bc_modes": [SYMMETRIC_BC, SYMMETRIC_BC],
        "fixed_bcs": None,
        "user_bcs": None,
        "min_temperature": 1e3,
        # "conduction_suppression_Tc": 2e5,
        # "conduction_suppression_Tlow": 25e4,
        # "htc_order": 1,
        # "htc_hyperdiffusion": 1e-2,
        # "htc_use_riemann_flux": True,
        "tracer_cma_flatten": True,
    }
    if config["suppress_conduction"]:
        sim_config["conduction_suppression_Tc"] = 2.5e5
        sim_config["conduction_suppression_Tlow"] = 35e4

    def condensation_ics(x, gamma):
        if x.shape[0] % 2 != 0:
            raise ValueError("length of x must be even")

        base_pressure = config["base_pressure"]
        mass_per_h = sim_config['avg_mass']
        h_mass = sim_config['h_mass']
        y = 1.0
        total_abund = sim_config['total_abund']
        k_B = const.k_B.value

        tr_pos = config["condensation_width"] * 0.5
        tr_width = config["transition_region_width"]
        T_cor = config["base_temperature"]
        T_cc = config["condensation_temperature"]

        temp_arg = (np.abs(x) - tr_pos) / tr_width
        temperature_profile = T_cc + 0.5 * (T_cor - T_cc) * (np.tanh(temp_arg) + 1.0)

        rho = base_pressure / ((total_abund + y) * k_B * temperature_profile) * mass_per_h * h_mass
        pressure = np.ones_like(rho) * base_pressure

        w = np.stack([
            rho,
            np.zeros_like(x),
            pressure,
            np.zeros_like(x),
        ])


        q = prim_to_cons(w, gamma=gamma)
        state = dict(xcc=x, dx=x[1] - x[0], Q=q, gamma=gamma)
        pw_interface = PwInterface(
            state,
            sim_config,
            prom_bc=pw.TabulatedPromBcProvider(**bc_table),
            active_atoms=active_atoms_condensation,
            atomic_models=atomic_models_condensation,
            background_params=BackgroundParams,
            threshold_temperature=ThresholdTemperature,
            stat_eq=config["stat_eq"],
            num_rays=5,
            bc_type=pw.UniformJPromBc,
            quiet=True,
            buffer_cells=3,
            shrink_threshold=0.85,
            shrink_factor=0.9,
            growth_factor=1.15,
            evaluate_radiative_losses=True,
            pop_tol=8e-4,
            num_threads=12,
            use_lw_fdiv=config["fdiv"],
        )
        pw_interface.update_initial_density_profile(state, sim_config)
        pw_interface.set_initial_tracers(state, sim_config, setup_cma=config['use_cma'])
        pw_interface.update_tracers(state, sim_config)
        tracer_eos(state, sim_config, evaluate_initial_ion_e=True)
        return state, pw_interface


    initial_state, pw_interface = condensation_ics(grid, gamma)

    # # Create state dictionary
    state = initial_state | {
        "xcc": grid,
        "dx": grid[1] - grid[0],
        "sources": [
            SpongeLayer(
                config["x_min"] + 0.5e6,
                config["x_max"] - 0.5e6,
                0.03,
                6e-6, # Ramp damping to exp(3) over 500 km
                q0_full=np.copy(initial_state['Q'])
            ),
            TownsendThinLoss('DM', min_temperature=37e3),
            # hyperbolic_thermal_conduction,
            sts_thermal_conduction,
        ],
        "split_sources": [
            pw_interface,
        ],
        "gamma": gamma,
        "time": 0.0,
    }

    snapshot_dir = f"snapshots_37kKThinCutoff_{"CMA_" if config["use_cma"] else ""}{config["base_pressure"]:.04f}Pa_{"se" if config["stat_eq"] else "td"}{"_fdiv" if config["fdiv"] else ""}{"_sat" if sim_config["saturate_conductive_flux"] else ""}{"_sup" if config["suppress_conduction"] else ""}"
    # do_restart = True
    # snap_num = 100
    # if do_restart:
    #     state = load_snapshot(snapshot_dir + f"/snap_{snap_num:05d}.nc", state, decrement_snap_num=True)

    # Run simulation
    num_iter = run_sim(
        state,
        sim_config,
        max_time=config["max_time"],
        output_cadence=config["output_cadence"],
        max_cfl=config["max_cfl"],
        snapshot_dir=snapshot_dir,
    )

