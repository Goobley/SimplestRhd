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
active_atoms_condensation = ["H", "Ca", "Mg"]
# active_atoms_condensation = ["H"]

config = dict(
    # x_min = -2.2e6,
    # x_max = 2.2e6,
    # num_grid_points = 1500,
    x_min = -20e6,
    x_max = 20e6,
    num_grid_points = 10_000,
    gamma = 5/3,
    max_time = 2000.0,
    output_cadence = 0.5,
    max_cfl = 0.3,
    base_pressure = 0.023,
    base_density = 1.7e-12,
    # base_density = 3.4e-12,
    blob_density = 4e-11,
    blob_delta = 0.7e6,
    # dip_depth = 0.725e6 * 2,
    dip_depth = 0.00e6,
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
SOLAR_G = 2.74e2 # m/s2

class CosineProjectedGravity:
    def __init__(
            self,
            grid,
            max_alt,
            solar_g,
            cos_start = 0.75 * np.pi,
            cos_end = 1.25 * np.pi,
    ):
        y = np.cos(np.linspace(cos_start, cos_end, grid.shape[0]))
        y -= np.min(y)
        y /= np.max(y)
        y *= max_alt
        self.y = y

        x = np.sqrt(grid**2 - y**2)
        self.x = x

        # NOTE(cmo): Slope in each cell
        dy = np.zeros_like(grid)
        dx = np.zeros_like(grid)
        dy[1:-1] = (y[2:] - y[:-2])
        dx[1:-1] = (x[2:] - x[:-2])
        dy[0] = (y[1] - y[0])
        dx[0] = (x[1] - x[0])
        dy[-1] = (y[-1] - y[-2])
        dx[-1] = (x[-1] - x[-2])

        # NOTE(cmo): projection angle
        alpha = np.atan2(dy, dx)
        self.alpha = alpha
        self.projected_grav = solar_g * np.sin(alpha)

    def __call__(
            self,
            state,
            sim_config,
            sources,
            ts
    ):
        Q = state["Q"]
        sources[IMOM, :] += Q[IRHO, :] * self.projected_grav
        sources[IENE, :] += Q[IMOM, :] * self.projected_grav



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
        "reconstruction_fn": reconstruct_plm,
        # TODO(cmo): Look at some artificial (shock) viscosity to smear the TR?
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
        "conduction_suppression_Tc": 2e5,
        "conduction_suppression_Tlow": 25e4,
        "htc_order": 1,
        "htc_hyperdiffusion": 1e-2,
        "htc_use_riemann_flux": True,
    }

    def condensation_ics(x, gamma, grav):
        base_pressure = config["base_pressure"]
        base_density = config["base_density"]
        blob_density = config["blob_density"]
        blob_delta = config["blob_delta"]
        mass_per_h = sim_config['avg_mass']
        h_mass = sim_config['h_mass']
        n_h_base = base_density / (mass_per_h * h_mass)
        y = 1.0
        total_abund = sim_config['total_abund']
        temp_base = temperature_si(base_pressure, n_h_base, total_abund=total_abund, y=y)
        k_B = const.k_B.value

        if x.shape[0] % 2 != 0:
            raise ValueError("length of x must be even")

        # NOTE(cmo): Integrate HSE over 1 side of the dip
        dx = x[1] - x[0]
        rho = np.zeros_like(x)
        pressure = np.zeros_like(x)
        i_start = x.shape[0] // 2
        rho[i_start] = base_density
        pressure[i_start] = base_pressure
        for i in range(i_start + 1, x.shape[0]):
            dP_dx_base = rho[i - 1] * grav.projected_grav[i - 1]
            P_half = pressure[i - 1] + dP_dx_base * 0.5 * dx
            T_half = temp_base
            rho_half = (mass_per_h * h_mass * P_half) / ((total_abund + y) * k_B * T_half)
            dP_dx_mid = rho_half * 0.5 * (grav.projected_grav[i-1] + grav.projected_grav[i])

            pressure[i] = pressure[i - 1] + dP_dx_mid * dx
            rho[i] = (mass_per_h * h_mass * pressure[i]) / ((total_abund + y) * k_B * temp_base)
            # for iter in range(100):
            #     old_pressure = pressure[i]
            #     if i == i_start + 1:
            #         pressure[i] = pressure[i-1] + (
            #             (0.5 * (grav.projected_grav[i-1] + grav.projected_grav[i]))
            #             * dx
            #             * (0.5 * (rho[i] + rho[i-1]))
            #         )
            #     else:
            #         pressure[i] = pressure[i-1] + (
            #             1.0 / 12.0
            #             * (0.5 * (grav.projected_grav[i-1] + grav.projected_grav[i]))
            #             * dx
            #             * (5.0 * rho[i] + 8 * rho[i-1] - rho[i-2])
            #         )
            #     if (np.abs(1.0 - pressure[i] / old_pressure) < 1e-5):
            #         break
            #     rho[i] = (mass_per_h * h_mass * pressure[i]) / ((total_abund + y) * k_B * temp_base)
            # else:
            #     print(f"No converge @ {i}")

        rho[:i_start] = rho[i_start:][::-1]
        pressure[:i_start] = pressure[i_start:][::-1]

        w = np.stack([
            rho,
            np.zeros_like(x),
            pressure,
            np.zeros_like(x),
        ])

        w[IRHO, :] += blob_density * np.exp(-x**2 / blob_delta**2)
        # NOTE(cmo): Should be approximately thermally balanced
        # Don't ask questions you don't want answers to.
        # w[IRHO, :] += 1.818e-11 * 10**(-0.14078 * np.log10(np.abs(x))) - config['base_density']
        # w[IRHO, :] += 4e-11 * 10**(-0.3 * np.log10(np.abs(x))) - config['base_density']


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
            stat_eq=False,
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
        )
        pw_interface.update_initial_density_profile(state, sim_config)
        pw_interface.set_initial_tracers(state, sim_config)
        pw_interface.update_tracers(state, sim_config)
        tracer_eos(state, sim_config, evaluate_initial_ion_e=True)
        return state, pw_interface

        # lte_eos(state, sim_config, temp_err_bound=1e-7, find_initial_ion_e=True)
        # return state

    grav = CosineProjectedGravity(
        grid,
        config["dip_depth"],
        -SOLAR_G,
        0.5 * np.pi,
        1.5 * np.pi
    )
    if (config['dip_depth'] == 0.0):
        grav.projected_grav[...] = 0.0
    initial_state, pw_interface = condensation_ics(grid, gamma, grav)

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
            # grav,
            TownsendThinLoss('DM', min_temperature=24e3),
            hyperbolic_thermal_conduction,
        ],
        "split_sources": [
            pw_interface,
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


    # # Convert to primitive variables
    # w = cons_to_prim(state["Q"], gamma=gamma)
    # final_time = state["time"]

    # # Extract interior points (excluding ghost cells)
    # interior_slice = slice(NUM_GHOST, -NUM_GHOST)
    # x_plot = grid[interior_slice]
    # rho = w[IRHO, interior_slice]
    # v = w[IVEL, interior_slice]
    # p = w[IPRE, interior_slice]

    # # Create plots
    # fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # axes[0].plot(x_plot, rho)
    # axes[0].set_ylabel("Density")
    # axes[0].set_xlabel("x")
    # axes[0].set_title(f"t = {final_time:.3f}")
    # axes[0].grid(True)

    # axes[1].plot(x_plot, v)
    # axes[1].set_ylabel("Velocity")
    # axes[1].set_xlabel("x")
    # axes[1].grid(True)

    # axes[2].plot(x_plot, p)
    # axes[2].set_ylabel("Pressure")
    # axes[2].set_xlabel("x")
    # axes[2].grid(True)

