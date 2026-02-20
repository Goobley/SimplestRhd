import numpy as np
import astropy.constants as const
import lightweaver as lw
import promweaver as pw

from .indices import (
    IRHO,
    IVEL,
    IPRE,
    IENE,
    NUM_GHOST,
)
from .eos import temperature_si, cons_to_prim

M_P = const.m_p.value

class PwInterface:
    def __init__(
            self,
            state,
            sim_config,
            prom_bc,
            active_atoms,
            atomic_models,
            background_params,
            threshold_temperature = 50e3,
            altitude=10e6,
            num_rays=10,
            bc_type=pw.ConePromBc,
            num_threads=8,
            growth_factor=1.6,
            shrink_factor=0.5,
            shrink_threshold=0.4,
            initial_conserve_pressure=True,
            stat_eq=True,
            quiet=False,
            total_abund=1.0,
        ):
        self.threshold_temperature = threshold_temperature
        self.background_params = background_params
        self.altitude = altitude
        self.active_atoms = active_atoms
        self.atomic_models = atomic_models
        self.num_threads = num_threads
        self.num_rays = num_rays
        self.bc_type = bc_type
        self.stat_eq = stat_eq
        self.quiet = quiet

        if total_abund is None:
            total_abund = lw.DefaultAtomicAbundance.totalAbundance
        self.total_abund = total_abund

        self.growth_factor = growth_factor
        self.shrink_factor = shrink_factor
        self.shrink_threshold = shrink_threshold

        self.prom_bc = prom_bc
        self.context_length = 0
        mask = self.mask_region(state, sim_config)
        self.create_new_context(np.sum(mask) * growth_factor, conserve_pressure=initial_conserve_pressure)
        self.update_atmos(state, sim_config)
        self.model.iterate_se()
        if initial_conserve_pressure:
            self.model.conserve_pressure = False

        self.num_tracers = 1 + sum(self.model.eq_pops[a].shape[0] for a in self.active_atoms)

        # TODO(cmo): Add the extra wavelengths for the losses to be correct.


    def create_new_context(self, context_length, conserve_pressure=False):
        context_length = int(context_length)
        print(f"Changing context length: {self.context_length}->{context_length}")
        self.context_length = context_length
        z_full = np.ascontiguousarray(np.arange(context_length, dtype=np.float64)[::-1])
        temperature_full = np.zeros(context_length)
        vlos_full = np.zeros(context_length)
        vturb_full = np.zeros(context_length)
        pressure_full = np.zeros(context_length)
        nh_tot_full = np.zeros(context_length)
        ne_full = np.zeros(context_length)

        temperature_full[:] = self.background_params["temperature"]
        vlos_full[:] = self.background_params["vlos"]
        vturb_full[:] = self.background_params["vturb"]
        pressure_full[:] = self.background_params["pressure"]
        nh_tot_full[:] = self.background_params["nh_tot"]
        ne_full[:] = self.background_params["ne"]

        self.model = pw.StratifiedPromModel(
            projection="prominence",
            z=z_full,
            temperature=temperature_full,
            vlos=vlos_full,
            vturb=vturb_full,
            nh_tot=nh_tot_full,
            ne=ne_full,
            altitude=self.altitude,
            active_atoms=self.active_atoms,
            atomic_models=self.atomic_models,
            bc_provider=self.prom_bc,
            Nthreads=self.num_threads,
            Nrays=self.num_rays,
            BcType=self.bc_type,
            ctx_kwargs=dict(formalSolver="piecewise_linear_1d"),
            conserve_charge=True,
            conserve_pressure=conserve_pressure,
        )
        self.model.ctx.depthData.fill = True

    def update_atmos(self, state, sim_config):
        """Update the atmosphere from the state"""
        mask = self.mask_region(state, sim_config)
        mask_count = np.sum(mask)
        if mask_count > self.context_length:
            self.create_new_context(
                min(
                    self.growth_factor * self.context_length,
                    state['xcc'].shape[0]
                )
            )
        elif mask_count < self.shrink_threshold * self.context_length:
            self.create_new_context(self.shrink_factor * self.context_length)

        y = state.get("y", 1.0)
        h_mass = sim_config.get('h_mass', M_P)
        q = state["Q"]
        w = cons_to_prim(q, gamma=state["gamma"])
        nh_tot = q[IRHO] / h_mass
        ne = nh_tot * y
        # TODO(cmo): Look at pushing the temperature through the state dict
        temperature = temperature_si(w[IPRE], nh_tot, y, total_abund=self.total_abund)

        z = state["xcc"][mask]
        temperature = temperature[mask]
        vlos = w[IVEL, mask]
        pressure = w[IPRE, mask]
        nh_tot = nh_tot[mask]
        ne = ne[mask]

        atmos = self.model.atmos
        z_full = atmos.z
        z_full[:mask_count] = z[::-1]
        dx = state["xcc"][1] - state["xcc"][0]
        for i in range(mask_count, self.context_length):
            z_full[i] = z_full[i-1] - dx

        temperature_full = atmos.temperature
        vlos_full = atmos.vlos
        nh_tot_full = atmos.nHTot
        ne_full = atmos.ne
        pressure_full = self.model.pressure

        temperature_full[:mask_count] = temperature[::-1]
        vlos_full[:mask_count] = vlos[::-1]
        nh_tot_full[:mask_count] = nh_tot[::-1]
        ne_full[:mask_count] = ne[::-1]
        pressure_full[:mask_count] = pressure[::-1]
        self.model.ctx.update_deps()

    def solve_rt(self, dt=None):
        if self.stat_eq:
            self.model.iterate_se(quiet=self.quiet, Nscatter=1, printInterval=-1)
        else:
            prev_state = None
            for i in range(100):
                ctx = self.model.ctx
                ctx.formal_sol_gamma_matrices()
                pops_update, prev_state = ctx.time_dep_update(dt, prev_state)
                nr_update = ctx.nr_pops_update(timeDependentDdata=dict(dt=dt, nPrev=prev_state))
                if not self.quiet:
                    print(f"-- Iteration {i}")
                    print(nr_update.compact_representation())
                    print('-' * 80)
                # TODO(cmo): Convergence

    def mask_region(self, state, sim_config):
        y = state.get("y", 1.0)
        h_mass = sim_config.get('h_mass', M_P)
        q = state["Q"]
        w = cons_to_prim(q, gamma=state["gamma"])
        nh_tot = q[IRHO] / h_mass
        temperature = temperature_si(w[IPRE], nh_tot, y, total_abund=self.total_abund)

        mask = temperature > self.threshold_temperature
        # NOTE(cmo): Shrink high temperature "islands" by one on each side to preserve gradients
        mask[1:] &= mask[:-1]
        mask[:-1] &= mask[1:]
        mask = ~mask
        return mask

    def set_initial_tracers(self, state, sim_config):
        """Fill the tracers array everywhere with LTE -- participating regions will be overwritten later"""
        y = state.get("y", 1.0)
        h_mass = sim_config.get('h_mass', M_P)
        q = state["Q"]
        w = cons_to_prim(q, gamma=state["gamma"])
        nh_tot = q[IRHO] / h_mass
        ne = nh_tot * y
        temperature = temperature_si(w[IPRE], nh_tot, y, total_abund=self.total_abund)

        tracers = state.get("tracers", np.zeros((self.num_tracers, state["xcc"].shape[0])))
        tracers[0, :] = ne
        start_idx = 1
        for a in self.active_atoms:
            pops = lw.lte_pops(
                self.model.rad_set[a],
                temperature,
                ne,
                nh_tot * lw.DefaultAtomicAbundance[a]
            )
            tracers[start_idx:start_idx + pops.shape[0], :] = pops
            start_idx += pops.shape[0]
        state["tracers"] = tracers

    def update_initial_density_profile(self, state, sim_config):
        h_mass = sim_config.get('h_mass', M_P)
        mask = self.mask_region(state, sim_config)
        mask_count = np.sum(mask)
        q = state["Q"]
        q[IRHO, mask] = self.model.atmos.nHTot[:mask_count][::-1] * h_mass

    def update_tracers(self, state, sim_config):
        tracers = state.get("tracers", np.zeros((self.num_tracers, state["xcc"].shape[0])))
        mask = self.mask_region(state, sim_config)
        mask_count = np.sum(mask)
        tracers[0, mask] = self.model.atmos.ne[:mask_count][::-1]
        start_idx = 1
        for a in self.active_atoms:
            pops = self.model.eq_pops[a]
            tracers[start_idx:start_idx + pops.shape[0], mask] = pops[:, :mask_count][:, ::-1]
            start_idx += pops.shape[0]
        state["tracers"] = tracers
        ratio = tracers[0] / tracers[1:].sum(axis=0)
        print(f"On update: {ratio.min()}, {ratio.max()}")

    def fill_from_tracers(self, state, sim_config):
        tracers = state["tracers"]
        ratio = tracers[0] / tracers[1:].sum(axis=0)
        print(f"On fill: {ratio.min()}, {ratio.max()}")
        mask = self.mask_region(state, sim_config)
        mask_count = np.sum(mask)
        # TODO(cmo): Check consistency between the two n_e's. Our EOS needs to
        # be responsible for aligning them
        ne = tracers[0, mask]
        self.model.atmos.ne[:mask_count] = ne[::-1]
        start_idx = 1
        for a in self.active_atoms:
            num_level = self.model.eq_pops[a].shape[0]
            pops = tracers[start_idx:start_idx+num_level, mask]
            self.model.eq_pops[a][:, :mask_count] = pops[:, ::-1]
            start_idx += num_level

        self.model.atmos.nHTot[:] = self.model.eq_pops["H"].sum(axis=0)

    def __call__(self, state, sim_config, sources, ts):
        self.update_atmos(state, sim_config)
        if "tracers" in state:
            self.fill_from_tracers(state, sim_config)
        self.solve_rt(dt=ts.dt)
        if "tracers" in state:
            self.update_tracers(state, sim_config)
        print("RT update complete")
