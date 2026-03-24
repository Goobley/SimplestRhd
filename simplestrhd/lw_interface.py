import numpy as np
import astropy.constants as const
import astropy.units as u
import lightweaver as lw
import lightweaver.LwCompiled as lw_cpp
import promweaver as pw
from pathlib import Path

from .indices import (
    IRHO,
    IVEL,
    IPRE,
    IENE,
    IIONE,
    NUM_GHOST,
)
from .eos import temperature_si, cons_to_prim

M_P = const.m_p.value

FS_LIB = Path(__file__).absolute().parent.parent / "fvm_formalsolver" / "build" / "libpragmatic_fvm_fs_1st_1d.so"
lw_cpp.FormalSolvers.load_fs_from_path(str(FS_LIB))

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
            evaluate_radiative_losses=True,
            add_edge_wavelengths=None,
            add_vertical_ray=True,
            buffer_cells=1,
            dump_full_rad_loss=False,
            pop_tol=1e-3,
            min_size=30,
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
        self.buffer_cells = buffer_cells
        self.dump_full_rad_loss = dump_full_rad_loss
        self.pop_tol = pop_tol
        self.min_size = min_size

        total_abund = sim_config.get("total_abund", 1.0)
        if total_abund is None:
            total_abund = lw.DefaultAtomicAbundance.totalAbundance
        self.total_abund = total_abund
        mass_per_h = sim_config.get("avg_mass", 1.0)
        self.mass_per_h = mass_per_h

        self.growth_factor = growth_factor
        self.shrink_factor = shrink_factor
        self.shrink_threshold = shrink_threshold

        if add_edge_wavelengths is None:
            add_edge_wavelengths = evaluate_radiative_losses
        self.evaluate_radiative_losses = evaluate_radiative_losses
        self.extra_wavelengths = None
        if add_edge_wavelengths:
            self.extra_wavelengths = self.compute_extra_wavelengths()
        self.add_vertical_ray = add_vertical_ray

        self.prom_bc = prom_bc
        self.context_length = 0
        mask = self.mask_region(state, sim_config)
        self.create_new_context((np.sum(mask) + 2 * self.buffer_cells) * growth_factor, conserve_pressure=initial_conserve_pressure)
        self.update_atmos(state, sim_config)
        self.model.iterate_se(popsTol=self.pop_tol)
        if initial_conserve_pressure:
            self.model.conserve_pressure = False

        self.num_tracers = 1 + sum(self.model.eq_pops[a].shape[0] for a in self.active_atoms)
        state['wavelength'] = self.model.ctx.spect.wavelength.copy()
        state['intensity'] = self.model.ctx.spect.I[:, -1][:, None].copy()

    def compute_extra_wavelengths(self):
        rad_set = lw.RadiativeSet(atoms=self.atomic_models)
        rad_set.set_active(*self.active_atoms)

        extra_waves = []
        for atom in rad_set.activeAtoms:
            for l in atom.lines:
                w = l.wavelength()
                extra_waves.append(w[0] - (w[1] - w[0]))
                extra_waves.append(w[-1] + (w[-1] - w[-2]))

            if atom.element == lw.PeriodicTable["H"]:
                # NOTE(cmo): handle longest H continuum
                edge = atom.continua[-1].lambdaEdge
                end = atom.lines[-1].lambda0
                extra_waves += np.linspace(edge, end, 7)[1:-1].tolist()
        extra_waves = np.array(sorted(extra_waves))
        return extra_waves


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
            # ctx_kwargs=dict(formalSolver="piecewise_linear_1d"),
            # ctx_kwargs=dict(formalSolver="piecewise_besser_1d"),
            ctx_kwargs=dict(formalSolver="pragmatic_1_fvm_1d"),
            conserve_charge=True,
            conserve_pressure=conserve_pressure,
            extra_wavelengths=self.extra_wavelengths,
            add_vertical_ray=self.add_vertical_ray,
        )
        self.model.ctx.depthData.fill = True
        self.hz_edges = (lw.compute_wavelength_edges(self.model.ctx) << u.nm).to(u.Hz, equivalencies=u.spectral()).value
        self.hz_bins = np.abs(self.hz_edges[1:] - self.hz_edges[:-1])

    def update_atmos(self, state, sim_config):
        """Update the atmosphere from the state"""
        mask = self.mask_region(state, sim_config)
        mask_count = np.sum(mask)
        if mask_count + 2 * self.buffer_cells > self.context_length:
            self.create_new_context(
                min(
                    max(
                        self.growth_factor * self.context_length,
                        mask_count + 2 * self.buffer_cells,
                        self.min_size,
                    ),
                    state['xcc'].shape[0]
                )
            )
        elif mask_count + 2 * self.buffer_cells < self.shrink_threshold * self.context_length:
            if self.context_length > self.min_size:
                self.create_new_context(
                    max(
                        self.shrink_factor * self.context_length,
                        self.min_size,
                    )
                )

        y = state.get("y", 1.0)
        h_mass = sim_config.get('h_mass', M_P)
        q = state["Q"]
        w = cons_to_prim(q, gamma=state["gamma"])
        nh_tot = q[IRHO] / (h_mass * self.mass_per_h)
        ne = nh_tot * y
        # TODO(cmo): Look at pushing the temperature through the state dict
        temperature = temperature_si(
            w[IPRE],
            nh_tot,
            y,
            total_abund=self.total_abund,
        )

        z = state["xcc"][mask]
        temperature = temperature[mask]
        vlos = w[IVEL, mask]
        pressure = w[IPRE, mask]
        nh_tot = nh_tot[mask]
        ne = ne[mask]

        bc = self.buffer_cells
        atmos = self.model.atmos
        z_full = atmos.z
        z_full[bc:mask_count+bc] = z[::-1]
        dx = state["xcc"][1] - state["xcc"][0]
        for i in range(bc - 1, -1, -1):
            z_full[i] = z_full[i+1] + dx
        for i in range(mask_count + bc, self.context_length):
            z_full[i] = z_full[i-1] - dx

        temperature_full = atmos.temperature
        vlos_full = atmos.vlos
        nh_tot_full = atmos.nHTot
        ne_full = atmos.ne
        pressure_full = self.model.pressure

        temperature_full[:] = self.background_params["temperature"]
        vlos_full[:] = self.background_params["vlos"]
        pressure_full[:] = self.background_params["pressure"]
        nh_tot_full[:] = self.background_params["nh_tot"]
        ne_full[:] = self.background_params["ne"]

        temperature_full[bc:mask_count+bc] = temperature[::-1]
        vlos_full[bc:mask_count+bc] = vlos[::-1]
        nh_tot_full[bc:mask_count+bc] = nh_tot[::-1]
        ne_full[bc:mask_count+bc] = ne[::-1]
        pressure_full[bc:mask_count+bc] = pressure[::-1]

        temperature_full[mask_count+bc:] = self.background_params["temperature"]
        vlos_full[mask_count+bc:] = self.background_params["vlos"]
        pressure_full[mask_count+bc:] = self.background_params["pressure"]
        nh_tot_full[mask_count+bc:] = self.background_params["nh_tot"]
        ne_full[mask_count+bc:] = self.background_params["ne"]

        # NOTE(cmo): Update the outer regions to use LTE populations (remove
        # time-dependence due to reshuffling of points around context for those
        # above threshold temperature).
        for a in self.active_atoms:
            pops = lw.lte_pops(
                self.model.rad_set[a],
                temperature_full,
                ne_full,
                nh_tot_full * lw.DefaultAtomicAbundance[a]
            )
            self.model.eq_pops[a][:, :bc] = pops[:, :bc]
            self.model.eq_pops[a][:, mask_count+bc:] = pops[:, mask_count+bc:]
        self.model.ctx.update_deps()

    def solve_rt(self, dt=None):
        if self.stat_eq or dt is None:
            self.model.iterate_se(quiet=self.quiet, Nscatter=1, popsTol=self.pop_tol)
        else:
            prev_state = None
            for i in range(2000):
                ctx = self.model.ctx
                ctx.formal_sol_gamma_matrices(lambdaIterate=i<1)
                pops_update, prev_state = ctx.time_dep_update(dt, prev_state)
                nr_update = ctx.nr_post_update(timeDependentData=dict(dt=dt, nPrev=prev_state))
                if not self.quiet and i > 50:
                    print(nr_update.compact_representation())
                    print('-' * 80)
                # NOTE(cmo): Only need to check nr_update as changes have not
                # been "committed" to the Ng state since before the start of the
                # time-dep update
                if i > 0 and nr_update.dPopsMax < self.pop_tol:
                    if not self.quiet and i > 50:
                        print('-- Done')
                    break
            else:
                breakpoint()

    def compute_rad_loss(self):
        ctx = self.model.ctx
        chi_tot = ctx.depthData.chi
        Sfn = (ctx.depthData.eta + (ctx.background.sca * ctx.spect.J)[:, None, None, :]) / chi_tot
        I_depth = ctx.depthData.I
        full_rad_loss_dir = (chi_tot * (Sfn - I_depth))
        # NOTE(cmo): flatten over mu and up/down
        full_rad_loss_dir = full_rad_loss_dir.reshape(full_rad_loss_dir.shape[0], -1, full_rad_loss_dir.shape[3])

        atmos = self.model.atmos
        wmu_stack = np.zeros(2 * atmos.wmu.shape[0])
        wmu_stack[0::2] = 0.5 * 4.0 * np.pi * atmos.wmu
        wmu_stack[1::2] = 0.5 * 4.0 * np.pi * atmos.wmu
        full_rad_loss = full_rad_loss_dir.transpose(0, 2, 1) @ wmu_stack
        full_rad_loss_bins = full_rad_loss * self.hz_bins[:, None]
        return full_rad_loss_bins

    def mask_region(self, state, sim_config):
        y = state.get("y", 1.0)
        h_mass = sim_config.get('h_mass', M_P)
        q = state["Q"]
        w = cons_to_prim(q, gamma=state["gamma"])
        nh_tot = q[IRHO] / (h_mass * self.mass_per_h)
        temperature = temperature_si(
            w[IPRE],
            nh_tot,
            y,
            total_abund=self.total_abund,
        )

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
        nh_tot = q[IRHO] / (h_mass * self.mass_per_h)
        ne = nh_tot * y
        temperature = temperature_si(
            w[IPRE],
            nh_tot,
            y,
            total_abund=self.total_abund,
        )

        tracers = state.get("tracers", np.zeros((self.num_tracers, state["xcc"].shape[0])))
        tracer_energy = state.get("tracer_energy", np.zeros(self.num_tracers))
        tracer_is_h = state.get("tracer_is_h", np.zeros(self.num_tracers, dtype=bool))
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
            for l in range(pops.shape[0]):
                tracer_energy[start_idx + l] = self.model.rad_set[a].levels[l].E_SI
            tracer_is_h[start_idx:start_idx + pops.shape[0]] = True
            start_idx += pops.shape[0]
        state["tracers"] = tracers
        state["tracer_energy"] = tracer_energy
        state["tracer_is_h"] = tracer_is_h

    def update_initial_density_profile(self, state, sim_config):
        h_mass = sim_config.get('h_mass', M_P)
        mask = self.mask_region(state, sim_config)
        mask_count = np.sum(mask)
        bc = self.buffer_cells
        q = state["Q"]
        q[IRHO, mask] = self.model.atmos.nHTot[bc:mask_count+bc][::-1] * (h_mass * self.mass_per_h)

    def update_tracers(self, state, sim_config):
        tracers = state.get("tracers", np.zeros((self.num_tracers, state["xcc"].shape[0])))
        mask = self.mask_region(state, sim_config)
        mask_count = np.sum(mask)
        bc = self.buffer_cells
        tracers[0, mask] = self.model.atmos.ne[bc:mask_count+bc][::-1]
        start_idx = 1
        for a in self.active_atoms:
            pops = self.model.eq_pops[a]
            tracers[start_idx:start_idx + pops.shape[0], mask] = pops[:, bc:mask_count+bc][:, ::-1]
            start_idx += pops.shape[0]
        state["tracers"] = tracers
        h_levels = self.model.eq_pops["H"].shape[0]
        y = tracers[0] / tracers[1:h_levels+1].sum(axis=0)
        state["y"] = y

    def fill_from_tracers(self, state, sim_config):
        tracers = state["tracers"]
        mask = self.mask_region(state, sim_config)
        mask_count = np.sum(mask)
        bc = self.buffer_cells
        # TODO(cmo): Check consistency between the two n_e's. Our EOS needs to
        # be responsible for aligning them
        ne = tracers[0, mask]
        self.model.atmos.ne[bc:mask_count+bc] = ne[::-1]
        start_idx = 1
        for a in self.active_atoms:
            num_level = self.model.eq_pops[a].shape[0]
            pops = tracers[start_idx:start_idx+num_level, mask]
            self.model.eq_pops[a][:, bc:mask_count+bc] = pops[:, ::-1]
            start_idx += num_level

        nh_tot_from_mass = self.model.atmos.nHTot
        nh_tot_from_n = self.model.eq_pops["H"].sum(axis=0)
        ratio = nh_tot_from_mass / nh_tot_from_n
        self.model.eq_pops["H"][...] *= ratio
        self.model.atmos.ne[...] *= ratio

    def __call__(self, state, sim_config, sources, ts):
        self.update_atmos(state, sim_config)
        if "tracers" in state:
            self.fill_from_tracers(state, sim_config)

        mask = self.mask_region(state, sim_config)
        mask_count = np.sum(mask)
        bc = self.buffer_cells
        start_of_timestep_losses = True
        if self.evaluate_radiative_losses and start_of_timestep_losses:
            for i in range(2):
                self.model.ctx.formal_sol_gamma_matrices()

            rad_loss = self.compute_rad_loss()
            gain_minus_loss_start = -np.sum(rad_loss[:, bc:mask_count+bc], axis=0)[::-1]

        self.solve_rt(dt=ts.dt)
        state['intensity'] = self.model.ctx.spect.I[:, -1][:, None].copy()
        if "tracers" in state:
            self.update_tracers(state, sim_config)
        if self.evaluate_radiative_losses:
            # NOTE(cmo): Remove mask evaluation here, because update_tracers
            # updates y and could mess with it. Otherwise could move the y
            # update to the EOS.
            rad_loss = self.compute_rad_loss()
            gain_minus_loss = -np.sum(rad_loss[:, bc:mask_count+bc], axis=0)[::-1]
            if start_of_timestep_losses:
                theta = 0.55
                gain_minus_loss = theta * gain_minus_loss + (1.0 - theta) * gain_minus_loss_start
            if not self.dump_full_rad_loss:
                rad_loss = np.zeros(sources.shape[1])
                rad_loss[mask] = gain_minus_loss
                state["rad_loss"] = rad_loss
            else:
                rad_loss_full = np.zeros((rad_loss.shape[0], sources.shape[1]))
                rad_loss_full[:, mask] = rad_loss[:, bc:mask_count+1][:, ::-1]
                state["full_rad_loss"] = rad_loss_full

            sources[IENE, mask] += gain_minus_loss
            # NOTE(cmo): Update e_ion consistently so we're not messing up the eos
            tracers = state["tracers"]
            tracer_energy = state["tracer_energy"]
            Q = state["Q"]
            W = state["W"]
            ion_e = np.sum(tracers * tracer_energy[:, None], axis=0)
            spec_ion_e = ion_e / Q[IRHO]
            Q[IIONE] = spec_ion_e
            W[IIONE] = spec_ion_e
            # NOTE(cmo): If this is only used as a split source, the IIONE update could be done as a source term, but eh.
