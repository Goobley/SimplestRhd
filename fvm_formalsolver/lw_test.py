import lightweaver as lw
import lightweaver.LwCompiled as lw_cpp
from lightweaver.fal import Falc82
import promweaver as pw
import matplotlib.pyplot as plt

lw_cpp.FormalSolvers.load_fs_from_path("build/libpragmatic_fvm_fs_1st_1d.so")

atmos = Falc82()
atmos.quadrature(5)

rad_set = lw.RadiativeSet(pw.default_atomic_models())
rad_set.set_active("H", "Ca")

spect = rad_set.compute_wavelength_grid()
eq_pops = rad_set.compute_eq_pops(atmos)

SplitEqPops = True
if SplitEqPops:
    eq_pops_bez = rad_set.compute_eq_pops(atmos)
else:
    eq_pops_bez = eq_pops

ctx_prag = lw.Context(atmos, spect, eq_pops, formalSolver="pragmatic_1_fvm_1d")
ctx_bez = lw.Context(atmos, spect, eq_pops_bez)

# ctx_prag.formal_sol_gamma_matrices()

lw.iterate_ctx_se(ctx_prag)
lw.iterate_ctx_se(ctx_bez)
# ctx_prag.formal_sol_gamma_matrices()


