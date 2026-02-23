"""SimplestRhd: A minimal 1D relativistic hydrodynamics solver with modular physics."""

# Import everything from all modules to maintain backward compatibility
from .conduction import *
from .config_template import *
from .eos import *
from .indices import *
from .io import *
from .lte_eos import *
from .lw_interface import *
from .reconstruction import *
from .riemann_flux import *
from .simulation import *
from .thin_loss import *
from .tracer_eos import *
from .tracers import *

__all__ = [
    # This will include everything exported by the submodules
]
