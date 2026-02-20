import numpy as np

def normalise_tracers(t, rho):
    """Normalises the tracers relative to the mass density"""
    return t / rho[None, :]

def tracer_flux(normalised_tracers_L, normalised_tracers_R, density_flux):
    """Computes the flux for the tracers (using values reconstructed around the
    interface). Upwind off the mass density flux."""
    return np.where(
        density_flux[None, :] >= 0.0,
        normalised_tracers_L * density_flux[None, :],
        normalised_tracers_R * density_flux[None, :],
    )