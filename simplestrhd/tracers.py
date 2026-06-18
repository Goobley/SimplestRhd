import numpy as np
from .indices import NUM_GHOST

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

def tracer_cma_normalisation(face_val, tracer_start_idx, tracer_end_idx, tracer_sum):
    """
    Normalise the tracers as per CMA (Consistent multifluid advection). Performs
    normalisation of each set of tracers defined by a pair of indices to the
    provided sum. This is performed in the reconstructed space (i.e. relative to
    mass density).
    """
    for (start, end, norm) in zip(tracer_start_idx, tracer_end_idx, tracer_sum):
        face_val_sum = np.sum(face_val[start:end, :], axis=0)
        face_val[start:end, :] *= norm / face_val_sum

def tracer_cma_validate(sim_config):
    use_tracer_cma = sim_config.get("use_tracer_cma", False)
    if not use_tracer_cma:
        return

    t_start = sim_config["tracer_cma_start_idx"]
    t_end = sim_config["tracer_cma_end_idx"]
    t_sum = sim_config["tracer_cma_inv_sum"]

    if len(t_start) != len(t_end) or len(t_start) != len(t_sum):
        raise ValueError("Inconsistency in CMA normalisation terms")


# def tracer_cma_flattening(n_tracer_cc, n_tracer_l, n_tracer_r, alpha_thresh=0.75, eps_thresh=0.01):
#     # # NOTE(cmo): Steepen -- I don't believe the steepening is very helpful. If anything, look at using a solver with a contact wave
#     # alpha_i = np.zeros_like(n_tracer_cc)
#     # alpha_i[:, 2:-2] = (n_tracer_cc[:, 3:-1] - n_tracer_cc[:, 1:-3]) / (n_tracer_cc[:, 4:None] - n_tracer_cc[:, None:-4])
#     # jump_size_cond = np.zeros_like(n_tracer_cc)
#     # jump_size_cond[:, 1:-1] = np.abs(n_tracer_cc[:, 2:] - n_tracer_cc[:, :-2]) - eps_thresh * np.minimum(n_tracer_cc[:, 2:], n_tracer_cc[:, :-2])

#     # extremum_cond = np.zeros_like(n_tracer_cc)
#     # extremum_cond[:, 2:-2] = (n_tracer_cc[:, 4:None] - n_tracer_cc[:, 3:-1]) * (n_tracer_cc[:, 1:-3] - n_tracer_cc[:, None:-4])

#     # steepen = (alpha_i > alpha_thresh) & (jump_size_cond > 0) & (extremum_cond > 0)
#     # theta = 0.25
#     # n_tracer_l[...] = np.where(
#     #     steepen,
#     #     n_tracer_cc + (1.0 + theta) * (n_tracer_l - n_tracer_cc),
#     #     n_tracer_l
#     # )
#     # n_tracer_r[...] = np.where(
#     #     steepen,
#     #     n_tracer_cc + (1.0 + theta) * (n_tracer_r - n_tracer_cc),
#     #     n_tracer_r
#     # )
#     # np.clip(
#     #     n_tracer_l[:, 1:-1],
#     #     a_min=np.minimum(n_tracer_cc[:, :-2], n_tracer_cc[:, 1:-1]),
#     #     a_max=np.maximum(n_tracer_cc[:, :-2], n_tracer_cc[:, 1:-1]),
#     #     out=n_tracer_l[:, 1:-1],
#     # )
#     # np.clip(
#     #     n_tracer_r[:, 1:-1],
#     #     a_min=np.minimum(n_tracer_cc[:, 1:-1], n_tracer_cc[:, 2:]),
#     #     a_max=np.maximum(n_tracer_cc[:, 1:-1], n_tracer_cc[:, 2:]),
#     #     out=n_tracer_r[:, 1:-1],
#     # )



#     # NOTE(cmo): Method 1, Doesn't seem great
#     # local_extremum = np.zeros_like(n_tracer_cc)
#     # local_extremum[:, 1:-1] = (n_tracer_cc[:, 2:] - n_tracer_cc[:, 1:-1]) * (n_tracer_cc[:, 1:-1] - n_tracer_cc[:, 0:-2])
#     # local_extremum = (local_extremum < 0)
#     # # expand condition
#     # local_extremum[:, 1:] |= local_extremum[:, :-1]
#     # local_extremum[:, :-1] |= local_extremum[:, 1:]

#     # w = 0.5
#     # n_tracer_l[...] = np.where(
#     #     local_extremum,
#     #     w * n_tracer_cc + (1.0 - w) * n_tracer_l,
#     #     n_tracer_l
#     # )
#     # n_tracer_r[...] = np.where(
#     #     local_extremum,
#     #     w * n_tracer_cc + (1.0 - w) * n_tracer_r,
#     #     n_tracer_r
#     # )

#     # NOTE(cmo): Method 2
#     S_L_plus = np.sum(np.maximum(0.0, n_tracer_l - n_tracer_cc), axis=0)
#     S_L_minus = np.sum(np.maximum(0.0, n_tracer_cc - n_tracer_l), axis=0)
#     S_R_plus = np.sum(np.maximum(0.0, n_tracer_r - n_tracer_cc), axis=0)
#     S_R_minus = np.sum(np.maximum(0.0, n_tracer_cc - n_tracer_r), axis=0)

#     delta_i_min_L = np.minimum(S_L_plus, S_L_minus)
#     delta_i_min_R = np.minimum(S_R_plus, S_R_minus)
#     delta_i_max_L = np.maximum(S_L_plus, S_L_minus)
#     delta_i_max_R = np.maximum(S_R_plus, S_R_minus)
#     s_L = 0.5 * np.abs(
#         np.sign(n_tracer_r - n_tracer_l) - np.sign(S_L_plus - S_L_minus)
#     )
#     s_R = 0.5 * np.abs(
#         np.sign(n_tracer_r - n_tracer_l) + np.sign(S_R_plus - S_R_minus)
#     )
#     beta = 0.25
#     w_L = s_L * np.maximum(0.0, np.minimum(1.0, beta * (delta_i_max_L - delta_i_min_L) / (delta_i_min_L + 1e-20)))
#     w_R = s_R * np.maximum(0.0, np.minimum(1.0, beta * (delta_i_max_R - delta_i_min_R) / (delta_i_min_R + 1e-20)))

#     n_tracer_l[...] = w_L * n_tracer_cc + (1.0 - w_L) * n_tracer_l
#     n_tracer_r[...] = w_R * n_tracer_cc + (1.0 - w_R) * n_tracer_r



def tracer_cma_flattening(n_tracer_cc, n_tracer_l, n_tracer_r):
    # NOTE(cmo): Method 2
    S_L_plus = np.sum(np.maximum(0.0, n_tracer_l - n_tracer_cc), axis=0)
    S_L_minus = np.sum(np.maximum(0.0, n_tracer_cc - n_tracer_l), axis=0)
    S_R_plus = np.sum(np.maximum(0.0, n_tracer_r - n_tracer_cc), axis=0)
    S_R_minus = np.sum(np.maximum(0.0, n_tracer_cc - n_tracer_r), axis=0)

    delta_i_min_L = np.minimum(S_L_plus, S_L_minus)
    delta_i_min_R = np.minimum(S_R_plus, S_R_minus)
    delta_i_max_L = np.maximum(S_L_plus, S_L_minus)
    delta_i_max_R = np.maximum(S_R_plus, S_R_minus)
    s_L = 0.5 * np.abs(
        np.sign(n_tracer_r - n_tracer_l) - np.sign(S_L_plus - S_L_minus)
    )
    s_R = 0.5 * np.abs(
        np.sign(n_tracer_r - n_tracer_l) + np.sign(S_R_plus - S_R_minus)
    )
    beta = 0.25
    w_L = s_L * np.maximum(0.0, np.minimum(1.0, beta * (delta_i_max_L - delta_i_min_L) / (delta_i_min_L + 1e-20)))
    w_R = s_R * np.maximum(0.0, np.minimum(1.0, beta * (delta_i_max_R - delta_i_min_R) / (delta_i_min_R + 1e-20)))

    n_tracer_l[...] = w_L * n_tracer_cc + (1.0 - w_L) * n_tracer_l
    n_tracer_r[...] = w_R * n_tracer_cc + (1.0 - w_R) * n_tracer_r