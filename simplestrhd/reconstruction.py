"""
Spatial reconstruction methods
"""
from typing import Tuple
import numpy as np

Array = np.ndarray


def reconstruct_fog(W: Array) -> Tuple[Array, Array]:
    """First-order Godunov (no reconstruction).

    Returns cell-center values at both left and right faces.

    Args:
        W: Primitive variables at cell centers

    Returns:
        WL, WR: Primitive variables at left and right faces
    """
    return (W.copy(), W.copy())


def slope_limiter(um, up):
    """Monotonized central difference slope limiter.

    Args:
        um: Left slope
        up: Right slope

    Returns:
        Limited slope
    """
    # MC limiter
    return (np.copysign(1.0, um) + np.copysign(1.0, up)) * np.minimum(
        np.abs(um), np.minimum(0.25 * np.abs(um + up), np.abs(up))
    )


def reconstruct_plm(W: Array) -> Tuple[Array, Array]:
    """Piecewise linear reconstruction.

    Uses the MC slope limiter for monotonicity preservation.

    Args:
        W: Primitive variables at cell centers

    Returns:
        WL, WR: Primitive variables at left and right faces
    """
    # Left edge of cell
    WL = W.copy()
    # Right edge of cell
    WR = W.copy()

    dwL = W[:, 1:-1] - W[:, :-2]
    dwR = W[:, 2:] - W[:, 1:-1]

    # slopes
    delta = slope_limiter(dwL, dwR)

    WL[:, 1:-1] = WL[:, 1:-1] - 0.5 * delta
    WR[:, 1:-1] = WR[:, 1:-1] + 0.5 * delta
    return WL, WR


def reconstruct_ppm(W: Array, num_ghost: int = 2) -> Tuple[Array, Array]:
    """Piecewise parabolic reconstruction.

    Uses cubic interpolation with steepness constraints.

    Args:
        W: Primitive variables at cell centers
        num_ghost: Number of ghost cells

    Returns:
        WL, WR: Primitive variables at left and right faces
    """
    def limited_slope(i):
        dwL = (
            W[:, num_ghost + i : -num_ghost + i]
            - W[:, num_ghost + i - 1 : -num_ghost + i - 1]
        )
        dwR = (
            W[:, num_ghost + i + 1 : None if i == num_ghost - 1 else -num_ghost + i + 1]
            - W[:, num_ghost + i : -num_ghost + i]
        )
        return slope_limiter(dwL, dwR)

    dw_m = limited_slope(-1)
    dw_0 = limited_slope(0)
    dw_p = limited_slope(1)

    WL = W.copy()
    WR = W.copy()
    # NOTE(cmo): Cubic reconstruction
    WL[:, num_ghost:-num_ghost] = 0.5 * (
        W[:, num_ghost - 1 : -num_ghost - 1] + W[:, num_ghost:-num_ghost]
    ) - (1.0 / 6.0) * (dw_0 - dw_m)
    WR[:, num_ghost:-num_ghost] = 0.5 * (
        W[:, num_ghost:-num_ghost] + W[:, num_ghost + 1 : -num_ghost + 1]
    ) - (1.0 / 6.0) * (dw_p - dw_0)

    mask = ((WR - W) * (W - WL)) <= 0.0
    WL = np.where(mask, W, WL)
    WR = np.where(mask, W, WR)

    mask = np.abs(WR - W) >= 2.0 * np.abs(WL - W)
    WR = np.where(mask, 3.0 * W - 2.0 * WL, WR)
    mask = np.abs(WL - W) >= 2.0 * np.abs(WR - W)
    WL = np.where(mask, 3.0 * W - 2.0 * WR, WL)
    return WL, WR
