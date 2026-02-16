from typing import Tuple
import numpy as np

from config import NUM_GHOST
Array = np.ndarray

def reconstruct_fog(W: Array) -> Tuple[Array, Array]:
    return (W.copy(), W.copy())

def slope_limiter(um, up):
    # MC
    return (np.copysign(1.0, um) + np.copysign(1.0, up)) * np.minimum(
        np.abs(um),
        np.minimum(
            0.25 * np.abs(um + up),
            np.abs(up)
        )
    )

def reconstruct_plm(W: Array) -> Tuple[Array, Array]:
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

def reconstruct_ppm(W: Array) -> Tuple[Array, Array]:
    def limited_slope(i):
        s = W.shape[1]
        dwL = W[:, NUM_GHOST+i:-NUM_GHOST+i] - W[:, NUM_GHOST+i-1:-NUM_GHOST+i-1]
        dwR = W[:, NUM_GHOST+i+1:None if i == NUM_GHOST - 1 else -NUM_GHOST+i+1] - W[:, NUM_GHOST+i:-NUM_GHOST+i]
        return slope_limiter(dwL, dwR)

    dw_m = limited_slope(-1)
    dw_0 = limited_slope(0)
    dw_p = limited_slope(1)

    WL = W.copy()
    WR = W.copy()
    # NOTE(cmo): Cubic reconstruction
    WL[:, NUM_GHOST:-NUM_GHOST] = 0.5 * (W[:, NUM_GHOST-1:-NUM_GHOST-1] + W[:, NUM_GHOST:-NUM_GHOST]) - (1.0 / 6.0) * (dw_0 - dw_m)
    WR[:, NUM_GHOST:-NUM_GHOST] = 0.5 * (W[:, NUM_GHOST:-NUM_GHOST] + W[:, NUM_GHOST+1:-NUM_GHOST+1]) - (1.0 / 6.0) * (dw_p - dw_0)

    mask = ((WR - W) * (W - WL)) <= 0.0
    WL = np.where(mask, W, WL)
    WR = np.where(mask, W, WR)

    mask = np.abs(WR - W) >= 2.0 * np.abs(WL - W)
    WR = np.where(mask, 3.0 * W - 2.0 * WL, WR)
    mask = np.abs(WL - W) >= 2.0 * np.abs(WR - W)
    WL = np.where(mask, 3.0 * W - 2.0 * WR, WL)
    return WL, WR
