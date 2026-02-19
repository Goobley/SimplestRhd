import numpy as np
import astropy.constants as const

from .indices import (
    IRHO,
    IPRE,
    IENE,
    NUM_GHOST,
)
from .eos import temperature_si

M_P = const.m_p.value

logt_DM = np.array([
    2.0, 2.1, 2.2, 2.3, 2.4,
    2.5, 2.6, 2.7, 2.8, 2.9,
    3.0, 3.1, 3.2, 3.3, 3.4,
    3.5, 3.6, 3.7, 3.8, 3.9,
    4.0, 4.1, 4.2, 4.3, 4.4,
    4.5, 4.6, 4.7, 4.8, 4.9,
    5.0, 5.1, 5.2, 5.3, 5.4,
    5.5, 5.6, 5.7, 5.8, 5.9,
    6.0, 6.1, 6.2, 6.3, 6.4,
    6.5, 6.6, 6.7, 6.8, 6.9,
    7.0, 7.1, 7.2, 7.3, 7.4,
    7.5, 7.6, 7.7, 7.8, 7.9,
    8.0, 8.1, 8.2, 8.3, 8.4,
    8.5, 8.6, 8.7, 8.8, 8.9,
    9.0
])

lambda_DM = np.array([
    -26.523, -26.398, -26.301, -26.222, -26.097,
    -26.011, -25.936, -25.866, -25.807, -25.754,
    -25.708, -25.667, -25.630, -25.595, -25.564,
    -25.534, -25.506, -25.479, -25.453, -25.429,
    -25.407, -23.019, -21.762, -21.742, -21.754,
    -21.730, -21.523, -21.455, -21.314, -21.229,
    -21.163, -21.126, -21.092, -21.060, -21.175,
    -21.280, -21.390, -21.547, -21.762, -22.050,
    -22.271, -22.521, -22.646, -22.660, -22.676,
    -22.688, -22.690, -22.662, -22.635, -22.609,
    -22.616, -22.646, -22.697, -22.740, -22.788,
    -22.815, -22.785, -22.754, -22.728, -22.703,
    -22.680, -22.630, -22.580, -22.530, -22.480,
    -22.430, -22.380, -22.330, -22.280, -22.230,
    -22.180
]) - 13.0 # cgs to SI

logt_simple = np.array([
    2.0,
    4.45,
    4.477,
    5.0,
    5.7,
    6.0,
    7.0,
    8.0,
])
log_lambda_simple = np.array([
    -26.5,
    -26.4,
    -22.819,
    -21.25,
    -21.600,
    -21.75,
    -22.25,
    -22.75,
]) - 13.0

class TownsendThinLoss:
    def __init__(self, name: str="DM"):
        valid_names = ["DM", "simple"]
        if name not in valid_names:
            raise ValueError(f"Got cooling curve {name}, not in {valid_names}")

        self.name = name
        if name == "DM":
            self.log_lambda = lambda_DM
            self.log_t = logt_DM
        elif name == "simple":
            self.log_lambda = log_lambda_simple
            self.log_t = logt_simple

        # NOTE(cmo): Following athenapk impl
        lambdas = 10**self.log_lambda
        temps = 10**self.log_t
        n_bins = temps.shape[0] - 1
        townsend_Y_k = np.zeros(n_bins)
        townsend_alpha_k = (self.log_lambda[1:] - self.log_lambda[:-1]) / (self.log_t[1:] - self.log_t[:-1])
        if np.any(townsend_alpha_k == 1.0):
            raise ValueError("Special alpha=1 case not implemented! Tweak your cooling curve a little.")

        # Calculate temporal evolution function Y_k
        for i in range(n_bins - 2, -1, -1):
            alpha_k_m1 = townsend_alpha_k[i] - 1.0
            step = (
                (lambdas[n_bins] / lambdas[i])
                * (temps[i] / temps[n_bins])
                * ((temps[i] / temps[i + 1])**alpha_k_m1 - 1.0) / alpha_k_m1
            )
            townsend_Y_k[i] = townsend_Y_k[i + 1] - step
        self.lambdas = lambdas
        self.temps = temps
        self.townsend_Y_k = townsend_Y_k
        self.townsend_alpha_k = townsend_alpha_k

    def __call__(self, state, sim_config, sources, ts):
        y = state.get('y', 1.0)
        h_mass = sim_config.get('h_mass', M_P)
        gamma = state['gamma']
        Q = state["Q"]
        W = state["W"]
        nh_tot = Q[IRHO] / h_mass
        ne = nh_tot * y
        temperature = temperature_si(W[IPRE], nh_tot, y)

        lambdas = self.lambdas
        temps = self.temps
        townsend_Y_k = self.townsend_Y_k
        townsend_alpha_k = self.townsend_alpha_k

        # Find temperature bin
        frac_idx = np.interp(temperature, temps, np.arange(temps.shape[0]))
        idx = frac_idx.astype(np.int32)

        # Compute temporal evolution function
        alpha_k_m1 = townsend_alpha_k[idx] - 1.0
        tef = townsend_Y_k[idx] + (
            (lambdas[-1] / lambdas[idx])
            * (temps[idx] / temps[-1])
            * ((temps[idx] / temperature)**alpha_k_m1 - 1.0) / alpha_k_m1
        )

        tef_adj = tef + lambdas[-1] * ts.dt_sub / temps[-1] * (nh_tot * ne) / (nh_tot + ne) * (gamma - 1.0) / (const.k_B.value)
        done = np.zeros(tef_adj.shape[0], dtype=bool)
        while not np.all(done) and np.any(tef_adj > townsend_Y_k[idx]):
            mask = tef_adj > townsend_Y_k[idx]
            idx = np.where(mask, idx - 1, idx)
            done = (~mask) | (idx == 0)

        new_temperature = temps[idx] * (
            1.0 - (1.0 - townsend_alpha_k[idx]) * (lambdas[idx] / lambdas[-1]) * (temps[-1] / temps[idx]) * (tef_adj - townsend_Y_k[idx])
            ) ** (1.0/(1.0 - townsend_alpha_k[idx]))

        delta_t = new_temperature - temperature
        delta_e = 1.0 / (gamma - 1.0) * (nh_tot + ne) * const.k_B.value * delta_t
        sources[IENE, NUM_GHOST:-NUM_GHOST] += delta_e[NUM_GHOST:-NUM_GHOST] / ts.dt_sub

def rad_loss_dm(state, sim_config, sources, time):
    y = state.get('y', 1.0)
    h_mass = sim_config.get('h_mass', M_P)
    Q = state["Q"]
    W = state["W"]
    nh_tot = Q[IRHO] / h_mass
    temperature = temperature_si(W[IPRE], nh_tot, y)

    # u.Unit('erg cm3 s-1').to('J m3 s-1')
    # Out[6]: 1.0000000000000003e-13
    log_lambda_si = np.interp(np.log10(temperature), logt_DM, lambda_DM)
    loss = nh_tot**2 * y * 10**log_lambda_si

    # lambda_cgs = np.where(
    #     temperature < 29.5e3,
    #     0.0,
    #     np.where(
    #         temperature < 30e3,
    #         10**(-25.518) * (temperature - 29.5e3),
    #         np.where(
    #             temperature < 100e3,
    #             10**(-36.25) * temperature**3,
    #             10**(-18.75) * temperature**(-0.5)
    #         )
    #     )
    # )
    # lambda_si = lambda_cgs * 1e-13
    # loss = nh_tot**2 * y * lambda_si

    sources[IENE, NUM_GHOST:-NUM_GHOST] -= loss[NUM_GHOST:-NUM_GHOST]

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    try:
        get_ipython().run_line_magic("matplotlib", "")
    except:
        plt.ion()
    plt.figure()
    plt.plot(logt_DM, lambda_DM)

    # y = 1.0
    # nh_tot = W[IRHO] / P_MASS
    # temperature = temperature_si(W[IPRE], nh_tot, y)

    # lambda_si = rad_loss_thin(temperature)
    # # NOTE(cmo): Fully ionised pure H for now
    # loss = nh_tot**2 * lambda_si
    # S[IENE, NUM_GHOST:-NUM_GHOST] -= loss[NUM_GHOST:-NUM_GHOST]