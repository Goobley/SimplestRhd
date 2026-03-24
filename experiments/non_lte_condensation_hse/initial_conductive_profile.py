import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
try:
    get_ipython().run_line_magic("matplotlib", "")
except:
    plt.ion()
from scipy.integrate import RK45
import lightweaver as lw
import astropy.constants as const

# Add parent directory to path so we can import simplestrhd
sys.path.insert(0, str(Path(__file__).absolute().parent.parent.parent))
from simplestrhd import logt_DM, lambda_DM

KAPPA_0 = 8e-12
T_EDGE = 1.8257e6
T_MIN = 25e3
PRESSURE = 0.023
EDGE_DENSITY = 1e-12

def gradients(z, y):
    temperature = y
    k_B = const.k_B.value
    nh = PRESSURE / ((lw.DefaultAtomicAbundance.totalAbundance + 1) * k_B * temperature)
    ne = nh

    loss_lambda = 10**np.interp(np.log10(temperature), logt_DM, lambda_DM)
    loss = loss_lambda * ne * nh

    gamma = 5.0 / 3.0
    alpha = 1.0 / (gamma - 1.0) * k_B * nh * (lw.DefaultAtomicAbundance.totalAbundance + 1)
    d_temperature = loss / (alpha * KAPPA_0 * temperature**2.5)
    return np.array([d_temperature])


initial_state = np.array([
    T_MIN,
])

sample_every = 1000.0
next_sample = 0.0
samples = []

rk = RK45(gradients, 0.0, initial_state, 5e6)
while rk.t < rk.t_bound:
    rk.step()
    print(rk.t)
    dense = rk.dense_output()
    while next_sample >= dense.t_min and next_sample <= dense.t_max:
        samples.append(dense(next_sample))
        next_sample += sample_every


temperature = np.array(samples)
k_B = const.k_B.value
rho = PRESSURE / ((lw.DefaultAtomicAbundance.totalAbundance + 1) * k_B * temperature) * (lw.DefaultAtomicAbundance.massPerH * const.m_p.value)



