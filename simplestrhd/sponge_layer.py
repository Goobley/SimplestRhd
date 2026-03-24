import numpy as np

class SpongeLayer:
    """
    Simple sacrificial sponge layer aas per Wilson 2016 (https://theses.gla.ac.uk/7209/)

    Parameters
    ----------
    x_min:
        damp for x <= x_min
    x_max:
        damp for x >= x_max
    a:
        amplitude on exp
    b:
        decay param in exp
    q0:
        parameters to damp towards as a NUM_EQ array. Different left and right
        states can be provided as a two element sequence of arrays.
    q0_full:
        a stratified state to damp towards. Ignored if `q0` provided.
    """
    def __init__(self, x_min, x_max, a, b, q0=None, q0_full=None):
        self.x_min = x_min
        self.x_max = x_max
        self.a = a
        self.b = b
        self.left_right = True
        if q0 is not None:
            if len(q0) == 2:
                self.q0_left = q0[0]
                self.q0_right = q0[1]
            else:
                self.q0_left = q0
                self.q0_right = q0
        elif q0_full is not None:
            self.left_right = False
            self.q0_full = q0_full
        else:
            raise ValueError("Must provide one of q0 or q0_full")

    def __call__(self, state, sim_config, sources, ts):
        pos = state['xcc']
        q = state["Q"]
        left_mask = pos <= self.x_min
        right_mask = pos >= self.x_max

        sigma_left = self.a * np.exp(self.b * np.abs(pos[left_mask] - self.x_min))
        sigma_right = self.a * np.exp(self.b * np.abs(pos[right_mask] - self.x_max))
        if self.left_right:
            sources[:, left_mask] += - sigma_left * (q[:, left_mask] - self.q0_left[:, None]) / ts.dt
            sources[:, right_mask] += - sigma_right * (q[:, right_mask] - self.q0_right[:, None]) / ts.dt
        else:
            sources[:, left_mask] += - sigma_left * (q[:, left_mask] - self.q0_full[:, left_mask]) / ts.dt
            sources[:, right_mask] += - sigma_right * (q[:, right_mask] - self.q0_full[:, right_mask]) / ts.dt
