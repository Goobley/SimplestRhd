import jax
import jax.numpy as jnp
import optimistix as optx
import lineax
import matplotlib.pyplot as plt
try:
    get_ipython().run_line_magic("matplotlib", "")
except:
    plt.ion()

jax.config.update("jax_enable_x64", True)

def harmonic_mean(a, b):
    return 2.0 * a * b / (a + b + 1e-20)

def residual(temperature, dx, kappa0=1.0, alpha=1.0, beta=2.5):
    resid = jnp.zeros_like(temperature)
    kappa = kappa0 * temperature**beta
    kappa_if = jnp.zeros(resid.shape[0] + 1)
    kappa_if = kappa_if.at[1:-1].set(harmonic_mean(kappa[:-1], kappa[1:]))

    resid = resid.at[1:-1].set(
        alpha / dx**2 * (
            kappa_if[2:-1] * (temperature[2:] - temperature[1:-1])
            - kappa_if[1:-2] * (temperature[1:-1] - temperature[:-2])
        )
    )
    return resid

@jax.jit
def rempel_residual(temp, prev_temp, dx, dt, theta=0.55):
    r_new = residual(temp, dx)
    r_old = residual(prev_temp, dx)
    return temp - prev_temp - dt * (theta * r_new + (1.0 - theta) * r_old)

# @jax.jit
# def saturated_residual(temp, prev_temp, dx, kappa0=1.0, alpha=1.0, beta=2.5):



@jax.jit
def solve_step(prev_temperature, dx, dt):
    sol = optx.root_find(lambda y, args: rempel_residual(y, *args), solver=optx.Newton(rtol=1e-4, atol=1e-4, linear_solver=lineax.GMRES(rtol=1e-4, atol=1e-4)), y0=prev_temperature, args=(prev_temperature, dx, dt))
    return sol


x_interfaces = jnp.linspace(0.0, 1.0, 251)
x = 0.5 * (x_interfaces[1:] + x_interfaces[:-1])
temperature = 0.1 + 0.9 * x**5
prev_temperature = temperature.copy()

dx = x[1] - x[0]
dt = 1e-3

t_max = 1.0

temps = [temperature]

def compute_temps(temps):
    prev_temperature = temperature.copy()
    t = 0.0
    while t < t_max:
        sol = solve_step(prev_temperature, dx, dt)
        temps.append(sol.value)
        prev_temperature = sol.value

        t += dt
        # print(sol.stats)

compute_temps(temps)
