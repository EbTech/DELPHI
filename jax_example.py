from jax import jit, grad
from jax.experimental.ode import odeint
import jax.numpy as jnp

def abs_val(x):
    return x**2.0

def f(x, t, a, b, c, k1, k2):
    x1, x2 = x
    return jnp.array([a * x1 + b * x2 + c, k1 * x1 + k2 * x2])

def myfunc(ode_params, k1, k2):
    a, b, c = ode_params
    x_init = jnp.array([1.1, 3.2])
    times = jnp.array([1., 2., 3., 4., 5.])
    x_final = odeint(f, x_init, times, a, b, c, k1, k2)
    print(x_final.shape)
    return x_final[4, 1]

# odeint: fn()

if __name__ == "__main__":
    fgrad = grad(myfunc)
    print("f(params) = ", myfunc((3.1, 1.5, 2.7), 5.1, 6.2))
    print("f'(params) = ", fgrad((3.1, 1.5, 2.7), 5.1, 6.2))
