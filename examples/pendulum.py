import numpy as np
import matplotlib.pyplot as plt
from pyRadau import radau


m = 1.356
l = 1
g = 10


def Pendulum(index=True):
    """Robertson problem of semi-stable chemical reaction, see mathworks and Shampine2005.

    References:
    -----------
    mathworks: https://de.mathworks.com/help/matlab/math/solve-differential-algebraic-equations-daes.html#bu75a7z-5 \\
    Shampine2005: https://doi.org/10.1016/j.amc.2004.12.011
    """
    assert index in [1, 2, 3, "GGL"]

    def fun(t, vy):
        """Cartesian pendulum, see Hairer1996 Section VII Example 2."""
        if index == "GGL":
            x, y, u, v, la, mu = vy
            raise NotImplementedError
            # f = np.zeros(6, dtype=vy.dtype)
            # f[0] = u
            # f[1] = v
            # f[2] = 2 * x * la / m
            # f[3] = 2 * y * la / m + g
        else:
            x, y, u, v, la = vy
            f = np.zeros(5, dtype=vy.dtype)
            f[0] = u
            f[1] = v
            f[2] = 2 * x * la / m
            f[3] = 2 * y * la / m - g

        if index == 3:
            f[4] = x * x + y * y - l * l
        elif index == 2:
            f[4] = 2 * x * u + 2 * y * v
        elif index == 1:
            raise NotImplementedError
            f[4] = 2 * x * u_dot + 2 * y * v_dot + 2 * u * u + 2 * v * v
        else:
            raise NotImplementedError

        return f

    if index == "GGL":
        mass_matrix = np.eye(6)
        mass_matrix[4, 4] = 0
        mass_matrix[5, 5] = 0
        # algebraic_equations = np.array(
        #     [2, 3, 4, 5], dtype=int
        # )  # position error only
        var_index = np.array([4, 5], dtype=int)  # position and velocity error
        # algebraic_equations = np.array([], dtype=int)
        # algebraic_equations = np.array([0, 0, 0, 0, 0, 0], dtype=int)
    else:
        mass_matrix = np.eye(5)
        mass_matrix[4, 4] = 0
        # algebraic_equations = np.array([2, 3, 4], dtype=int)  # position error only
        # algebraic_equations = np.array([4], dtype=int)  # position and velocity error
        # algebraic_equations = np.array([], dtype=int)
        # acutally this is the DAE index
        # algebraic_equations = np.array([0, 0, 0, 0, 0], dtype=int)
        # algebraic_equations = np.array([0, 0, 2, 2, 3], dtype=int)
        # algebraic_equations = np.array([0, 0, 2, 2, 2], dtype=int)
        var_index = np.array([0, 0, 3, 3, 3], dtype=int)

    def jac(t, y):
        raise NotImplementedError

    return mass_matrix, fun, jac, var_index


if __name__ == "__main__":
    index = 3
    mass_matrix, fun, jac, var_index = Pendulum(index)

    # # time span
    # t0 = 0
    # t1 = 2.0
    # # t1 *= 10
    # t1 *= 3
    # t_span = (t0, t1)

    # initial conditions
    if index == "GGL":
        y0 = np.array([l, 0, 0, 0, 0, 0], dtype=float)
    else:
        y0 = np.array([l, 0, 0, 0, 0], dtype=float)

    t0 = 0
    t1 = 5e1
    num = int(1e3)
    t = np.linspace(0, t1, num=num)
    h0 = 1e-6
    max_steps = int(1e6)
    # min_order = 5
    # max_order = 9
    # min_order = 1
    # max_order = 5
    # min_order = 5
    # max_order = 13
    min_order = 1
    max_order = 13
    # max_order = 1
    # min_order = 1
    # max_order = 5
    # min_order = 5
    # min_order = 9
    # max_order = 9
    # min_order = 13
    # max_order = 13
    reltol = 1e-8
    abstol = 1e-8

    y = radau(
        rhs_fn=fun,
        y0=y0,
        t0=t0,
        t=t,
        h0=h0,
        min_order=min_order,
        max_order=max_order,
        max_steps=max_steps,
        reltol=reltol,
        abstol=abstol,
        mass_matrix=mass_matrix,
        dae_index_1_count=2,
        dae_index_2_count=2,
        dae_index_3_count=1,
        # dae_index_1_count=4,
        # dae_index_2_count=1,
        # dae_index_3_count=0,
        # classical_step_size_control=False,
        newton_start_zero=True,  # this works for index 3 DAE
        # newton_start_zero=False,
    )
    y = np.array(y).T

    # visualization
    fig, ax = plt.subplots(3, 1)

    print(f"t: {t}")
    print(f"y:\n{y}")

    ax[0].plot(t, y[0], "-b", label="x")
    ax[0].plot(t, y[1], "-r", label="y")
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(t, y[2], "-b", label="u")
    ax[1].plot(t, y[3], "-r", label="v")
    ax[1].legend()
    ax[1].grid()

    ax[2].plot(y[0], y[1], "-k", label="x-y")
    ax[2].grid()
    ax[2].set_aspect("equal")

    plt.show()
