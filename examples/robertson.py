import numpy as np
import matplotlib.pyplot as plt
from pyRadau import radau


def Robertson(DAE=True):
    """Robertson problem of semi-stable chemical reaction, see mathworks and Shampine2005.

    References:
    -----------
    mathworks: https://de.mathworks.com/help/matlab/math/solve-differential-algebraic-equations-daes.html#bu75a7z-5 \\
    Shampine2005: https://doi.org/10.1016/j.amc.2004.12.011
    """

    if DAE:
        var_index = np.array([0, 0, 1])
    else:
        var_index = None

    mass_matrix = np.eye(3)
    if DAE:
        mass_matrix[2, 2] = 0

    def fun(t, y):
        y1, y2, y3 = y
        f = np.zeros(3, dtype=float)
        f[0] = -0.04 * y1 + 1e4 * y2 * y3
        f[1] = 0.04 * y1 - 1e4 * y2 * y3 - 3e7 * y2**2
        if DAE:
            f[2] = y1 + y2 + y3 - 1
        else:
            f[2] = 3e7 * y2**2
        return f

    def jac(t, y, fjac, rpar, ipar, n, ldjac):
        raise NotImplementedError
        # # print(f"jac called")
        # def fun(y):
        #     f = np.zeros(3)
        #     fcn(t, y, f, rpar, ipar, n)
        #     return f

        # fjac = approx_fprime(y, fun, method="2-point")
        # # print(f"jac finished")

    return mass_matrix, fun, jac, var_index


if __name__ == "__main__":
    mass_matrix, fun, jac, var_index = Robertson()

    y0 = np.array([1, 0, 0], dtype=float)
    t0 = 0
    t1 = 1e7
    num = 100
    t = np.logspace(-5, 7, num=num)
    max_steps = int(1e4)
    # max_steps = int(1e6)
    # min_order = 5
    # max_order = 13
    min_order = 5
    max_order = 17
    reltol = 1e-14
    abstol = 1e-14

    # TODO: Compute dense output using collocation polynomial
    sol_t = []
    sol_h = []
    sol_y = []

    def dense_callback(told, t, y, cont):
        sol_t.append(t)
        sol_h.append(t - told)  # TODO: This is flawed
        sol_y.append(y)

    y = radau(
        rhs_fn=fun,
        y0=y0,
        t0=t0,
        t=t,
        min_order=min_order,
        max_order=max_order,
        max_steps=max_steps,
        reltol=reltol,
        abstol=abstol,
        mass_matrix=mass_matrix,
        dense_callback=dense_callback,
    )
    y = np.array(y).T

    # print(f"y.shape: {y.shape}")

    # exit()

    # t = np.array(sol_t)
    h = np.array(sol_h)
    # y = np.array(sol_y).T

    # visualization
    fig, ax = plt.subplots(2, 1)

    # print(f"t: {t}")
    # print(f"y:\n{y}")

    ax[0].plot(t, y[0], "-b", label="y1")
    ax[0].plot(t, y[1] * 1e4, "-r", label="y2")
    ax[0].plot(t, y[2], "-y", label="y3")
    ax[0].set_xscale("log")
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(sol_t, sol_h, "-ok", label="h")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].legend()
    ax[1].grid()

    plt.show()
