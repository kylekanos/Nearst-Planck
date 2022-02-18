import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple


# the coefficients parameter
Params = namedtuple('Params', ['D', 'beta'])


def solve_phi(dx: float, rho: np.ndarray) -> np.ndarray:
    """ use matrix inversion to solve Poisson equation """
    mat = np.zeros((rho.size, rho.size))
    dx2 = 1 / dx**2
    mat[0, 0] = 2 * dx2
    mat[0, 1] = -1 * dx2
    mat[-1, -2] = -1 * dx2
    mat[-1, -1] = 2 * dx2
    for i in range(1, rho.size-1):
        mat[i, i] = 2 * dx
        mat[i, i-1] = -1 * dx2
        mat[i, i+1] = -1 * dx2
    return np.linalg.solve(mat, rho)


def nearst_planck(dt: float, dx: float, rho: np.ndarray, phi: np.ndarray, coefs: Params) -> np.ndarray:
    """ solve the Nearst-Planck equation one step """
    q = np.copy(rho)
    for idx in range(1, len(rho)-1):
        dif = (rho[idx+1] + rho[idx-1] - 2 * rho[idx]) * dt / dx**2
        adv = coefs.beta * 0.25 * (phi[idx+1] - phi[idx-1]) * (rho[idx+1] - rho[idx-1]) * dt /dx
        q[idx] = rho[idx] + coefs.D * (dif + adv - dt * coefs.beta * rho[idx]**2)
    return q


def evolve(tend: float, dx: float, rho: np.ndarray, coefs: Params) -> np.ndarray:
    """ evolve the initial density rho from 0 until tend """
    # the timestep, set by CFL condition
    t, dt = 0, 0.5 * dx**2 / coefs.D
    print(f"dt = {dt}; should take {int(tend/dt)+1} iterations")

    # determine the potential
    phi = solve_phi(dx, rho)

    out_t = 0.05

    while t < tend:
        eta = nearst_planck(dt, dx, rho, phi, coefs)
        phi = solve_phi(dx, eta)
        rho = eta
        # make last step exact to end time
        if t + dt > tend:
            dt = tend - t
        t += dt
        if t >= out_t:
            print(f"Reached {t}")
            out_t += 0.05

    return rho


def main():
    """ main caller, runs the system """
    # our coefficient
    coefs = Params(D=1, beta=0.1)

    # build our grid
    num_points = 100
    x = np.linspace(-5, 5, num=num_points, endpoint=True)
    
    # distribute the charge via normal
    rho = np.exp(-x**2)
    # store in plotter
    plt.plot(x, rho, label='rho(t=0)')

    for d, b in [(1, 1), (1, 2), (2, 1)]:
        new_rho = evolve(0.5, x[1] - x[0], np.exp(-x**2), Params(d, b))
        plt.plot(x, new_rho, label=f'rho(0.5,{d}, {b})') 

    # evolve until t = 0.5
    # new_rho = evolve(0.5, x[1] - x[0], rho, coefs)
    # plt.plot(x, new_rho, label='rho(t=0.5)')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
