import numpy as np
import time
from IPython.display import clear_output

def Gi(r, eps, lambd, Nq=1001, t=1, m=1):
    q = np.linspace(-np.pi, np.pi, Nq)
    qx, qy, qz = np.meshgrid(q, q, q, indexing='ij')

    epsp = eps + 1j * lambd

    # Precompute reusable terms to avoid redundant calculations
    cos_qx = np.cos(qx)
    cos_qy = np.cos(qy)
    sin_qx = np.sin(qx)
    sin_qy = np.sin(qy)
    sin_qz = np.sin(qz)
    m_cos_term = m + 2 - cos_qx - cos_qy

    # Combine terms to reduce the number of operations
    dx = t * sin_qx
    dy = t * sin_qy
    dzp = t * sin_qz + m_cos_term
    dzm = t * sin_qz - m_cos_term

    # Combine common factors to reduce computations
    Eqp2 = dx**2 + dy**2 + dzp**2
    Eqm2 = dx**2 + dy**2 + dzm**2
    common_factor = 1 / (epsp**2 - Eqp2) + 1 / (epsp**2 - Eqm2)

    # Vectorized calculation for gk
    gk = np.array([
        epsp * common_factor,
        dx * common_factor,
        dy * common_factor,
        dzp / (epsp**2 - Eqp2) + dzm / (epsp**2 - Eqm2)
    ])

    dk = np.abs(q[0] - q[1])
    Omega = (2 * np.pi)**3

    # Vectorized exponential calculation for gr
    gr = np.exp(1j * (r[0] * qx + r[1] * qy + r[2] * qz)) * gk / Omega
    # Corrected integration using np.trapz
    gr_integrated = np.trapz(np.trapz( np.trapz(gr, q, dk, axis=-1) , q, dk, axis=-1) , q, dk, axis=-1)

    return gr_integrated

def calculate_2d_eps(lambd, axis="x", epsmin=-10, epsmax=0, Neps=301, rmax=10, Nr=301, Nq=101, t=1, m=1):
    epsvec = np.linspace(epsmin, epsmax, Neps)
    rvec = np.linspace(-rmax, rmax, Nr)

    tzero = time.perf_counter()

    # Determine the direction vector based on the axis
    direction_vectors = {"x": np.array((1, 0, 0)), "y": np.array((0, 1, 0)), "z": np.array((0, 0, 1))}
    r = direction_vectors.get(axis, np.array((1, 0, 0)))

    g0, gx, gy, gz = (np.zeros((Nr, Neps), dtype=complex) for _ in range(4))

    for i, eps in enumerate(epsvec):
        clear_output(wait=True)
        print(f"{i + 1} of {Neps}")
        print(time.perf_counter() - tzero)
        for j, rj in enumerate(rvec):
            r_vec = rj * r
            g = Gi(r_vec, eps, lambd, Nq, t, m)
            g0[j, i], gx[j, i], gy[j, i], gz[j, i] = g

    # Save the results in a loop
    file_prefix = f"{axis}_lamb={lambd}_rmax={rmax}_Nr={Nr}_Neps={Neps}_Nq={Nq}_t={t}_m={m}_epslimits={epsmin}~{epsmax}"
    for g, suffix in zip([g0, gx, gy, gz], ['s0', 'sx', 'sy', 'sz']):
        filename = f'txt/[{suffix}][TRS]{file_prefix}.txt'
        np.savetxt(filename, g)

    return g0, gx, gy, gz

# Removed multiple calls to calculate_2d_eps function. They should be called as needed, not in the optimized function itself.

calculate_2d_eps(lambd=0.01,axis = "z",epsmin = -2,epsmax=-1, Neps = 100, rmax = 5, Nr = 500, Nq = 100, t=1, m=0.5)
calculate_2d_eps(lambd=0.01,axis = "z",epsmin = -3,epsmax=-2, Neps = 100, rmax = 5, Nr = 500, Nq = 100, t=1, m=0.5)
calculate_2d_eps(lambd=0.01,axis = "z",epsmin = -4,epsmax=-3, Neps = 100, rmax = 5, Nr = 500, Nq = 100, t=1, m=0.5)
calculate_2d_eps(lambd=0.01,axis = "z",epsmin = -5,epsmax=-4, Neps = 100, rmax = 5, Nr = 500, Nq = 100, t=1, m=0.5)