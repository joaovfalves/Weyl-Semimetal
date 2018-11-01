# Tutorial 2.4.1. Band structure calculations
# ===========================================
#
# Physics background
# ------------------
#  band structure of a simple quantum wire in tight-binding approximation
#
# Kwant features highlighted
# --------------------------
#  - Computing the band structure of a finalized lead.

import kwant
import scipy.sparse.linalg as sla
import numpy as np

# For plotting.
from matplotlib import pyplot


import tinyarray

sigma_0 = tinyarray.array([[1, 0], [0, 1]])
sigma_x = tinyarray.array([[0, 1], [1, 0]])
sigma_y = tinyarray.array([[0, -1j], [1j, 0]])
sigma_z = tinyarray.array([[1, 0], [0, -1]])

def make_lead(m, k0= 0, kx=0, ky=0, a=1, t_x=0.5, t_y=0.5, t_z=0.5, L=30):
    # Start with an empty lead with a single square lattice
    lat = kwant.lattice.chain(a)

    sym_lead = kwant.TranslationalSymmetry(*lat._prim_vecs)
    lead = kwant.Builder(sym_lead)

    # build up one unit cell of the lead, and add the hoppings
    # to the next unit cell
   #### Define and attach the leads. ####
    # Construct the left lead.
    lead[(lat(j) for j in range(L))] = (2 * t_x * (np.cos(kx) - np.cos(k0)) + m *(2 - np.cos(ky)))*sigma_x +2* t_y*np.sin(ky)*sigma_y
    lead[lat.neighbors()] = -1j * t_z * sigma_z - m/2 * sigma_x

    return lead

def closed_system(m, a = 1, k0= 0, kx=0, ky=0, t_x=0.5, t_y=0.5, t_z=0.5, L=300):

    lat = kwant.lattice.chain(a, norbs = 2)

    syst = kwant.Builder()

    #### Define the scattering region. ####
    syst[(lat(x) for x in range(L))] = (2 * t_x * (np.cos(kx) - np.cos(k0)) + m *(2 - np.cos(ky)))*sigma_x +2* t_y*np.sin(ky)*sigma_y
    syst[lat.neighbors()] = -1j * t_z * sigma_z - m/2 * sigma_x

    # It's a closed system for a change, so no leads
    return syst

def sorted_eigs(ev):
    evals, evecs = ev
    evals, evecs = map(np.array, zip(*sorted(zip(evals, evecs.transpose()))))
    return evals, evecs.transpose()


def plot_wave_function(syst, m = 0.9, a = 1, t = 0.5, L = 300):
    # Calculate the wave functions in the system.
    ham_mat = syst.hamiltonian_submatrix(sparse=True, args=[m,a,t,L])
    evals, evecs = sorted_eigs(sla.eigsh(ham_mat, k=5, which='SM'))

    # Plot the probability density of the 10th eigenmode.
    evecs_up = evecs[:, 2][1::2]
    evecs_dn = evecs[:, 2][::2]

    z = np.array(range(300))
    up_sq = abs(evecs_up)**2
    dn_sq = abs(evecs_dn)**2

    pyplot.plot(z, up_sq + dn_sq)
    pyplot.show()

def calc_2_ends(syst, a = 1, kx=0, ky=0, L = 300):

    '''
    Essa função retorna o valor do módulo da
    função de onda nos pontos inicial e final
    da cadeia unidimensional que compõe o
    sistema. Note que a função retorna uma 'tuple'
    com duas entradas.

    '''

    # Calculate the wave functions in the system.
    ham_mat = syst.hamiltonian_submatrix(sparse=True, args=[kx, ky])
    # ham_mat = syst.hamiltonian_submatrix(sparse=True)
    evals, evecs = sorted_eigs(sla.eigsh(ham_mat, k=5, which='SM'))

    # Plot the probability density of the 10th eigenmode.
    evecs_up = evecs[:, 0][0::2]
    evecs_dn = evecs[:, 0][1::2]

    z = np.array(range(L))
    up_sq = abs(evecs_up)**2
    dn_sq = abs(evecs_dn)**2
    mod_sqrd = up_sq + dn_sq
    norm = a * sum(up_sq + dn_sq)
    psi_t = (1/norm) * mod_sqrd

    return (psi_t[0], psi_t[-1])

def main():

    m = 0.9

    syst = closed_system(m)

    # Check that the system looks as intended.
    # kwant.plot(syst)

    # Finalize the system.
    syst = syst.finalized()

    plot_wave_function(syst)
    a, b = calc_2_ends(syst, a = 1, kx=0, ky=0, L = 300)
    print(a)
    print(b)

    # lead1 = make_lead(m).finalized()
    # kwant.plotter.bands(lead1, show=False)
    # pyplot.xlabel("momentum [(lattice constant)^-1]")
    # pyplot.ylabel("energy [t]")
    # pyplot.title("m=" + str(m))
    # pyplot.show()


# Call the main function if the script gets executed (as opposed to imported).
# See <http://docs.python.org/library/__main__.html>.
if __name__ == '__main__':
    main()
