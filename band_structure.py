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

import numpy as np

# For plotting.
from matplotlib import pyplot

import tinyarray

sigma_0 = tinyarray.array([[1, 0], [0, 1]])
sigma_x = tinyarray.array([[0, 1], [1, 0]])
sigma_y = tinyarray.array([[0, -1j], [1j, 0]])
sigma_z = tinyarray.array([[1, 0], [0, -1]])

def make_lead(m= 0, k0=0, kx=0, ky=0, a=1, t_x=0.5, t_y=0.5, t_z=0.5, L=30):
    # Start with an empty lead with a single square lattice
    lat = kwant.lattice.chain(a, norbs=2)

    sym_lead = kwant.TranslationalSymmetry(*lat._prim_vecs)
    lead = kwant.Builder(sym_lead)

    # build up one unit cell of the lead, and add the hoppings
    # to the next unit cell
   #### Define and attach the leads. ####
    # Construct the left lead.
    lead[(lat(j) for j in range(L))] = (2 * t_x * (np.cos(kx) - np.cos(k0)) + m *(2 - np.cos(ky)))*sigma_x +2* t_y*np.sin(ky)*sigma_y
    lead[lat.neighbors()] = -1j * t_z * sigma_z - m/2 * sigma_x

    return lead


def main():
    
    m=1
    lead1 = make_lead(m).finalized()
    kwant.plotter.bands(lead1, show=False)
    pyplot.xlabel("momentum [(lattice constant)^-1]")
    pyplot.ylabel("energy [t]")
    pyplot.title("m=" + str(m)) 
    pyplot.show()


# Call the main function if the script gets executed (as opposed to imported).
# See <http://docs.python.org/library/__main__.html>.
if __name__ == '__main__':
    main()
