# TRIQS application maxent
# Copyright (C) 2018 Gernot J. Kraberger
# Copyright (C) 2018 Simons Foundation
# Authors: Gernot J. Kraberger and Manuel Zingl
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


from __future__ import absolute_import, print_function
from triqs_maxent.triqs_support import *
if if_triqs_1():
    from pytriqs.gf.local import *
elif if_triqs_2():
    from pytriqs.gf import *
if if_triqs_2():
    # TRIQS 2.0 gives data with higher error
    level = 1.e-12
else:  # 1.0 and USE_TRIQS=OFF
    level = 1.e-15
from triqs_maxent import *
from triqs_maxent.elementwise_maxent import *
from itertools import product
import numpy as np

em = ElementwiseMaxEnt()

if not if_no_triqs():

    G_iw = GfImFreq(beta=40, indices=[0, 1])
    G_iw[0, 0] << inverse(Omega - 1)
    G_iw[1, 1] << inverse(Omega + 1)
    G_tau = GfImTime(beta=40, indices=[0, 1], n_points=2051)
    G_tau.set_from_inverse_fourier(G_iw)

    em.set_G_iw(G_iw)
    for element in product(range(2), range(2)):
        em.set_G_element(em.maxent_diagonal,
                         em.G_mat, element, True)

        assert np.max(np.abs(em.maxent_diagonal.G -
                             G_tau.data[:, element[0], element[1]])) < level,\
            "set_G_iw not correct"

    em.set_G_tau(G_tau)
    for element in product(range(2), range(2)):
        em.set_G_element(em.maxent_diagonal,
                         em.G_mat, element, True)
        assert np.max(np.abs(em.maxent_diagonal.G -
                             G_tau.data[:, element[0], element[1]])) < level,\
            "set_G_tau not correct"

tau = np.linspace(0, 40, 10)
G_elems = dict()
for element in product(range(2), range(2)):
    G_elems[element] = np.random.rand(len(tau))
    np.savetxt('elementwise_set_G_{}_{}.dat'.format(*element),
               np.column_stack((tau, G_elems[element])))

em.set_G_tau_filename_pattern('elementwise_set_G_{i}_{j}.dat', (2, 2))
for element in product(range(2), range(2)):
    em.set_G_element(em.maxent_diagonal,
                     em.G_mat, element, True)
    assert np.max(np.abs(em.maxent_diagonal.G - G_elems[element])) < level,\
        "set_filename_pattern not correct"

em.set_G_tau_filenames(
    [['elementwise_set_G_{i}_{j}.dat'.format(i=i, j=j) for j in range(2)]
        for i in range(2)])
for element in product(range(2), range(2)):
    em.set_G_element(em.maxent_diagonal,
                     em.G_mat, element, True)
    assert np.max(np.abs(em.maxent_diagonal.G - G_elems[element])) < level,\
        "set_filenames not correct"

em.set_G_tau_data(tau, np.array([[G_elems[i, j]
                                  for j in range(2)] for i in range(2)]))
for element in product(range(2), range(2)):
    em.set_G_element(em.maxent_diagonal,
                     em.G_mat, element, True)
    assert np.max(np.abs(em.maxent_diagonal.G - G_elems[element])) < level,\
        "set_G_tau_data not correct"
