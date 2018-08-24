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


from __future__ import absolute_import, print_function, division
import numpy as np
from triqs_maxent.triqs_support import *
if if_triqs_1():
    from pytriqs.gf.local import *
elif if_triqs_2():
    from pytriqs.gf import *
from triqs_maxent.tau_maxent import TauMaxEnt
from triqs_maxent.alpha_meshes import *

np.random.seed(298347923)

if if_no_triqs():
    G_tau_data = np.loadtxt('g_tau_semicircular.dat')
    tau = G_tau_data[:, 0]
    G_tau_data = G_tau_data[:, 1]
else:
    # First, generate an exact G(tau)
    G_iw = GfImFreq(beta=40, indices=[0], n_points=100)
    G_iw << SemiCircular(1)
    G_tau = GfImTime(beta=40, indices=[0], n_points=201)
    G_tau.set_from_inverse_fourier(G_iw)
    try:
        tau = np.array(list(G_tau.mesh.values()))
    except:
        tau = np.array(list(G_tau.mesh))
    G_tau_data = G_tau.data[:, 0, 0].real
# we assume that the error is proportional to G(tau)
err = -1.e-3 * G_tau_data
# we calculate some noise to add to G(tau)
dG_tau = np.random.randn(len(err))
G_tau_data += err * dG_tau  # add it to the Green function

# perform MaxEnt
tm = TauMaxEnt(cost_function='bryan')
tm.set_G_tau_data(tau, G_tau_data)
tm.set_err(err)
tm.alpha_mesh = LogAlphaMesh(alpha_min=0.01, alpha_max=2000, n_points=5)
result = tm.run()
