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
import numpy as np
from triqs_maxent.tau_maxent import *
from triqs_maxent.triqs_support import *
if if_triqs_1():
    from pytriqs.gf.local import *
elif if_triqs_2():
    from pytriqs.gf import *

np_tau = 10000
np_tau_c = 2500
beta = 50.0

# This is the initial Matusbara Green function for this test
giw = GfImFreq(indices=[1], beta=beta)
giw << SemiCircular(half_bandwidth=1)

# Create an imaginary-time Green function
gt = GfImTime(indices=[1], beta=beta)
gt << InverseFourier(giw)

# Save G_tau to a file
skip = 1
try:
    # this will work in TRIQS 2.0
    mesh = np.array(list(gt.mesh.values())).reshape(-1, 1).real
except AttributeError:
    # this will work in TRIQS 1.4
    mesh = np.array(list(gt.mesh)).reshape(-1, 1).real
m = np.vstack((mesh[0::skip]))
re_data = gt.data.real.reshape((gt.data.shape[0], -1))
re_data = np.vstack((re_data[0::skip]))
#err_data = np.array([[error] for i in mesh])
#err_data = np.vstack((err_data[0::skip]))
mesh_a_data = np.hstack((m, re_data))
np.savetxt('gt.dat', mesh_a_data)

# Set G_tau directly
M1 = TauMaxEnt()
M1.set_G_tau(gt)

# Set G_tau from file
M2a = TauMaxEnt()
M2a.set_G_tau_file('gt.dat')
np.testing.assert_almost_equal(M1.G, M2a.G)
np.testing.assert_almost_equal(M1.tau, M2a.tau)

# Set G_tau from data
M2b = TauMaxEnt()
M2b.set_G_tau_data(np.transpose(m)[0], np.transpose(re_data)[0])
np.testing.assert_almost_equal(M1.G, M2b.G)
np.testing.assert_almost_equal(M1.tau, M2b.tau)

# Set G_tau via G_iw
M3 = TauMaxEnt()
M3.set_G_iw(giw, np_tau=np_tau)
np.testing.assert_almost_equal(M1.G, M3.G)
np.testing.assert_almost_equal(M1.tau, M3.tau)

# Create the imaginary-time Green function on a coarse grid
gt_c = GfImTime(indices=[1], beta=beta, n_points=np_tau_c)
gt_c << InverseFourier(giw)

# G_tau on coarse grid
M6 = TauMaxEnt()
M6.set_G_tau(gt_c)

# G_tau interpolated on coarse grid
M7 = TauMaxEnt()
M7.set_G_tau(gt, tau_new=np.linspace(0, beta, np_tau_c))
np.testing.assert_almost_equal(M6.G, M7.G)
