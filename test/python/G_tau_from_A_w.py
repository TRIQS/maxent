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


from triqs_maxent.maxent_util import *
from triqs_maxent.triqs_support import *
if if_triqs_1():
    from pytriqs.gf.local import *
elif if_triqs_2():
    from pytriqs.gf import *
from pytriqs.utility.comparison_tests import *

beta = 20.0
np_tau = 101

G_tau_1 = GfImTime(beta=beta, indices=[0], n_points=np_tau)
G_iw_1 = GfImFreq(beta=beta, indices=[0], n_points=np_tau / 2)
G_w = GfReFreq(indices=[0], window=(-1.1, 1.1), n_points=30000)
G_iw_1 << SemiCircular(1.0)
G_tau_1 << InverseFourier(G_iw_1)
G_w << SemiCircular(1.0)

w_points = np.array([w.real for w in G_w.mesh])
A_w = -G_w.data[:, 0, 0].imag / np.pi
G_tau_2 = get_G_tau_from_A_w(A_w=A_w,
                             w_points=w_points,
                             beta=beta,
                             np_tau=np_tau)

assert_arrays_are_close(G_tau_1.data[:, 0, 0].real,
                        G_tau_2.data[:, 0, 0].real, precision=1e-6)
assert_arrays_are_close(G_tau_1.data[:, 0, 0].imag,
                        G_tau_2.data[:, 0, 0].imag, precision=1e-10)
