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
from triqs.gf import *
from triqs.utility.comparison_tests import *
import copy

n_points = 700
w_min = -20
w_max = -w_min

# This is the initial Matusbara Green function for this test
gw = GfReFreq(indices=[1], n_points=n_points, window=(w_min, w_max))
#gw << SemiCircular(half_bandwidth=1)
gw << inverse(Omega + 4j) * inverse(Omega + 3j)
A_w = -gw.data[:, 0, 0].imag / np.pi
w_points = np.array([w.real for w in gw.mesh])
gw_rec = get_G_w_from_A_w(A_w,
                          w_points,
                          np_omega=n_points,
                          w_min=w_min,
                          w_max=w_max)

assert_arrays_are_close(gw.data[:, 0, 0].real,
                        gw_rec.data[:, 0, 0].real, precision=1e-2)
assert_arrays_are_close(gw.data[:, 0, 0].imag,
                        gw_rec.data[:, 0, 0].imag, precision=1e-2)

# Do the same for a matrix-valued GF
n_size = 2
gw = GfReFreq(indices=list(range(n_size)), n_points=n_points, window=(w_min, w_max))
gw[0, 0] << inverse(Omega + 4j) * inverse(Omega + 3j)
gw[1, 1] << inverse(Omega + 2j) * inverse(Omega + 3j)

# rotate GF to generate off-diagonals
theta = np.pi / 3
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])
gw_rot = copy.deepcopy(gw)
gw_rot.from_L_G_R(R, gw, R.conjugate().transpose())
A_w_rot = np.zeros((n_size, n_size, n_points), dtype=complex)
for n in range(n_points):
    A_w_rot[:, :, n] = \
        (-1.0 / (2 * np.pi * 1j)) * (gw_rot.data[n, :, :] -
                                     np.transpose(np.conjugate(gw_rot.data[n, :, :])))

w_points = np.array([w.real for w in gw.mesh])
gw_rot_rec = get_G_w_from_A_w(A_w_rot,
                              w_points,
                              np_omega=n_points,
                              w_min=w_min,
                              w_max=w_max)

assert_arrays_are_close(gw_rot.data[:, :, :].real,
                        gw_rot_rec.data[:, :, :].real, precision=1e-2)
assert_arrays_are_close(gw_rot.data[:, :, :].imag,
                        gw_rot_rec.data[:, :, :].imag, precision=1e-2)

# Calculate G_w again, using np_interp_A
gw_rot_rec = get_G_w_from_A_w(A_w_rot,
                              w_points,
                              np_interp_A = int(n_points*1.2),
                              np_omega=n_points,
                              w_min=w_min,
                              w_max=w_max)

w_points_double = np.array([w.real for w in gw_rot_rec.mesh])
for i in range(n_size):
    for j in range(n_size):
        assert_arrays_are_close(gw_rot.data[:, i, j].real,
                                np.interp(w_points, w_points_double,
                                gw_rot_rec.data[:, i, j].real), precision=1e-2)
        assert_arrays_are_close(gw_rot.data[:, i, j].imag,
                                np.interp(w_points, w_points_double,
                                gw_rot_rec.data[:, i, j].imag), precision=1e-2)
