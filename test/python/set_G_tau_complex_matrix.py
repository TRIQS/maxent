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
from triqs_maxent import *
from triqs_maxent.elementwise_maxent import *
import numpy as np
from triqs_maxent.triqs_support import *
if if_triqs_1():
    from pytriqs.gf.local import *
elif if_triqs_2():
    from pytriqs.gf import *

noise = 1e-3


def numpy_assert(a, b, dec): return np.testing.assert_almost_equal(
    a, b, decimal=dec)

G_iw = GfImFreq(beta=10, indices=[0, 1], n_points=1000)
G_iw[1, 1] << (4.0 * Flat(0.4) - 2.0 * Flat(0.2)) / 2.0
G_iw[0, 0] << (8.0 * Flat(0.8) - 2.0 * Flat(0.2)) / 6.0

# Rotate
theta = np.pi / 4  # np.pi/2.0
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta) * np.exp(0.1j), np.cos(theta) * np.exp(0.1j)]])
G_iw_rot = copy.deepcopy(G_iw)
G_iw_rot.from_L_G_R(R, G_iw, R.conjugate().transpose())

# Generate G_tau_noise
np_tau = len(G_iw_rot.mesh) + 1
G_tau = GfImTime(beta=G_iw_rot.mesh.beta, indices=G_iw_rot.indices,
                 n_points=np_tau)
G_tau.set_from_inverse_fourier(G_iw_rot)
# add some noise to G_tau
np.random.seed(666)
G_tau_noise = G_tau.data + noise * np.random.randn(*np.shape(G_tau.data))
# Symmetrize G_tau_noise
G_tau_noise[:, 0, 1] = G_tau_noise[:, 1, 0].conjugate()
try:
    # this will work in TRIQS 2.0
    tau = np.array(list(G_tau.mesh.values())).real
except:
    # this will work in TRIQS 1.4
    tau = np.array(list(G_tau.mesh)).real

# put it back into TRIQS G_tau and G_iw
G_tau.data[:, :, :] = G_tau_noise[:, :, :]

try:
    # this is necessary in TRIQS unstabel but will fail in 1.4
    # We use the known tail from G_iw_rot for the FT of the noisy data
    G_iw_rot.set_from_fourier(G_tau, G_iw_rot.fit_tail()[0])
    level = 9 
except:
    G_iw_rot.set_from_fourier(G_tau)
    level = 13
    try:
	# this is necessary in TRIQS 2.0 but will fail in 1.4
	from pytriqs.gf.gf_fnt import replace_by_tail
	tail, err = G_iw_rot.fit_tail(
	    known_moments=np.zeros((1, 2, 2), dtype=np.complex_))
	replace_by_tail(G_iw_rot, tail, 200)
	# in TRIQS 2.0 we can only check up to 6 digits because the FT
	# gives such a high error due to the uncertainty of the tail fit
	level = 6
    except:
    	pass

G_tau.set_from_inverse_fourier(G_iw_rot)

save_Gtau = np.zeros((len(tau), 3))
fn = 'set_G_tau_complex_matrix_'
for i in [0, 1]:
    for j in [0, 1]:
        save_Gtau[:, 0] = tau
        save_Gtau[:, 1] = np.real(G_tau.data[:, i, j])
        save_Gtau[:, 2] = np.imag(G_tau.data[:, i, j])
        np.savetxt(fn + str(i) + '_' + str(j) + '.dat', save_Gtau)


def check_elementwise(ew, dec=13):
    for i in [0, 1]:
        for j in [0, 1]:
            if i == j:
                # real part
                ew.set_G_element(ew.maxent_diagonal, ew.G_mat, (i, i), True)
                numpy_assert(ew.maxent_diagonal.G,
                             np.real(G_tau.data[:, i, i]), dec)
                # imaginary part
                ew.set_G_element(ew.maxent_diagonal, ew.G_mat, (i, i), False)
                numpy_assert(
                    ew.maxent_diagonal.G,
                    np.imag(G_tau.data[:, i, i]), dec)
            else:
                # real part
                ew.set_G_element(ew.maxent_offdiagonal, ew.G_mat, (i, j), True)
                numpy_assert(G_tau.data[:, i, j], G_tau.data[:, i, j], dec)
                numpy_assert(ew.maxent_offdiagonal.G,
                             np.real(G_tau.data[:, i, j]), dec)
                # imaginary part
                ew.set_G_element(
                    ew.maxent_offdiagonal, ew.G_mat, (i, j), False)
                numpy_assert(ew.maxent_offdiagonal.G,
                             np.imag(G_tau.data[:, i, j]), dec)

# From TRIQS G_iw
ew0 = ElementwiseMaxEnt(use_hermiticity=True, use_complex=True)
ew0.set_G_iw(G_iw_rot)
check_elementwise(ew0, level)

# From TRIQS G_tau
ew1 = ElementwiseMaxEnt(use_hermiticity=True, use_complex=True)
ew1.set_G_tau(G_tau)
check_elementwise(ew1)

# From data
ew2 = ElementwiseMaxEnt(use_hermiticity=True, use_complex=False)
ew2.set_G_tau_data(tau, G_tau.data[:, :, :].transpose([1, 2, 0]))
check_elementwise(ew2)

# From file (names)
ew3 = ElementwiseMaxEnt(use_hermiticity=True, use_complex=True)
ew3.set_G_tau_filenames([[fn + str(0) + '_' + str(0) + '.dat',
                          fn + str(0) + '_' + str(1) + '.dat'],
                         [fn + str(1) + '_' + str(0) + '.dat',
                          fn + str(1) + '_' + str(1) + '.dat']],
                        tau_col=0, G_col_re=1, G_col_im=2)

check_elementwise(ew3)
# From file (pattern)
ew4 = ElementwiseMaxEnt(use_hermiticity=True, use_complex=True)
ew4.set_G_tau_filename_pattern(
    fn + '{i}_{j}.dat', (2, 2), tau_col=0, G_col_re=1, G_col_im=2)
check_elementwise(ew4)
