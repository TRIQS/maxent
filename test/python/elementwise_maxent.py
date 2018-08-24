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


def object_dict_equal(a, b):
    print(a.__dict__ == b.__dict__)
    assert a.__dict__ == b.__dict__, 'maxent_results are not equal'

if if_no_triqs():
    with np.load('elementwise_g_tau.npz') as data:
        tau = data['tau']
        G_tau_noise = data['G_tau_noise']
        G_w_rot = data['G_w_rot']
        w = data['w']
else:
    # Create matrix valued Green functions
    G_w = GfReFreq(indices=[0, 1], window=(-10, 10), n_points=5000)
    G_w[1, 1] << (4.0 * Flat(0.4) - 2.0 * Flat(0.2)) / 2.0
    G_w[0, 0] << (8.0 * Flat(0.8) - 2.0 * Flat(0.2)) / 6.0

    G_iw = GfImFreq(beta=400, indices=[0, 1], n_points=1000)
    G_iw[1, 1] << (4.0 * Flat(0.4) - 2.0 * Flat(0.2)) / 2.0
    G_iw[0, 0] << (8.0 * Flat(0.8) - 2.0 * Flat(0.2)) / 6.0

    # Rotate
    theta = np.pi / 4  # np.pi/2.0
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    G_iw_rot = copy.deepcopy(G_iw)
    G_iw_rot.from_L_G_R(R, G_iw, R.transpose())
    G_w_rot = copy.deepcopy(G_w)
    G_w_rot.from_L_G_R(R, G_w, R.transpose())

    # Check that the density is indeed 1
    numpy_assert(G_iw.density(), 0.5 * np.eye(2), 11)
    numpy_assert(G_iw_rot.density(), 0.5 * np.eye(2), 11)
    # Density for GfReFreq is still a TRIQS pull request sitting
    # around. We check the norm (problem is anyhow ph-symmetric):
    w = [np.real(w) for w in G_w_rot[0, 0].mesh]
    for i in [0, 1]:
        for j in [0, 1]:
            dta = G_w_rot[i, j].data[:]
            if len(dta.shape) > 1:
                dta = dta[:, 0, 0]
            numpy_assert(np.trapz(-1.0 / (np.pi) *
                                  np.imag(dta), w), float(i == j), 3)

    np_tau = len(G_iw_rot.mesh) + 1
    G_tau = GfImTime(beta=G_iw_rot.mesh.beta, indices=G_iw_rot.indices,
                     n_points=np_tau)
    G_tau.set_from_inverse_fourier(G_iw_rot)
    # add some noise to G_tau
    np.random.seed(666)
    G_tau_noise = G_tau.data[::10].real + noise * \
        np.random.randn(*np.shape(G_tau.data[::10]))
    # Symmetrize G_tau_noise
    G_tau_noise[:, 0, 1] = G_tau_noise[:, 1, 0]
    G_tau_noise = G_tau_noise.transpose([1, 2, 0])
    try:
        # this will work in TRIQS 2.0
        tau = np.array(list(G_tau.mesh.values())).real[::10]
    except AttributeError:
        # this will work in TRIQS 1.4
        tau = np.array(list(G_tau.mesh)).real[::10]

    # np.savez_compressed('elementwise_g_tau', tau=tau,
    #    G_tau_noise=G_tau_noise, G_w_rot=G_w_rot.data, w=w)

# Maxent for all matrix elements
ew = ElementwiseMaxEnt(use_hermiticity=False)
ew.set_G_tau_data(tau, G_tau_noise)
ew.omega = HyperbolicOmegaMesh(omega_min=-10, omega_max=10, n_points=80)
ew.alpha_mesh = LogAlphaMesh(alpha_min=0.05, alpha_max=500, n_points=8)
ew.set_error(noise)
result_ew = ew.run()

# Elementwise use hermiticity
ew = ElementwiseMaxEnt(use_hermiticity=True)
# add some noise to G_tau
ew.set_G_tau_data(tau, G_tau_noise)
ew.omega = HyperbolicOmegaMesh(omega_min=-10, omega_max=10, n_points=80)
ew.alpha_mesh = LogAlphaMesh(alpha_min=0.05, alpha_max=500, n_points=8)
# we test a different way to set the error
ew.set_error(noise * np.ones(G_tau_noise.shape))
result_ew_herm = ew.run()

# Maxent for the diagonals
di = DiagonalMaxEnt()
# use same G_tau
di.set_G_tau_data(tau, G_tau_noise)
di.omega = HyperbolicOmegaMesh(omega_min=-10, omega_max=10, n_points=80)
di.alpha_mesh = LogAlphaMesh(alpha_min=0.05, alpha_max=500, n_points=8)
di.set_error(noise)
result_di = di.run()

# Poorman's (better than elementwise!)
pm = PoormanMaxEnt(use_hermiticity=False)
# use same G_tau
pm.set_G_tau_data(tau, G_tau_noise)
pm.omega = HyperbolicOmegaMesh(omega_min=-10, omega_max=10, n_points=80)
pm.alpha_mesh = LogAlphaMesh(alpha_min=0.05, alpha_max=500, n_points=8)
pm.set_error(noise)
result_pm = pm.run()

# Poorman's use_hermiticity
pm = PoormanMaxEnt(use_hermiticity=True)
# use same G_tau
pm.set_G_tau_data(tau, G_tau_noise)
pm.omega = HyperbolicOmegaMesh(omega_min=-10, omega_max=10, n_points=80)
pm.alpha_mesh = LogAlphaMesh(alpha_min=0.05, alpha_max=500, n_points=8)
pm.set_error(noise)
result_pm_herm = pm.run()

# We check whether we can call .data
result_pm_herm.data

assert np.all(result_ew_herm.A_out[0, 1] == result_ew_herm.A_out[1, 0]), \
    "hermiticity does not work correcly"
assert np.all(result_pm_herm.A_out[0, 1] == result_pm_herm.A_out[1, 0]), \
    "hermiticity does not work correcly"

# Check if the resulting A are equal
for j in [0, 1]:
    for i in [0, 1]:
        lfa = result_pm.analyzer_results[i][j]['LineFitAnalyzer']
        numpy_assert(result_pm.A_out[i, j], lfa['A_out'], 16)
        numpy_assert(result_ew.A[i, j], result_ew_herm.A[i, j], 13)
        numpy_assert(result_pm.A[i, j], result_pm_herm.A[i, j], 13)
        if i == j:
            numpy_assert(result_di.A[i, i], result_ew.A[i, i], 13)
            numpy_assert(result_di.A[i, i], result_pm.A[i, i], 13)

# Check if Poorman's is doing better (at least a bit)
if if_triqs_1():
    # in TRIQS 1.4, .data will have 3 indices and we will have to use an extra
    # 0, 0
    A_w_data = -1.0 / np.pi * np.imag(G_w_rot[0, 1].data[:, 0, 0])
elif if_triqs_2():
    A_w_data = -1.0 / np.pi * np.imag(G_w_rot[0, 1].data[:])
elif if_no_triqs():
    A_w_data = -1.0 / np.pi * np.imag(G_w_rot[:, 0, 1])
w_pm = np.array(result_pm.omega)
w_ew = np.array(result_ew.omega)
A_01_pm = result_pm.analyzer_results[0][1]['LineFitAnalyzer']['A_out']
A_01_ew = result_ew.analyzer_results[0][1]['LineFitAnalyzer']['A_out']
A_00_ew = result_ew.analyzer_results[0][0]['LineFitAnalyzer']['A_out']
A_11_ew = result_ew.analyzer_results[1][1]['LineFitAnalyzer']['A_out']

assert (np.sum(np.abs(A_w_data - np.interp(w, w_pm, A_01_pm))) - \
        np.sum(np.abs(A_w_data - np.interp(w, w_ew, A_01_ew)))) < 0, 'Poorman did worse.'

# Check normalization of A
numpy_assert(np.trapz(A_01_pm, w_pm), 0, 2)
numpy_assert(np.trapz(A_01_ew, w_ew), 0, 2)
numpy_assert(np.trapz(A_00_ew, w_ew), 1, 2)
numpy_assert(np.trapz(A_11_ew, w_ew), 1, 2)
