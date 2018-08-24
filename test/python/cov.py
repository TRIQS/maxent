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
import numpy as np
from triqs_maxent import *
from triqs_maxent.tau_maxent import *

np.random.seed(3466983071)

tm = TauMaxEnt()

if if_no_triqs():
    tm.set_G_tau_file('g_tau_semicircular.dat')
else:
    # First, generate an exact G(tau)
    G_iw = GfImFreq(beta=40, indices=[0], n_points=100)
    G_iw << SemiCircular(1)
    G_tau = GfImTime(beta=40, indices=[0], n_points=201)
    G_tau.set_from_inverse_fourier(G_iw)
    # fix for TRIQS 2.0
    G_tau.data[:] -= G_tau.data.imag * 1.0j
    tm.set_G_tau(G_tau)
    assert np.max(np.abs(tm.G - G_tau.data[:, 0, 0])) < 1.e-14, \
        "G(tau) inconsistency"
# we assume that the error is proportional to G(tau)
err = -1.e-3 * tm.G
# we calculate some noise to add to G(tau)
G_tau_data = tm.G.copy()

# perform MaxEnt
tm.analyzers = []
tm.minimizer.maxiter = 100
tm.alpha_mesh = LogAlphaMesh(alpha_min=160, n_points=1)
K0 = tm.K.K.copy()
tm.set_err(err)
result = tm.run()

# we can also give the error as covariance matrix
C = np.diag(err**2)
tm.set_cov(C)
assert np.max(np.abs(np.sort(tm.err) - np.sort(err))) < 1.e-14, \
    "err is different"
result_C = tm.run()
if if_no_triqs():
    tm.set_G_tau_file('g_tau_semicircular.dat')
else:
    tm.set_G_tau(G_tau)
result_C2 = tm.run()

T = tm._T.copy()

print('Undoing transformation')
tm.err = np.diag(np.dot(tm._T.conjugate().transpose(),
                        np.dot(np.diag(tm.err), tm._T)))
tm._transform(None)
assert np.max(np.abs(K0 - tm.K.K)) < 1.e-14, "Kernel inconsistency"
assert np.max(np.abs(tm.err - err)) < 1.e-14, \
    "err is different {}".format(np.max(np.abs(tm.err - err)))
assert np.max(np.abs(tm.G - G_tau_data)) < 1.e-14, \
    "G(tau) inconsistency"
result_C3 = tm.run()

assert np.max(np.abs(result.A[0] - result_C.A[0])) < 0.05, \
    "spectral functions differ"
assert np.max(np.abs(result.A[0] - result_C2.A[0])) < 0.05, \
    "spectral functions differ"
assert np.max(np.abs(result.A[0] - result_C3.A[0])) < 0.05, \
    "spectral functions differ"


# Second part of the test: we check if the transformation survives
# parameter changes

K_orig = copy.deepcopy(tm.K)
for i in range(2):
    N_om = 80
    n_cov = 9
    err[n_cov:] = 0.0
    C = np.diag(err**2)
    tm.err = None
    tm.K = copy.deepcopy(K_orig)
    tm.set_cov(C)
    assert (tm.K.K.shape == (n_cov, 100))
    assert (tm.K.U.shape[0] == n_cov)
    assert (tm.K.V.shape[0] == 100)
    assert (tm.K.K_delta.shape == (201, 100))
    if i == 1:
        tm.err = None
        tm.K = copy.deepcopy(K_orig)
    tm.omega = HyperbolicOmegaMesh(n_points=N_om)
    if i == 1:
        tm.set_cov(C)
    assert (tm.K.K.shape == (n_cov, N_om))
    assert (tm.K.U.shape[0] == n_cov)
    assert (tm.K.V.shape[0] == N_om)
    assert (tm.K.K_delta.shape == (201, N_om))
    if i == 1:
        tm.err = None
        tm.K = copy.deepcopy(K_orig)
        N_om = 100
    tm.tau = np.linspace(0, 50, 201)
    if i == 1:
        tm.set_cov(C)
    assert (tm.K.K.shape == (n_cov, N_om))
    assert (tm.K.U.shape[0] == n_cov)
    assert (tm.K.V.shape[0] == N_om)
    assert (tm.K.K_delta.shape == (201, N_om))
    if i == 1:
        tm.err = None
        tm.K = copy.deepcopy(K_orig)
    tm.K.reduce_singular_space(1.e-2)
    if i == 1:
        tm.set_cov(C)
    assert (tm.K.K.shape == (n_cov, N_om))
    assert (tm.K.U.shape[0] == n_cov)
    assert (tm.K.V.shape[0] == N_om)
    assert (tm.K.K_delta.shape == (201, N_om))
    if i == 1:
        tm.err = None
        tm.K = copy.deepcopy(K_orig)
    tm.K.reduce_singular_space(1.e-14)
    if i == 1:
        tm.set_cov(C)
    assert (tm.K.K.shape == (n_cov, N_om))
    assert (tm.K.U.shape[0] == n_cov)
    assert (tm.K.V.shape[0] == N_om)
    assert (tm.K.K_delta.shape == (201, N_om))
    if i == 1:
        tm.err = None
        tm.K = copy.deepcopy(K_orig)
    tm.K = PreblurKernel(tm.K, 0.1)
    if i == 1:
        tm.set_cov(C)
    assert (tm.K.K.shape == (n_cov, N_om))
    assert (tm.K.U.shape[0] == n_cov)
    assert (tm.K.V.shape[0] == N_om)
    assert (tm.K.K_delta.shape == (201, N_om))
    if i == 1:
        tm.err = None
        tm.K = copy.deepcopy(K_orig)
    tm.K.b = 0.2
    tm.K.parameter_change()
    if i == 1:
        tm.set_cov(C)
    assert (tm.K.K.shape == (n_cov, N_om))
    assert (tm.K.U.shape[0] == n_cov)
    assert (tm.K.V.shape[0] == N_om)
    assert (tm.K.K_delta.shape == (201, N_om))
    if i == 1:
        tm.err = None
        tm.K = copy.deepcopy(K_orig)
    tm.G = G_tau_data.copy()
    if i == 1:
        tm.set_cov(C)
    assert (tm.K.K.shape == (n_cov, N_om))
    assert (tm.K.U.shape[0] == n_cov)
    assert (tm.K.V.shape[0] == N_om)
    assert (tm.K.K_delta.shape == (201, N_om))
