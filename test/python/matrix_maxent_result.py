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
#from pytriqs.archive import HDFArchive
#from pytriqs.utility.h5diff import h5diff

# to make it reproducible
np.random.seed(658436166)

# first, we construct tau, omega, K
beta = 40
tau = np.linspace(0, beta, 100)
omega = HyperbolicOmegaMesh(omega_min=-10, omega_max=10, n_points=100)
K = TauKernel(tau=tau, omega=omega, beta=beta)
K.reduce_singular_space()

# next, we construct the G(tau)
A = np.exp(-omega**2)
A /= np.trapz(A, omega)
G = np.dot(K.K, A)
G += 1.e-4 * np.random.randn(len(G))
err = 1.e-4 * np.ones(len(G))

# a flat default model
D = FlatDefaultModel(omega=omega)

# the three main ingredients for the cost function
chi2 = NormalChi2(K=K, G=G, err=err)
S = NormalEntropy(D=D)
H_of_v = NormalH_of_v(D=D, K=K)
A_of_H = IdentityA_of_H(omega=D.omega)

# the cost function
Q = MaxEntCostFunction(chi2=chi2, S=S, H_of_v=H_of_v, A_of_H=A_of_H)
Q.set_alpha(0.1)

# a random singular-space vector
v1 = np.random.rand(len(K.S))
v2 = np.random.rand(len(K.S))

######################################################################
mr = MaxEntResult()
mr.add_result(Q(v1))

gives_error = False
try:
    mr.add_result(Q(v1), matrix_element=(1, 1))
except AssertionError:
    gives_error = True
assert gives_error, "giving matrix_element should give an error"

assert mr.alpha == [0.1], "alpha values incorrect"
assert mr.alpha.shape == (1,), "alpha shape incorrect"
assert mr._n_alphas == 1, "no. of alpha values incorrect"
for qtyname in ("chi2", "S", "Q"):
    qty = getattr(mr, qtyname)
    assert qty.shape == (1,), "{} shape incorrect".format(qtyname)
assert mr.A.shape == (1, 100), "A shape incorrect"
assert mr.v.shape == (1, 48), "v shape incorrect"
assert mr.run_times.shape == (1,), "run_times has wrong shape"

######################################################################
mr = MaxEntResult(matrix_structure=(2, 2))

gives_error = False
try:
    mr.add_result(Q(v1))
except AssertionError:
    gives_error = True
assert gives_error, "leaving out matrix_element should give an error"

assert mr._get_empty(fill_with=lambda: 1) == [[1, 1], [1, 1]], \
    "get_empty does not return the right result"

mr.add_result(Q(v1), matrix_element=(1, 1))
mr.add_result(Q(v2), matrix_element=(1, 1))

assert np.all(mr.alpha == 0.1), "alpha values incorrect"
assert mr.alpha.shape == (2,), "alpha shape incorrect"
assert np.all(mr._n_alphas == [[0, 0], [0, 2]]), \
    "no. of alpha values incorrect"
for qtyname in ("chi2", "S", "Q"):
    qty = getattr(mr, qtyname)
    assert qty.shape == (2, 2, 2), "{} shape incorrect".format(qtyname)
assert mr.A.shape == (2, 2, 2, 100), "A shape incorrect"
assert mr.v.shape == (2, 2, 2, 48), "v shape incorrect"

assert mr.run_times.shape == (2, 2, 2), "run_times has wrong shape"
assert mr.omega.shape == (100,), "omega shape incorrect"

######################################################################
mr = MaxEntResult(matrix_structure=(2, 2))
mr.add_result(Q(v1), matrix_element=(0, 0))
mr.add_result(Q(v2), matrix_element=(1, 1))
mr.add_result(Q(v1), matrix_element=(0, 1))
mr.add_result(Q(v2), matrix_element=(1, 0))
assert mr.alpha == [0.1], "alphas incorrect"

for qtyname in ("chi2", "S", "Q"):
    qty = getattr(mr, qtyname)
    assert qty.shape == (2, 2, 1), "{} shape incorrect".format(qtyname)
assert mr.A.shape == (2, 2, 1, 100), "A shape incorrect"
assert mr.v.shape == (2, 2, 1, 48), "v shape incorrect"
assert mr.omega.shape == (100,), "omega shape incorrect"

assert mr.run_times.shape == (2, 2, 1), "run_times has wrong shape"

assert mr.run_time_total.shape == (2, 2), "run_time_total shape incorrect"
