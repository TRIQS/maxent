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
from triqs_maxent import *
import matplotlib.pyplot as plt

"""In this test we check if a function evaluated on different
omega-grids gives the same chi2 and S."""

# to make it reproducible
np.random.seed(658436166)

beta = 40
tau = np.linspace(0, beta, 100)

omega = LinearOmegaMesh(omega_min=-10, omega_max=10, n_points=100)
K = TauKernel(tau=tau, omega=omega, beta=beta)
A = np.exp(-omega**2)
A /= np.trapz(A, omega)
G = np.dot(K.K_delta, A)
err = np.ones(len(G))


omega_grids = [
    HyperbolicOmegaMesh(omega_min=-10, omega_max=10, n_points=1000),
    LinearOmegaMesh(omega_min=-10, omega_max=10, n_points=1000)]

for test_A in [lambda omega: omega / omega, lambda omega: np.exp(-omega**2)]:
    c2 = []
    s = []
    for iom in xrange(len(omega_grids)):
        omega = omega_grids[iom]
        K = TauKernel(tau=tau, omega=omega, beta=beta)
        # here we construct the G(tau)
        A = test_A(omega)
        A /= np.trapz(A, omega)
        D = FlatDefaultModel(omega=omega)
        chi2 = NormalChi2(K=K, G=G, err=err)
        S = NormalEntropy(D=D)
        c2.append(chi2.f(A * omega.delta))
        s.append(S.f(A * omega.delta))
    assert abs(c2[1] - c2[0]) < 1.e-4, "chi2 not equal"
    assert abs(s[1] - s[0]) < 1.e-4, "S not equal"
