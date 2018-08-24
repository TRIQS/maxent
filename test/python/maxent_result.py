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
from pytriqs.archive import HDFArchive
from pytriqs.utility.h5diff import h5diff
from triqs_maxent.maxent_result import _get_empty

# A(v) depends on the SVD result because it is parametrized
# using the matrix K.V. However, K.V is not uniquely defined
# and due to loss of numerical precision for singular values
# close to 10^-16 (we use a threshold of 10^-14 in
# K.reduce_singular_space) the K.V is different on different
# machines.

# Therefore, we perform the test twice. Once, using the
# calculated K.V and a criterion that is not very strict,
# and once using a read-in K.V and a much stricter criterion.

produce_ref = False

for i in range(2):
    # to make it reproducible
    np.random.seed(658436166)

    # first, we construct tau, omega, K
    beta = 40
    tau = np.linspace(0, beta, 100)
    omega = HyperbolicOmegaMesh(omega_min=-10, omega_max=10, n_points=100)
    K = TauKernel(tau=tau, omega=omega, beta=beta)
    K.reduce_singular_space()
    if i == 0 and produce_ref:
        with HDFArchive('maxent_result.ref.h5', 'a') as ar:
            ar['V'] = K._V
    if i == 1:
        with HDFArchive('maxent_result.ref.h5', 'r') as ar:
            K._V = ar['V']

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

    # the cost function
    Q = MaxEntCostFunction(chi2=chi2, S=S, H_of_v=H_of_v)
    Q.set_alpha(0.1)

    # a random singular-space vector
    v = np.random.rand(len(K.S))

    mr = MaxEntResult()
    mr.add_result(Q(v))
    mr.exclude(('analyzer_results', 'probability', 'H'))
    mr.include(('H',))

    if produce_ref:
        with HDFArchive('maxent_result.ref.h5', 'a') as ar:
            ar['result'] = mr.data
    with HDFArchive('maxent_result.out.h5', 'w' if i == 0 else 'a') as ar:
        ar['result{}'.format(i + 1)] = mr.data

# compare the two files
with HDFArchive('maxent_result.out.h5', 'r') as ar1, \
        HDFArchive('maxent_result.ref.h5', 'r') as ar2:
    a1r = ar1.get_raw('result1')
    a2r = ar2.get_raw('result')
    for key in a1r:
        assert key in a2r, "key {} not in ref".format(key)
    for key in a2r:
        assert key in a1r, "key {} not in out".format(key)
        # we ignore the run_time keys because it does not make sense to compare
        # them
        if key.startswith('run_time'):
            continue
        if isinstance(a1r[key], np.ndarray):
            if key == 'data_variable':
                assert np.max( np.abs( a1r[key] - a2r[key])) < 1.e-2,\
                    "{} not equal".format(key)
            else:
                assert np.max( np.abs( a1r[key] / a2r[key] - 1.0)) < 1.e-2,\
                    "{} not equal".format(key)
        else:
            assert a1r[key] == a2r[key], "{} not equal".format(key)

    a1r = ar1.get_raw('result2')
    for key in a1r:
        assert key in a2r, "key {} not in ref".format(key)
    for key in a2r:
        level = 1.e-12
        if key == 'G_rec':
            level = 1.e-4
        assert key in a1r, "key {} not in out".format(key)
        # we ignore the run_time keys because it does not make sense to compare
        # them
        if key.startswith('run_time'):
            continue
        if isinstance(a1r[key], np.ndarray):
            if key == 'data_variable':
                assert np.max(np.abs(a1r[key] - a2r[key])) < level,\
                    "{} not equal".format(key)
            else:
                assert np.max(np.abs(a1r[key] / a2r[key] - 1)) < level,\
                    "{} not equal (difference {})".format(key,
                                                          np.max(np.abs(a1r[key] - a2r[key])))
        else:
            assert a1r[key] == a2r[key], "{} not equal".format(key)

# Test _get_empty
matrix = _get_empty((2, 3, 4))
assert np.array(matrix).shape == (2, 3, 4, 0), "Problem with _get_empty"
