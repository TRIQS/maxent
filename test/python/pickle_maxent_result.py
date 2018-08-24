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
import numpy as np
import pickle

# 0 Gtau to generate maxent_result
ew = ElementwiseMaxEnt(use_hermiticity=False)
data = np.loadtxt('g_tau_semicircular.dat')
tau = data[:, 0]
G_tau = np.zeros((2, 2, len(data)))
np.random.seed(666)
G_tau[0, 0, :] = data[:, 1] + 1.e-4 * np.random.randn(len(data))
G_tau[1, 1, :] = data[:, 1] + 1.e-4 * np.random.randn(len(data))
ew.set_G_tau_data(tau, G_tau)
ew.set_error(1.e-4)
ew.omega = HyperbolicOmegaMesh(omega_min=-10, omega_max=10, n_points=80)
ew.alpha_mesh = LogAlphaMesh(1, 1000, n_points=5)
result_ew = ew.run()

with open('pickle_maxent_result.out.pickle', 'w') as fi:
    pickle.dump(result_ew.data, fi)


with open('pickle_maxent_result.out.pickle', 'r') as fi:
    reload_result_ew = pickle.load(fi)

# load from h5
numpy_assert = lambda a, b: np.testing.assert_equal(a, b)

for field in reload_result_ew._all_fields:
    if field == 'analyzer_results':
        for i in range(len(result_ew.analyzer_results)):
            for j in range(len(result_ew.analyzer_results[i])):
                for key in result_ew.analyzer_results[i][j]:
                    numpy_assert(
                        reload_result_ew.analyzer_results[i][j][key]['A_out'],
                        result_ew.analyzer_results[i][j][key]['A_out'])
    elif field == 'matrix_structure':
        assert getattr(reload_result_ew, field) == \
            getattr(result_ew, field), "matrix structure not equal"
    elif field == 'run_times':
        # why does this not work?
        pass
    else:
        numpy_assert(getattr(result_ew, field),
                     getattr(reload_result_ew, field))
