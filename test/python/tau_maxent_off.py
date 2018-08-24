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
from triqs_maxent.tau_maxent import *
from triqs_maxent.triqs_support import *
if if_triqs_1():
    from pytriqs.gf.local import *
elif if_triqs_2():
    from pytriqs.gf import *
import numpy as np
import copy

# Create 2x2 test GfImFreq
G_iw = GfImFreq(beta=40, indices=[0, 1], n_points=100)
G_iw[0, 0] << SemiCircular(0.2)
G_iw[1, 1] << SemiCircular(0.5)

# Rotate
theta = np.pi / 8.0
R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
G_iw_rot = copy.deepcopy(G_iw)
G_iw_rot.from_L_G_R(R, G_iw, R.transpose())

# TauMaxEnt for the 0,1 element
tm_0_1 = TauMaxEnt(cost_function='PlusMinus')
tm_0_1.set_G_iw(G_iw_rot[0, 1])
np.random.seed(666)
tm_0_1.set_G_tau_data(tm_0_1.tau, tm_0_1.G + 1.e-6 *
                      np.random.randn(len(tm_0_1.G)))
tm_0_1.maxent_loop.cost_function.d_dv = False
tm_0_1.alpha_mesh = LogAlphaMesh(alpha_min=0.5, alpha_max=500, n_points=6)
tm_0_1.set_error(1.e-6)

# TauMaxEnt for the 1,0 element
tm_1_0 = TauMaxEnt(cost_function='PlusMinus')
tm_1_0.set_G_iw(G_iw_rot[0, 1])
np.random.seed(7405926)
tm_1_0.set_G_tau_data(tm_1_0.tau, tm_1_0.G + 1.e-6 *
                      np.random.randn(len(tm_1_0.G)))
tm_1_0.maxent_loop.cost_function.d_dv = False
tm_1_0.alpha_mesh = LogAlphaMesh(alpha_min=0.5, alpha_max=500, n_points=6)
tm_1_0.set_error(1.e-6)

assert np.max(np.abs(TauKernel(tm_0_1.tau, tm_0_1.omega).K -
                     tm_0_1.K.K)) < 1.e-14, "the kernel is highly suspicious"

# Checking if tm_1_0 and tm_0_1 are the same - as long as the input G is the same this
# should be always fulfilled.


numpy_assert = lambda a, b, d: np.testing.assert_almost_equal(a, b, decimal=d)


numpy_assert_rtol = lambda a, b, r: np.testing.assert_allclose(a, b, rtol=r)


numpy_assert(tm_1_0.alpha_mesh, tm_0_1.alpha_mesh, 13)
numpy_assert(tm_1_0.data_variable, tm_0_1.data_variable, 13)
numpy_assert(tm_1_0.err, tm_0_1.err, 13)
numpy_assert(tm_1_0.omega, tm_0_1.omega, 13)
numpy_assert(tm_1_0.D.D, tm_0_1.D.D, 13)
numpy_assert(tm_1_0.H_of_v.D.D, tm_0_1.H_of_v.D.D, 13)
numpy_assert(tm_1_0.H_of_v.K.V, tm_0_1.H_of_v.K.V, 13)
numpy_assert(tm_1_0.K.K, tm_0_1.K.K, 13)
numpy_assert(tm_1_0.K.U, tm_0_1.K.U, 13)
numpy_assert(tm_1_0.K.S, tm_0_1.K.S, 13)
numpy_assert(tm_1_0.K.V, tm_0_1.K.V, 13)
omega = HyperbolicOmegaMesh(omega_min=-10, omega_max=10, n_points=100)
random_A = np.random.rand(len(omega))
numpy_assert(tm_1_0.S(random_A).f(), tm_0_1.S(random_A).f(), 13)
# take noise into account
numpy_assert(tm_1_0.chi2(random_A).f(), tm_0_1.chi2(random_A).f(), -10)
numpy_assert(tm_1_0.G, tm_0_1.G, 5)

result1 = tm_1_0.run()
result2 = tm_0_1.run()

# Check the results. We cannot be too strict on them - it is MaxEnt after all.
# For equal seeds equal the results would be the same up to machine precision.

for field in result1._all_fields:
    if field == 'analyzer_results':
        for key in result1.analyzer_results:
            numpy_assert(
                result1.analyzer_results[key]['A_out'],
                result2.analyzer_results[key]['A_out'],
                2)
    elif field.startswith('run_time'):
        # run times will never be equal
        pass
    elif field in ('matrix_structure', 'effective_matrix_structure'):
        assert getattr(result1, field) == getattr(
            result2, field), "matrix structure not equal"
    elif field == 'v':
        continue
    elif field == 'A' or field == 'H':
        continue
    elif field == 'G':
        continue
    elif field == 'G_rec':
        continue
    elif field == 'G_orig':
        continue
    elif isinstance(getattr(result1, field), str):
        assert getattr(result1, field) == getattr(result2, field),\
            "values not equal: {} = {} != {}".format(field,
                                                     getattr(result1, field),
                                                     getattr(result2, field))
    else:
        numpy_assert_rtol(getattr(result1, field),
                          getattr(result2, field), 0.2)
