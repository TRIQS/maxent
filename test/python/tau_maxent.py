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
import matplotlib.pyplot as plt
import copy

np.random.seed(9)

tm = TauMaxEnt(probability='normal')
if if_no_triqs():
    tm.set_G_tau_file('g_tau_semicircular.dat')
else:
    G_iw = GfImFreq(beta=40, indices=[0], n_points=100)
    G_iw << SemiCircular(1)
    tm.set_G_iw(G_iw)
# this just adds some artificial noise, usually we wouldn't do that
tm.set_G_tau_data(tm.tau, tm.G + 1.e-3 * np.random.randn(len(tm.G)))
# use a thinner alpha mesh for test purposes
tm.alpha_mesh = LogAlphaMesh(alpha_min=0.08, n_points=5)
tm.omega = HyperbolicOmegaMesh(omega_min=-10, omega_max=10, n_points=200)
tm.set_error(1.e-3)

assert np.max(np.abs(TauKernel(tm.tau, tm.omega).K - tm.K.K)) < 1.e-14, \
    "the kernel is highly suspicious"

# here, we benchmark the TauKernel, which allows the user to use a
# standard version of MaxEnt quite using a simple interface, to the
# more difficult setup using the MaxEntLoop

# just to trigger SVD
tm.K.S

beta = 40
tau = tm.tau
omega = HyperbolicOmegaMesh(omega_min=-10, omega_max=10, n_points=200)
K = copy.deepcopy(tm.K)
# here we construct the G(tau)
G = tm.G
err = 1.e-3 * np.ones(len(G))
D = FlatDefaultModel(omega=omega)
chi2 = NormalChi2(K=K, G=G, err=err)
S = NormalEntropy(D=D)
H_of_v = NormalH_of_v(D=D, K=K)
Q = MaxEntCostFunction(chi2=chi2, S=S, H_of_v=H_of_v)
logtaker = Logtaker()
minimizer = LevenbergMinimizer()
alpha_values = LogAlphaMesh(alpha_min=0.08, n_points=5)
ml = MaxEntLoop(cost_function=Q,
                minimizer=minimizer,
                alpha_mesh=alpha_values,
                logtaker=logtaker,
                probability='normal')


numpy_assert = lambda a, b, d=13: np.testing.assert_almost_equal(
    a, b, decimal=d)
numpy_assert(ml.G, tm.maxent_loop.G)
numpy_assert(ml.alpha_mesh, tm.maxent_loop.alpha_mesh)
numpy_assert(ml.data_variable, tm.maxent_loop.data_variable)
numpy_assert(ml.err, tm.maxent_loop.err)
numpy_assert(ml.omega, tm.maxent_loop.omega)
numpy_assert(ml.D.D, tm.maxent_loop.D.D)
numpy_assert(ml.H_of_v.D.D, tm.maxent_loop.H_of_v.D.D)
numpy_assert(ml.H_of_v.K.V, tm.maxent_loop.H_of_v.K.V)
numpy_assert(ml.K.K, tm.maxent_loop.K.K)
numpy_assert(ml.K.U, tm.maxent_loop.K.U)
numpy_assert(ml.K.S, tm.maxent_loop.K.S)
numpy_assert(ml.K.V, tm.maxent_loop.K.V)
random_A = np.random.rand(len(omega))
numpy_assert(ml.chi2(random_A).f(), tm.maxent_loop.chi2(random_A).f())
numpy_assert(ml.chi2(random_A).d(), tm.maxent_loop.chi2(random_A).d())
numpy_assert(ml.chi2(random_A).dd(), tm.maxent_loop.chi2(random_A).dd())
numpy_assert(ml.S(random_A).f(), tm.maxent_loop.S(random_A).f())
numpy_assert(ml.S(random_A).d(), tm.maxent_loop.S(random_A).d())
numpy_assert(ml.S(random_A).dd(), tm.maxent_loop.S(random_A).dd())
random_v = np.random.rand(len(ml.K.S))
numpy_assert(ml.H_of_v(random_v).f(), tm.maxent_loop.H_of_v(random_v).f())
numpy_assert(ml.H_of_v(random_v).d(), tm.maxent_loop.H_of_v(random_v).d())
numpy_assert(ml.H_of_v(random_v).dd(), tm.maxent_loop.H_of_v(random_v).dd())

result1 = ml.run()
result2 = tm.run()

assert np.all(np.max(np.abs(result1.A_out - result2.A_out))
              < 1.e-14), "A_out different"
assert np.all(result1.A_out ==
              result1.analyzer_results['LineFitAnalyzer']['A_out']), \
    "A_out not from the LineFitAnalyzer"

for field in result1._all_fields:
    if field == 'analyzer_results':
        for key in result1.analyzer_results:
            numpy_assert(result1.analyzer_results[key]['A_out'],
                         result2.analyzer_results[key]['A_out'])
    elif field.startswith('run_time'):
        # run times will never be equal
        pass
    elif field in ('matrix_structure', 'effective_matrix_structure'):
        assert getattr(result1, field) == getattr(result2, field), \
            "matrix structure not equal"
    elif isinstance(getattr(result1, field), str):
        assert getattr(result1, field) == getattr(result2, field), \
            "values not equal: {} = {} != {}".format(field,
                                                     getattr(result1, field),
                                                     getattr(result2, field))
    else:
        numpy_assert(getattr(result1, field), getattr(result2, field))

# check probability
numpy_assert(result2.probability, [-8476.5281283264321246, -2343.02752795091601, - \
             704.2831835069059707, -280.2662732287696485, -175.305925543312469], 6)
