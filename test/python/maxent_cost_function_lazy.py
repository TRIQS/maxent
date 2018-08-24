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


from __future__ import absolute_import, print_function, division
import numpy as np
from triqs_maxent.cost_functions import MaxEntCostFunction
from triqs_maxent.functions import NullFunction, IdentityH_of_v
from triqs_maxent.minimizers import *
from triqs_maxent.kernels import *
from triqs_maxent.omega_meshes import *
from triqs_maxent.default_models import DataDefaultModel

# This test might not work under some python environments, as it is
# makes use of the sys.settrace feature. "Its behavior is part of the
# implementation platform, rather than part of the language definition,
# and thus may not be available in all Python implementations."
# (python doc)
# If the test fails due to that problem, feel free to ignore it.

# Here we set up a dummy MaxEntCostFunction
me = MaxEntCostFunction(S=NullFunction(), H_of_v=IdentityH_of_v())
tau = np.linspace(0, 1, 101)
design_matrix = np.ones((len(tau), 3))
design_matrix[:, 0] = tau**2
design_matrix[:, 1] = tau
solution = [1.0, 2.0, 3.0]
data = np.dot(design_matrix, solution)

me.chi2.K = DataKernel(tau, DataOmegaMesh(range(3)), design_matrix)
me.chi2.G = data
me.chi2.err = 1.e-4 * (0 * data + 1)
me.chi2.parameter_change()

me.H_of_v.D = DataDefaultModel(solution, DataOmegaMesh(range(3)))
me.H_of_v.parameter_change()

me.A_of_H.omega = me.H_of_v.D.omega

me.set_alpha(0.0)

# Let's minimize it
print("Searching solution")
lm = LevenbergMinimizer(convergence=MaxDerivativeConvergenceMethod(1.e-7),
                        verbose_callback=print)
min_v = lm.minimize(me, np.ones(len(solution)))
print(min_v)
assert np.max(np.abs(min_v - solution)) < 1.e-15,\
    "Solution not found " + str(min_v)

v = np.random.rand(len(solution))

# now we set up the trace
import sys
called_functions = []


def trace(frame, event, arg):
    if event == "call":
        filename = frame.f_code.co_filename
        if ('function' in filename and
            frame.f_code.co_name != "new_func" and
                not frame.f_code.co_name.startswith('_') and
                not frame.f_code.co_name.startswith('get_')):
            called_functions.append('{}.{}'.format(
                frame.f_locals["self"].__class__.__name__,
                frame.f_code.co_name))
    return trace

# first call the functions directly
sys.settrace(trace)
me.dH(v)
me.dH(v)
sys.settrace(None)
print("\nFunctions called in the first block (each should appear twice):")
print("\n".join(called_functions))
assert len(called_functions) == 2 * len(set(called_functions)),\
    "the number of called functions is not correct"

# then supply a v first, then call the functions
# all results should be cached
called_functions = []
sys.settrace(trace)
me1 = me(v)
me1.d()
me1.dH()
me1.ddH()
me1.dd()
me1.f()
me1.chi2.f()
me1.H_of_v.f()
me1.A_of_H.f()
sys.settrace(None)
print("\nFunctions called in the second block (each should appear just once):")
print("\n".join(called_functions))
assert len(called_functions) == len(set(called_functions)),\
    "functions were called more than once"
