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
from triqs_maxent.minimizers import LevenbergMinimizer, MaxDerivativeConvergenceMethod
from triqs_maxent.functions import DoublyDerivableFunction, cached


class Cube(DoublyDerivableFunction):

    @cached
    def f(self, v):
        return np.sum(v**3)

    @cached
    def d(self, v):
        return 3 * v**2

    @cached
    def dd(self, v):
        return 6 * np.diag(v)

cube = Cube()

for J_squared in [True, False]:
    for marquardt in [True, False]:
        lm = LevenbergMinimizer(
            J_squared=J_squared,
            marquardt=marquardt,
            convergence=MaxDerivativeConvergenceMethod(1.e-10))
        min_v = lm.minimize(cube, np.array([200.0]))
        assert min_v < 1.e-5, \
            """
            wrong minimum found ({}, should be 0)
            Variant: J_squared {}
                     marquardt {}
            """.format(min_v,
                       J_squared,
                       marquardt)
        conv = np.max(np.abs(cube.d(min_v)))
        assert conv < lm.convergence.convergence_criterion, \
            """
            convergence criterion not met
            Variant: J_squared {}
                     marquardt {}
            """.format(J_squared, marquardt)
