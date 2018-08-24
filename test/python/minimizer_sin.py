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
from triqs_maxent.minimizers import LevenbergMinimizer
from triqs_maxent.functions import DoublyDerivableFunction, cached


class Sin(DoublyDerivableFunction):

    @cached
    def f(self, v):
        return np.sin(v)[0]

    @cached
    def d(self, v):
        return np.cos(v)

    @cached
    def dd(self, v):
        return -np.diag(np.sin(v))

sin = Sin()

for J_squared in [True, False]:
    for marquardt in [True, False]:
        lm = LevenbergMinimizer(J_squared=J_squared,
                                marquardt=marquardt)
        min_v = lm.minimize(sin, np.array([0.1]))
        assert np.abs(sin.f(min_v) + 1) < 1.e-9, \
            """
            minimum not found
            value is {}
            Variant: J_squared {}
                     marquardt {}""".format(sin.f(min_v),
                                            J_squared,
                                            marquardt)
