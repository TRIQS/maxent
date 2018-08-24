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
from triqs_maxent.functions import DoublyDerivableFunction, cached
from triqs_maxent.minimizers import *


class Circle(DoublyDerivableFunction):

    def __init__(self, x, y):
        self.center = np.array([x, y])

    @cached
    def f(self, v):
        x, y = v
        return np.sqrt((x - self.center[0])**2 + (y - self.center[1])**2)

    @cached
    def d(self, v):
        return (v - self.center) / self.f(v)

    @cached
    def dd(self, v):
        x, y = v
        ret = np.empty((2, 2))
        ret[0, 0] = (y - self.center[1])**2
        ret[0, 1] = -(x - self.center[0]) * (y - self.center[1])
        ret[1, 0] = ret[0, 1]
        ret[1, 1] = (x - self.center[0])**2
        ret /= self.f(v)**3
        return ret

center = (3.0, 2.0)

circle = Circle(*center)

lm = LevenbergMinimizer(convergence=FunctionChangeConvergenceMethod(1.e-10),
                        verbose_callback=print)
min_v = lm.minimize(circle, np.array([200.0, 150.0]))
diff = np.max(np.abs(min_v - center))
assert diff < 1.e-9, \
    "wrong minimum found, {} should be 0".format(diff)
