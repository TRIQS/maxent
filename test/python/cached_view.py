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
from triqs_maxent.functions import DoublyDerivableFunction
from triqs_maxent.functions import cached
import copy

nf = 0
nd = 0
ndd = 0
nsq = 0


class Sin(DoublyDerivableFunction):

    @cached
    def f(self, v):
        global nf
        nf += 1
        return np.sin(v)[0]

    @cached
    def d(self, v):
        global nd
        nd += 1
        return np.cos(v)

    @cached
    def dd(self, v):
        global ndd
        ndd += 1
        return -np.diag(np.sin(v))

    @cached
    def square(self, v):
        global nsq
        nsq += 1
        return v**2

    @cached
    def fourth1(self, v):
        return self.square(v) * self.square(v)

    @cached
    def fourth2(self, v):
        return self.square(self.square(v))


sin = Sin()

v = np.random.rand(1,)

# this raises the function evaluation counters
f1 = sin.f(v)
d1 = sin.d(v)
dd1 = sin.dd(v)
assert nf == 1, "wrong number of function evaluations (f) " + str(nf)
assert nd == 1, "wrong number of function evaluations (d) " + str(nd)
assert ndd == 1, "wrong number of function evaluations (dd) " + str(ndd)
# this raises the function evaluation counters
f1 = sin.f(v)
assert nf == 2, "wrong number of function evaluations (f) " + str(nf)
assert nd == 1, "wrong number of function evaluations (d) " + str(nd)
assert ndd == 1, "wrong number of function evaluations (dd) " + str(ndd)

sin1 = sin(v)
# this raises the function evaluation counters
f2 = sin1.f()
d2 = sin1.d()
dd2 = sin1.dd()

# the following two statements should NOT raise the function evaluation
# counters
f2 = sin1.f()
d2 = sin1.d()

assert f1 == f2, "the values of f are different"
assert d1 == d2, "the values of d are different"
assert dd1 == dd2, "the values of dd are different"

assert nf == 3, "wrong number of function evaluations (f) " + str(nf)
assert nd == 2, "wrong number of function evaluations (d) " + str(nd)
assert ndd == 2, "wrong number of function evaluations (dd) " + str(ndd)

# check that using a different v in the function does not change the
# cached value(s)
sin1.f(2 * v)
f3 = sin1.f()
assert f1 == f3, "the values of f are different"
assert nf == 4, "wrong number of function evaluations (f) " + str(nf)

sin1.fourth2()
sin1.fourth2()
sin1.fourth1()
sin1.fourth1()
assert nsq == 2, "wrong number of function evaluations (sq) " + str(nsq)

v[0] += 0.01
sin1.f(v)
assert nf == 5, "wrong number of function evaluations (f) " + str(nf)
