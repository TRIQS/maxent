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
from triqs_maxent.functions import cached, Entropy,\
    PlusMinusEntropy, safelog
from triqs_maxent.omega_meshes import LinearOmegaMesh
from triqs_maxent.default_models import DataDefaultModel


class ComplicatedPlusMinusEntropy(Entropy):
    """ the PM entropy """
    @cached
    def f(self, A):
        return np.sum((-2.0 * self.D.D - (-A / 2.0 + np.sqrt(A**2.0 + 4 * self.D.D**2.0) / 2.0) * safelog((-A / 2.0 + np.sqrt(A**2.0 + 4 * self.D.D**2.0) / 2.0) / self.D.D) - (
            A / 2.0 + np.sqrt(A**2.0 + 4 * self.D.D**2.0) / 2.0) * safelog((A / 2.0 + np.sqrt(A**2.0 + 4 * self.D.D**2.0) / 2.0) / self.D.D) + np.sqrt(A**2.0 + 4 * self.D.D**2.0)))

    @cached
    def d(self, A):
        return (-(A / (2.0 * np.sqrt(A**2.0 + 4 * self.D.D**2.0)) - 1 / 2.0) * safelog((-A / 2.0 + np.sqrt(A**2.0 + 4 * self.D.D**2.0) / 2.0) / self.D.D) -
                (A / (2.0 * np.sqrt(A**2.0 + 4 * self.D.D**2.0)) + 1 / 2.0) * safelog((A / 2.0 + np.sqrt(A**2.0 + 4 * self.D.D**2.0) / 2.0) / self.D.D))

    @cached
    def dd(self, A):
        return np.diag((((A**2.0 / (A**2.0 + 4 * self.D.D**2.0) - 1) * safelog(-(A - np.sqrt(A**2.0 + 4 * self.D.D**2.0)) / (2.0 * self.D.D)) / np.sqrt(A**2.0 + 4 * self.D.D**2.0) + (A**2.0 / (A**2.0 + 4 * self.D.D**2.0) - 1) * safelog((A + np.sqrt(A**2.0 + 4 * self.D.D**2.0)) / \
                       (2.0 * self.D.D)) / np.sqrt(A**2.0 + 4 * self.D.D**2.0) - (A / np.sqrt(A**2.0 + 4 * self.D.D**2.0) + 1)**2.0 / (A + np.sqrt(A**2.0 + 4 * self.D.D**2.0)) + (A / np.sqrt(A**2.0 + 4 * self.D.D**2.0) - 1)**2.0 / (A - np.sqrt(A**2.0 + 4 * self.D.D**2.0))) / 2.0))


w = LinearOmegaMesh(-10, 10, 101)
D = DataDefaultModel(0 * w + 0.9, w)
Entropy1 = ComplicatedPlusMinusEntropy(D=D)
Entropy2 = PlusMinusEntropy(D=D)
np.random.seed(6666)
A = np.random.rand(len(w))
S1 = Entropy1(A)
S2 = Entropy2(A)
diff = np.abs(S1.f() - S2.f())
assert diff < 1.e-14, "value of f is different " + str(diff)
diff = np.max(np.abs(S1.d() - S2.d()))
assert diff < 1.e-14, "value of d is different " + str(diff)
diff = np.max(np.abs(S1.dd() - S2.dd()))
assert diff < 1.e-14, "value of dd is different " + str(diff)
