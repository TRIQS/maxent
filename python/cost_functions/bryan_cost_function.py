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
import numpy as np
from .cost_function import CostFunction
from ..functions import cached


class BryanCostFunction(CostFunction):
    r""" The usual MaxEnt cost function

    This cost function contains some simplifications that are only
    valid if the normal :math:`\chi^2` and :math:`S` are used, i.e.,
    it does only work for diagonal entries of the spectral function.
    For a general cost function, use :py:class:`.MaxEntCostFunction`.

    The expression for the cost function is

    .. math::

        Q_\alpha(v) = \frac12 \chi^2(H(v)) \eta - \alpha S(H(v)),

    where :math:`\eta` is an additional factor for the :math:`\chi^2` term,
    which can be given as ``chi2_factor`` (default: ``1.0``).

    The derivatives ``d`` and ``dd`` do not actually give the derivatives
    of the cost function, but a simplified version that is obtained when
    multiplying from the left by :math:`V^\dagger`.

    Parameters
    ----------
    chi2_factor : float
        an additional factor for the :math:`\chi^2` term of the cost function
        (default: 1.0)
    """

    def __init__(self, chi2_factor=1.0):
        super(BryanCostFunction, self).__init__(chi2_factor=chi2_factor)

    @cached
    def f(self, v):
        r""" Calculate the function to be minimized, :math:`Q_{\alpha}(v)`.

        Parameters
        ==========
        v : array
            vector in singular space giving the solution
        """

        H = self.H_of_v.f(v)
        chi2 = self.chi2.f(H)
        S = self.S.f(H)

        Q = 0.5 * chi2 * self.chi2_factor - self._alpha * S
        return Q

    @cached
    def dH(self, v):
        r""" Calculate the derivative of the function with respect to H"""

        H = self.H_of_v.f(v)
        dchi2 = self.chi2.d(H)
        dS = self.S.d(H)

        return 0.5 * dchi2 * self.chi2_factor - self._alpha * dS

    @cached
    def d(self, v):
        r""" Calculate the derivative of the function to be minimized

        Parameters
        ==========
        v : array
            vector in singular space giving the solution
        """

        H = self.H_of_v.f(v)

        # dchi2 / d(KA)
        dchi2 = 2 * (np.dot(self.K.K, H) - self.G) / self.err**2

        ret = self.chi2.K.S * \
            np.dot(self.chi2.K.U.conjugate().transpose(), 0.5 * dchi2 * self.chi2_factor)

        return -(-ret - self._alpha * v)

    @cached
    def ddH(self, v):
        r""" Calculate the 2nd derivative of the function with respect to H"""

        H = self.H_of_v.f(v)
        ddchi2 = self.chi2.dd(H)
        ddS = self.S.dd(H)

        return 0.5 * ddchi2 * self.chi2_factor - self._alpha * ddS

    @cached
    def dd(self, v):
        r""" Calculate the 2nd derivative of the function to be minimized

        Parameters
        ==========
        v : array
            vector in singular space giving the solution
        """

        H = self.H_of_v.f(v)
        ddchi2 = self.chi2.dd(H)
        ret = np.dot(self.chi2.K.V.conjugate().transpose(), ddchi2)
        ret = np.einsum('ij,j,jk->ik', ret, H, self.chi2.K.V)
        return 0.5 * ret * self.chi2_factor

    def get_H_of_v(self):
        return self._H_of_v

    def set_H_of_v(self, H_of_v, update_Q=True):
        raise NotImplementedError('Cannot change H_of_v in BryanCostFunction.')

    H_of_v = property(get_H_of_v, set_H_of_v)

    def get_A_of_H(self):
        return self._A_of_H

    def set_A_of_H(self, A_of_H, update_Q=True):
        raise NotImplementedError('Cannot change A_of_H in BryanCostFunction.')

    A_of_H = property(get_A_of_H, set_A_of_H)
