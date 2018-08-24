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


class MaxEntCostFunction(CostFunction):
    r""" The usual MaxEnt cost function

    The expression for the cost function is

    .. math::

        Q_\alpha(v) = \frac12 \chi^2(H(v)) \eta - \alpha S(H(v)),

    where :math:`\eta` is an additional factor for the :math:`\chi^2` term,
    which can be given as ``chi2_factor`` (default: ``1.0``).

    This function implements the usual derivatives ``d`` and ``dd``,
    where by default (i.e., with ``d_dv = False``) ``d`` is
    :math:`\frac{\partial Q_\alpha}{\partial H}` and ``dd`` is
    :math:`\frac{\partial^2 Q_\alpha}{\partial H \partial v}`.
    If ``dA_projection>0``, the derivatives are projected into singular
    space; either by multiplying from the left by :math:`V^\dagger` (i.e., the
    matrix with the right singular vectors, if ``dA_projection=1``) or by
    :math:`\partial H/\partial v` (if ``dA_projection=2``).

    If ``d_dv = True``, we have ``d`` giving :math:`\frac{\partial Q_\alpha}{\partial v}`;
    ``dd`` is again the derivative of ``d`` with respect to ``v``.

    This cost function should be used for general :math:`\chi^2` and :math:`S`.
    If you use the normal values, it is possible to use the :py:class:`.BryanCostFunction` instead.

    Parameters
    ----------
    d_dv : bool
        see explanation above
    dA_projection : int (0, 1, 2)
        see explanation above; it is ignored if ``d_dv = True``
    **kwargs
        gets passed to :py:class:`.CostFunction`
    """

    def __init__(self, d_dv=False, dA_projection=2, **kwargs):
        self.d_dv = d_dv
        self.dA_projection = dA_projection
        super(MaxEntCostFunction, self).__init__(**kwargs)

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

        dQ_dH = self.dH(v)

        if self.d_dv:
            dH_dv = self.H_of_v.d(v)
            retval = np.dot(dH_dv.transpose(), dQ_dH)
        else:
            if self.dA_projection == 1:
                retval = np.dot(self.chi2.K.V.conjugate().transpose(), dQ_dH)
            elif self.dA_projection == 2:
                dH_dv = self.H_of_v.d(v)
                retval = np.dot(dH_dv.transpose(), dQ_dH)
            else:
                retval = dQ_dH
        return retval

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

        ddQ_dHdH = self.ddH(v)
        dH_dv = self.H_of_v.d(v)

        if self.d_dv:
            dQ_dH = self.dH(v)
            ddH_dvdv = self.H_of_v.dd(v)
            # we make use of the fact that the Jacobian is symmetric
            retval = (np.dot(np.dot(ddQ_dHdH, dH_dv).transpose(), dH_dv) +
                      np.einsum('k,kij->ij', dQ_dH, ddH_dvdv))
        else:
            if self.dA_projection == 1:
                retval = np.dot(self.chi2.K.V.conjugate().transpose(),
                                np.dot(ddQ_dHdH, dH_dv))
            elif self.dA_projection == 2:
                retval = np.dot(dH_dv.transpose(), np.dot(ddQ_dHdH, dH_dv))
            else:
                retval = np.dot(ddQ_dHdH, dH_dv)
        return retval
