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
from .minimizer import Minimizer
from .convergence_methods import *


class LevenbergMinimizer(Minimizer):
    r""" The Levenberg minimization algorithm.

    The task of this algorithm is to minimize a function :math:`Q`
    with respect to a quantity :math:`H`. That is equivalent to searching
    a solution to :math:`f := \partial Q/\partial H = 0`.
    We assume that the equation is parametrized by a solution vector
    :math:`v` (i.e., we are looking for a solution in :math:`H(v)`).
    We then calculate the matrix :math:`J = \frac{\partial}{\partial v} \frac{\partial Q}{\partial H}`.

    Depending on the settings of the parameters, the solution is searched
    for in the following way:

        * ``J_squared=False, marquardt = False``:
          :math:`(J + \mu 1)\delta v = f`
        * ``J_squared=False, marquardt = True``:
          :math:`(J + \mathrm{diag} J)\delta v = f`
        * ``J_squared=True, marquardt = False``:
          :math:`(J^TJ + \mu 1)\delta v = f`
        * ``J_squared=True, marquardt = True``:
          :math:`(J^TJ + \mathrm{diag} J^T J)\delta v = f`

    Then, :math:`v` is updated by subtracting :math:`\delta v`.

    The step length is determined by the damping parameter :math:`\mu`,
    which is chosen so that :math:`Q` is minimal.

    Parameters
    ==========
    convergence : :py:mod:`ConvergenceMethod <.convergence_methods>`
        method to check convergence; the default is
        ``MaxDerivativeConvergenceMethod(1.e-4) | RelativeFunctionChangeConvergenceMethod(1.e-16)``
    maxiter : int
        the maximum number of iterations
    miniter : int
        the minimum number of iterations
    J_squared : bool
        if ``True``, the algorithm solves
        :math:`(J^TJ +\mu 1) \delta v = J^T f`, else
        :math:`(J +\mu 1) \delta v = f`.
    marquardt : bool
        if ``True``, the algorithm uses
        :math:`\mathrm{diag}(J)` or :math:`\mathrm{diag}(J^T J)` (depending on the
        parameter ``J_squared``), instead of the identity matrix.
    mu0 : float
        the initial value of the Levenberg damping parameter
    nu : float
        the relative increase/decrease of mu when necessary
    max_mu : float
        the maximum number of mu, to prevent infinite loops
    verbose_callback : function
        a function used to print verbosity information
        e.g., the print function

    Attributes
    ----------
    n_iter : int
        the total number of iterations performed since the creation
        of the class
    n_iter_last : int
        the number of iterations performed when ``minimize`` was last called
    converged : bool
        whether the algorithm converged in fewer than ``maxiter`` loops
        when ``minimze`` was last called
    """

    def __init__(self,
                 convergence=None,
                 maxiter=1000,
                 miniter=0,
                 J_squared=False,
                 marquardt=False,
                 mu0=1.e-18,
                 nu=1.3,
                 max_mu=1.e20,
                 verbose_callback=None):

        if convergence is None:
            self.convergence = OrConvergenceMethod(
                MaxDerivativeConvergenceMethod(1.e-4),
                RelativeFunctionChangeConvergenceMethod(1.e-16))
        else:
            self.convergence = convergence
        self.maxiter = maxiter
        self.miniter = miniter
        self.J_squared = J_squared
        self.marquardt = marquardt
        self.mu0 = mu0
        self.nu = nu
        self.max_mu = max_mu
        self.verbose_callback = verbose_callback

        # the number of iterations performed in total
        self.n_iter = 0
        # the number of iterations performed in last call
        self.n_iter_last = 0

    def minimize(self, function, v0):
        """ Perform the minimization.

        Parameters
        ==========
        function : DoublyDerivableFunction
            the function to be minimized
        v0 : array
            the initial function argument :math:`v`

        Returns
        =======
        v : array
            the vector :math:`v` at the minimum
        """

        if self.nu <= 1.0:
            raise Exception('If nu <= 1, there will be an infinite loop.')

        # obviously, we are not converged yet
        self.converged = False
        # initialize the damping parameter
        mu = self.mu0
        # initialize v
        v = v0

        # we set the initial function value
        func_val = function(v)
        Q1 = func_val.f()
        Q0 = np.nan

        # the main iteration
        for i in xrange(self.maxiter):
            # here, f is the Jacobian of the function wrt v
            f = func_val.d()
            # the Jacobian of f
            J = func_val.dd()

            # check convergence
            conv_status, self.converged = self.convergence(func_val,
                                                           v, Q0=Q0, Q1=Q1)

            if self.verbose_callback is not None:
                msg = '{:6d} '.format(i + 1)
                msg += "Q: {:12.6e}, ".format(func_val.f(v))
                msg += 'max_f: {:12.6e}, '.format(np.max(np.abs(f)))
                msg += 'conv: {:12.6e}'.format(conv_status)
                self.verbose_callback(msg)

            # declare converged
            if self.converged and i >= self.miniter:
                break

            # change J and f by multiplying the equation with J^T
            if self.J_squared:
                f = np.dot(J.transpose(), f)
                J = np.dot(J.transpose(), J)

            # define the identity matrix
            if self.marquardt:
                Id = np.diag(np.diag(J))
            else:
                Id = np.eye(len(J))

            # adjusting mu until the function value Q is minimal

            # let Q0 be the last Q-value
            Q0 = Q1
            # the change in v to the next iteration
            dv = np.linalg.solve(J + mu * Id, f)

            # in the following, there might be some unphysical values of
            # v that are supplied to the function. Turn off warnings.
            old_seterr = np.seterr(all='ignore')

            # first, we check what the Q of the new iteration will be
            Q1 = function(v - dv).f()
            # as Q has to decrease in every iteration, we increase mu
            # until this is the case (Q0 is the function value from the
            # last iteration)
            while (Q1 > Q0 or np.isnan(Q1)) and mu < self.max_mu:
                mu *= self.nu
                dv = np.linalg.solve(J + mu * Id, f)
                Q1 = function(v - dv).f()

            # we check what increasing mu does to Q
            dv2 = np.linalg.solve(J + self.nu * mu * Id, f)
            Q2 = function(v - dv2).f()

            # if Q decreased when mu was increased, we will try to
            # further decrease Q by further increasing mu
            if Q2 < Q1:
                nuf = self.nu
                mu *= self.nu
                Q2 = Q1
                dvnew = dv2
            # if Q increased when mu was increased, we will try to
            # further increase Q by decreasing mu
            else:
                nuf = 1.0 / self.nu
                mu /= nuf
                dvnew = dv

            Q1 = np.inf  # just to make sure the loop runs at least once
            while (Q2 < Q1 and mu < self.max_mu
                    and mu > self.nu * np.finfo(float).eps):
                Q1 = Q2
                dv = dvnew
                mu *= nuf
                dvnew = np.linalg.solve(J + mu * Id, f)
                Q2 = function(v - dvnew).f()

            # here, we require v to be physical again. Reset the seterr status.
            np.seterr(**old_seterr)

            # update v
            v -= dv
            func_val = function(v)
            # just to be sure that Q1 is accurate (will be used in the next
            # loop)
            Q1 = func_val.f()

        self.n_iter_last = i + 1
        self.n_iter += i + 1

        return v
