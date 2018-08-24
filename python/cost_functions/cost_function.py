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
from ..functions import DoublyDerivableFunction, cached, \
    NormalChi2, NormalEntropy, NormalH_of_v, IdentityA_of_H


class CostFunction(DoublyDerivableFunction):
    r""" The base class for the function to be minimized

    Parameters
    ----------
    chi2 : Chi2
        the expression for chi2 (e.g. :py:class:`.NormalChi2`)
    S : Entropy
        the expression for S (e.g. :py:class:`.NormalEntropy`)
    H_of_v : H_of_v
        the expression for :math:`H(v)` (e.g. :py:class:`.NormalH_of_v`)
    A_of_H : A_of_H
        the expression for :math:`A(H)` (e.g. :py:class:`.IdentityA_of_H`)
    chi2_factor : float
        an additional factor for the :math:`\chi^2` term of the cost function
        (default: 1.0)
    """

    def __init__(self,
                 chi2=None,
                 S=None,
                 H_of_v=None,
                 A_of_H=None,
                 chi2_factor=1.0):
        self._chi2 = chi2
        self._S = S
        self._H_of_v = H_of_v
        self._A_of_H = A_of_H
        self.chi2_factor = chi2_factor
        if self._chi2 is None:
            self._chi2 = NormalChi2()
        if self._S is None:
            self._S = NormalEntropy()
        if self._H_of_v is None:
            self._H_of_v = NormalH_of_v()
        if self._A_of_H is None:
            omega = None
            try:
                omega = self._chi2.omega
            except:
                pass
            self._A_of_H = IdentityA_of_H(omega)
        self._alpha = None

    def set_alpha(self, alpha):
        """ Set the hyper-parameter. """
        self._alpha = alpha

    def __call__(self, x):
        ret = super(CostFunction, self).__call__(x)
        ret._H_of_v = ret.H_of_v(ret._x)
        H = ret._H_of_v.f()
        ret._A_of_H = ret.A_of_H(H)
        # we explicitly set it to the same object
        ret._H_of_v._x = ret._x
        ret._A_of_H._x = H
        ret._chi2 = ret.chi2(H)
        ret._chi2._x = H
        ret._S = ret.S(H)
        ret._S._x = H
        return ret

    def f(self, v):
        r""" Calculate the function to be minimized, :math:`Q_{\alpha}(v)`.

        Parameters
        ==========
        v : array
            vector in singular space giving the solution;
            if None, the last supplied value should be reused
        """

        raise NotImplementedError('Please use a subclass of CostFunction.')

    def d(self, v):
        r""" Calculate the derivative of the function to be minimized

        Parameters
        ==========
        v : array
            vector in singular space giving the solution
            if None, the last supplied value should be reused
        """

        raise NotImplementedError('Please use a subclass of CostFunction.')

    def dd(self, v):
        r""" Calculate the 2nd derivative of the function to be minimized

        Parameters
        ==========
        v : array
            vector in singular space giving the solution
            if None, the last supplied value should be reused
        """

        raise NotImplementedError('Please use a subclass of CostFunction.')

    def parameter_change(self):
        pass

    ####### Helper functions #######

    def get_K(self):
        return self.chi2.K

    def set_K(self, K, update_chi2=True, update_H_of_v=True, update_Q=True):
        self.chi2.set_K(K, update_chi2=update_chi2)
        self.H_of_v.set_K(K, update_H_of_v=update_H_of_v)
        if update_Q:
            self.parameter_change()

    K = property(get_K, set_K)

    def get_G(self):
        return self.chi2.G

    def set_G(self, G, update_chi2=True, update_Q=True):
        self.chi2.set_G(G, update_chi2=update_chi2)
        if update_Q:
            self.parameter_change()

    G = property(get_G, set_G)

    def get_err(self):
        return self.chi2.err

    def set_err(self, err, update_chi2=True, update_Q=True):
        self.chi2.set_err(err, update_chi2=update_chi2)
        if update_Q:
            self.parameter_change()

    err = property(get_err, set_err)

    def get_omega(self):
        return self.chi2.K.omega

    def set_omega(self,
                  omega,
                  update_K=True,
                  update_chi2=True,
                  update_D=True,
                  update_S=True,
                  update_H_of_v=True,
                  update_A_of_H=True,
                  update_Q=True):
        self.chi2.set_omega(omega, update_K=update_K, update_chi2=update_chi2)
        if update_K:
            # we use update_H_of_v = False here, because it gets updated
            # anyhow when omega is set later on
            self.H_of_v.set_K(self.K, update_H_of_v=False)
        self.S.set_omega(omega, update_D=update_D, update_S=update_S)
        self.H_of_v.set_omega(omega,
                              update_D=update_D,
                              update_H_of_v=update_H_of_v)
        self.A_of_H.set_omega(omega, update_A_of_H=update_A_of_H)
        if update_Q:
            self.parameter_change()

    omega = property(get_omega, set_omega)

    # ``data_variable`` is a name we use for tau
    # because in general we might want to use a different kernel
    # and then the supplied data is not G(tau) but, eg, G(iw)
    def get_data_variable(self):
        return self.chi2.K.data_variable

    def set_data_variable(self,
                          data_variable,
                          update_K=True,
                          update_chi2=True, update_Q=True,
                          update_H_of_v=True):
        self.chi2.set_data_variable(data_variable, update_K=update_K,
                                    update_chi2=update_chi2)
        if update_K:
            self.H_of_v.set_K(self.K, update_H_of_v=update_H_of_v)
        if update_Q:
            self.parameter_change()

    data_variable = property(get_data_variable, set_data_variable)

    def get_D(self):
        return self.S.D

    def set_D(self,
              D,
              update_S=True,
              update_H_of_v=True,
              update_Q=True,
              update_A_of_H=True):
        self.S.set_D(D, update_S=update_S)
        self.H_of_v.set_D(D, update_H_of_v=update_H_of_v)
        self.A_of_H.set_omega(D.omega, update_A_of_H=update_A_of_H)
        if update_Q:
            self.parameter_change()

    D = property(get_D, set_D)

    def get_chi2(self):
        return self._chi2

    def set_chi2(self, chi2, update_Q=True):
        self._chi2 = chi2
        if update_Q:
            self.parameter_change()

    chi2 = property(get_chi2, set_chi2)

    def get_S(self):
        return self._S

    def set_S(self, S, update_Q=True):
        self._S = S
        if update_Q:
            self.parameter_change()

    S = property(get_S, set_S)

    def get_H_of_v(self):
        return self._H_of_v

    def set_H_of_v(self, H_of_v, update_Q=True):
        self._H_of_v = H_of_v
        if update_Q:
            self.parameter_change()

    H_of_v = property(get_H_of_v, set_H_of_v)

    def get_A_of_H(self):
        return self._A_of_H

    def set_A_of_H(self, A_of_H, update_Q=True):
        self._A_of_H = A_of_H
        if update_Q:
            self.parameter_change()

    A_of_H = property(get_A_of_H, set_A_of_H)

    @property
    def G_orig(self):
        if hasattr(self, "_G_orig"):
            return self._G_orig
        else:
            return self.G
