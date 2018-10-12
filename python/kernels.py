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


r"""
This module provides kernels for the analytic continuation.
In general, we have :math:`G_i = \sum_j K_{ij} H_j`.
Note that for non-uniform :math:`\omega`-grids, :math:`H = A \Delta\omega`.

At the moment, only the kernel for the continuation of :math:`G(\tau)`
is implemented, i.e., the :py:class:`.TauKernel`.

For any kernel the preblur version can be used by the :py:class:`.PreblurKernel`
(see :ref:`preblur`).
"""

from __future__ import absolute_import, print_function
import numpy as np
from .preblur import *


class KernelSVD(object):
    """ A kernel object with a singular value decomposition

    Parameters
    ----------
    K : array
        the kernel matrix
    """

    def __init__(self, K=None):
        self._U = None
        self._S = None
        self._V = None
        self._K = K
        self._last_threshold = None

    def svd(self):
        """ Perform the SVD if not yet performed

        Usually, this function does not need to be called by the user.

        We have :math:`K = USV^\dagger`.
        """
        if self._U is None:
            self._U, self._S, self._V = np.linalg.svd(self.K,
                                                      full_matrices=False)
            self._V = self._V.transpose()  # due to different conventions
        return (self._U, self._S, self._V)

    @property
    def U(self):
        """ Get the matrix of left-singular vectors of the kernel

        If not performed already, the SVD is done.
        """
        if self._U is None:
            self.svd()
        return self._U

    @property
    def S(self):
        """ Get the vector of singular values of the kernel

        If not performed already, the SVD is done.
        """
        if self._S is None:
            self.svd()
        return self._S

    @property
    def V(self):
        """ Get the matrix of right-singular vectors of the kernel

        If not performed already, the SVD is done.
        """
        if self._V is None:
            self.svd()
        return self._V

    @property
    def K(self):
        """ The actual kernel matrix """
        return self._K

    def reduce_singular_space(self, threshold=1.e-14):
        """ Reduce the singular space

        All singular values smaller than the ``threshold`` are dropped
        from ``S``, ``U``, ``V``.

        Parameters
        ----------
        threshold : float
            the threshold for dropping singular values
        """
        if self._last_threshold is not None:
            if threshold is None or threshold < self._last_threshold:
                self._U = None
                self._V = None
                self._S = None
        self._last_threshold = threshold
        L = np.where(self.S >= threshold)[0]
        self._U = self._U[:, L]
        self._S = self._S[L]
        self._V = self._V[:, L]
        return self


class Kernel(KernelSVD):
    """ The kernel for the analytic continuation
    """

    def __init__(self):
        super(Kernel, self).__init__()
        self.omega = None
        self._T = None

    @property
    def K_delta(self):
        """ The kernel including a :math:`\Delta\omega`.

        Use this to get the reconstructed Green function as
        :math:`G_{rec} = K_{delta} A`.
        Note that it does not get rotated with :py:meth:`.transform`,
        i.e. it gives back the original :math:`G_{rec}`.
        """
        return self._K_delta

    @property
    def data_variable(self):
        raise NotImplemented("Use a subclass of Kernel")

    def parameter_change(self):
        r""" Notify the kernel of a parameter change

        This should be called when, e.g., the data variable (e.g., :math:`\tau`)
        or the omega-mesh is changed.
        """
        self._fill_values()

    def _fill_values(self):
        raise NotImplemented("Use a subclass of Kernel")

    def transform(self, T_):
        """ multiply the kernel from the left with the matrix ``T_``

        ``T_`` is the absolute rotation with respect to the unrotated
        quantities
        """

        if T_ is None:
            if self._T is not None:
                T = self._T.conjugate().transpose()
            else:
                return
        else:
            if self._T is not None:
                T = np.dot(T_, self._T.conjugate().transpose())
            else:
                T = T_
        self._T = T_

        self._U = np.dot(T, self.U)
        self._K = np.dot(T, self._K)


class DataKernel(Kernel):
    r""" A kernel given by a matrix

    Parameters
    ----------
    data_variable : array
        the array of the data variable, i.e. the variable that the
        input-data depends on; typically, one continues :math:`G(\tau)`,
        then this would correspond to the :math:`\tau`-grid.
    omega : OmegaMesh
        the frequency mesh
    K : array
        the kernel matrix
    """

    def __init__(self, data_variable, omega, K):
        super(DataKernel, self).__init__()
        self._data_variable = data_variable
        self.omega = omega
        self._K = K
        self._K_delta = np.einsum('ij,j->ij', self._K, self.omega.delta)

    @property
    def data_variable(self):
        return self._data_variable


class TauKernel(Kernel):
    r""" A kernel for continuing :math:`G(\tau)`

    This kernel is defined as

    .. math::

        K(\tau, \omega) =  - \frac{\exp(-\tau\omega)}{1 + \exp(\beta \omega)}.

    With this, we have

    .. math::

        G(\tau) = \int d\omega\, K(\tau, \omega) A(\omega).

    Parameters
    ----------
    tau : array
        the :math:`\tau`-mesh where the data is given
    omega : OmegaMesh
        the :math:`\omega`-mesh where the spectral function should be
        calculated
    beta : float
        the inverse temperature; if not given, it is taken to be the
        last ``tau``-value
    """

    def __init__(self, tau, omega, beta=None):
        super(TauKernel, self).__init__()
        self.tau = tau
        self.omega = omega
        self.beta = beta
        self._fill_values()

    def _fill_values(self):
        # invalidate U, S, V
        self._U = None
        self._S = None
        self._V = None
        beta = self.beta
        if beta is None:
            beta = self.tau[-1]

        oomega, ttau = np.meshgrid(self.omega, self.tau)
        # we implement two different (mathematically equivalent) expressions
        # for K depending on the sign of omega, for numerical reasons
        L = oomega >= 0.0
        iL = np.where(L)
        nL = np.where(np.logical_not(L))
        self._K = np.empty(oomega.shape)
        self._K[iL] = -np.exp(-oomega[iL] * ttau[iL]) / \
            (np.exp(-beta * oomega[iL]) + 1.0)
        self._K[nL] = -np.exp(oomega[nL] * (beta - ttau[nL])) / \
            (1.0 + np.exp(beta * oomega[nL]))

        # include trapz integration in the kernel
        self._K_delta = np.einsum('ij,j->ij', self._K, self.omega.delta)

        T = self._T
        self._T = None
        # if self._T is set, the kernel should be transformed
        self.transform(T)

    @property
    def data_variable(self):
        r""" :math:`\tau` """
        return self.tau

    @data_variable.setter
    def data_variable(self, value):
        self.tau = value


class IOmegaKernel(Kernel):
    r""" A kernel for continuing :math:`G(i\omega)`

    This kernel is defined as

    .. math::

        K(i\omega, \omega) = \frac{1}{i\omega - \omega}

    With this, we have

    .. math::

        G(i\omega) = \int d\omega\, K(i\omega, \omega) A(\omega).

    Parameters
    ----------
    iomega : array
        the :math:`iomega`-mesh where the data is given
        (as a real array, i.e. it is internally multiplied by ``1.0j``)
    omega : OmegaMesh
        the :math:`\omega`-mesh where the spectral function should be
        calculated
    beta : float
        the inverse temperature; if not given, it is taken from the difference
        of the first two :math:`i\omega` values
    """

    def __init__(self, iomega, omega, beta=None):
        super(IOmegaKernel, self).__init__()
        self.iomega = iomega
        self.omega = omega
        self.beta = beta
        self._fill_values()

    def _fill_values(self):
        # invalidate U, S, V
        self._U = None
        self._S = None
        self._V = None
        beta = self.beta
        if beta is None:
            beta = 2 * np.pi / (self.iomega[1] - self.iomega[0])

        oomega, iiomega = np.meshgrid(self.omega, self.iomega)
        self._K = np.empty(oomega.shape)
        self._K = 1.0 / (1.0j * iiomega - oomega)

        # include trapz integration in the kernel
        self._K_delta = np.einsum('ij,j->ij', self._K, self.omega.delta)

        T = self._T
        self._T = None
        # if self._T is set, the kernel should be transformed
        self.transform(T)

    @property
    def data_variable(self):
        r""" :math:`i\omega` """
        return self.iomega

    @data_variable.setter
    def data_variable(self, value):
        self.iomega = value


class PreblurKernel(Kernel):
    """ A kernel for the preblur formalism

    In the preblur formalism, the equation :math:`G = KA` is replaced
    by :math:`G = KBH`, with a hidden image :math:`H` and a blur matrix
    :math:`B`.

    The dicretization of :math:`\omega` has to be accounted for by
    including a :math:`\Delta\omega`. In fact, the total equation is
    :math:`G_i = \sum_{jk} K_{ij} \Delta\omega_j B_{jk} H_k`, where
    :math:`H_k` already includes a :math:`\Delta\omega_k`.

    Note that the ``PreblurKernel`` should always be used together
    with the :py:class:`.PreblurA_of_H`.

    Parameters
    ----------
    K : :py:class:`Kernel`
        the kernel to use up to the preblur matrix (e.g. a :py:class:`.TauKernel` instance)
    b : float
        the blur parameter, i.e., the width of the Gaussian that the hidden
        image is convolved with to get the blurred spectral function
    """

    def __init__(self, K, b):
        KernelSVD.__init__(self)
        self._T = None
        self.kernel = K
        self._b = b
        self._fill_values()

    def parameter_change(self):
        self.kernel.parameter_change()
        self._fill_values()

    def _fill_values(self):
        # invalidate U, S, V
        self._U = None
        self._S = None
        self._V = None

        self._B = get_preblur(self.omega, self._b)
        self._K = np.dot(self.kernel.K,
                         np.einsum('ij,i->ij', self._B, self.omega.delta))
        self._K_delta = self.kernel.K_delta

    def transform(self, T):
        self.kernel.transform(T)
        self._fill_values()

    def get_omega(self):
        return self.kernel.omega

    def set_omega(self, omega):
        self.kernel.omega = omega

    omega = property(get_omega, set_omega)

    @property
    def data_variable(self):
        return self.kernel.data_variable

    @data_variable.setter
    def data_variable(self, value):
        self.kernel.data_variable = value
