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


"""
This file defines a bunch of functions that represent physical
functions in the MaxEnt formalism.

The base classes for all these are :class:`.DoublyDerivableFunction`
and/or :class:`.InvertibleFunction`.

The most important functions defined are

  * the definition of the :math:`\chi^2`, which comes as :py:class:`.NormalChi2`.
  * the definition of the entropy; for diagonal elements of Green functions
    the :py:class:`.NormalEntropy` should be used, for off-diagonals the
    :py:class:`.PlusMinusEntropy`.
  * the definition of the parametrization of :math:`H(v)` in singular
    space (which maps a vector :math:`v` in singular space to a spectral
    function :math:`H`); here, again we have a :py:class:`.NormalH_of_v`
    and a :py:class:`.PlusMinusH_of_v`.
  * the definition of the parametrization of :math:`A(H)`; for normal
    calculations the :py:class:`.IdentityA_of_H` takes care of the factor
    :math:`\Delta\omega` in non-uniform :math:`\omega` meshes. For preblur
    calculations (see :ref:`preblur`), the :py:class:`.PreblurA_of_H` additionally
    blurs the hidden image :math:`H` to get the spectral function :math:`A`.
"""
from __future__ import absolute_import, print_function

import numpy as np
from .maxent_util import *
from .preblur import *
from .kernels import KernelSVD
from functools import wraps
import copy


def safelog(A):
    # a logarithm that does not produce as many nans
    A[np.where(np.abs(A) <= 1.e-100)] = 1.e-100
    return np.log(A)


def view_complex(A):
    return A.view(np.complex_).reshape(A.shape[:-1])


def view_real(A):
    return A.view(float).reshape(A.shape + (2,))


def cached(func):
    """ A descriptor for cached functions """
    @wraps(func)
    def new_func(self, x=None):
        # use an x that has been supplied before
        if x is None or x is getattr(self, "_x", None):
            if func not in self._cached:
                # cache the result
                self._cached[func] = func(self, self._x)
            return self._cached[func]
        else:
            return func(self, x)
    return new_func

# =====================================================================
#  Generic
# =====================================================================


class GenericFunction(object):

    def parameter_change(self):
        """ Notify the function that parameters have changed

        This allows to reprocess certain values.
        """
        pass


class CachedFunction(GenericFunction):
    """ A function that remembers its values

    The general way to use the functions that are cached, which here are
    all the functions derived from GenericFunction, is to supply the
    argument to the function class and then get either the function value
    as ``.f()``, the derivative as ``.d()``, or the second derivative as ``.dd()``.
    Note that it is advisable to supply the argument once (e.g. ``cf(x)``) and evaluate
    everything afterwards as then the results are being cached.

    Note that just the reference of the supplied argument is stored, i.e.
    if you change it there might be inconsistent results.
    """

    def __call__(self, x):
        ret = copy.copy(self)  # a shallow copy
        ret._cached = dict()
        # the following is a trick to ensure that id(self._x) != id(x)
        # thus, when calling a function with self._x, the cached version
        # is used, but with any other x (including the original object,
        # which might have been modified without changing the id, i.e.
        # the memory pointer) the function is reevaluated
        ret._x = x.view()
        return ret


class DoublyDerivableFunction(CachedFunction):
    """ Template for a double derivable function

    This function has the methods ``f``, ``d`` and ``dd``, representing
    the function values and its two derivatives.
    """

    def __init__(self, **kwargs):
        self.extra_args = kwargs
        self.parameter_change()

    @cached
    def f(self, x):
        """ function value """
        raise NotImplementedError()

    @cached
    def d(self, x):
        """ first derivative """
        raise NotImplementedError()

    @cached
    def dd(self, x):
        """ second derivative """
        raise NotImplementedError()

    def check_derivatives(self, around, renorm=False, prec=1.e-8):
        """ check derivatives using numerical derivation

        Parameters
        ----------
        around : array
            the value that should be inserted for ``x`` in the functions
        renorm : bool or float
            if bool: if False, do not renormalize, if True: renormalize
            by the function value; if float, renormalize by the value
            of the float; this allows to get some kind of relative error
        prec : float
            the precision of the check, i.e. if the error is larger than
            ``prec``, a  warning will be issued

        Returns
        -------
        success : bool
            whether the test passed (True) or not (False)
        """
        return (self.check_d(around, renorm, prec) and
                self.check_dd(around, renorm, prec))

    def check_d(self, around, renorm=False, prec=1.e-8):
        """ check first derivative

        see :py:meth:`.check_derivatives`
        """
        return check_der(self.f, self.d, around, renorm, prec,
                         "1st derivative " + type(self).__name__)

    def check_dd(self, around, renorm=False, prec=1.e-8):
        """ check second derivative

        see :py:meth:`.check_derivatives`
        """

        return check_der(self.d, self.dd, around, renorm, prec,
                         "2nd derivative " + type(self).__name__)


class InvertibleFunction(CachedFunction):
    """ Template for an invertible function

    This function has the methods ``f`` and ``inv``, representing
    the function values and the inverse function
    """

    def __init__(self, **kwargs):
        self.extra_args = kwargs
        self.parameter_change()

    @cached
    def f(self, x):
        """ function value """
        raise NotImplementedError()

    @cached
    def inv(self, y):
        """ inverse function value """
        raise NotImplementedError()

    def check_inv(self, x, prec=1.e-8):
        """ check whether inv is really the inverse of f """
        y = self.f(x)
        x2 = self.inv(y)
        if np.max(np.abs(x - x2)) > prec:
            error_message(
                """Inverse of function is not correct:
                   {} - difference: {}
                """.format(type(self).__name__,
                           np.max(np.abs(x - x2))))


class NullFunction(DoublyDerivableFunction):
    """ A constant function that is zero """

    @cached
    def f(self, x):
        return 0

    @cached
    def d(self, x):
        return 0 * x

    @cached
    def dd(self, x):
        return np.diag(0 * x)

# =====================================================================
#  Chi2
# =====================================================================


class Chi2(DoublyDerivableFunction):
    r""" A function giving the least squares

    Parameters
    ----------
    K : :py:class:`.Kernel`
        the kernel to use
    G : array
        the Green function data
    err : array
        the error of the Green function data (must have the same length
        as G)
    """

    def __init__(self, K=None, G=None, err=None):
        self._K = K
        self._G = G
        self._err = err
        self.parameter_change()

    ####### Helper functions #######

    def get_K(self):
        return self._K

    def set_K(self, K, update_chi2=True):
        self._K = K
        if update_chi2:
            self.parameter_change()

    K = property(get_K, set_K)

    def get_G(self):
        return self._G

    def set_G(self, G, update_chi2=True):
        self._G = G
        if update_chi2:
            self.parameter_change()

    G = property(get_G, set_G)

    def get_err(self):
        return self._err

    def set_err(self, err, update_chi2=True):
        self._err = err
        if update_chi2:
            self.parameter_change()

    err = property(get_err, set_err)

    def get_omega(self):
        return self.K.omega

    def set_omega(self, omega, update_K=True, update_chi2=True):
        self.K.omega = omega
        if update_K:
            self.K.parameter_change()
        if update_chi2:
            self.parameter_change()

    omega = property(get_omega, set_omega)

    def get_data_variable(self):
        return self.K.data_variable

    def set_data_variable(self,
                          data_variable,
                          update_K=True,
                          update_chi2=True):
        self.K.data_variable = data_variable
        if update_K:
            self.K.parameter_change()
        if update_chi2:
            self.parameter_change()

    data_variable = property(get_data_variable, set_data_variable)


class NormalChi2(Chi2):
    r""" A function giving the usual least squares

    This is calculated as

    .. math::

        \chi^2 = \sum_i \frac{(G_i - \sum_j K_{ij} H_j)^2}{\sigma_i^2}

    Note that :math:`H = A\Delta\omega` (in the usual case, see :ref:`preblur` for a different definition).

    Parameters
    ----------
    K : :py:class:`.Kernel`
        the kernel to use
    G : array
        the Green function data
    err : array
        the error of the Green function data (must have the same length
        as G)
    """

    @cached
    def f(self, A):
        return sum(np.abs(np.dot(self.K.K, A) - self.G)**2 / self.err**2)

    @cached
    def d(self, A):
        return np.dot(2 * (np.dot(self.K.K, A) - self.G) /
                      self.err**2, np.conjugate(self.K.K))

    @cached
    def dd(self, A):
        # this is constant
        return self.d2

    def parameter_change(self):
        """ Notify that the parameters (either ``K`` or ``err``) have changed """
        # we calculate the value of the second derivative as it is constant
        if self.K is not None and self.err is not None:
            self.d2 = 2 * np.einsum('il,ik,i->kl', np.conjugate(self.K.K),
                                    self.K.K, 1. / self.err**2)


class ComplexChi2(Chi2):
    r""" A function giving the usual least squares

    This is calculated as

    .. math::

        \chi^2 = \sum_i \frac{|G_i - \sum_j K_{ij} H_j|^2}{\sigma_i^2}

    Note that :math:`H = A\Delta\omega` (in the usual case, see :ref:`preblur` for a different definition).

    Parameters
    ----------
    K : :py:class:`.Kernel`
        the kernel to use
    G : array
        the Green function data
    err : array
        the error of the Green function data (must have the same length
        as G)
    """

    @cached
    def f(self, A):
        diff = np.dot(self.K.K, view_complex(A)) - self.G
        return sum(np.abs(diff)**2 / self.err**2)

    @cached
    def d(self, A):
        diff = np.dot(self.K.K, view_complex(A)) - self.G
        return view_real(2 * np.dot(diff / self.err**2,
                                    np.conjugate(self.K.K)))

    @cached
    def dd(self, A):
        # this is constant
        return self.d2

    def parameter_change(self):
        """ Notify that the parameters (either ``K`` or ``err``) have changed """
        # we calculate the value of the second derivative as it is constant
        if self.K is not None and self.err is not None:
            N_w = self.K.K.shape[-1]
            E = 2 * np.einsum('il,ik,i->kl', np.conjugate(self.K.K),
                              self.K.K, 1. / self.err**2)
            self.d2 = np.zeros((N_w, 2, N_w, 2))
            self.d2[:, 0, :, 0] = np.real(E)
            self.d2[:, 0, :, 1] = np.imag(E)
            self.d2[:, 1, :, 1] = -np.imag(E)
            self.d2[:, 1, :, 1] = np.real(E)


# =====================================================================
#  Entropy
# =====================================================================


class Entropy(DoublyDerivableFunction):
    """ A function giving an entropy term for regularization

    Parameters
    ----------
    D : DefaultModel
        the default model
    """

    def __init__(self, D=None):
        self._D = D
        self.parameter_change()

    ####### Helper functions #######

    def get_D(self):
        return self._D

    def set_D(self, D, update_S=True):
        self._D = D
        if update_S:
            self.parameter_change()

    D = property(get_D, set_D)

    def get_omega(self):
        return self.D.omega

    def set_omega(self, omega, update_D=True, update_S=True):
        self.D.omega = omega
        if update_D:
            self.D.parameter_change()
        if update_S:
            self.parameter_change()

    omega = property(get_omega, set_omega)


class NormalEntropy(Entropy):
    """ The usual entropy

    This calculates the entropy as

    .. math ::

        S = \sum_i (H_i - D_i - H_i \log(H_i/D_i)).

    Note that :math:`H = A\Delta\omega` (in the usual case, see :ref:`preblur` for a different definition).
    Also, the default model usually includes the :math:`\Delta\omega`.

    Parameters
    ----------
    D : DefaultModel
        the default model
    """
    @cached
    def f(self, A):
        return np.sum((A - self.D.D - A * safelog(A / self.D.D)))

    @cached
    def d(self, A):
        return - (safelog(A) - safelog(self.D.D))

    @cached
    def dd(self, A):
        # we need this to prevent a NaN in the calculation
        A[np.where(np.abs(A) <= 1.e-100)] = 1.e-100
        return -np.diag(1.0 / A)


class PlusMinusEntropy(NormalEntropy):
    """ The Plus-Minus entropy

    This calculates the entropy as

    .. math ::

        S = S_{normal}(H^+) + S_{normal}(H^-),

    where :math:`S_{normal}` is the :py:class:`NormalEntropy`. We have
    :math:`H = H^+ - H^-`.

    Note that :math:`H = A\Delta\omega` (in the usual case, see :ref:`preblur` for a different definition).
    Also, the default model usually includes the :math:`\Delta\omega`.

    Parameters
    ----------
    D : DefaultModel
        the default model
    """

    @cached
    def _A_plus(self, A):
        return (np.sqrt(A**2.0 + 4.0 * self.D.D**2) + A) / 2.0

    @cached
    def _A_minus(self, A):
        return (np.sqrt(A**2.0 + 4.0 * self.D.D**2) - A) / 2.0

    @cached
    def f(self, A):
        return super(PlusMinusEntropy, self).f(self._A_plus(A)) + \
            super(PlusMinusEntropy, self).f(self._A_minus(A))

    @cached
    def d(self, A):
        return super(PlusMinusEntropy, self).d(self._A_plus(A))

    @cached
    def dd(self, A):
        return super(PlusMinusEntropy, self).dd(
            self._A_plus(A) + self._A_minus(A))


class AbsoluteEntropy(Entropy):
    """ The entropy with ``|A|``

    .. warning:: This entropy is not convex!
    """
    @cached
    def f(self, A):
        return np.sum((np.abs(A) - self.D.D -
                       np.abs(A) * safelog(np.abs(A) / self.D.D)))

    @cached
    def d(self, A):
        return - safelog(np.abs(A) / self.D.D) * np.sign(A)

    @cached
    def dd(self, A):
        return -np.diag(1.0 / np.abs(A))


class ShiftedAbsoluteEntropy(Entropy):
    """ The entropy with ``|A|+D`` """
    @cached
    def f(self, A):
        return np.sum((np.abs(A) + self.D.D - self.D.D - (np.abs(A) + \
                      self.D.D) * safelog((np.abs(A) + self.D.D) / self.D.D)))

    @cached
    def d(self, A):
        return - safelog((np.abs(A) + self.D.D) / self.D.D) * np.sign(A)

    @cached
    def dd(self, A):
        return -np.diag(1.0 / (np.abs(A) + self.D.D))

# =====================================================================
#  H(v)
# =====================================================================


class GenericH_of_v(DoublyDerivableFunction, InvertibleFunction):
    """ A function giving the parametrization :math:`H(v)`

    Parameters
    ----------
    D : DefaultModel
        the default model to use
    K : :py:class:`.Kernel`
        the kernel to use
    """

    def __init__(self, D=None, K=None):
        self._D = D
        self._K = K
        self._B = 1.0
        self.parameter_change()

    ####### Helper functions #######

    def get_D(self):
        return self._D

    def set_D(self, D, update_H_of_v=True):
        self._D = D
        if update_H_of_v:
            self.parameter_change()

    D = property(get_D, set_D)

    def get_K(self):
        return self._K

    def set_K(self, K, update_H_of_v=True):
        self._K = K
        if update_H_of_v:
            self.parameter_change()

    K = property(get_K, set_K)

    def get_omega(self):
        return self.D.omega

    def set_omega(self, omega, update_D=True, update_H_of_v=True):
        self.D.omega = omega
        if update_D:
            self.D.parameter_change()
        if update_H_of_v:
            self.parameter_change()

    omega = property(get_omega, set_omega)


class NormalH_of_v(GenericH_of_v):
    """ Bryan's parametrization H(v)

    This parametrization uses

    .. math::

        H(v) = D \exp(Vv),

    where :math:`V` is the matrix of the right-singular vectors of the
    kernel.

    Parameters
    ----------
    D : DefaultModel
        the default model to use
    K : :py:class:`.Kernel`
        the kernel to use
    """
    @cached
    def f(self, v):
        return self.D.D * np.exp(np.dot(self.K.V, v))

    @cached
    def d(self, v):
        return self.D.D[:, np.newaxis] * self.K.V * \
            np.exp(np.dot(self.K.V, v))[:, np.newaxis]

    @cached
    def dd(self, v):
        return np.einsum('k,kj,kl,k->kjl', self.D.D, self.K.V, self.K.V,
                         np.exp(np.dot(self.K.V, v)))

    @cached
    def inv(self, A):
        return np.dot(self.K.V.transpose(), safelog(A / self.D.D))


class PlusMinusH_of_v(GenericH_of_v):
    """ Plus/minus parametrization H(v)

    This should be used with the :py:class:`.PlusMinusEntropy`.
    The parametrization is

    .. math::

        H(v) = D (e^{Vv} - e^{-Vv})

    where :math:`V` is the matrix of the right-singular vectors of the
    kernel.

    Parameters
    ----------
    D : DefaultModel
        the default model to use
    K : :py:class:`.Kernel`
        the kernel to use
    """
    @cached
    def f(self, v):
        return self.D.D * (np.exp(np.dot(self.K.V, v)) -
                           np.exp(-np.dot(self.K.V, v)))

    @cached
    def d(self, v):
        return self.D.D[:, np.newaxis] * self.K.V * (np.exp(
            np.dot(self.K.V, v))[:, np.newaxis] + np.exp(-np.dot(self.K.V, v))[:, np.newaxis])

    @cached
    def dd(self, v):
        return np.einsum('k,kj,kl,k->kjl', self.D.D, self.K.V, self.K.V,
                         np.exp(np.dot(self.K.V, v)) - np.exp(-np.dot(self.K.V, v)))

    @cached
    def inv(self, A):
        return np.dot(self.K.V.transpose(), safelog(
            (A + np.sqrt(A**2 + 4 * self.D.D**2)) / (2 * self.D.D)))


class NoExpH_of_v(GenericH_of_v):
    """ Parametrization H(v) without the exponential """
    @cached
    def f(self, v):
        return self.D.D * np.dot(self.K.V, v)

    @cached
    def d(self, v):
        return self.D.D[:, np.newaxis] * self.K.V

    @cached
    def dd(self, v):
        return np.zeros((len(self.D.D), self.K.V.shape[1], self.K.V.shape[1]))

    @cached
    def inv(self, A):
        return np.dot(self.K.V.transpose(), A / self.D.D)


class IdentityH_of_v(GenericH_of_v):
    """ Parametrization H(v)=v """

    def parameter_change(self):
        if self.D is not None:
            self.d2 = np.zeros((len(self.D.D), len(self.D.D), len(self.D.D)))

    @cached
    def f(self, v):
        return v

    @cached
    def d(self, v):
        return np.eye(len(v))

    @cached
    def dd(self, v):
        return self.d2

    @cached
    def inv(self, A):
        return A

# =====================================================================
#  A(H)
# =====================================================================


class GenericA_of_H(DoublyDerivableFunction, InvertibleFunction):
    """ A parametrization :math:`A(H)` """

    def get_omega(self):
        return self._omega

    def set_omega(self, omega, update_A_of_H=True):
        self._omega = omega
        if update_A_of_H:
            self.parameter_change()

    omega = property(get_omega, set_omega)


class IdentityA_of_H(GenericA_of_H):
    """ Parametrization A(H)=H

    For non-uniform omega meshes, this takes care of the :math:`\Delta \omega`.
    Use this whenever you don't use the :py:class:`.PreblurKernel`
    """

    def __init__(self, omega):
        self._omega = omega

    @cached
    def f(self, H):
        return H / self._omega.delta

    @cached
    def d(self, H):
        return 1.0  # np.eye(len(H)), but faster

    @cached
    def dd(self, H):
        return 0.0  # np.zeros((len(H), len(H), len(H))), but faster

    @cached
    def inv(self, A):
        return A * self._omega.delta


class PreblurA_of_H(GenericA_of_H):
    """ A_of_H using preblur

    With preblur, we have :math:`A(H) = BH` (up to a :math:`\Delta\omega`
    for non-uniform omega meshes).

    Use this whenever you use the :py:class:`.PreblurKernel`.

    Parameters
    ----------
    b : float
        blur parameter (width of Gaussian)
    omega : array
        the omega mesh used for ``H_of_v``
    """

    def __init__(self, b, omega):
        """ A_of_H using preblur

        b : float
            blur parameter (width of Gaussian)
        omega : array
            the omega mesh used for H_of_v
        """

        self._omega = omega
        self._b = b
        self.parameter_change()

    def parameter_change(self):
        self._B = get_preblur(self._omega, self._b)

    @cached
    def f(self, H):
        return np.dot(self._B, H)

    @cached
    def d(self, H):
        return self._B

    @cached
    def dd(self, H):
        return np.zeros((len(H), len(H), len(H)))

    @cached
    def inv(self, A):
        return np.linalg.lstsq(self._B, A)[0]

    def get_b(self):
        return self._b

    def set_b(self, b, update_A_of_H=True):
        self._b = b
        if update_A_of_H:
            self.parameter_change()

    b = property(get_b, set_b)
