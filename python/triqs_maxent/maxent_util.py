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
This file defines a bunch of functions that facilitate the use of
MaxEnt.
"""



import numpy as np
from itertools import product
from triqs.gf import *
from triqs.utility import mpi
from .kernels import TauKernel
from .omega_meshes import DataOmegaMesh


def get_G_w_from_A_w(A_w,
                     w_points,
                     np_interp_A=None,
                     np_omega=2000,
                     w_min=-10,
                     w_max=10,
                     broadening_factor=1.0):
    r""" Use Kramers-Kronig to determine the retarded Green function :math:`G(\omega)`

    This calculates :math:`G(\omega)` from the spectral function :math:`A(\omega)`.
    A numerical broadening of :math:`bf * i\Delta\omega`
    is used, with the adjustable broadening factor bf (default is 1).
    This function normalizes :math:`A(\omega)`.
    Use mpi to save time.

    Parameters
    ----------
    A_w : array
        Real-frequency spectral function.
    w_points : array
        Real-frequency grid points.
    np_interp_A : int
        Number of grid points A_w should be interpolated on before
        G_w is calculated. The interpolation is performed on a linear
        grid with np_interp_A points from min(w_points) to max(w_points).
    np_omega : int
        Number of equidistant grid points of the output Green function.
    w_min : float
        Start point of output Green function.
    w_max : float
        End point of output Green function.
    broadening_factor : float
        Factor multiplying the broadening :math:`i\Delta\omega`

    Returns
    -------
    G_w : GfReFreq
        TRIQS retarded Green function.
    """

    shape_A = np.shape(A_w)

    if len(shape_A) == 1:
        indices = [0]
        matrix_valued = False
    elif (len(shape_A) == 3) and (shape_A[0] == shape_A[1]):
        indices = list(range(0, shape_A[0]))
        matrix_valued = True
    else:
        raise Exception('A_w has wrong shape, must be n x n x n_w')

    if w_min > w_max:
        raise Exception('w_min must be smaller than w_max')

    if np_interp_A:
        w_points_interp = np.linspace(np.min(w_points),
                                      np.max(w_points), np_interp_A)
        if matrix_valued:
            A_temp = np.zeros((shape_A[0], shape_A[1], np_interp_A), dtype=complex)
            for i in range(shape_A[0]):
                for j in range(shape_A[1]):
                    A_temp[i, j, :] = np.interp(w_points_interp,
                                                w_points, A_w[i, j, :])
            A_w = A_temp
        else:
            A_w = np.interp(w_points_interp, w_points, A_w)
        w_points = w_points_interp

    G_w = GfReFreq(indices=indices, window=(w_min, w_max), n_points=np_omega)
    G_w.zero()

    iw_points = np.array(list(range(len(w_points))))

    for iw in mpi.slice_array(iw_points):
        domega = (w_points[min(len(w_points) - 1, iw + 1)] -
                  w_points[max(0, iw - 1)]) * 0.5
        if matrix_valued:
            for i in range(shape_A[0]):
                for j in range(shape_A[1]):
                    G_w[i, j] << G_w[i, j] + A_w[i, j, iw] * \
                        inverse(Omega - w_points[iw] + 1j * domega * broadening_factor) * domega
        else:
            G_w << G_w + A_w[iw] * \
                inverse(Omega - w_points[iw] + 1j * domega * broadening_factor) * domega

    G_w << mpi.all_reduce(G_w)
    mpi.barrier()

    return G_w


def get_G_tau_from_A_w(A_w, w_points, beta, np_tau):
    r""" Calculate :math:`G(\tau)` for a given :math:`A(\omega)`.

    Parameters
    ----------
    A_w : array
        Real-frequency spectral function.
    w_points : array or maxent mesh
        Real-frequency grid points.
    beta : float
        Inverse Temperature.
    np_tau : int
        Number of equidistant grid points of the output Green function.
        The tau grid runs from 0 to beta.

    Returns
    -------
    G_tau : GfImTime
        TRIQS imaginary-time Green function.
    """

    if not hasattr(w_points, 'delta'):
        w_points = DataOmegaMesh(w_points)

    K = TauKernel(tau=np.linspace(0.0, beta, np_tau),
                  omega=w_points,
                  beta=beta)
    G_tau = GfImTime(indices=[0], beta=beta, n_points=np_tau)
    G_tau.data[:, 0, 0] = np.dot(K.K_delta, A_w)
    # We don't care about the tail here as it will be removed in the next
    # TRIQS release

    return G_tau


def numder(fun, x, delta=1.e-6):
    r""" Calculate the numerical derivative (i.e., Jacobian) of fun
    around x.

    Parameters
    ----------
    fun : function
        a function :math:`\mathbb{R}^n \to \mathbb{R}`
    x : array
        the function argument where the numerical derivative should
        be evaluated
    delta : float
        the :math:`\Delta x` that is used in the approximation of the
        derivative
    """
    x2 = np.empty(x.shape)
    dfun = None
    for i in product(*map(range, x.shape)):
        x2[:] = x
        x2[i] += delta
        funplus = fun(x2)
        x2[i] -= 2 * delta
        funminus = fun(x2)
        if dfun is None:
            if len(funplus.shape) == 0:
                dfun = np.empty((1,) + x2.shape, dtype=funplus.dtype)
            else:
                dfun = np.empty(funplus.shape + x2.shape, dtype=funplus.dtype)
        dfun[(Ellipsis,) + i] = (funplus - funminus) / (2.0 * delta)
    return dfun


def check_der(f, d, around, renorm=False, prec=1.e-8, name=''):
    r""" check whether d is the analytical derivative of f by comparing
    with the numerical derivative

    Parameters
    ----------
    f : function
        we want to calculate the derivative of this function;
        a function :math:`\mathbb{R}^n \to \mathbb{R}` of :math:`x`
    d : function
        a function :math:`\mathbb{R}^n \to \mathbb{R}^n` which gives the
        analytic derivative of :math:`f` with respect to the elements
        of :math:`x`
    renorm : bool or float
        if bool: if False, do not renormalize, if True: renormalize
        by the function value; if float, renormalize by the value
        of the float; this allows to get some kind of relative error
    prec : float
        the precision of the check, i.e. if the error is larger than
        ``prec``, a  warning will be issued
    name : str
        the name; this will be used in the error message if the derivatives
        are not equal to allow you to identify the culprit
    """
    d1 = numder(f, around)
    d2 = d(around)
    maxerr = np.abs(d1 - d2)
    if renorm is True:
        maxerr /= np.abs(f(around))
    elif renorm is False:
        pass
    else:
        maxerr /= np.abs(renorm)
    maxerr = np.max(maxerr)
    if maxerr > prec:
        print('numerical derivative does not fit analytic derivative: {} - difference {}'.format(name, maxerr))
        return False
    return True
