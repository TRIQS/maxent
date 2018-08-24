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
from triqs_maxent.kernels import TauKernel
from triqs_maxent.omega_meshes import DataOmegaMesh
import numpy as np


# the old get_kernel method
def get_kernel(tau, omega, beta=None):
    """ Calculate the kernel

    Parameters
    ----------
    tau : numpy array
        list of tau values
    omega : numpy array
        list of omega values
    beta : float
        beta value (default: last tau value)

    Returns
    -------
    K : numpy array
        Kernel matrix
    """
    if beta is None:
        beta = tau[-1]
    oomega, ttau = np.meshgrid(omega, tau)
    K = np.empty(oomega.shape)
    # the following is a trick to avoid overflows and similar
    L = oomega >= 0.0
    iL = np.where(L)
    nL = np.where(np.logical_not(L))
    K[iL] = -np.exp(-oomega[iL] * ttau[iL]) / \
        (np.exp(-beta * oomega[iL]) + 1.0)
    K[nL] = -np.exp(oomega[nL] * (beta - ttau[nL])) / \
        (1.0 + np.exp(beta * oomega[nL]))
    return K

tau = 10 * np.random.rand(10)
omega = DataOmegaMesh(np.random.rand(20))
beta = 10.0
K1 = TauKernel(tau=tau, omega=omega, beta=beta)
K2 = get_kernel(tau, omega, beta)

assert np.max(np.abs(K1.K - K2)) < 1.e-15, "kernel not the same"

maxdiff = np.max(np.abs(K1.K -
                        np.dot(K1.U, np.dot(np.diag(K1.S), K1.V.transpose()))))
assert  maxdiff < 1.e-14, \
    "Kernel is not the same as its SVD U*S*V^T (diff = {})".format(maxdiff)

L1 = len(K1.S)
threshold = np.median(K1.S)
K1.reduce_singular_space(threshold)
L2 = len(K1.S)
assert L2 == L1 / 2, "error in reduce_singular_space"

K3 = TauKernel(tau=tau, omega=omega, beta=beta)
K3.omega = omega[::2]
assert np.max(np.abs(K1.K - K3.K)) < 1.e-15, "kernel not the same"
K3.parameter_change()
assert K3.K.shape == (K1.K.shape[0], K1.K.shape[1] / 2), "wrong kernel shape"
# cannot compare K3 to K1[:,::2] because the delta omega is different
