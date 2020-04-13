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
In the preblur formalism, use :py:class:`.PreblurKernel` and
:py:class:`.PreblurA_of_H`. For a description, see :ref:`preblur`.

This is just a helper function that is called by these classes to get
the blur matrix :math:`B` from the blur parameter :math:`b`.
"""



import numpy as np


def get_preblur(omega, b):
    r""" Get the blur matrix

    The blur matrix is calculated as

    .. math::

        B_{ij} = \frac{1}{2\pi b^2} e^{-\frac{(\omega_i-\omega_j)^2}{2b^2}}

    with a blur parameter :math:`b`. The matrix :math:`B` gets normalized
    both along the rows and columns before being returned.

    Parameters
    ----------
    omega : OmegaMesh
        the omega mesh used for the calculation
    b : float
        the blur parameter (standard deviation of the Gaussian function)
    """
    w1, w2 = np.meshgrid(omega, omega)
    B = np.exp(-(w1 - w2)**2 / 2.0 / b**2) / np.sqrt(2.0 * np.pi * b**2)
    renorm = (np.dot(omega.delta, B))
    B = B / renorm[:, np.newaxis]
    renorm = (np.dot(B, omega.delta))
    B = B / renorm[np.newaxis, :]
    return B
