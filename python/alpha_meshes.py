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
import warnings


class BaseAlphaMesh(np.ndarray):
    """ Base class for alpha meshes.
        All meshes inherit from this class.
    """

    def __new__(cls, alpha_min=0.0001, alpha_max=20, n_points=20,
                *args, **kwargs):
        obj = super(BaseAlphaMesh, cls).__new__(cls, shape=(n_points))
        return obj

    def __init__(self, alpha_min=0.0001, alpha_max=20, n_points=20):
        if n_points > 1:
            if alpha_min > alpha_max:
                raise Exception('alpha_min must be smaller than alpha_max')
            if (alpha_min <= 0) or (alpha_max <= 0):
                raise Exception('All alpha values must be positive')
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.n_points = n_points


class DataAlphaMesh(BaseAlphaMesh):
    """ Alpha mesh from data array

    The :math:`\\alpha`-points are picked from a user-supplied array.

    Parameters
    ----------
    data : array
        the alpha points
    """

    def __new__(cls, data):
        return super(DataAlphaMesh, cls).__new__(cls, np.min(data),
                                                 np.max(data), len(data))

    def __init__(self, data):
        super(DataAlphaMesh, self).__init__(np.min(data),
                                            np.max(data), len(data))
        self[:] = sorted(data, reverse=True)


class LogAlphaMesh(BaseAlphaMesh):
    """ Alpha mesh with logarithmically spaced data points

    Parameters
    ----------
    alpha_min : float
        the minimal alpha (NOT its log)
    alpha_max: float
        the maximal alpha (NOT its log)
    n_points : int
        the number of points in the alpha mesh
    """

    def __init__(self, alpha_min=0.0001, alpha_max=20, n_points=20):
        super(LogAlphaMesh, self).__init__(alpha_min, alpha_max, n_points)
        self[:] = np.logspace(np.log10(alpha_min),
                              np.log10(alpha_max),
                              n_points)[::-1]


class LinearAlphaMesh(BaseAlphaMesh):
    """ Alpha mesh with linearly spaced data points

    Parameters
    ----------
    alpha_min : float
        the minimal alpha
    alpha_max: float
        the maximal alpha
    n_points : int
        the number of points in the alpha mesh
    """

    def __init__(self, alpha_min=0.0001, alpha_max=20, n_points=20):
        super(LinearAlphaMesh, self).__init__(alpha_min, alpha_max, n_points)
        self[:] = np.linspace(alpha_min, alpha_max, n_points)[::-1]
