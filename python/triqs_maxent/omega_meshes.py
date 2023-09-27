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




import numpy as np


class BaseOmegaMesh(np.ndarray):
    """ Base class for omega meshes.
        All meshes inherit from this class.
    """

    def __new__(cls, omega_min=-10, omega_max=10, n_points=100,
                *args, **kwargs):
        self = super(BaseOmegaMesh, cls).__new__(cls, shape=(n_points))
        return self

    def __init__(self, omega_min=-10, omega_max=10, n_points=100,
                 *args, **kwargs):
        if omega_min > omega_max:
            raise Exception('omega_min must be smaller than omega_max')
        self.omega_min = omega_min
        self.omega_max = omega_max
        self.n_points = n_points
        self._delta = None

    def __array_finalize__(self, obj):
        if obj is not None:
            try:  # Error with numpy 1.13
                self.omega_min = obj.omega_min
                self.omega_max = obj.omega_max
                self.n_points = obj.n_points
            except AttributeError as e:
                pass
        self._delta = None

    @property
    def delta(self):
        if self._delta is None:
            delta = np.empty(len(self))
            delta[1:-1] = (self[2:] - self[:-2]) / 2.0
            delta[0] = (self[1] - self[0]) / 2.0
            delta[-1] = (self[-1] - self[-2]) / 2.0
            self._delta = delta
        return self._delta


class LinearOmegaMesh(BaseOmegaMesh):
    """ Omega mesh with linear spacing

    The :math:`i`-th :math:`\\omega`-point is given by

    .. math::

        \\omega_i = \\omega_{min} + i \\frac{\\omega_{max}-\\omega_{min}}{n_{max}-1},

    where :math:`i` runs from :math:`0` to :math:`n_{max}-1`.

    Parameters
    ----------
    omega_min : float
        the minimal omega
    omega_max : float
        the maximal omega
    n_points : int
        the number of omega points
    """

    def __init__(self, omega_min=-10, omega_max=10, n_points=100):
        super(LinearOmegaMesh, self).__init__(omega_min, omega_max, n_points)
        self[:] = np.linspace(omega_min, omega_max, n_points)


class DataOmegaMesh(BaseOmegaMesh):
    """ Omega mesh from data array

    The :math:`\\omega`-points are picked from a user-supplied array.

    Parameters
    ----------
    data : array
        an array giving the omega points
    """

    def __new__(cls, data):
        return super(DataOmegaMesh, cls).__new__(cls, np.min(data),
                                                 np.max(data), len(data))

    def __init__(self, data):
        super(DataOmegaMesh, self).__init__(np.min(data),
                                            np.max(data), len(data))
        self[:] = data


class LorentzianOmegaMesh(BaseOmegaMesh):
    """ Omega mesh with Lorentzian spacing

    This mesh is a lot denser than the linear mesh around :math:`\\omega=0`
    and far less denser for high :math:`|\omega|`.
    The lowest value is at :math:`\omega_{min}`, the largest at :math:`\omega_{max}`.

    Parameters
    ----------
    omega_min : float
        the minimal omega
    omega_max : float
        the maximal omega
    n_points : int
        the number of omega points
    cut : float
        a parameter influencing the relative density between the middle
        and the edge of the interval
    """

    def __init__(self, omega_min=-10, omega_max=10, n_points=100, cut=0.01):
        super(LorentzianOmegaMesh, self).__init__(omega_min,
                                                  omega_max,
                                                  n_points)
        self.cut = cut

        u = np.linspace(0, 1, n_points + 1)
        temp = np.tan(np.pi * (u * (1. - 2 * cut) + cut - 0.5))
        t = (temp - temp[0]) / (temp[-1] - temp[0])
        w = omega_min + (omega_max - omega_min) * t
        w = (w[:-1] + w[1:]) / 2.0
        w = (w - w[0]) / (w[-1] - w[0]) * (omega_max - omega_min) + omega_min
        self[:] = w

    def __array_finalize__(self, obj):
        super(LorentzianOmegaMesh, self).__array_finalize__(obj)
        if obj is not None:
            try:  # Error with numpy 1.13
                self.cut = obj.cut
            except AttributeError as e:
                pass


class LorentzianSmallerOmegaMesh(BaseOmegaMesh):
    """ Omega mesh with Lorentzian spacing

    This mesh is a lot denser than the linear mesh around :math:`\\omega=0`
    and far less denser for high :math:`|\omega|`.
    The lowest value is not at :math:`\omega_{min}`, the largest at :math:`\omega_{max}`;
    this is the main difference related to :py:class:`.LorentzianOmegaMesh`.

    Parameters
    ----------
    omega_min : float
        the minimal omega
    omega_max : float
        the maximal omega
    n_points : int
        the number of omega points
    cut : float
        a parameter influencing the relative density between the middle
        and the edge of the interval
    """

    def __init__(self, omega_min=-10, omega_max=10, n_points=100, cut=0.01):
        # as in Levy, Gull code
        super(LorentzianSmallerOmegaMesh, self).__init__(omega_min,
                                                         omega_max,
                                                         n_points)
        self.cut = cut

        u = np.linspace(0, 1, n_points + 1)
        temp = np.tan(np.pi * (u * (1. - 2 * cut) + cut - 0.5))
        t = (temp - temp[0]) / (temp[-1] - temp[0])
        w = omega_min + (omega_max - omega_min) * t
        w = (w[:-1] + w[1:]) / 2.0
        self[:] = w

    def __array_finalize__(self, obj):
        super(LorentzianSmallerOmegaMesh, self).__array_finalize__(obj)
        if obj is not None:
            try:  # Error with numpy 1.13
                self.cut = obj.cut
            except AttributeError as e:
                pass


class HyperbolicOmegaMesh(BaseOmegaMesh):
    """ Omega mesh with hyperbolic spacing

    This mesh is denser than the linear mesh around :math:`\\omega=0`
    and behaves like a sparser variant of a linear mesh at :math:`|\\omega|\\to\\infty`.

    Parameters
    ----------
    omega_min : float
        the minimal omega
    omega_max : float
        the maximal omega
    n_points : int
        the number of omega points
    """

    def __init__(self, omega_min=-10, omega_max=10, n_points=100):
        super(HyperbolicOmegaMesh, self).__init__(omega_min,
                                                  omega_max,
                                                  n_points)
        u = np.linspace(-1, 1, n_points)
        w = np.sign(u) * (np.sqrt(1 + u**2) - 1)
        w = omega_min + (omega_max - omega_min) * (w - w[0]) / (w[-1] - w[0])
        self[:] = w
