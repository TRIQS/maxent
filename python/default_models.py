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


class BaseDefaultModel(object):
    """ Base class for default models.
        All default models inherit from this class.
    """

    def __init__(self, omega):
        self.omega = omega
        self._D = None

    @property
    def D(self):
        return self._D

    def parameter_change(self):
        self._fill_values()

    def _fill_values(self):
        raise NotImplemented("Use a subclass of BaseDefaultModel")

    def __len__(self):
        return len(self._D)


class FlatDefaultModel(BaseDefaultModel):
    """ A flat default model for total absence of knowledge

    Parameters
    ----------
    omega : OmegaMesh
        the omega mesh used for the calculation
    """

    def __init__(self, omega):
        super(FlatDefaultModel, self).__init__(omega)
        self._fill_values()

    def _fill_values(self):
        self._D = np.ones(self.omega.shape) / \
            np.sum(self.omega.delta) * self.omega.delta


class DataDefaultModel(BaseDefaultModel):
    """ A default model given on a grid

    Parameters
    ----------
    default : array
        the default model data
    omega_in : array or OmegaMesh
        the omega-values corresponding to the default model data
    omega : OmegaMesh
        the omega mesh used for the calculation
        (if None, omega_in is used)
    """

    def __init__(self, default, omega_in, omega=None):
        if omega is None:
            omega = omega_in
        super(DataDefaultModel, self).__init__(omega)
        self.omega_in = omega_in
        self.default = default
        self._fill_values()

    def _fill_values(self):
        if np.all(self.omega_in == self.omega):
            self._D = self.default
        else:
            self._D = np.interp(self.omega, self.omega_in, self.default)
        self._D = self._D * self.omega.delta


class FileDefaultModel(DataDefaultModel):
    """ A default model given on a grid read from a file

    Parameters
    ----------
    filename : str
        the name of the data file; the first column contains the
        omega values, the second column the corresponding value
        of the default model
    omega : OmegaMesh
        the omega mesh used for the calculation
        (if None, the data omega mesh is used)
    """

    def __init__(self, filename, omega=None):
        data = np.loadtxt(filename)
        self = super(FileDefaultModel, cls).__new__(cls,
                                                    default=data[:, 1],
                                                    omega_in=data[:, 0],
                                                    omega=omega)
