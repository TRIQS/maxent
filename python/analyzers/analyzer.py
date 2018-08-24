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
from ..plot_utils import *
import sys
import numpy as np


class AnalyzerResult(dict):
    r""" Keep the result of the analyzer

    An analyzer gets the results of the MaxEnt optimization for different
    values of :math:`\alpha` and outputs one single :math:`A(\omega)`.
    This inherits from dict, the most important keys are:

    - A_out : the output spectral function
    - name : the name of the analyzer
    - info : human-readable info about what the analyzer did
    - alpha_index : if applicable, the index of the best :math:`A(\omega)`

    """

    def __reduce_to_dict__(self):
        """ for saving to h5 """
        return self

    @classmethod
    def __factory_from_dict__(cls, name, D):
        """ for reading from h5 """
        self = cls()
        for d, v in D.iteritems():
            self[d] = v
        return self

    def _get_maxent_result(self, maxent_result):
        if maxent_result is None:
            try:
                maxent_result = self.maxent_result
            except AttributeError:
                print('.--------------------------------------------------.',
                      file=sys.stderr)
                print('| Please supply the keyword argument maxent_result |',
                      file=sys.stderr)
                print("'--------------------------------------------------'",
                      file=sys.stderr)
                raise
        return maxent_result

    @plot_function
    def plot_A_out(self, maxent_result=None, **kwargs):
        """ Plot the spectral function

        Parameters
        ----------
        maxent_result : :py:class:`.MaxEntResult`
            the corresponding MaxEntResult, where the omega mesh etc.
            are saved.
            If None, ``AnalyzerResult.maxent_result`` has to be set.
        label : str
            the label of the curve (for a legend)
        x_label : str
            the label of the x-axis
        y_label : str
            the label of the y-axis
        log_x : bool
            whether the x-axis should be log-scaled (default: False)
        log_y : bool
            whether the y-axis should be log-scaled (default: False)
        """
        maxent_result = self._get_maxent_result(maxent_result)
        return (maxent_result.omega, self['A_out'],
                dict(label=r'$A(\omega)$ {}'.format(self['name']),
                     x_label=r'$\omega$',
                     y_label=r'$A(\omega)$',
                     log_x=False,
                     log_y=False))

    @plot_function
    def plot_curvature(self, maxent_result=None, **kwargs):
        r""" Plot the curvature of :math:`\log \chi^2` vs :math:`\log \alpha`

        This is not available for all analyzers.

        Parameters
        ----------
        maxent_result : :py:class:`.MaxEntResult`
            the corresponding MaxEntResult, where the omega mesh etc.
            are saved.
            If None, ``AnalyzerResult.maxent_result`` has to be set.
        label : str
            the label of the curve (for a legend)
        x_label : str
            the label of the x-axis
        y_label : str
            the label of the y-axis
        log_x : bool
            whether the x-axis should be log-scaled (default: False)
        log_y : bool
            whether the y-axis should be log-scaled (default: False)
        """

        maxent_result = self._get_maxent_result(maxent_result)
        return (maxent_result.alpha, self['curvature'],
                dict(label=r'curvature {}'.format(self['name']),
                     x_label=r'$\alpha$',
                     y_label=r'curvature',
                     log_x=True,
                     log_y=False))

    @plot_function
    def plot_dS_dalpha(self, maxent_result=None, **kwargs):
        r""" Plot the derivative of the entropy with respect to :math:`\alpha`

        This is not available for all analyzers.

        Parameters
        ----------
        maxent_result : :py:class:`.MaxEntResult`
            the corresponding MaxEntResult, where the omega mesh etc.
            are saved.
            If None, ``AnalyzerResult.maxent_result`` has to be set.
        label : str
            the label of the curve (for a legend)
        x_label : str
            the label of the x-axis
        y_label : str
            the label of the y-axis
        log_x : bool
            whether the x-axis should be log-scaled (default: True)
        log_y : bool
            whether the y-axis should be log-scaled (default: False)
        """

        maxent_result = self._get_maxent_result(maxent_result)
        return (maxent_result.alpha, self['dS_dalpha'],
                dict(label=r'dS_dalpha {}'.format(self['name']),
                     x_label=r'$\alpha$',
                     y_label=r'dS_dalpha',
                     log_x=True,
                     log_y=False))

    @plot_function
    def plot_linefit(self, maxent_result=None, element=None, **kwargs):
        r""" Plot the fitted lines of :math:`\log \chi^2` vs :math:`\log \alpha`

        This is not available for all analyzers.

        Parameters
        ----------
        maxent_result : :py:class:`.MaxEntResult`
            the corresponding MaxEntResult, where the omega mesh etc.
            are saved.
            If None, ``AnalyzerResult.maxent_result`` has to be set.
        label : str
            the label of the curve (for a legend)
        x_label : str
            the label of the x-axis
        y_label : str
            the label of the y-axis
        log_x : bool
            whether the x-axis should be log-scaled (default: True)
        log_y : bool
            whether the y-axis should be log-scaled (default: True)
        """

        maxent_result = self._get_maxent_result(maxent_result)
        idx = slice(None) if element is None else element
        return (maxent_result.alpha,
                np.column_stack((maxent_result.chi2[idx],
                                 np.exp(np.polyval(self['linefit_params'][0], np.log(maxent_result.alpha))),
                                 np.exp(np.polyval(self['linefit_params'][1], np.log(maxent_result.alpha))))),
                dict(label=r'linefit {}'.format(self['name']),
                     x_label=r'$\alpha$',
                     y_label=r'linefit',
                     log_x=True,
                     log_y=True))

try:
    from pytriqs.archive.hdf_archive_schemes import register_class
    register_class(AnalyzerResult)
except ImportError:  # notriqs
    pass


class Analyzer(object):
    r""" Analyzer base class

    The base class for analyzing the values :math:`A_{\alpha}` and getting
    the one true (:math:`\alpha`-independent) solution from the data.
    """

    def __init__(self, name=None, **kwargs):
        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name

    def analyze(self, maxent_result, matrix_element=None):
        raise NotImplementedError('Please use a subclass of Analyzer.')
