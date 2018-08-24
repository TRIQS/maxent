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
from .analyzer import Analyzer, AnalyzerResult


class ClassicAnalyzer(Analyzer):
    """ Analyzer using the classic MaxEnt method

    This returns the spectrum :math:`A_\\alpha(\\omega)` that has the
    maximum probability (see :py:class:`.BryanAnalyzer` for a plot of
    the probability).

    Parameters
    ==========
    name : str
        the name of the method, defaults to `ClassicAnalyzer`.

    Attributes
    ==========
    A_out : array (vector)
        the output, i.e. the one true spectrum
    alpha_index : int
        the index of the output in the ``A_values`` array
    info : str
        some information about what the analyzer did
    """

    def __init__(self, name=None):
        super(ClassicAnalyzer, self).__init__(name=name)

    def analyze(self, maxent_result, matrix_element=None):
        r""" Perform the analysis

        Parameters
        ----------
        maxent_result : :py:class:`.MaxEntResult`
            the result where the :math:`\alpha`-dependent data is taken
            from
        matrix_element : tuple
            the matrix element (if applicable) that should be analyzed

        Returns
        -------
        result : :py:class:`AnalyzerResult`
            the result of the analysis, including the :math:`A_{out}`
        """

        def elem(what):
            return maxent_result._get_element(what, matrix_element)
        res = AnalyzerResult()
        res['name'] = self.name
        if np.all(np.isnan(elem(maxent_result.probability))):
            res['info'] = 'Probability not calculated. Cannot use ClassicAnalyzer.'
            return res

        res['alpha_index'] = np.nanargmax(elem(maxent_result.probability))

        res['A_out'] = elem(maxent_result.A)[res['alpha_index']]
        res['info'] = 'Ideal alpha (classic): {} (= index {} zero-based)'.format(
            maxent_result.alpha[res['alpha_index']], res['alpha_index'])
        return res
