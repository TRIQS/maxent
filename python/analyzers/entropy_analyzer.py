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


class EntropyAnalyzer(Analyzer):
    r""" Analyzer searching a flat feature in the entropy

    This analyzer chooses the spectrum :math:`A_\alpha(\omega)` where
    the derivative of the entropy with respect to :math:`\log\alpha`
    is minimal.

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        S = [-0.8456575058822821, -0.8658403520963711, -0.8844467584579873, -0.9015179804032627, -0.917151672328798, -0.9314863556664439, -0.944683164425082, -0.956907556102252, -0.9683133476777639, -0.9790307680896819, -0.9891593596567183, -0.9987657170111451, -1.0078854019950367, -1.0165280011549969, -1.024684177382945, -1.0323336228063789, -1.0394529560517198, -1.0460227711524488, -1.0520332288852523, -1.057487795325951, -1.062404977107112, -1.0668181517074484, -1.070773798901126, -1.0743285656102701, -1.0775456259846594, -1.08049074893449, -1.0832283965498788, -1.0858180930339953, -1.0883112550281395, -1.0907486650570901, -1.0931587807617578, -1.095557070693374, -1.0979465250594018, -1.1003193968995932, -1.1026600976500056, -1.1049490343932176, -1.1071670792456967, -1.1093003459253266, -1.111345024505977, -1.1133121363430305, -1.1152320138828, -1.1171577437178764, -1.1191656036718212, -1.1213502283233436, -1.1238160022650803, -1.1266718510795215, -1.1300348922182275, -1.1340406305366701, -1.1388547911555023, -1.1446847433171343, -1.1517907169027382, -1.1604971815494127, -1.1712045022410134, -1.1844004970446838, -1.2006707511139032, -1.220706440371579, -1.2453068384914605, -1.2753730596069384, -1.311887106342708, -1.355868667039715]
        S = np.array(S)
        alpha = np.logspace(0, np.log10(2.e5) , len(S))[::-1]
        plt.subplot(2,1,1)
        plt.semilogx(alpha, S)
        plt.xlabel(r'$\alpha$')
        plt.ylabel(r'$S$')
        plt.subplot(2,1,2)
        deriv = np.full(len(alpha), np.nan)
        deriv[1:-1] = ((S[2:] - S[:-2]) / (
            np.log(alpha[2:]) - np.log(alpha[:-2])))
        plt.semilogx(alpha, deriv)
        plt.xlabel(r'$\alpha$')
        plt.ylabel(r'$dS/d\log\alpha$')


    Parameters
    ==========
    name : str
        the name of the method, defaults to `EntropyAnalyzer`.

    Attributes
    ==========
    A_out : array (vector)
        the output, i.e. the one true spectrum
    alpha_index : int
        the index of the output in the ``A_values`` array
    dS_dalpha : array
        the derivative of the entropy with respect to :math:`\\log\\alpha`
    info : str
        some information about what the analyzer did
    """

    def __init__(self, name=None):
        super(EntropyAnalyzer, self).__init__(name=name)

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
        res['dS_dalpha'] = np.full(len(maxent_result.alpha), np.nan)
        res['dS_dalpha'][1:-1] = ((elem(maxent_result.S)[2:] - elem(maxent_result.S)[:-2]) / (
            np.log(maxent_result.alpha[2:]) - np.log(maxent_result.alpha[:-2])))
        res['alpha_index'] = np.nanargmin(res['dS_dalpha']**2)
        if np.isnan(res['alpha_index']):
            raise ValueError('dS_dalpha is all NaN')

        res['A_out'] = elem(maxent_result.A)[res['alpha_index']]
        res['name'] = self.name
        res['info'] = 'Ideal alpha (entropy): {} (= index {} zero-based)'.format(
            maxent_result.alpha[res['alpha_index']], res['alpha_index'])
        return res
