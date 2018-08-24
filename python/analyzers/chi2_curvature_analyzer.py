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


def curv(x, y):
    r""" calculate the curvature of a curve given by x / y data points

    The curvature is given by

    .. math::

        c = \frac{\partial^2 y}{\partial x^2} \Bigg/ \left(1 + \left(\frac{\partial y}{\partial x}\right)^2\right)^{3/2}.

    The derivatives are calculated using a central finite difference
    approximation with second-order accuracy.
    Therefore, the resulting curvature contains ``nan`` as first and
    last entry.
    """
    l = len(x)
    der2 = np.full(l, np.nan)
    der1 = np.full(l, np.nan)
    curvature = np.full(l, np.nan)
    for k in range(1, len(x) - 1):
        der2[k] = (y[k + 1] - 2 * y[k] + y[k - 1]) / \
            ((x[k + 1] - x[k]) * (x[k] - x[k - 1]))
        der1[k] = ((y[k + 1] - y[k]) / (x[k + 1] - x[k]) +
                   (y[k] - y[k - 1]) / (x[k] - x[k - 1])) / 2
    curvature = der2 / (1 + der1 * der1)**(3. / 2.)
    return curvature, der1, der2


class Chi2CurvatureAnalyzer(Analyzer):
    r""" Analyzer using the curvature of :math:`\chi^2(\alpha)`.

    In analogy to the procedure used in the OmegaMaxEnt code, this
    analyzer chooses the spectral function by searching for the
    maximum of the curvature of :math:`\log\chi^2(\gamma \log\alpha)`.

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        from triqs_maxent.analyzers.chi2_curvature_analyzer import curv
        chi2 = [29349.131935651938, 22046.546280176568, 16571.918154748197, 12487.464413222642, 9445.982385619374, 7178.502063034175, 5481.286491560124, 4203.095677764095, 3233.499702682774, 2492.7723059042005, 1923.6134333677408, 1484.7009011237717, 1145.8924661597946, 884.8005310524965, 684.4330049956029, 531.6146648070567, 415.9506388789646, 329.14872296705806, 264.5685376022979, 216.90786277834727, 181.96883131892676, 156.46999971120204, 137.88618534909426, 124.30790112614721, 114.31767388342703, 106.88286008781692, 101.26499523088836, 96.94522657033058, 93.56467085587808, 90.87798882482437, 88.71820342038981, 86.97077896408142, 85.5551277292544, 84.41192628409034, 83.49484873710317, 82.76553387129734, 82.19079735392212, 81.74128618303814, 81.39095719753091, 81.11694546963328, 80.89956952736446, 80.72238218084304, 80.57227364014592, 80.43956708639224, 80.31784061686754, 80.20324577429295, 80.09354393085312, 79.98731507910087, 79.88352134738581, 79.78132986379869, 79.68005975635513, 79.57917682677997, 79.47830114687353, 79.3772153699245, 79.27587158258149, 79.17439295434937, 79.07307358453718, 78.97237532888748, 78.87292433093153, 78.77550599359282]
        alpha = np.logspace(0, np.log10(2.e5) , len(chi2))[::-1]
        plt.subplot(2,1,1)
        plt.loglog(alpha, chi2)
        plt.xlabel(r'$\alpha$')
        plt.ylabel(r'$\chi^2$')
        plt.subplot(2,1,2)
        curv = curv(0.2*np.log10(alpha), np.log10(chi2))[0]
        plt.semilogx(alpha, curv)
        plt.xlabel(r'$\alpha$')
        plt.ylabel(r'$\mathrm{curvature}(\log \chi^2(\gamma \log\alpha))$')


    Parameters
    ==========
    gamma : float
        the parameter by which the argument of the curve is multiplied
        before calculating the curvature, defaults to ``0.2``
    name : str
        the name of the method, defaults to `Chi2CurvatureAnalyzer`.

    Attributes
    ==========
    A_out : array (vector)
        the output, i.e. the one true spectrum
    alpha_index : int
        the index of the output in the ``A_values`` array
    curvature : array
        the curvature of :math:`\log\chi^2(\gamma \log\alpha)`
    info : str
        some information about what the analyzer did
    """

    def __init__(self, gamma=0.2, name=None):
        self.gamma = gamma
        super(Chi2CurvatureAnalyzer, self).__init__(name=name)

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
        res['curvature'], dchi2_1, dchi2_2 = curv(
            self.gamma * np.log10(maxent_result.alpha), np.log10(elem(maxent_result.chi2)))
        res['alpha_index'] = np.nanargmax(res['curvature'])
        res['A_out'] = elem(maxent_result.A)[res['alpha_index']]
        res['gamma'] = self.gamma
        res['name'] = self.name
        res['info'] = \
            'Ideal alpha (curvature): {} (= index {} zero-based)'.format(
            maxent_result.alpha[res['alpha_index']], res['alpha_index'])
        return res
