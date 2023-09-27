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
from .analyzer import Analyzer, AnalyzerResult


def get_delta(v):
    """ Get the integration delta for arbitrarily-spaced vector

    Parameters
    ----------
    v : numpy array
        values of e.g. omega

    Returns
    -------
    delta : numpy array
        integration delta v for the generalized trapezoidal integration
    """
    delta = np.empty(len(v))
    delta[1:-1] = (v[2:] - v[:-2]) / 2.0
    delta[0] = (v[1] - v[0]) / 2.0
    delta[-1] = (v[-1] - v[-2]) / 2.0
    return delta


class BryanAnalyzer(Analyzer):
    r""" Bryan's analyzer

    This analyzer averages the spectral :math:`A_\alpha(\omega)` over
    :math:`\alpha`, weighted by the probability :math:`p(\alpha)`.
    This is known as the Bryan MaxEnt method.

    Given the probability (e.g., the one in the following plot, which is not normalized),
    first the correct normalization is performed and then an average over all
    spectral functions weighted by the probability is done.

    The normalized probability is

    .. math::

        \bar{p} = \sum_i p_i \Delta\alpha_i

    and the output spectral function is

    .. math::

        A_{out}(\omega) = \sum_i \bar{p}_i A_{\alpha_i}(\omega) \Delta\alpha_i.

    If ``average_by_integration`` is ``True``, :math:`\Delta\alpha` is
    calculated using the trapezoidal rule; else, it is just taken to be one.

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        log_prob = [ -133.77644825014883, -131.1811908969303, -128.87341313304904, -126.82613548679048, -125.01469769984743, -123.41663004550679, -122.01152204906518, -120.78088130473434, -119.70797803270561, -118.77769170465523, -117.9763548787618, -117.29161778070646, -116.71230318712861, -116.22830157384657, -115.83044965526712, -115.51044678567999, -115.26075975426504, -115.07455020581381, -114.94560323182378, -114.86826719005813, -114.83739408789381, -114.84829352389285, -114.89667888116442, -114.97863680101291, -115.0905799690475, -115.22921464699886, -115.39151433606989, -115.5746861915961, -115.77614736900517, -115.99350342514697, -116.2245185175958, -116.4671120942432, -116.71933049377914, -116.97932853127824, -117.24536855479826, -117.5158025086968, -117.78907087999886, -118.06368576040379, -118.33825708524627, -118.61146172746481, -118.88209449357468, -119.14906945217739, -119.41146583894628, -119.66854741331817, -119.91990114496151, -120.16545905975504, -120.4056296141437, -120.64136582811162, -120.87420189716848, -121.10607386134298, -121.3389742988314, -121.57441157558964, -121.81295817443836, -122.05409935782784, -122.29644776767404, -122.53822785898488, -122.77762927908799, -123.01305326403487, -123.24343091919015, -123.46821951105106]
        alpha = np.logspace(-1, np.log10(30), len(log_prob))[::-1]
        p = np.exp(log_prob - np.max(log_prob))
        plt.semilogx(alpha, p)
        plt.xlabel(r'$\alpha$')
        plt.ylabel(r'$p(\alpha)$')


    Parameters
    ==========
    average_by_integration : bool
        if True, the average spectral function is calculated by integrating
        over all alphas weighted by the probability using the trapezoidal
        rule;
        if False (the default), it is calculated by summing over all alphas weighted
        by the probability
    name : str
        the name of the method, defaults to `BryanAnalyzer`.

    Attributes
    ==========
    A_out : array (vector)
        the output, i.e. the one true spectrum
    info : str
        some information about what the analyzer did
    """

    def __init__(self, average_by_integration=False, name=None):
        self.average_by_integration = average_by_integration
        super(BryanAnalyzer, self).__init__(name=name)

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
            res['info'] = 'Probability not calculated. Cannot use BryanAnalyzer.'
            return res

        res['A_out'] = np.zeros(maxent_result.A.shape[-1])

        prob = np.exp(elem(maxent_result.probability) -
                      np.nanmax(elem(maxent_result.probability)))
        prob_L = np.where(np.logical_not(np.isnan(prob)))
        # normalize probability
        if self.average_by_integration:
            prob[prob_L] /= np.trapz(prob[prob_L], maxent_result.alpha[prob_L])
            delta_alpha = np.full(len(prob), np.nan)
            delta_alpha[prob_L] = get_delta(maxent_result.alpha[prob_L])
        else:
            prob[prob_L] /= np.sum(prob[prob_L])

        for i in range(len(maxent_result.alpha)):
            if np.isnan(prob[i]):
                continue
            if self.average_by_integration:
                res['A_out'] += prob[i] * \
                    elem(maxent_result.A)[i, :] * delta_alpha[i]
            else:
                res['A_out'] += prob[i] * elem(maxent_result.A)[i, :]

        res['info'] = 'Bryan analyzer: average of A weighted by probability calculated.'
        return res
