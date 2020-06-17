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

if not hasattr(np, 'full'):
    # polyfill full for older numpy:
    np.full = lambda a, f: np.zeros(a) + f

def fit_piecewise(logx, logy, p2_deg=0):
    """ piecewise linear fit (e.g. for chi2)

    The curve is fitted by two linear curves; for the first few indices,
    the fit is done by choosing both the slope and the y-axis intercept.
    If ``p2_deg`` is ``0``, a constant curve is fitted (i.e., the only
    fit parameter is the y-axis intercept) for the last few indices,
    if it is ``1`` also the slope is fitted.

    The point (i.e., data point index) up to where the first linear curve
    is used (and from where the second linear curve is used) is also determined
    by minimizing the misfit. I.e., we search the minimum misfit with
    respect to the fit parameters of the linear curve and with respect
    to the index up to where the first curve is used.

    Parameters
    ----------
    logx : array
        the x-coordinates of the curve that should be fitted piecewise
    logy : array
        the y-corrdinates of the curve that should be fitted piecewise
    p2_deg : int
        ``0`` or ``1``, giving the polynomial degree of the second curve
    """
    chi2 = np.full(len(logx), np.nan)
    p1 = [0] * len(logx)
    p2 = [0] * len(logx)

    def denan(what, check=None):
        if check is None:
            check = what
        return what[np.logical_not(np.isnan(check))]
    for i in range(2, len(logx) - 2):
        chi2[i] = 0.0
        try:
            p1[i], residuals, rank, singular_values, rcond = np.polyfit(
                denan(logx[:i], logy[:i]), denan(logy[:i]), deg=1, full=True)
            if len(residuals) > 0:
                chi2[i] += residuals[0]
            p2[i], residuals, rank, singular_values, rcond = np.polyfit(
                denan(logx[i:], logy[i:]), denan(logy[i:]),
                deg=p2_deg, full=True)
            if len(residuals) > 0:
                chi2[i] += residuals[0]
        except TypeError:
            p1[i] = np.nan
            p2[i] = np.nan
            chi2[i] = np.nan
    i = np.nanargmin(chi2)
    if np.isnan(i):
        raise ValueError('chi2 is all NaN')

    X_x = ((p2[i][1] if p2_deg == 1 else p2[i][0]) - p1[i][1]) / \
        (p1[i][0] - (p2[i][0] if p2_deg == 1 else 0.0))
    idx = np.nanargmin(np.abs(logx - X_x))

    if np.isnan(idx):
        raise ValueError('abs(logx - X_x) is all NaN')

    return idx, (p1[i], p2[i])


class LineFitAnalyzer(Analyzer):
    r""" Analyzer searching the kink in :math:`\chi^2`

    This analyzer fits a piecewise linear function consisting of two
    straight lines to the function :math:`\log\chi^2(\log\alpha)` (the
    blue curve in the example plot below),
    using the slope and y-intercept of the linear functions (the orange and
    green curves in the example plot below) as fit
    parameters, together with the point where the description changes
    from one piece to the other.
    The :math:`\alpha`-value closest to their intersection is chosen
    to give the final spectral function.

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        from triqs_maxent.analyzers.linefit_analyzer import fit_piecewise
        chi2 = [29349.131935651938, 22046.546280176568, 16571.918154748197, 12487.464413222642, 9445.982385619374, 7178.502063034175, 5481.286491560124, 4203.095677764095, 3233.499702682774, 2492.7723059042005, 1923.6134333677408, 1484.7009011237717, 1145.8924661597946, 884.8005310524965, 684.4330049956029, 531.6146648070567, 415.9506388789646, 329.14872296705806, 264.5685376022979, 216.90786277834727, 181.96883131892676, 156.46999971120204, 137.88618534909426, 124.30790112614721, 114.31767388342703, 106.88286008781692, 101.26499523088836, 96.94522657033058, 93.56467085587808, 90.87798882482437, 88.71820342038981, 86.97077896408142, 85.5551277292544, 84.41192628409034, 83.49484873710317, 82.76553387129734, 82.19079735392212, 81.74128618303814, 81.39095719753091, 81.11694546963328, 80.89956952736446, 80.72238218084304, 80.57227364014592, 80.43956708639224, 80.31784061686754, 80.20324577429295, 80.09354393085312, 79.98731507910087, 79.88352134738581, 79.78132986379869, 79.68005975635513, 79.57917682677997, 79.47830114687353, 79.3772153699245, 79.27587158258149, 79.17439295434937, 79.07307358453718, 78.97237532888748, 78.87292433093153, 78.77550599359282]
        alpha = np.logspace(0, np.log10(2.e5) , len(chi2))[::-1]
        plt.loglog(alpha, chi2)
        plt.xlabel(r'$\alpha$')
        plt.ylabel(r'$\chi^2$')
        _, p = \
            fit_piecewise(np.log(alpha),
                          np.log(chi2))
        yl = plt.ylim()
        plt.loglog(alpha, np.exp(p[0][1]+p[0][0]*np.log(alpha)))
        plt.loglog(alpha, np.exp(p[1][0]*np.ones(len(alpha))))
        plt.ylim(yl)

    Please be careful to include a sufficient number of :math:`\alpha`
    points; however, when using too large :math:`\alpha`-values, the
    curve starts to flatten again - then, a piecewise fit using three
    linear curves would be necessary (which is not implemented).

    Parameters
    ==========
    linefit_deg : int
        whether the leftmost piece should have zero slope (``linefit_deg=0``)
        or whether its slope should be fitted (``linefit_deg=1``);
        defaults to ``0``.
    name : str
        the name of the method, defaults to `LineFitAnalyzer`.

    Attributes
    ==========
    A_out : array (vector)
        the output, i.e. the one true spectrum
    alpha_index : int
        the index of the output in the ``A_values`` array
    linefit_params : list
        the polyfit parameters of the two lines
    info : str
        some information about what the analyzer did
    """

    def __init__(self, linefit_deg=0, name=None):
        self.linefit_deg = linefit_deg
        super(LineFitAnalyzer, self).__init__(name=name)

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
        res['alpha_index'], res['linefit_params'] = \
            fit_piecewise(np.log(maxent_result.alpha),
                          np.log(elem(maxent_result.chi2)),
                          self.linefit_deg)

        res['A_out'] = elem(maxent_result.A)[res['alpha_index']]
        res['linefit_deg'] = self.linefit_deg
        res['name'] = self.name
        res['info'] = \
            'Ideal alpha (linefit): {} (= index {} zero-based)'.format(
            maxent_result.alpha[res['alpha_index']],
            res['alpha_index'])
        return res
