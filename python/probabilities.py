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



from .functions import GenericFunction
import numpy as np
import copy


class NormalLogProbability(GenericFunction):
    r""" calculate the :math:`\log` of the probability of :math:`\alpha`

    Parameters
    ----------
    log_norm_S : function
        normalization of the entropy as a function of (:math:`\alpha`, :math:`N_{\omega}`),

    log_measure : function
        measure for the integration as a function of :math:`A`,

        The default is

        .. math ::

            \frac{1}{2} \det(-\frac{\partial^2 S}{\partial A_i \partial A_j}).

        Skilling, Classic Maximum Entropy, Maximum Entropy and Bayesian Methods, Kluwer 1989
        gives this as a general expression for the measure.

    log_prior_alpha : function
        prior of alpha as a function of alpha, default is :math:`-\log(\alpha)`, i.e.
        Jeffrey's prior (a constant prior according to S.F. Gull, Developments in Maximum Entropy Data Analysis,
        in J. Skilling (ed.) Maximum Entropy and Baysian Methods, p. 57, Kluwer 1989 is possible by
        setting it to ``lambda alpha : 0``)
    """

    def __init__(self,
                 log_measure=None,
                 log_norm_S=None,
                 log_prior_alpha=None):
        self.log_measure = log_measure
        self.log_norm_S = log_norm_S
        self.log_prior_alpha = log_prior_alpha

        if self.log_norm_S is None:
            self.log_norm_S = lambda alpha, N_w: (N_w / 2.0) * np.log(alpha)
        if self.log_measure is None:
            self.log_measure = \
                lambda S: 1 / 2.0 * np.linalg.slogdet(-S.dd())[1]
        if self.log_prior_alpha is None:
            self.log_prior_alpha = lambda alpha: -np.log(alpha)

        self.cost_function = None

    def __call__(self, cost_function):
        ret = copy.copy(self)  # a shallow copy
        ret.cost_function = cost_function
        return ret

    def f(self):
        ddQ = self.cost_function.ddH()
        _, p = np.linalg.slogdet(ddQ)
        p = -0.5 * p
        p += self.log_measure(self.cost_function.S)
        p += self.log_norm_S(self.cost_function._alpha,
                             len(self.cost_function.H_of_v.f()))
        p -= self.cost_function.f()
        p += self.log_prior_alpha(self.cost_function._alpha)
        return p
