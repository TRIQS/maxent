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
from triqs_maxent import *
from triqs_maxent.triqs_support import *
import matplotlib.pyplot as plt

import sys
import traceback
import warnings


def warn_with_traceback(message,
                        category,
                        filename,
                        lineno,
                        file=None,
                        line=None):
    log = file if hasattr(file, 'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message,
                                     category,
                                     filename,
                                     lineno,
                                     line))
warnings.showwarning = warn_with_traceback

# to make it reproducible
np.random.seed(658436166)

beta = 40
tau = np.linspace(0, beta, 100)
omega = HyperbolicOmegaMesh(omega_min=-10, omega_max=10, n_points=100)
K = TauKernel(tau=tau, omega=omega, beta=beta)
# here we construct the G(tau)
A = np.exp(-omega**2)
A /= np.trapz(A, omega)
G = np.dot(K.K, A)
G += 1.e-4 * np.random.randn(len(G))
err = 1.e-4 * np.ones(len(G))

D = FlatDefaultModel(omega=omega)

chi2 = NormalChi2(K=K, G=G, err=err)
S = NormalEntropy(D=D)
H_of_v = NormalH_of_v(D=D, K=K)

Q = MaxEntCostFunction(chi2=chi2, S=S, H_of_v=H_of_v)

logtaker = Logtaker()

minimizer = LevenbergMinimizer(maxiter=10000)

alpha_values = LogAlphaMesh(alpha_max=6000, alpha_min=8, n_points=5)

ml = MaxEntLoop(cost_function=Q, minimizer=minimizer,
                alpha_mesh=alpha_values, logtaker=logtaker)
result = ml.run()

if not if_no_triqs():
    from pytriqs.archive import HDFArchive
    with HDFArchive('maxent_loop.h5', 'a') as ar:
        ar['result'] = result.data
else:
    result.data
