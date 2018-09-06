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
from triqs_maxent.omega_meshes import *
from triqs_maxent.default_models import *

w = HyperbolicOmegaMesh(omega_min=-10, omega_max=10, n_points=100)

D1 = FlatDefaultModel(omega=w)
assert np.all(np.abs(D1.D / w.delta - 1.0 / 20.0) < 1.e-15), \
    "flat default model is wrong"
assert np.sum(D1.D) - 1.0 < 2.e-15, "flat default model integral is wrong"

D2 = DataDefaultModel(D1.D / w.delta, w)
assert np.all(D1.D == D2.D), "data default model is wrong"

w_lin = LinearOmegaMesh(omega_min=-10, omega_max=10, n_points=50)
D3 = DataDefaultModel(D1.D / w.delta, w, w_lin)
assert len(D3) == 50, "data default model has wrong length"
assert np.all(np.abs(D3.D / w_lin.delta - 1.0 / 20.0) < 1.e-15), \
    "data default model is wrong"

w_2p = DataOmegaMesh([-10, 10])
D4 = DataDefaultModel([-10, 10], w_2p, w_lin)
assert np.all(np.abs(D4.D / w_lin.delta - np.linspace(-10, 10, 50))
              < 1.e-15), "data default model is wrong"

D5 = DataDefaultModel([-9, 11], w_2p, w)
# the integral of x+1 is x^2/2+x
# the definite integral in the interval [-10,10] is 20.0
# note that the trapezoidal rule is exact for a linear function
assert np.abs(np.sum(D5.D) - 20.0) < 1.e-13, \
    "data default model integral is wrong {}".format(np.abs(np.trapz(D5.D, w) - 20.0))

D5.omega = w_2p
assert len(D5.D) == len(w), "parameter change without parameter_change"
D5.parameter_change()
assert len(D5.D) == 2, "no parameter change after parameter_change"
assert np.abs(np.sum(D5.D) - 20.0) < 1.e-13, \
    "data default model integral is wrong"
