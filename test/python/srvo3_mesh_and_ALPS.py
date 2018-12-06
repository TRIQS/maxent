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
from triqs_maxent import *
from pytriqs.archive import *
import numpy as np

# This test runs TauMaxEnt for srvo3 at beta = 37.5.
# Additionally the lower alpha value is choosen such that it
# coincides with the maxspec result of the ALPS/maxent code
# Note that we have removed the factor 2/ntau in the ALPS/maxent code,
# as we think that this is a bug. To be checked with the developers.

generate_ref = False


def numpy_assert(a, b, dec):
    return np.testing.assert_almost_equal(a, b, decimal=dec)

alpha_mesh = LogAlphaMesh(alpha_min=5.514845959 / 500,
                          alpha_max=100,
                          n_points=2)
om_max = 9.18000974

tm = TauMaxEnt(cost_function=BryanCostFunction())
tm.set_G_tau_file(filename='srvo3_mesh_and_ALPS_gtau.dat',
                  tau_col=0,
                  G_col=1,
                  err_col=2)
tm.omega = LorentzianOmegaMesh(omega_min=-om_max,
                               omega_max=om_max,
                               n_points=500)
tm.alpha_mesh = alpha_mesh
res_lor = tm.run()

tm = TauMaxEnt(cost_function=BryanCostFunction())
tm.set_G_tau_file(filename='srvo3_mesh_and_ALPS_gtau.dat',
                  tau_col=0,
                  G_col=1,
                  err_col=2)
tm.omega = LinearOmegaMesh(omega_min=-om_max, omega_max=om_max, n_points=500)
tm.alpha_mesh = alpha_mesh
res_lin = tm.run()

tm = TauMaxEnt(cost_function=BryanCostFunction())
tm.set_G_tau_file(filename='srvo3_mesh_and_ALPS_gtau.dat',
                  tau_col=0,
                  G_col=1,
                  err_col=2)
tm.omega = HyperbolicOmegaMesh(omega_min=-om_max,
                               omega_max=om_max,
                               n_points=500)
tm.alpha_mesh = alpha_mesh
res_hyp = tm.run()

if generate_ref:
    with HDFArchive('srvo3_mesh_and_ALPS.ref.h5', 'w') as ar:
        alps_data = np.loadtxt('srvo3_mesh_and_ALPS_maxspec.dat')
        ar['A_maxspec_ALPS'] = [alps_data[:, 0], alps_data[:, 1]]
        ar['A_maxent_lin'] = [res_lin.omega, res_lin.A[1, :]]
        ar['A_maxent_hyp'] = [res_hyp.omega, res_hyp.A[1, :]]
        ar['A_maxent_lor'] = [res_lor.omega, res_lor.A[1, :]]

# compare different meshes
lin_omega = np.linspace(-om_max, om_max, 500)
numpy_assert(np.interp(lin_omega, res_lin.omega, res_lin.A[1, :]),
             np.interp(lin_omega, res_lor.omega, res_lor.A[1, :]), 2)
numpy_assert(np.interp(lin_omega, res_lin.omega, res_lin.A[1, :]),
             np.interp(lin_omega, res_hyp.omega, res_hyp.A[1, :]), 2)
numpy_assert(np.interp(lin_omega, res_lor.omega, res_lor.A[1, :]),
             np.interp(lin_omega, res_hyp.omega, res_hyp.A[1, :]), 2)

# compare to reference data
with HDFArchive('srvo3_mesh_and_ALPS.ref.h5', 'r') as ar:
    numpy_assert(res_lin.omega, ar['A_maxent_lin'][0], 6)
    numpy_assert(res_lor.omega, ar['A_maxent_lor'][0], 6)
    numpy_assert(res_hyp.omega, ar['A_maxent_hyp'][0], 6)
    numpy_assert(res_lor.omega, ar['A_maxspec_ALPS'][0], 6)

    numpy_assert(res_lin.A[1, :], ar['A_maxent_lin'][1], 6)
    numpy_assert(res_lor.A[1, :], ar['A_maxent_lor'][1], 6)
    numpy_assert(res_hyp.A[1, :], ar['A_maxent_hyp'][1], 6)
    numpy_assert(res_lor.A[1, :], ar['A_maxspec_ALPS'][1], 2)
