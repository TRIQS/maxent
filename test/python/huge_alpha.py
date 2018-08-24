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
from triqs_maxent.tau_maxent import *
from triqs_maxent.triqs_support import *
if if_triqs_1():
    from pytriqs.gf.local import *
elif if_triqs_2():
    from pytriqs.gf import *
import numpy as np

# This test checks if for a huge alpha the default model is reproduced

tm = TauMaxEnt()

if if_no_triqs():
    tm.set_G_tau_file('g_tau_semicircular.dat')
else:
    G_iw = GfImFreq(beta=40, indices=[0], n_points=200)
    G_iw << SemiCircular(1.0)
    tm.set_G_iw(G_iw)
tm.set_G_tau_data(tm.tau, tm.G + 1.e-4 * np.random.randn(len(tm.G)))
tm.maxent_loop.cost_function.d_dv = False
# Trick use huge alpha in mesh
tm.alpha_mesh = LogAlphaMesh(alpha_min=1e10 - 1, alpha_max=1e10, n_points=5)
tm.set_error(5.e-4)
tm.reduce_singular_space = 1.e-16
result = tm.run()
result._all_fields.append('H')

assert np.max(result.H - tm.D.D) < 1e-6,\
    'For a huge alpha default model is not reproduced (err = {}).'.format(np.max(result.H - tm.D.D))
