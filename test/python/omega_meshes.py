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


from __future__ import print_function
from triqs_maxent.omega_meshes import *
import numpy as np
import copy
from triqs_maxent.triqs_support import assert_text_files_equal
import sys


# the old get_delta function
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


def testfunction():
    for mesh in [LinearOmegaMesh, LorentzianOmegaMesh,
                 LorentzianSmallerOmegaMesh, HyperbolicOmegaMesh]:
        print(mesh.__name__)
        m = mesh(omega_min=-10, omega_max=10, n_points=10)
        for w in m:
            print('%.8f' % w)
        print('-' * 80)

        maxdiff = np.max(np.abs(get_delta(m) - m.delta))
        assert maxdiff < 1.e-15, "delta not correct (diff {})".format(maxdiff)

        func = np.random.rand(len(m))
        integral1 = np.trapz(func, m)
        integral2 = np.sum(func * m.delta)
        assert np.abs(integral1 - integral2) < 1.e-14,\
            "integration does not match"

    meshcopy = m.copy()
    assert np.all(meshcopy == m), "m.copy(): data not equal"
    assert meshcopy.omega_min == m.omega_min, "m.copy(): omega_min not equal"
    assert meshcopy.omega_max == m.omega_max, "m.copy(): omega_max not equal"
    assert meshcopy.n_points == m.n_points, "m.copy(): n_points not equal"

    meshcopy = copy.deepcopy(m)
    assert np.all(meshcopy == m), "copy.deepcopy: data not equal"
    assert meshcopy.omega_min == m.omega_min, "copy.deepcopy: omega_min not equal"
    assert meshcopy.omega_max == m.omega_max, "copy.deepcopy: omega_max not equal"
    assert meshcopy.n_points == m.n_points, "copy.deepcopy: n_points not equal"

    # The following does not work. We call it a feature now.
    # meshcopy = np.copy(m)
    # assert np.all(meshcopy == m), "np.copy: data not equal"
    # assert meshcopy.omega_min == m.omega_min, "np.copy: omega_min not equal"
    # assert meshcopy.omega_max == m.omega_max, "np.copy: omega_max not equal"
    # assert meshcopy.n_points == m.n_points, "np.copy: n_points not equal"


with open('omega_meshes.out', 'w') as out:
    sys.stdout = out

    testfunction()

assert_text_files_equal('omega_meshes.ref', 'omega_meshes.out')
