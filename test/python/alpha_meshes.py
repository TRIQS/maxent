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
from triqs_maxent.alpha_meshes import *
from triqs_maxent.triqs_support import assert_text_files_equal
import sys

with open('alpha_meshes.out', 'w') as out:
    sys.stdout = out

    for mesh in [LinearAlphaMesh, LogAlphaMesh]:
        print(mesh.__name__)
        m = mesh(alpha_min=1.e-4, alpha_max=1.e2, n_points=10)
        for w in m:
            print('%.8f' % w)
        print('-' * 80)

        try:
            mesh(alpha_min=2, alpha_max=1)
            raise Exception('No error thrown when alpha_min > alpha max')
        except:
            pass

        try:
            mesh(alpha_min=-1, alpha_max=2)
            raise Exception('No error thrown when alpha < 0')
        except:
            pass

    mesh = DataAlphaMesh
    print(mesh.__name__)
    m = mesh(np.linspace(1.e-3, 100, 10))
    for w in m:
        print('%.8f' % w)
    print('-' * 80)

    try:
        mesh(np.linspace(-1, 10))
        raise Exception('No error thrown when alpha < 0')
    except:
        pass

assert_text_files_equal('alpha_meshes.out', 'alpha_meshes.ref')
