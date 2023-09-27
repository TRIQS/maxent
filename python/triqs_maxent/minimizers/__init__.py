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



from .levenberg_minimizer import LevenbergMinimizer
# yes, it looks strange, but this is the only way the automatic addition
# to __all__ works
from . import convergence_methods
from .convergence_methods import *

__all__ = ['LevenbergMinimizer']
# add all classes from convergence_methods to __all__
for elem in dir(convergence_methods):
    if elem[0].isupper():
        __all__.append(elem)
