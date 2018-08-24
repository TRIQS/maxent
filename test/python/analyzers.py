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
from triqs_maxent.analyzers import *
from triqs_maxent import MaxEntResult
from pytriqs.archive import HDFArchive, HDFArchiveGroup
#from pytriqs.utility.h5diff import h5diff
from pytriqs.utility.h5diff import compare

# to ensure reproducibility
np.random.seed(658436166)

res = MaxEntResult()
# create mock data
alpha = np.logspace(-4, 2, 20)[::-1]
res._saved['alpha'] = alpha
res._saved['chi2'] = 1.0 - 1.0 / (1 + np.exp((alpha - 100) / 10.0))
res._saved['A'] = np.random.rand(20, 100)
for i in range(20):
    res._saved['A'][i, :] /= np.trapz(res._saved['A'][i, :])
res._saved['S'] = (np.log(alpha) - 0.1)**3.0
res._saved['probability'] = -((np.log(alpha) - 1) / 1.0)**2

# check the individual analyzers
results = []
for ana in [LineFitAnalyzer(), Chi2CurvatureAnalyzer(), EntropyAnalyzer(),
            BryanAnalyzer(average_by_integration=False), ClassicAnalyzer()]:
    result = ana.analyze(res)
    assert np.abs(np.trapz(result['A_out']) - 1.0) < 1.e-10, \
        "{}: Norm of A(w) not one".format(result['name'])
    results.append(result)

with HDFArchive('analyzers.out.h5', 'w') as ar:
    for result in results:
        ar[result['name']] = result

# h5diff does not work with custom types
# h5diff('analyzers.out.h5','analyzers.ref.h5')
# basically copied the file pytriqs/utility/h5diff.py from the library
# apart from the lines marked with !!!
from pytriqs.archive import *
from pytriqs.utility.comparison_tests import *
try:
    from pytriqs.gf import GfImFreq, GfImTime, GfReFreq, GfReTime, GfLegendre, BlockGf
except:
    from pytriqs.gf.local import GfImFreq, GfImTime, GfReFreq, GfReTime, GfLegendre, BlockGf
from pytriqs.operators import *
from pytriqs.arrays import BlockMatrix
import sys
import numpy
failures = []
verbose = 0


def compare(key, a, b, level, precision):
    """Compare two objects named key"""

    if verbose and key:
        print(level * '  ' + "Comparing %s ...." % key)

    try:
        t = type(a)
        assert isinstance(b, t), "%s have different types" % key

        # !!! added AnalyzerResult in the following line
        if t == dict or isinstance(a, (HDFArchiveGroup, AnalyzerResult)):
            if list(a.keys()) != list(b.keys()):
                failures.append(
                    "Two archive groups '%s' with different keys \n %s \n vs\n %s" %
                    (key, list(
                        a.keys()), list(
                        b.keys())))
            for k in set(a.keys()).intersection(b.keys()):
                compare(key + '/' + k, a[k], b[k], level + 1, precision)

        # The TRIQS object which are comparable starts here ....
        elif t in [GfImFreq, GfImTime, GfReFreq, GfReTime, GfLegendre]:
            assert_gfs_are_close(a, b, precision)

        elif t in [BlockGf]:
            assert_block_gfs_are_close(a, b, precision)

        elif t in [Operator]:
            assert (a - b).is_zero(), "Many body operators not equal"

        elif t in [BlockMatrix]:
            for i in range(len(a.matrix_vec)):
                assert_arrays_are_close(a(i), b(i))

        # ... until here
        elif isinstance(a, numpy.ndarray):
            # !!! changed this so that it allows NaN values
            np.testing.assert_almost_equal(
                a, b, -int(np.round(np.log10(precision / 1.5))))

        elif t in [int, float, complex]:
            assert abs(a - b) < 1.e-10, " a-b = %" % (a - b)

        elif t in [bool, numpy.bool_]:
            assert a == b

        elif t in [list, tuple]:
            assert len(a) == len(b), "List of different size"
            for x, y in zip(a, b):
                compare(key, x, y, level + 1, precision)

        elif t in [str]:
            assert a == b, "Strings '%s' and '%s' are different" % (a, b)

        else:
            raise NotImplementedError(
                "The type %s for key '%s' is not comparable by h5diff" %
                (t, key))

    except (AssertionError, RuntimeError, ValueError) as e:
        # eliminate the lines starting with .., which are not the main error
        # message
        mess = '\n'.join([l for l in e.message.split('\n')
                          if l.strip() and not l.startswith('..')])
        failures.append("Comparison of key '%s'  has failed:\n "
                        "" % key + mess)

compare('', HDFArchive('analyzers.out.h5', 'r'),
        HDFArchive('analyzers.ref.h5', 'r'), 0, 1.e-6)

if failures:
    print ('-' * 50, file=sys.stderr)
    print ('-' * 20 + '  FAILED  ' + '-' * 20, file=sys.stderr)
    print ('-' * 50, file=sys.stderr)
    for x in failures:
        print (x, file=sys.stderr)
        print ('-' * 50, file=sys.stderr)
    raise RuntimeError("FAILED")
