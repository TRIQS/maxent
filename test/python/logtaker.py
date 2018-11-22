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
from triqs_maxent.logtaker import Logtaker, VerbosityFlags
import hashlib

logflags = [
    VerbosityFlags.Quiet,                              # 0
    VerbosityFlags.Header,                             # 1
    VerbosityFlags.ElementInfo,                        # 2
    VerbosityFlags.Timing,                             # 3
    VerbosityFlags.AlphaLoop,                          # 4
    VerbosityFlags.SolverDetails,                      # 5
    VerbosityFlags.Errors,                             # 6
    VerbosityFlags.Header | VerbosityFlags.Timing,     # 7
    VerbosityFlags.Default,                            # 8
]

l = Logtaker()
l.open_logfile('logtaker.dat', False)

for i, flag in enumerate(logflags):
    l.verbose = flag
    l.message(VerbosityFlags.Quiet, "=== Test #{} ===", i)
    l.message(VerbosityFlags.Header, "This is a header message.")
    l.message(VerbosityFlags.ElementInfo, "This is an element info message.")
    l.message(VerbosityFlags.Timing, "This is a timing message.")
    l.message(VerbosityFlags.AlphaLoop, "This is an alpha loop message.")
    l.message(VerbosityFlags.SolverDetails,
              "This is a solver details message.")
    l.error_message("This is an error message")
    l.message(VerbosityFlags.Timing | VerbosityFlags.Header,
              "This is a header + timing message.")

l.close_logfile()


def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

md5_1 = md5('logtaker.dat')
md5_2 = md5('logtaker.dat.ref')
assert md5_1 == md5_2, "logtaker file output differs"
