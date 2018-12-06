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


from triqs_maxent.sigma_continuator import *
from triqs_maxent.tau_maxent import *
from triqs_maxent.triqs_support import *
if if_triqs_1():
    from pytriqs.gf.local import *
elif if_triqs_2():
    from pytriqs.gf import *
from pytriqs.archive import *
from pytriqs.utility.comparison_tests import *

np_tau = 10000
beta = 50.0

# Matusbara Green function for this test
giw = GfImFreq(indices=[1], beta=beta)
giw << SemiCircular(half_bandwidth=1)

# Create an imaginary-time Green function
gt = GfImTime(indices=[1], beta=beta)
gt << InverseFourier(giw)

# set G_tau in MaxEnt
M = TauMaxEnt()
M.set_G_tau(gt)

# Semi-circular GfReFreq
gw = GfReFreq(indices=[1], window=(-2, 2), n_points=1000)
gw << SemiCircular(half_bandwidth=1)

# Check construction of Gaux with inversion method
shift_inv = 2.4
siw = giw.copy()
siw << Omega + shift_inv - inverse(giw)

SC1 = InversionSigmaContinuator(siw, shift_inv)
M1 = TauMaxEnt()
M1.set_G_iw(SC1.Gaux_iw, np_tau=np_tau)

assert_arrays_are_close(M.G, M1.G)
assert_arrays_are_close(M.tau, M1.tau)

# Check if construction works for BlockGf
Siw = BlockGf(name_list=('b1', 'b2'), block_list=(siw, siw), make_copies=False)

SC1b = InversionSigmaContinuator(Siw, shift_inv)

assert_arrays_are_close(SC1b.Gaux_iw['b1'].data, giw.data)
assert_arrays_are_close(SC1b.Gaux_iw['b2'].data, giw.data)


# Check construction of Gaux with direct method
shift_cor = -1.3
siw = giw.copy()
siw << siw + shift_cor

SC2 = DirectSigmaContinuator(siw)
M2 = TauMaxEnt()
M2.set_G_iw(SC2.Gaux_iw, np_tau=np_tau)

assert_arrays_are_close(M.G, M2.G)
assert_arrays_are_close(M.tau, M2.tau)

# Check if construction works for BlockGf
Siw = BlockGf(name_list=('b1', 'b2'), block_list=(siw, siw), make_copies=False)
SC2b = DirectSigmaContinuator(Siw)

assert_arrays_are_close(SC2b.Gaux_iw['b1'].data, giw.data)
assert_arrays_are_close(SC2b.Gaux_iw['b2'].data, giw.data)

# check if S_w is calculated correctly
sw = gw.copy()
sw << Omega + shift_inv - inverse(gw)

Gw = BlockGf(name_list=('b1', 'b2'), block_list=(gw, gw), make_copies=False)
SC1b.set_Gaux_w(Gw)

assert_gfs_are_close(SC1b.S_w['b1'], sw)
assert_gfs_are_close(SC1b.S_w['b2'], sw)

sw = gw.copy()
sw << gw + shift_cor

SC2b.set_Gaux_w(Gw)

assert_gfs_are_close(SC2b.S_w['b1'], sw)
assert_gfs_are_close(SC2b.S_w['b2'], sw)

# Inversion with different shifts for different blocks
shift_inv1 = 2.4
shift_inv2 = 3.3
siw1 = giw.copy()
siw1 << Omega + shift_inv1 - inverse(giw)
siw2 = giw.copy()
siw2 << Omega + shift_inv2 - inverse(giw)

Siw = BlockGf(name_list=('b1', 'b2'),
              block_list=(siw1, siw2), make_copies=False)

SC3 = InversionSigmaContinuator(Siw, {'b1': shift_inv1, 'b2': shift_inv2})

assert_arrays_are_close(SC3.Gaux_iw['b1'].data, giw.data)
assert_arrays_are_close(SC3.Gaux_iw['b2'].data, giw.data)

# Write to h5
with HDFArchive('sigma_continuation.out.h5', 'w') as ar:
    ar['SC1b'] = SC1b
    ar['SC2b'] = SC2b

# Read from h5
with HDFArchive('sigma_continuation.out.h5', 'r') as ar:
    SC1c = ar['SC1b']
    SC2c = ar['SC2b']

def object_dict_equal(a, b):
    for key in a:
        if key in ['S_iw', 'S_w', 'Gaux_w', 'Gaux_iw']:
            pass
        else:
            assert a[key] == b[key], 'SigmaContinuator h5 save/load failed'


object_dict_equal(SC1b.__dict__, SC1c.__dict__)
object_dict_equal(SC2b.__dict__, SC2c.__dict__)

assert_block_gfs_are_close(SC1b.S_iw, SC1c.S_iw)
assert_block_gfs_are_close(SC1b.S_w, SC1c.S_w)
assert_block_gfs_are_close(SC2b.S_iw, SC2c.S_iw)
assert_block_gfs_are_close(SC2b.S_w, SC2c.S_w)
