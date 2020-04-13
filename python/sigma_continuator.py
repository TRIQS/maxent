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



from .triqs_support import *
if if_triqs_1():
    from pytriqs.gf.local import *
elif if_triqs_2():
    from pytriqs.gf import *
from .maxent_util import *


class SigmaContinuator(object):
    """ Base class for the analytic continuation of self-energies"""

    @require_triqs
    def __init__(self):
        self._BlockGf = False
        self._constant_shift = {}

    @require_triqs
    def set_S_iw(self, S_iw):
        """ Set Matsubara self-energy """
        self.check_S_iw(S_iw)
        self.S_iw = S_iw

    def check_S_iw(self, S_iw):
        """ Check if self-energy is a TRIQS Green function"""

        if isinstance(S_iw, BlockGf):
            self._BlockGf = True
            for name, s_iw in S_iw:
                self.check_S_iw(s_iw)
        elif not isinstance(S_iw.mesh, MeshImFreq):
            raise NotImplementedError(
                'SigmaContinuator takes only TRIQS Green functions.')

    def check_Gaux_w(self, Gaux_w):
        """ Check if Gaux_w is a TRIQS Green function"""

        if isinstance(Gaux_w, BlockGf):
            if set(Gaux_w.indices) != set(self.S_iw.indices):
                raise IOError(
                    'Block names of Gaux_w do not agree with S_iw')
            for name, gaux_w in Gaux_w:
                self.check_Gaux_w(gaux_w)
        elif not isinstance(Gaux_w.mesh, MeshReFreq):
            raise NotImplementedError(
                'SigmaContinuator takes only TRIQS Green functions.')

    @require_triqs
    def set_Gaux_w_from_Aaux_w(self, Aaux_w, w_points, *args, **kwargs):
        r""" Calculate the auxiliary Green function :math:`G_{aux}(\omega)` from
        the auxiliary spectral function :math:`A_{aux}(\omega)` with :py:func:`.get_G_w_from_A_w()`
        The methods calls :py:meth:`.set_Gaux_w()`. Arguments are passed on to :py:func:`.get_G_w_from_A_w()`.

        Parameters
        ==========
        Aaux_w : dict or array
            Real-frequency spectral function as numpy array or in case of BlockGfs a dict
            of arrays with same key as S_iw.
        w_points : array
            Real-frequency grid points.
        """

        if self._BlockGf:
            if set(self.S_iw.indices) != set(Aaux_w.keys()):
                raise Exception(
                    'Indices of Aaux dictionary are not the same as in S_iw')

            self.set_Gaux_w(BlockGf(name_block_generator=[(name, get_G_w_from_A_w(
                Aaux_w[name], w_points, *args, **kwargs)) for name in list(self.S_iw.indices)], make_copies=False))

        else:
            if not isinstance(Aaux_w, np.ndarray):
                raise Exception('Please supply Aaux_w as a numpy ndarray.')

            self.set_Gaux_w(get_G_w_from_A_w(
                Aaux_w, w_points, *args, **kwargs))

    @require_triqs
    def set_Gaux_w(self, Gaux_w):
        r""" Set the auxiliary real-frequency Green function :math:`G_{aux}(\omega)` and calculate
        the real-frequency self-energy :math:`\Sigma(\omega)`. The result is stored as S_w.

        Parameters
        ==========
        Gaux_w : GfReFreq
            TRIQS real-frequency Green function
        """

        self.check_Gaux_w(Gaux_w)
        self.Gaux_w = Gaux_w
        self._calculate_S_w()

    def _calculate_Gaux_iw(self):
        raise NotImplementedError('Please use a subclass of SigmaContinuator.')

    def _calculate_S_w(self):
        raise NotImplementedError('Please use a subclass of SigmaContinuator.')

    def __reduce_to_dict__(self):
        ret = self.__dict__
        return ret

    @classmethod
    def __factory_from_dict__(cls, name, D):
        self = cls(D['S_iw'])
        for key in D:
            setattr(self, key, D[key])
        return self


class DirectSigmaContinuator(SigmaContinuator):
    r""" Direct method to construct auxiliary Green function

    This class constructs an auxiliary Green function by subtracting the
    high-frequency term of the self-energy
    :math:`G_{aux}(z) = \Sigma(z) - \Sigma(i\infty)` and normalizing the
    resulting auxiliary Green function.

    Parameters
    ==========
    S_iw : GfImFreq
        TRIQS Matsubara Green function
    """

    @require_triqs
    def __init__(self, S_iw):
        super(DirectSigmaContinuator, self).__init__()
        self.set_S_iw(S_iw)
        self._norm = {}
        self._calculate_Gaux_iw()

    def _calculate_Gaux_iw(self):
        def _calculate_gaux_iw(g):
            if tuple(g[1].target_shape) not in [(1, 1), ()]:
                raise NotImplementedError(
                    'DirectSigmaContinuator not implemented for matrix-valued Sigma')
            try:
                # this will only work in TRIQS 1.4
                tail = g[1].tail
            except:
                # this will work with TRIQS 2.1
                tail = g[1].fit_tail()[0]
            self._constant_shift[g[0]] = tail[0][0][0]
            self._norm[g[0]] = tail[1][0][0]
            g[1] << (g[1] - self._constant_shift[g[0]]) / self._norm[g[0]]

        self.Gaux_iw = self.S_iw.copy()
        list(map(_calculate_gaux_iw, self.Gaux_iw)) if self._BlockGf else _calculate_gaux_iw(
            ('0', self.Gaux_iw))

    def _calculate_S_w(self):
        def _calculate_s_w(s):
            if tuple(s[1].target_shape) not in [(1, 1), ()]:
                raise NotImplementedError(
                    'DirectSigmaContinuator not implemented for matrix-valued Sigma')
            s[1] << s[1] * self._norm[s[0]] + self._constant_shift[s[0]]

        self.S_w = self.Gaux_w.copy()
        list(map(_calculate_s_w, self.S_w)) if self._BlockGf else _calculate_s_w(
            ('0', self.S_w))


class InversionSigmaContinuator(SigmaContinuator):
    r""" Inversion method to construct auxiliary Green function

    This class constructs an auxiliary Green function using
    :math:`1/ (\omega + C - \Sigma(i\omega_n))`.

    Parameters
    ==========
    S_iw : GfImFreq
        Self-energy :math:`\Sigma(i\omega_n)` as TRIQS Matsubara Green function
    constant_shift : float
        Constant C (usually set to the double counting)
    """

    @require_triqs
    def __init__(self, S_iw, constant_shift=0):
        super(InversionSigmaContinuator, self).__init__()
        self.set_S_iw(S_iw)

        if not self._BlockGf:
            self._constant_shift['0'] = constant_shift
        elif isinstance(constant_shift, dict) and set(constant_shift.keys()) == set(self.S_iw.indices):
            self._constant_shift = constant_shift
        else:
            self._constant_shift = dict.fromkeys(
                set(self.S_iw.indices), constant_shift)

        self._calculate_Gaux_iw()

    def _calculate_Gaux_iw(self):
        def _calculate_gaux_iw(g):
            g[1] << Omega + self._constant_shift[g[0]] - g[1]
            g[1].invert()

        self.Gaux_iw = self.S_iw.copy()
        list(map(_calculate_gaux_iw, self.Gaux_iw)) if self._BlockGf else _calculate_gaux_iw(
            ('0', self.Gaux_iw))

    def _calculate_S_w(self):
        def _calculate_s_w(s):
            s[1] << Omega + self._constant_shift[s[0]] - inverse(s[1])

        self.S_w = self.Gaux_w.copy()
        list(map(_calculate_s_w, self.S_w)) if self._BlockGf else _calculate_s_w(
            ('0', self.S_w))


try:
    from h5.formats import register_class
    register_class(InversionSigmaContinuator)
    register_class(DirectSigmaContinuator)
except ImportError:  # notriqs
    pass
