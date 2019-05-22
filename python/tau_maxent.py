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
from .triqs_support import *
if if_triqs_1():
    from pytriqs.gf.local import *
elif if_triqs_2():
    from pytriqs.gf import *
from .maxent_loop import MaxEntLoop
from .omega_meshes import HyperbolicOmegaMesh
from .default_models import FlatDefaultModel
from .kernels import TauKernel
from .cost_functions import BryanCostFunction, MaxEntCostFunction
from .functions import NormalEntropy, PlusMinusEntropy
from .functions import NormalH_of_v, PlusMinusH_of_v
import numpy as np
import copy


class TauMaxEnt(object):
    r""" Perform MaxEnt with a :math:`G(\tau)` kernel.

    The methods and properties of :py:class:`.MaxEntLoop` are, in general,
    shadowed by ``TauMaxEnt``, i.e., they can be used in a ``TauMaxEnt``
    object as well.

    Parameters
    ----------
    cov_threshold : float
        when setting a covariance using :py:meth:`.TauMaxEnt.set_cov`, this threshold
        is used to ignore small eigenvalues
    **kwargs :
        are passed on to :py:class:`.MaxEntLoop`
    """

    # this is needed to make the getattr/setattr magic work
    maxent_loop = None

    def __init__(self, cov_threshold=1.e-14, **kwargs):

        self.maxent_loop = MaxEntLoop(**kwargs)

        omega = HyperbolicOmegaMesh()
        self.D = FlatDefaultModel(omega)
        # just some artificial tau data
        self.K = TauKernel([0, 1], omega)
        # N.B.: can only set omega after having initialized a kernel
        self.omega = omega
        self.cov_threshold = cov_threshold

    # getattr and setattr allows us to access the MaxEntLoop attributes
    # as if they were TauMaxEnt attributes
    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, 'maxent_loop'), name)

    def __setattr__(self, name, value):
        if hasattr(self.maxent_loop, name):
            return setattr(self.maxent_loop, name, value)
        else:
            object.__setattr__(self, name, value)

    @require_triqs
    def set_G_tau(self, G_tau, re=True, tau_new=None):
        r""" Set :math:`G(\tau)` from TRIQS GfImTime

        Parameters
        ==========
        G_tau : GfImTime
            The data for the analytic continuation.
            For Green functions with more than 1x1 matrix structure,
            choose a particular matrix element.
        re : logical
            If True, the real part of the data is continued, else the
            imaginary part.
        tau_new : array
            G_tau is interpolated on a new tau grid as given by tau_new.
            If not given, the original tau grid of G_tau is used.
        """

        if isinstance(G_tau, BlockGf):
            raise NotImplementedError(
                'TRIQS BlockGfs are not supported by TauMaxEnt.\n' +
                'Consider looping over over the blocks and calling TauMaxEnt individually for each GfImTime.')

        if not isinstance(G_tau.mesh, MeshImTime):
            raise Exception(
                'set_G_tau only accepts TRIQS GfImTime objects.\n' +
                'Use the appropriate set_* method for other data formats.')

        if tuple(G_tau.target_shape) not in [(1, 1), ()]:
            raise Exception(
                'Please choose one matrix element of G(tau) or use ElementwiseMaxEnt.')

        try:
            # this will work in TRIQS 2.1
            tau = np.array(list(G_tau.mesh.values())).real
        except AttributeError:
            # this will work in TRIQS 1.4
            tau = np.array(list(G_tau.mesh)).real
        if re:
            # if the target shape is (1,1), we need to pick the (0,0)
            # element; else, the target shape is (), i.e. only a one-dim
            # array is returned
            if tuple(G_tau.target_shape) == (1, 1):
                G = G_tau.data[:, 0, 0].real
            else:
                G = G_tau.data.real
        else:
            if tuple(G_tau.target_shape) == (1, 1):
                G = G_tau.data[:, 0, 0].imag
            else:
                G = G_tau.data.imag

        if tau_new is not None:
            self.G = np.interp(tau_new, tau, G)
            self.tau = tau_new
        else:
            self.G = G
            self.tau = tau

        self._transform(self._T, G_original_basis=True)

    @require_triqs
    def set_G_iw(self, G_iw, np_tau=-1, **kwargs):
        r""" Set :math:`G(\tau)` from TRIQS GfImFreq

        Parameters
        ==========
        G_iw : GfImFreq
            The data for the analytic continuation. A Fourier transform is performed
        np_tau : int
            Number of target tau points (must be >= ``(3*len(G_iw.mesh)+1`` or
            -1; then ``(3*len(G_iw.mesh)+1)`` is chosen)
        **kwargs :
            arguments supplied to :py:meth:`set_G_tau`
        """

        if isinstance(G_iw, BlockGf):
            raise NotImplementedError(
                'TRIQS BlockGfs are not supported by TauMaxEnt.\n' +
                'Consider looping over over the blocks and calling TauMaxEnt individually for each GfImFreq.')

        if not isinstance(G_iw.mesh, MeshImFreq):
            raise Exception(
                'set_G_iw only accepts TRIQS GfImFreq objects.\n' +
                'Use the appropriate set_* method for other data formats.')

        if np_tau < 0:
            # this is the shortest mesh that does not provoke an error
            # in set_from_inverse_fourier
            np_tau = 3*len(G_iw.mesh)+1
        try:
            # this will work in TRIQS 2.1
            G_tau = GfImTime(beta=G_iw.mesh.beta,
                             target_shape=G_iw.target_shape,
                             n_points=np_tau)
        except AssertionError:
            # this will work in TRIQS 1.4
            G_tau = GfImTime(beta=G_iw.mesh.beta, indices=G_iw.indices,
                             n_points=np_tau)
        G_tau.set_from_inverse_fourier(G_iw)
        self.set_G_tau(G_tau, **kwargs)

    def set_G_tau_data(self, tau, G_tau):
        r""" Set :math:`G(\tau)` from array.

        Parameters
        ==========
        tau : array
            tau-grid
        G_tau : array
            The data for the analytic continuation.
        """

        assert len(tau) == len(G_tau), \
            "tau and G_tau don't have the same dimension"
        self.tau = tau
        self.G = G_tau
        self._transform(self._T, G_original_basis=True)

    def set_G_tau_file(self, filename, tau_col=0, G_col=1, err_col=None):
        r""" Set :math:`G(\tau)` from data file.

        Parameters
        ==========
        filename : str
            the name of the file to load.
            The first column (see ``tau_col``) is the :math:`\tau`-grid,
            the second column (see ``G_col``) is the :math:`G(\tau)`
            data.
        tau_col : int
            the 0-based column number of the :math:`\tau`-grid
        G_col : int
            the 0-based column number of the :math:`G(\tau)`-data
        err_col : int
            the 0-based column number of the error-data or None if the
            error is not supplied via a file
        """

        dat = np.loadtxt(filename)
        self.tau = dat[:, tau_col]
        self.G = dat[:, G_col]
        if err_col is not None:
            self.err = dat[:, err_col]
            # undo any transformation
            self._transform(None, G_original_basis=True)
        else:
            self._transform(self._T, G_original_basis=True)

    def set_error(self, error):
        r""" Set error from array.

        Parameters
        ==========
        error : scalar or array
            the error of the data, either in the same shape as the
            supplied ``G_tau`` or as a scalar (then it's the same for
            all :math:`\tau`-values).
        """
        if not np.all(np.isreal(error)):
            raise Exception('complex error supplied, only real accepted')
        else:
            error = np.real(error)

        try:
            if len(error) == len(self.G):
                self.err = error
            else:
                raise Exception('Supply scalar error or with length of G_tau.')
        except TypeError:
            self.err = error * np.ones(self.G.shape)

        # undo any transformation
        self._transform(None)

    def set_cov(self, cov):
        r""" Set covariance matrix from array.

        The covariance matrix is diagonalized and the analytic continuation
        problem is rotated into the eigenbasis. Thus, diagonal errors can
        be used. The errors are the square roots of the eigenvalues of the
        covariance matrix. Due to numerics, small eigenvalues have to be ignored;
        this is done according to the parameter ``cov_threshold``.

        Parameters
        ==========
        cov : array
            covariance matrix, :math:`N_\tau \times N_\tau`.
            It has to be symmetric.
        """
        self.cov = cov

        assert np.max(np.abs(cov - cov.transpose())) < 1.e-10, \
            'Supplied covariance matrix is not symmetric.'

        # diagonalize the covariance matrix
        e, v = np.linalg.eigh(cov)
        if np.any(e < 0):
            self.logtaker.error_message(
                "Eigenvalues of the covariance matrix are not all positive; they will be ignored. Smallest negative value: {}",
                np.min(e))
        L = e >= self.cov_threshold  # no abs
        e = e[L]
        v = v[:, L]
        # to avoid error in chi2 when updating the kernel
        self.err = None
        # rotate self.G and the kernel into the basis where it is diagonal
        if hasattr(self.cost_function, "_G_orig"):
            self.G = self.cost_function._G_orig
        self._transform(v.conjugate().transpose())
        self.err = np.sqrt(e)

    def _transform_G(self, T_to, T_from=None):
        if T_to is None:
            if T_from is None:
                T = 1
            else:
                T = T_from.conjugate().transpose()
        else:
            if T_from is None:
                T = T_to
            else:
                T = np.dot(T_to, T_from.conjugate().transpose())
        self.G = np.dot(T, self.G)

    def _transform(self, T_, G_original_basis=False):
        """ rotate G and K from the left with the matrix T_

        T_ is the absolute rotation with respect to the unrotated
        quantities

        If you reset G in the original (untransformed) basis, call this
        function with the desired transformation and G_original_basis=True.

        Note that the error is not transformed accordingly (as the use-case
        is that a covariance matrix is diagonalized). If required, this
        has to be done by the user (however, only diagonal errors will
        be handled).
        """

        if G_original_basis:
            self.cost_function._G_orig = copy.deepcopy(self.G)

        self._transform_G(T_, None if G_original_basis else self._T)
        # this also sets self._T, which is shadowed
        self.K.transform(T_)
        # this is to trigger updating e.g. chi2 when K changes
        self.K = self.K

    def set_cov_file(self, filename):
        r""" Set covariance matrix from data file.

        See :py:meth:`.TauMaxEnt.set_cov` for more info.

        Parameters
        ==========
        filename : str
            the name of the file to load.
        """

        self.set_cov(np.loadtxt(filename))

    def get_tau(self):
        return self.maxent_loop.get_data_variable()

    def set_tau(self, tau, update_K=True,
                update_chi2=True, update_Q=True,
                update_H_of_v=True):
        self.maxent_loop.set_data_variable(tau,
                                           update_K=update_K,
                                           update_chi2=update_chi2,
                                           update_Q=update_Q,
                                           update_H_of_v=update_H_of_v)

    tau = property(get_tau, set_tau)

    @property
    def _T(self):
        return self.K._T
