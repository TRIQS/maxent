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
    from triqs.gf.local import *
elif if_triqs_2():
    from triqs.gf import *
from .tau_maxent import TauMaxEnt
from .default_models import *
from .maxent_result import MaxEntResult
from .logtaker import VerbosityFlags
import numpy as np
import copy


class CallableMethodCheck(object):
    """ Allow to call two function and check for equality

    This class calls the two function, ``fun1`` and ``fun2``, that
    are given via its constructor, when called. If the result is equal,
    it is returned, if it is not, an exception is raised.
    """

    def __init__(self, name, fun1, fun2):
        self.name = name
        self.fun1 = fun1
        self.fun2 = fun2

    def __call__(self, *args, **kwargs):
        ret1 = self.fun1(*args, **kwargs)
        ret2 = self.fun2(*args, **kwargs)
        if np.all(ret1 == ret2):
            return ret1
        else:
            raise Exception('Element {n} not uniquely defined. '
                            'Use self.maxent_diagonal.{n} or '
                            'self.maxent_offdiagonal.{n}!'.format(n=self.name))


class ElementwiseMaxEnt(object):
    r""" Perform MaxEnt for a matrix, element-wise

    Parameters
    ----------
    use_hermiticity : bool
        whether matrix elements ij with i>j shall be taken from
        the complex conjugate of the element ji
    use_complex : bool
        whether complex numbers are used (i.e., whether the :math:`G(\tau)` data
        contains imaginary parts)

    Attributes
    ----------
    maxent_diagonal : e.g. TauMaxEnt
        the MaxEnt worker for the diagonal elements
        Note that setting this to a new object will influence all the
        variables that are derived from the maxent instance, e.g. omega,
        K, D, etc. Be sure to ensure compatibility of maxent_diagonal
        and maxent_offdiagonal and maxent_result!
    maxent_offdiagonal : e.g. TauMaxEnt
        the MaxEnt worker for the off-diagonal elements
    set_G_element : function, lambda
        function with arguments ``maxent``, ``G_mat``, ``elem``, ``re``
        that calls one of the ``set_G_***`` functions of ``maxent``
        to set its Green function to the element ``elem`` of the
        matrix ``G_mat``
    determine_shape : function, lambda
        function with arguments ``G_mat`` that returns the shape
        of the Green function matrix
    maxent_result : MaxEntResult
        the result of the calculation; set this to None in order to
        delete the preceding results
    """

    # this is needed to make the getattr/setattr magic work
    maxent_diagonal = None
    maxent_offdiagonal = None

    def __init__(self, use_hermiticity=True, use_complex=False, **kwargs):
        self.maxent_diagonal = TauMaxEnt(**kwargs)
        self.maxent_offdiagonal = TauMaxEnt(
            cost_function='plusminus', **kwargs)
        self.set_G_element = None
        self.determine_shape = None
        self.G_mat = None
        self.maxent_result = None
        self.use_hermiticity = use_hermiticity
        self.use_complex = use_complex

    # getattr and setattr allows us to access the MaxEnt (e.g. TauMaxEnt)
    # attributes as if they were ElementwiseMaxEnt attributes
    def __getattr__(self, name):
        # get the attribute both from the diagonal and the offdiagonal
        # MaxEnt object; if the result is different, raise an error
        qty_diagonal = getattr(
            object.__getattribute__(self, 'maxent_diagonal'), name)
        qty_offdiagonal = getattr(
            object.__getattribute__(self, 'maxent_offdiagonal'), name)
        if np.all(qty_diagonal == qty_offdiagonal):
            return qty_diagonal
        else:
            if hasattr(qty_diagonal, '__call__') and \
                    hasattr(qty_offdiagonal, '__call__'):
                return CallableMethodCheck(name, qty_diagonal, qty_offdiagonal)
            else:
                raise Exception('Element {n} not uniquely defined. '
                                'Use self.maxent_diagonal.{n} or '
                                'self.maxent_offdiagonal.{n}!'.format(n=name))

    def __setattr__(self, name, value):
        # we set the attribute of both the diagonal and the offdiagonal
        # MaxEnt object

        if hasattr(self.maxent_diagonal, name) and \
                hasattr(self.maxent_offdiagonal, name):
            # check that the value was equal before and issue a warning if not
            qty_diagonal = getattr(
                object.__getattribute__(self, 'maxent_diagonal'), name)
            qty_offdiagonal = getattr(
                object.__getattribute__(self, 'maxent_offdiagonal'), name)
            if np.any(qty_diagonal != qty_diagonal):
                self.maxent_diagonal.logtaker.error_message(
                    "Setting {n} which is not equal in maxent_diagonal and maxent_offdiagonal.")

            setattr(self.maxent_offdiagonal, name, value)
            return setattr(self.maxent_diagonal, name, value)
        else:
            object.__setattr__(self, name, value)

    def prepare_maxent_result(self, overwrite=False):
        """ Create a MaxEntResult object to hold the results

        This is called by the respective routines; there is usually no
        need for the user to call this.
        In order to discard the current results and start with a new
        one, either set maxent_result to None or call this method with
        ``overwrite=True``.

        Parameters
        ----------
        overwrite : bool
            whether the current result should be overwritten if already
            present
        """
        if self.maxent_result is None or overwrite:
            self.maxent_result = MaxEntResult(
                matrix_structure=self.determine_shape(self.G_mat),
                element_wise=True,
                use_hermiticity=self.use_hermiticity,
                complex_elements=self.use_complex)

    def run_element(self, element, re=True):
        """ Run MaxEnt for a matrix element

        This method fetches the element from the G matrix that was set
        using one of the ``set_G...`` methods and passes it to the correct
        MaxEnt worker (i.e., either the diagonal or off-diagonal MaxEnt).

        Parameters
        ----------
        element : tuple
            a tuple of two elements giving the 0-based index of the
            matrix element to analytically continue
        re : bool
            whether the real (True) or imaginary (False) part should be continued

        Returns
        -------
        maxent_result : MaxEntResult
            the result of the calculation; it has the correct matrix
            shape, with NaNs for the elements that were not calculated
            (yet)
        """
        self.prepare_maxent_result(overwrite=False)
        i, j = element
        if i == j:  # diagonal
            self.maxent_diagonal.logtaker.message(
                VerbosityFlags.ElementInfo,
                "Calling MaxEnt for element {i} {i}".format(i=i))
            self.set_G_element(self.maxent_diagonal,
                               self.G_mat, (i, i), True)
            self.put_error(self.maxent_diagonal, self.get_error((i, i)))
            self.maxent_diagonal.run(result=self.maxent_result,
                                     matrix_element=(i, i),
                                     complex_index=0 if re else 1)
        else:  # off-diagonal
            if (self.use_hermiticity and (i > j)):
                self.maxent_offdiagonal.logtaker.message(
                    VerbosityFlags.ElementInfo,
                    "Element {} {} not calculated, "
                    "can be determined from hermiticity".format(i, j))
                return self.maxent_result
            self.maxent_offdiagonal.logtaker.message(
                VerbosityFlags.ElementInfo,
                "Calling MaxEnt for element {} {} ".format(i, j))
            self.set_G_element(self.maxent_offdiagonal,
                               self.G_mat, (i, j), re)
            self.put_error(self.maxent_offdiagonal, self.get_error((i, j)))
            self.maxent_offdiagonal.run(result=self.maxent_result,
                                        matrix_element=(i, j),
                                        complex_index=0 if re else 1)

        return self.maxent_result

    def run_diagonal(self):
        """ Run MaxEnt for all diagonal elements

        Returns
        -------
        maxent_result : MaxEntResult
            the result of the calculation; it has the correct matrix
            shape, with NaNs for the elements that were not calculated
            (yet)
        """

        self.maxent_diagonal.logtaker.message(VerbosityFlags.ElementInfo,
                                              "Calculating diagonal elements.")
        for i in range(self.shape[0]):
            self.run_element((i, i))
            if self.use_complex and \
                    (i, i, 1) not in self.maxent_result.zero_elements:
                print('appending')
                self.maxent_result.zero_elements.append((i, i, 1))
        return self.maxent_result

    def run_offdiagonal(self):
        """ Run MaxEnt for all off-diagonal elements

        If ``use_hermiticity`` is True, half of the off-diagonal elements
        are not calculated, but inferred by using the hermiticity of the
        spectral function.

        Returns
        -------
        maxent_result : MaxEntResult
            the result of the calculation; it has the correct matrix
            shape, with NaNs for the elements that were not calculated
            (yet)
        """

        self.maxent_offdiagonal.logtaker.message(
            VerbosityFlags.ElementInfo,
            "Calculating off-diagonal elements.")
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if i == j:
                    continue
                for ri in ([True, False] if self.use_complex else [True]):
                    self.run_element((i, j), re=ri)
        return self.maxent_result

    def run(self):
        """ Run elementwise MaxEnt for all the matrix elements

        First, the diagonal elements, then the off-diagonal elements
        are calculated

        Returns
        -------
        maxent_result : MaxEntResult
            the result of the calculation; it has the correct matrix
            shape
        """
        self.run_diagonal()
        self.run_offdiagonal()

        return self.maxent_result

    def set_G(self, G_mat, set_G_element, determine_shape):
        """ Set Green function matrix

        Set the Green function matrix together with the function to
        feed one of its element to the underlying maxent object and
        the function to determine its shape.

        Consider using :py:meth:`.ElementwiseMaxEnt.set_G_tau`, :py:meth:`.ElementwiseMaxEnt.set_G_iw`, :py:meth:`.ElementwiseMaxEnt.set_G_tau_data`
        or :py:meth:`.ElementwiseMaxEnt.set_G_tau_filenames`.

        Parameters
        ----------
        G_mat : whatever
            Green function matrix
        set_G_element : function, lambda
            function with arguments ``maxent``, ``G_mat``, ``elem``, ``re``
            that calls one of the ``set_G_***`` functions of ``maxent``
            to set its Green function to the element ``elem`` of the
            matrix ``G_mat``
        determine_shape : function, lambda
            function with arguments ``G_mat`` that returns the shape
            of the Green function matrix
        """
        self.G_mat = G_mat
        self.set_G_element = set_G_element
        self.determine_shape = determine_shape
        self.maxent_result = None

    @require_triqs
    def set_G_tau(self, G_tau, *args, **kwargs):
        r""" Set matrix-valued :math:`G(\tau)` as TRIQS GfImTime

        The extra arguments are passed on to the :py:meth:`~.tau_maxent.TauMaxEnt.set_G_tau` method of
        ``maxent_diagonal`` and ``maxent_offdiagonal``.

        Parameters
        ----------
        G_tau : GfImTime
            The Green function
        """

        if isinstance(G_tau, BlockGf):
            raise NotImplementedError(
                'TRIQS BlockGfs are not supported by TauMaxEnt.\n' +
                'Consider looping over over the blocks and calling TauMaxEnt individually for each GfImTime.')

        if not isinstance(G_tau.mesh, MeshImTime):
            raise Exception(
                'set_G_tau only accepts TRIQS GfImTime objects.\n' +
                'Use the appropriate set_* method for other data formats.')

        set_G_element = lambda maxent, G_mat, elem, re: \
            maxent.set_G_tau(G_mat[elem], re=re, *args, **kwargs)
        determine_shape = lambda G_mat: G_mat.data.shape[1:]
        G_mat = G_tau
        self.set_G(G_mat, set_G_element, determine_shape)

    @require_triqs
    def set_G_iw(self, G_iw, *args, **kwargs):
        r""" Set matrix-valued :math:`G(i\omega_n)` as TRIQS GfImFreq

        The extra arguments are passed on to the :py:meth:`~.tau_maxent.TauMaxEnt.set_G_iw` method of
        ``maxent_diagonal`` and ``maxent_offdiagonal``.

        Parameters
        ----------
        G_iw : GfImFreq
            The Green function
        """

        if isinstance(G_iw, BlockGf):
            raise NotImplementedError(
                'TRIQS BlockGfs are not supported by TauMaxEnt.\n' +
                'Consider looping over over the blocks and calling TauMaxEnt individually for each GfImFreq.')

        if not isinstance(G_iw.mesh, MeshImFreq):
            raise Exception(
                'set_G_iw only accepts TRIQS GfImFreq objects.\n' +
                'Use the appropriate set_* method for other data formats.')

        set_G_element = lambda maxent, G_mat, elem, re: \
            maxent.set_G_iw(G_mat[elem], re=re, *args, **kwargs)
        determine_shape = lambda G_mat: G_mat.data.shape[1:]
        G_mat = G_iw
        self.set_G(G_mat, set_G_element, determine_shape)

    def set_G_tau_data(self, tau, G_tau, *args, **kwargs):
        r""" Set matrix-valued :math:`G(\tau)` from array

        The extra arguments are passed on to the :py:meth:`~.tau_maxent.TauMaxEnt.set_G_tau_data` method of
        ``maxent_diagonal`` and ``maxent_offdiagonal``.

        Parameters
        ----------
        tau : array
            a one-dimensional array of tau-values
        G_tau : array
            a three-dimensional array; the first two dimensions are the
            matrix dimensions, the last dimension is the tau grid
        """

        set_G_element = lambda maxent, G_mat, elem, re: \
            maxent.set_G_tau_data(G_mat[0],
                                  np.real(G_mat[1][elem]) if re else
                                  np.imag(G_mat[1][elem]),
                                  *args, **kwargs)
        determine_shape = lambda G_mat: G_mat[1].shape[:2]
        G_mat = (tau, G_tau)
        self.set_G(G_mat, set_G_element, determine_shape)

    def set_G_tau_filename_pattern(self,
                                   filename,
                                   dimension,
                                   tau_col=0,
                                   G_col_re=1,
                                   G_col_im=2,
                                   *args,
                                   **kwargs):
        r""" Set matrix-valued :math:`G(\tau)` from files

        For each matrix element, read the Green function from a file.
        The name of the files is given as a pattern.

        The extra arguments are passed on to the :py:meth:`~.tau_maxent.TauMaxEnt.set_G_tau_file` method of
        ``maxent_diagonal`` and ``maxent_offdiagonal``.

        Parameters
        ----------
        filename : str
            the filename pattern. It must contain {i} and {j}, placeholders
            which will be replaced by the index of the matrix element
            using the python ``format`` function.
        dimension : tuple of two ints
            the matrix dimension (shape)
        tau_col : int
            the 0-based column number of the :math:`\tau`-grid
        G_col_re : int
            the 0-based column number of the :math:`G(\tau)`-data (real part)
        G_col_im : int
            the 0-based column number of the :math:`G(\tau)`-data (imaginary part)
        """

        set_G_element = lambda maxent, G_mat, elem, re: \
            maxent.set_G_tau_file(G_mat.format(i=elem[0], j=elem[1]),
                                  tau_col, G_col_re if re else G_col_im,
                                  *args, **kwargs)
        determine_shape = lambda G_mat: dimension
        G_mat = filename
        self.set_G(G_mat, set_G_element, determine_shape)

    def set_G_tau_filenames(self,
                            filenames,
                            tau_col=0,
                            G_col_re=1,
                            G_col_im=2,
                            *args,
                            **kwargs):
        r""" Set matrix-valued :math:`G(\tau)` from files

        For each matrix element, read the Green function from a file.

        The extra arguments are passed on to the :py:meth:`~.tau_maxent.TauMaxEnt.set_G_tau_file` method of
        ``maxent_diagonal`` and ``maxent_offdiagonal``.

        Parameters
        ----------
        filenames : two-dimensional array of str
            for each matrix element, the name of the file from which
            the Green function should be read
        tau_col : int
            the 0-based column number of the :math:`\tau`-grid
        G_col_re : int
            the 0-based column number of the :math:`G(\tau)`-data (real part)
        G_col_im : int
            the 0-based column number of the :math:`G(\tau)`-data (imaginary part)
        """

        set_G_element = lambda maxent, G_mat, elem, re: \
            maxent.set_G_tau_file(G_mat[elem[0]][elem[1]],
                                  tau_col, G_col_re if re else G_col_im,
                                  *args, **kwargs)
        determine_shape = lambda G_mat: G_mat.shape
        G_mat = filenames
        self.set_G(G_mat, set_G_element, determine_shape)

    def set_error(self, error):
        r""" Set the error of :math:`G(\tau)`

        Parameters
        ----------
        error : float or array
            If the error is the same for every tau-point and matrix
            element, a float can be given. If it is the same for every
            matrix element, a one-dimensional array can be given. Else,
            an array with dimensions MxNxT is expected, where MxN is the
            matrix dimension of :math:`G(\tau)` and T is the number of tau-points.
        """
        self.error = error
        self.error_dimension = 1
        self.put_error = lambda maxent, error: maxent.set_error(error)

    def get_error(self, elem):
        """ Get the error of a matrix element

        elem : tuple
            a tuple of two elements giving the 0-based index of the
            matrix element
        """
        if isinstance(self.error, float):
            return self.error
        elif len(self.error.shape) == self.error_dimension:
            return self.error
        else:
            return self.error[elem]

    def set_cov(self, cov):
        r""" Set the covariance matrix of :math:`G(\tau)`

        Parameters
        ----------
        cov : array
            If the cov matrix is the same for every element, just the
            matrix can be given; else, an array with dimensions MxNxTxT
            is expected, where MxN is the matrix dimension of :math:`G(\tau)` and
            T is the number of tau-points
        """
        self.error_dimension = 2
        self.error = cov
        self.put_error = lambda maxent, error: maxent.set_cov(error)

    def get_tau(self):
        """ Get the data variable (e.g., tau) """
        qty_diagonal = self.maxent_diagonal.get_data_variable()
        qty_offdiagonal = self.maxent_offdiagonal.get_data_variable()
        if np.all(qty_diagonal == qty_offdiagonal):
            return qty_diagonal
        else:
            raise Exception('Element {n} not uniquely defined. '
                            'Use self.maxent_diagonal.{n} or '
                            'self.maxent_offdiagonal.{n}!'.format(n=name))

    def set_tau(self, tau, update_K=True,
                update_chi2=True, update_Q=True,
                update_H_of_v=True):
        self.set_data_variable(tau,
                               update_K=update_K,
                               update_chi2=update_chi2,
                               update_Q=update_Q,
                               update_H_of_v=update_H_of_v)

    tau = property(get_tau, set_tau)

    @property
    def shape(self):
        """ The shape of the Green function matrix """
        try:
            return self.determine_shape(self.G_mat)
        except Exception as e:
            print(e)
            raise Exception('Cannot determine shape.')


class DiagonalMaxEnt(ElementwiseMaxEnt):
    """ Perform MaxEnt for a matrix, element-wise for the diagonals.
    """

    def run(self):
        """ Run MaxEnt for all the diagonal elements of the matrix """
        self.run_diagonal()
        return self.maxent_result

    def run_offdiagonal(self):
        raise TypeError('DiagonalMaxEnt cannot run for off-diagonals.')


class PoormanMaxEnt(ElementwiseMaxEnt):
    r""" Perform poor man's MaxEnt for a matrix, element-wise.

    After calculating the diagonal elements, the off-diagonal elements
    are calculated using a default model derived from the diagonals:
    For element ``i, j``, we use

    .. math::
        D_{ij} = \sqrt{A_{ii} A_{jj}} + \varepsilon

    where :math:`\varepsilon` is a small additive constant to avoid
    that the default model becomes zero.

    This usually gives better results than just performing the
    ElementwiseMaxEnt for the off-diagonals.

    Parameters
    ----------
    analyzer_offdiag_D : Analyzer
        the name of the analyzer that should be used to get the spectral
        function of the diagonals for calculating the default model
    D_add_constant : float
        small constant that is added to the default model to ensure it
        is not zero
    """

    def __init__(self, analyzer_offdiag_D='LineFitAnalyzer',
                 D_add_constant=1.e-6, *args, **kwargs):
        super(PoormanMaxEnt, self).__init__(*args, **kwargs)
        self.analyzer_offdiag_D = analyzer_offdiag_D
        self.D_add_constant = D_add_constant

    def run(self):
        """ Run elementwise MaxEnt for all the matrix elements

        First, the diagonal elements, then the off-diagonal elements
        are calculated using the default model of the poor man's method

        Returns
        -------
        maxent_result : MaxEntResult
            the result of the calculation; it has the correct matrix
            shape
        """

        self.run_diagonal()
        self.run_offdiagonal()
        return self.maxent_result

    def run_offdiagonal(self):
        """ Run MaxEnt for all off-diagonal elements

        In the poor man's method, this requires the result of the diagonal
        elements. Make sure to call ``run_diagonal`` first.

        If ``use_hermiticity`` is True, half of the off-diagonal elements
        are not calculated, but inferred by using the hermiticity of the
        spectral function.

        Returns
        -------
        maxent_result : MaxEntResult
            the result of the calculation; it has the correct matrix
            shape, with NaNs for the elements that were not calculated
            (yet)
        """

        self.prepare_maxent_result(overwrite=False)
        self.maxent_offdiagonal.logtaker.message(
            VerbosityFlags.ElementInfo,
            "Calculating off-diagonal elements using default model from diagonal solution")
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if i == j:
                    continue
                if self.use_complex:
                    # the diagonal elements have to be real
                    A_1 = self.maxent_result.analyzer_results[i][i][0]\
                        [self.analyzer_offdiag_D]['A_out']
                    A_2 = self.maxent_result.analyzer_results[j][j][0]\
                        [self.analyzer_offdiag_D]['A_out']
                else:
                    A_1 = self.maxent_result.analyzer_results[i][i]\
                        [self.analyzer_offdiag_D]['A_out']
                    A_2 = self.maxent_result.analyzer_results[j][j]\
                        [self.analyzer_offdiag_D]['A_out']
                # add a small constant to avoid exactly zero default model
                self.maxent_offdiagonal.set_D(DataDefaultModel(
                    np.sqrt(A_1 * A_2) + self.D_add_constant, self.omega))
                for ri in ([True, False] if self.use_complex else [True]):
                    self.run_element((i, j), re=ri)
        return self.maxent_result
