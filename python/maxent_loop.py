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



from .alpha_meshes import LogAlphaMesh
from .cost_functions import MaxEntCostFunction, BryanCostFunction
from .minimizers import LevenbergMinimizer
from .logtaker import Logtaker, VerbosityFlags
from .maxent_result import MaxEntResult
from .analyzers import *
from .probabilities import NormalLogProbability
from .functions import PlusMinusEntropy, PlusMinusH_of_v, view_real
from datetime import datetime
import numpy as np


class MaxEntLoop(object):
    r""" The main class running the MaxEnt optimization and :math:`\alpha` loop

    Parameters
    ----------
    cost_function : CostFunction or str
        the cost function, e.g. :py:class:`.MaxEntCostFunction` (the default) or
        :py:class:`.BryanCostFunction`. This is :math:`\alpha` dependent
        and has to be minimized for each :math:`\alpha`.
        It can also be a string, i.e. 'normal' for MaxEntCostFunction,
        'plusminus' for ``MaxEntCostFunction`` with :py:class:`.PlusMinusEntropy`,
        or 'bryan' for ``BryanCostFunction``.
    minimizer : Minimizer
        the minimizer, e.g. :py:class:`.LevenbergMinimizer` (the default). This actually
        minimizes the ``cost_function`` for a particular :math:`\alpha`.
    alpha_mesh : AlphaMesh
        the array of :math:`\alpha` values that should be calculated.
        The convention is to start with the largest :math:`\alpha` value.
        The default is a :py:class:`.LogAlphaMesh`.
    probability : Probability or str
        the function, e.g. :py:class:`.NormalLogProbability`, that allows
        to calculate the :math:`\log` of the probability of a particular :math:`\alpha`.
        If 'normal', the :py:class:`.NormalLogProbability` is used.
        Default: ``None``
    analyzers : list of Analyzer
        the list of analyzers that are used to choose the one true spectral
        functions from :math:`A_\alpha(\omega)`.
        Default: ``[LineFitAnalyzer(), Chi2CurvatureAnalyzer(), EntropyAnalyzer()]``
        and additionally ``[BryanAnalyzer(), ClassicAnalyzer()]`` if
        ``probability`` is not ``None``
    logtaker : Logtaker
        for processing the log
    G_threshold : float
        if all values of the data Green function have an absolute value
        below that threshold, the calculation is not performed
    reduce_singular_space : float
        a threshold for reducing the singular space, see :py:meth:`.KernelSVD.reduce_singular_space`.
    A_init : array
        the initial spectral function (i.e., the starting value of the
        optimization for the first :math:`\alpha` value)
        (if it is ``None``, the default model is used)
    interactive : bool
        whether or not to pressing Ctrl+C during the run should show a
        menu to decide what to do instead of immediately exiting
        (default: True)
    scale_alpha : float or str
        scale all alpha values by this factor; if ``'Ndata'``, it is scaled
        by the number of data points (the default)
    """

    def __init__(self,
                 cost_function=None,
                 minimizer=None,
                 alpha_mesh=None,
                 probability=None,
                 analyzers=None,
                 logtaker=None,
                 G_threshold=1.e-10,
                 reduce_singular_space=1.e-14,
                 A_init=None,
                 interactive=True,
                 scale_alpha='Ndata'):

        self.cost_function = cost_function
        self.minimizer = minimizer
        self.alpha_mesh = alpha_mesh
        self.logtaker = logtaker
        self.probability = probability
        self.analyzers = analyzers

        self.G_threshold = G_threshold

        # supply default values
        if self.cost_function is None:
            self.cost_function = MaxEntCostFunction()
        else:
            if isinstance(self.cost_function, str):
                if self.cost_function.lower() == 'normal':
                    self.cost_function = MaxEntCostFunction()
                elif self.cost_function.lower() == 'plusminus':
                    self.cost_function = MaxEntCostFunction(
                        S=PlusMinusEntropy(),
                        H_of_v=PlusMinusH_of_v())
                elif self.cost_function.lower() == 'bryan':
                    self.cost_function = BryanCostFunction()
                else:
                    raise Exception(
                        'Unknown cost_function str {}.'.format(
                            self.cost_function))
        if self.minimizer is None:
            self.minimizer = LevenbergMinimizer()
        if self.alpha_mesh is None:
            self.alpha_mesh = LogAlphaMesh()
        if self.logtaker is None:
            self.logtaker = Logtaker()
        if isinstance(self.probability, str):
            if self.probability.lower() == 'normal':
                self.probability = NormalLogProbability()
        if self.analyzers is None:
            self.analyzers = [LineFitAnalyzer(), Chi2CurvatureAnalyzer(),
                              EntropyAnalyzer()]
            if self.probability is not None:
                self.analyzers += [BryanAnalyzer(), ClassicAnalyzer()]

        self.interactive = interactive
        self.A_init = A_init
        self.reduce_singular_space = reduce_singular_space
        self.scale_alpha = scale_alpha

    ####### Main functionality #######

    def run(self,
            result=None,
            matrix_element=None,
            complex_index=None):
        """ Run the MaxEnt loop

        Parameters
        ----------
        result : MaxEntResult
            the object where the result should be saved in; especially
            useful for elementwise calculations, where the results from
            different matrix elements should be written into only one
            result object. If ``None``, a new ``MaxEntResult`` is generated.
        matrix_element : tuple
            the index of the matrix element that should be calculated.
            This is needed when a ``result`` is given in an elementwise
            calculation, so that the result can be written for the correct
            matrix element
        complex_index : int
            the index of the complex number, either 0 (real) or 1 (imaginary);
            this is needed for complex elementwise calculations

        Returns
        -------
        result : :py:class:`.MaxEntResult`
            the result of the calculation
        """

        # check if all entries of G are below the threshold; then
        # no calculation should be performed
        if np.max(np.abs(self.G)) < self.G_threshold:
            if result is not None and matrix_element is not None:
                result.zero_elements.append(matrix_element)
            self.logtaker.error_message(
                'G below threshold, not performing the calculation.')
            return None

        self.logtaker.welcome_message()

        assert self.err is not None, 'No error specified'
        self.K.reduce_singular_space(self.reduce_singular_space)

        # calculate minimal chi2
        A_min = np.linalg.lstsq(self.K.K, self.G, rcond=-1)[0]
        if(np.any(np.iscomplex(A_min))):
            A_min = view_real(A_min)
        chi2_min = self.chi2(A_min).f()
        self.logtaker.message(VerbosityFlags.Header,
                              "Minimal chi2: {}",
                              chi2_min)

        # the initial value of v
        H = np.empty(self.chi2.input_size)
        right_side = (
            self.D.D if self.A_init is None else self.A_init) * self.omega.delta
        right_side_slice = [np.newaxis] * H.ndim
        for i in self.chi2.axes_preference[:right_side.ndim]:
            right_side_slice[i] = slice(None, None)
        H[:] = right_side[tuple(right_side_slice)]
        v = self.H_of_v(H).inv()

        # set up result
        if result is None:
            result = MaxEntResult()

        # set the default analyzer name to the first analyzer
        if result._default_analyzer_name is None:
            try:
                result._default_analyzer_name = self.analyzers[0].name
            except:
                pass

        if self.scale_alpha is None:
            scale_alpha = 1.0
        elif isinstance(self.scale_alpha, str):
            if self.scale_alpha.lower() == "ndata":
                scale_alpha = len(self.G)
            else:
                raise Exception(
                    "Unknown value {} for scale_alpha".format(
                        self.scale_alpha))
            self.logtaker.message(
                VerbosityFlags.Header,
                'scaling alpha by a factor {} (number of data points)'.format(scale_alpha))
        else:
            scale_alpha = self.scale_alpha
            self.logtaker.message(
                VerbosityFlags.Header,
                'scaling alpha by a factor {}'.format(scale_alpha))

        self.check_consistency()

        result.start_timing(matrix_element=matrix_element,
                            complex_index=complex_index)

        any_not_converged = False
        # the main loop
        for i_alpha, alpha in enumerate(self.alpha_mesh):
            try:
                self.cost_function.set_alpha(alpha * scale_alpha)
                # minimize the cost function
                v = self.minimizer.minimize(self.cost_function, v)
                Q_min = self.cost_function(v)
                # report
                self.logtaker.message(
                    VerbosityFlags.AlphaLoop,
                    "alpha[{:" + str(int(np.ceil(np.log10(len(self.alpha_mesh))))) + "d}] = {:16.8e}, chi2 = {:16.8e}, n_iter={:8d}{}",
                    i_alpha,
                    alpha * scale_alpha,
                    Q_min.chi2.f(),
                    self.minimizer.n_iter_last,
                    ' ' if self.minimizer.converged else '!')
                if not self.minimizer.converged:
                    any_not_converged = True
                if self.probability is None:
                    result.add_result(Q_min,
                                      matrix_element=matrix_element,
                                      complex_index=complex_index)
                else:
                    result.add_result(Q_min,
                                      self.probability(Q_min).f(),
                                      matrix_element=matrix_element,
                                      complex_index=complex_index)
            except KeyboardInterrupt:
                if not self.interactive:
                    raise
                print()
                print('Program interrupted.')
                print(' c  continue with next alpha (this alpha will be tossed!)')
                print(' a  continue with alpha analyzer (end alpha loop)')
                print(' r  return immediately')
                print(' x  exit code immediately')
                inp = input('How to proceed? ')
                if inp == 'c':
                    pass
                elif inp == 'a':
                    break
                elif inp == 'r':
                    return result
                elif inp == 'x':
                    raise

        if any_not_converged:
            self.logtaker.message(
                VerbosityFlags.AlphaLoop,
                "\n! ... The minimizer did not converge. Results might be wrong.\n")

        run_time = result.end_timing(matrix_element=matrix_element,
                                     complex_index=complex_index)

        self.logtaker.message(VerbosityFlags.Timing,
                              "MaxEnt loop finished in {}", run_time)

        # analyze to get the one true spectral function from A(alpha)
        result.analyze(self.analyzers,
                       matrix_element=matrix_element,
                       complex_index=complex_index)

        return result

    ####### Helper functions #######

    def check_consistency(self):
        """ check whether all child objects are consistent

        e.g., whether they all have the same ``omega`` values etc.

        Raises an error if not.

        This is automatically done before ``run``\ ning.
        """
        assert self.chi2 is self.cost_function.chi2
        assert self.S is self.cost_function.S
        assert self.D is self.cost_function.D
        assert self.K is self.cost_function.K
        assert self.G is self.cost_function.G
        assert self.err is self.cost_function.err
        assert self.omega is self.cost_function.omega
        assert self.data_variable is self.cost_function.data_variable
        assert self.H_of_v is self.cost_function.H_of_v
        assert self.A_of_H is self.cost_function.A_of_H
        assert self.cost_function.K is self.H_of_v.K
        assert self.chi2.K is self.H_of_v.K
        assert np.all(self.K.omega == self.omega)
        assert np.all(self.D.omega == self.omega)
        assert np.all(self.S.omega == self.omega)
        assert np.all(self.H_of_v.omega == self.omega)
        assert np.all(self.A_of_H.omega == self.omega)
        assert np.all(self.K.data_variable == self.data_variable)
        assert self.K is self.H_of_v.K
        assert np.all(self.chi2.data_variable == self.data_variable)
        assert np.all(self.H_of_v.K.data_variable == self.data_variable)
        assert np.all(self.H_of_v.D.D == self.D.D)
        assert np.all(self.S.D.D == self.D.D)

    def set_verbosity(self, verbosity=None, add=None, remove=None,
                      change_callback=True):
        """ Set the verbosity

        Parameters
        ----------
        verbosity : :py:class:`.VerbosityFlags`
            if not None, the verbosity is set to this value;
            e.g.::

                VerbosityFlags.Header | VerbosityFlags.Timing

            to show just the header and the timing information.
        add : :py:class:`.VerbosityFlags`
            if not None, the verbosity flags given are added to the
            verbosity flags already set. If also verbosity is not None,
            this is performed *after* setting ``verbosity``.
        remove : :py:class:`.VerbosityFlags`
            if not None, the verbosity flags given are removed from
            the verbosity flags already set. If also ``verbosity`` or
            ``add`` is not None, this is performed after the other two.
        change_callback : bool
            whether to turn on/off the callback in the minimizer as needed
            according to the verbosity flags given (the minimizer is faster
            if it does not have to produce verbose info)
        """

        if verbosity is not None:
            self.logtaker.verbose = verbosity

        if add is not None:
            self.logtaker.verbose |= add

        if remove is not None:
            self.logtaker.verbose &= ~ remove

        # generating the verbose message in the minimizer takes some
        # time that can be saved, therefore we explicitly set the
        # callback to None if not needed
        if change_callback:
            if self.logtaker.verbose & VerbosityFlags.SolverDetails:
                self.minimizer.verbose_callback = self.logtaker.solver_verbose_callback
            else:
                self.minimizer.verbose_callback = None

    def get_K(self):
        return self.cost_function.K

    def set_K(self, K, update_chi2=True, update_H_of_v=True, update_Q=True):
        self.cost_function.set_K(K,
                                 update_chi2=update_chi2,
                                 update_H_of_v=update_H_of_v,
                                 update_Q=update_Q)

    K = property(get_K, set_K)

    def get_G(self):
        return self.cost_function.G

    def set_G(self, G, update_chi2=True, update_Q=True):
        self.cost_function.set_G(G, update_chi2=update_chi2, update_Q=update_Q)

    G = property(get_G, set_G)

    def get_err(self):
        return self.cost_function.err

    def set_err(self, err, update_chi2=True, update_Q=True):
        self.cost_function.set_err(err,
                                   update_chi2=update_chi2,
                                   update_Q=update_Q)

    err = property(get_err, set_err)

    def get_omega(self):
        return self.cost_function.omega

    def set_omega(self,
                  omega,
                  update_K=True,
                  update_chi2=True,
                  update_D=True,
                  update_S=True,
                  update_H_of_v=True,
                  update_A_of_H=True,
                  update_Q=True):
        self.cost_function.set_omega(omega,
                                     update_K=update_K,
                                     update_chi2=update_chi2,
                                     update_D=update_D,
                                     update_S=update_S,
                                     update_H_of_v=update_H_of_v,
                                     update_A_of_H=update_A_of_H,
                                     update_Q=update_Q)

    omega = property(get_omega, set_omega)

    # ``data_variable`` is a name we use for tau
    # because in general we might want to use a different kernel
    # and then the supplied data is not G(tau) but, eg, G(iw)
    def get_data_variable(self):
        return self.cost_function.data_variable

    def set_data_variable(self,
                          data_variable,
                          update_K=True,
                          update_chi2=True, update_Q=True,
                          update_H_of_v=True):
        self.cost_function.set_data_variable(data_variable,
                                             update_K=update_K,
                                             update_chi2=update_chi2,
                                             update_Q=update_Q,
                                             update_H_of_v=update_H_of_v)

    data_variable = property(get_data_variable, set_data_variable)

    def get_D(self):
        return self.cost_function.D

    def set_D(self,
              D,
              update_S=True,
              update_H_of_v=True,
              update_A_of_H=True,
              update_Q=True):
        self.cost_function.set_D(D,
                                 update_S=update_S,
                                 update_H_of_v=update_H_of_v,
                                 update_A_of_H=update_A_of_H,
                                 update_Q=update_Q)

    D = property(get_D, set_D)

    def get_chi2(self):
        return self.cost_function.chi2

    def set_chi2(self, chi2, update_Q=True):
        self.cost_function.set_chi2(chi2, update_Q=update_Q)

    chi2 = property(get_chi2, set_chi2)

    def get_S(self):
        return self.cost_function.S

    def set_S(self, S, update_Q=True):
        self.cost_function.set_S(S, update_Q=update_Q)

    S = property(get_S, set_S)

    def get_H_of_v(self):
        return self.cost_function.H_of_v

    def set_H_of_v(self, H_of_v, update_Q=True):
        self.cost_function.set_H_of_v(H_of_v, update_Q=update_Q)

    H_of_v = property(get_H_of_v, set_H_of_v)

    def get_A_of_H(self):
        return self.cost_function.A_of_H

    def set_A_of_H(self, A_of_H, update_Q=True):
        self.cost_function.set_A_of_H(A_of_H, update_Q=update_Q)

    A_of_H = property(get_A_of_H, set_A_of_H)
