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


"""
This module contains the :py:class:`MaxEntResult` and the :py:class:`MaxEntResultData`
classes. They are very much alike and share many common methods and properties.
The latter is a bare-bones version of the former that can be written to h5-files.

It contains the result of a MaxEnt calculation.
In :py:class:`MaxEntResult`, the properties that can be read (and analyzed)
from that run are documented.
In :py:class:`MaxEntResultData`, the plot functions for visualizing them
are documented.
"""
from __future__ import absolute_import, print_function
import numpy as np
from .plot_utils import *
from .omega_meshes import DataOmegaMesh
from .alpha_meshes import DataAlphaMesh
from datetime import datetime, timedelta
from collections import Sequence, OrderedDict
from itertools import product, izip_longest
import copy
from functools import wraps


def saved(func):
    """ Cache the value of a function """
    try:
        func.__doc__ += "\n\n        This is also available in :py:class:`.MaxEntResultData` if :py:meth:`included <.MaxEntResultData.include>`."
    except:
        pass

    @property
    @wraps(func)
    def new_func(self):
        if func.__name__ not in self._all_fields and self._check_fieldnames:
            raise AttributeError(
                'Field {} not available'.format(func.__name__))
        if func.__name__ not in self._saved:
            self._saved[func.__name__] = func(self)
        return self._saved[func.__name__]
    return new_func


def recursive_map(seq, func):
    """ Apply a function to all elements recursively

    The list ``seq`` is iterated and the function ``func`` is applied
    to all elements that are not sequences.

    Parameters
    ----------
    seq : list
        the list that we iterate over
    func : function with one argument
        this function is applied to each item
    """
    for item in seq:
        if isinstance(item, list):
            yield type(item)(recursive_map(item, func))
        else:
            yield func(item)


def recursive_dtype(seq):
    """ Get the dtype of the first item of seq. """
    for item in seq:
        if isinstance(item, list):
            return recursive_dtype(item)
        else:
            return item.dtype


def _get_empty(matrix_structure, fill_with=list, element_wise=True):
    """ Return an empty object that has ``len(matrix_structure)`` dimensions

    The shape of the object is given by the ``matrix_structure``;
    every element is filled with the object returned by calling ``fill_with``.

    Parameters
    ----------
    matrix_structure : tuple or list
        the size of the matrix, as in array.shape
    fill_with : callable
        fill_with is called to initialize each matrix element
    element_wise : bool
        whether to prepare the structure for element-wise adding or not
        (i.e., whether there will be one saved scalar-valued cost function
        per element or only one saved matrix-valued cost function)
    """
    if matrix_structure is None or not element_wise:
        return fill_with()
    else:
        next_matrix_structure = None
        if len(matrix_structure) > 1:
            next_matrix_structure = matrix_structure[1:]
        ret = []
        for i in range(matrix_structure[0]):
            ret.append(_get_empty(next_matrix_structure, fill_with))
        return ret


def _find_shape(seq):
    """ Get shape of nested sequences

    This is a helper function to convert sequences to arrays
    """
    # from https://stackoverflow.com/a/27890978
    try:
        len_ = len(seq)
    except TypeError:
        return ()
    shapes = [_find_shape(subseq) for subseq in seq]
    return (len_,) + tuple(max(sizes) for sizes in izip_longest(*shapes,
                                                                fillvalue=1))


def _fill_array(arr, seq):
    """ Convert sequence to array with NaN for missing values

    This is a helper function to convert sequences to arrays
    """
    # from https://stackoverflow.com/a/27890978
    if arr.ndim == 1:
        try:
            len_ = len(seq)
        except TypeError:
            len_ = 0
        arr[:len_] = seq
        arr[len_:] = np.nan
    else:
        for subarr, subseq in izip_longest(arr, seq, fillvalue=()):
            _fill_array(subarr, subseq)


class MaxEntResultData(object):
    """ Hold the result of a MaxEnt calculation

    Note that some functions/attributes documented in :py:class:`.MaxEntResult`
    also apply to :py:class:`.MaxEntResultData`.

    Parameters
    ----------
    matrix_structure : tuple or list
        the size of the matrix, as in array.shape
    element_wise : bool
        whether to use element-wise adding or not
        (i.e., whether there will be one saved scalar-valued cost function
        per element or only one saved matrix-valued cost function)
    complex_elements : bool
        whether there are complex elements
    use_hermiticity : bool
        whether or not the Hermiticity of the spectral function should
        be used to get values of :py:meth:`A_out` that are were not calculated
    """

    def __init__(self, matrix_structure=None, element_wise=True,
                 complex_elements=False, use_hermiticity=True):
        # this is needed for saving to h5
        self._all_fields = ['alpha', 'v', 'chi2', 'S', 'A', 'Q', 'omega',
                            'probability', 'analyzer_results', 'run_times',
                            'run_time_total', 'matrix_structure',
                            'effective_matrix_structure',
                            'element_wise', 'complex_elements',
                            'use_hermiticity', 'G', 'data_variable',
                            'G_rec', 'H', 'default_analyzer_name',
                            'zero_elements', 'G_orig']
        self._matrix_structure = matrix_structure
        self._complex_elements = complex_elements
        self._element_wise = element_wise
        self._use_hermiticity = use_hermiticity
        # whether to check if a field is in _all_fields when calling the
        # function
        self._check_fieldnames = True
        # which analyzer is the default for :py:meth:`.A_out`
        self._default_analyzer_name = None
        # are there any elements that are zero (see G_threshold in MaxEntLoop)
        self._zero_elements = []
        # this holds the saved values
        self._saved = dict()

    def _get_element(self, array, matrix_element):
        """ Get the element specified by matrix_element

        Fetches a particular ``matrix_element`` from an ``array``.
        """
        if self.matrix_structure is None:
            assert matrix_element is None, "Cannot give matrix_element when matrix_structure is None"
            return array
        else:
            if not self._element_wise:
                assert matrix_element is None, "Cannot give matrix_element when element_wise is False"
                return array
            else:
                assert matrix_element is not None, "matrix_element must be given"
                ret = array
                for i in matrix_element:
                    ret = ret[i]
                return ret

    # if the value is in ``_saved``, it will be returned
    def __getattr__(self, name):
        if name == "_saved":
            raise AttributeError('this should never happen')
        if name in self._saved:
            return self._saved[name]
        raise AttributeError(
            "'MaxEntResultData' object has no attribute '{}'".format(name))

    def include_only(self, fields):
        """ Set fields that are saved to h5

        Parameters
        ----------
        fields : list of str
            the fields that should be saved; by excluding some fields,
            unnecessary data can be skipped and thus the memory consumption
            lowered, but some functionality might not work
        """
        self._all_fields = []
        self.include(fields)

    def include(self, fields):
        """ Add fields that are saved to h5

        Parameters
        ----------
        fields : list of str
            the fields that should be saved in addition to the ones
            included so far; by excluding some fields,
            unnecessary data can be skipped and thus the memory consumption
            lowered, but some functionality might not work
        """

        old_check_fieldnames = self._check_fieldnames
        self._check_fieldnames = False
        for field in fields:
            if not hasattr(self, field):
                raise AttributeError('Unknown field: {}'.format(field))
            if field not in self._all_fields:
                self._all_fields.append(field)
        self._check_fieldnames = old_check_fieldnames

    def exclude(self, fields):
        """ Remove fields that are saved to h5

        Parameters
        ----------
        fields : list of str
            the fields that should not be saved; by excluding some fields,
            unnecessary data can be skipped and thus the memory consumption
            lowered, but some functionality might not work
        """

        for field in fields:
            if not hasattr(self, field):
                raise AttributeError('Unknown field: {}'.format(field))
            if field in self._all_fields:
                self._all_fields.remove(field)

    def get_default_analyzer(self, analyzer=None):
        """ The default analyzer

        If the ``matrix_structure`` is not ``None``, this gives back
        a ``M x N`` array (for complex entries ``M x N x 2``) with the
        analyzer for that matrix element.

        Parameters
        ----------
        analyzer : str
            the name of the analyzer; if None, the ``default_analyzer_name`` is used
        """
        if analyzer is None:
            analyzer = self.default_analyzer_name
        if analyzer is None:
            analyzer = 'LineFitAnalyzer'
        if self.matrix_structure is None or not self.element_wise:
            return self.analyzer_results[analyzer]
        else:
            # get it in the correct matrix shape
            ret = np.empty(self.effective_matrix_structure, dtype=object)
            m = map(range, self.effective_matrix_structure)
            for elem in product(*m):
                try:
                    ret[elem] = self._get_element(
                        self.analyzer_results, elem)[analyzer]
                except KeyError:
                    ret[elem] = None
            return ret

    default_analyzer = property(get_default_analyzer)

    def get_A_out(self, analyzer=None):
        r"""Get the one true spectral function :math:`A(\omega)` from the default analyzer

        Parameters
        ----------
        analyzer : str
            the name of the analyzer to use; if not given, the default
            analyzer (as specified by default_analyzer_name) is used
        """

        da = self.get_default_analyzer(analyzer)
        if self.matrix_structure is None or not self.element_wise:
            return da['A_out']
        else:
            # first determine the size of the output spectral function
            len_A = len(self.omega)
            # prepare an array for the output spectral function
            A_out = np.full(self.effective_matrix_structure + (len_A,), np.nan)
            for elem in self.zero_elements:
                A_out[elem] = 0.0
            # loop over all elements in the effective_matrix_structure
            m = map(range, self.effective_matrix_structure)
            for elem in product(*m):
                # we copy the element index into get_elem because this
                # is the element we want to get
                get_elem = elem
                # usually we don't want to conjugate, i.e. no minus sign
                maybe_minus = lambda x: x
                # if the element is None and we want to use the hermiticity
                # of A, we want to use the transpose matrix element to set
                # our matrix element
                if da[elem] is None and self.use_hermiticity:
                    # we construct the index of the transpose element
                    conj_elem = list(elem)
                    conj_elem[0], conj_elem[1] = conj_elem[1], conj_elem[0]
                    # and conjugate it if necessary
                    if self.complex_elements and conj_elem[-1] == 1:
                        maybe_minus = lambda x: -x
                    # in that case we actually want to get the transpose element
                    # and conjugate it
                    get_elem = tuple(conj_elem)
                # we retrieve the element from the analyzer and write it to
                # A_out
                if da[get_elem] is not None:
                    A_out[elem] = maybe_minus(da[get_elem]['A_out'])
                # else: the element of A_out just remains NaN

            # in the complex case, we want to transform A_out from a
            # M x N x 2 real matrix to a M x N complex matrix
            if self.complex_elements:
                return A_out[..., 0, :] + 1.0j * A_out[..., 1, :]
            else:
                return A_out

    A_out = property(get_A_out)

    def _add_matrix_structure_to_dict(self, d, check_element_wise=True):
        """ A helper function for plotting

        This adds the keys ``n_m``, ``n_n`` and possibly ``n_c`` (for
        complex elements) to the dict ``d`` holding the size of the
        ``matrix_structure``.
        """
        dnew = OrderedDict()
        if self.matrix_structure is not None and (
                (not check_element_wise) or self.element_wise):
            dnew['n_m'] = self.matrix_structure[0]
            dnew['n_n'] = self.matrix_structure[1]
            if self.complex_elements:
                dnew['n_c'] = 2
        dnew.update(d)
        return dnew

    @plot_function
    def plot_chi2(self, element=None, **kwargs):
        r""" Plot the misfit :math:`\chi^2` as function of :math:`\alpha`

        Parameters
        ----------
        element : tuple
            matrix element
        label : str
            the label of the curve (for a legend)
        x_label : str
            the label of the x-axis
        y_label : str
            the label of the y-axis
        log_x : bool
            whether the x-axis should be log-scaled (default: True)
        log_y : bool
            whether the y-axis should be log-scaled (default: True)
        """
        idx = slice(None) if element is None else element
        return (self.alpha, self.chi2[idx],
                self._add_matrix_structure_to_dict(
            OrderedDict(label=r'$\chi^2$', x_label=r'$\alpha$',
                        y_label=r'$\chi^2$', log_x=True, log_y=True))
                )

    @plot_function
    def plot_S(self, element=None, **kwargs):
        r""" Plot the entropy :math:`S` as function of :math:`\alpha`

        Parameters
        ----------
        element : tuple
            matrix element
        label : str
            the label of the curve (for a legend)
        x_label : str
            the label of the x-axis
        y_label : str
            the label of the y-axis
        log_x : bool
            whether the x-axis should be log-scaled (default: True)
        log_y : bool
            whether the y-axis should be log-scaled (default: False)
        """

        idx = slice(None) if element is None else element
        return (self.alpha, self.S[idx],
                self._add_matrix_structure_to_dict(
                OrderedDict(label=r'$S$',
                            x_label=r'$\alpha$', y_label=r'$S$',
                            log_x=True, log_y=False)))

    @plot_function
    def plot_Q(self, element=None, **kwargs):
        r""" Plot the cost function :math:`Q` as function of :math:`\alpha`

        Parameters
        ----------
        element : tuple
            matrix element
        label : str
            the label of the curve (for a legend)
        x_label : str
            the label of the x-axis
        y_label : str
            the label of the y-axis
        log_x : bool
            whether the x-axis should be log-scaled (default: True)
        log_y : bool
            whether the y-axis should be log-scaled (default: True)
        """

        idx = slice(None) if element is None else element
        return (self.alpha, self.Q[idx],
                self._add_matrix_structure_to_dict(
                OrderedDict(label=r'$Q$',
                            x_label=r'$\alpha$', y_label=r'$Q$',
                            log_x=True, log_y=True)))

    @plot_function
    def plot_A(self, element=None, alpha_index=0, **kwargs):
        r""" Plot a particular :math:`A_{\alpha}` as a function of :math:`\omega`

        Parameters
        ----------
        element : tuple
            matrix element
        alpha_index : int
            the index of the alpha value
        label : str
            the label of the curve (for a legend)
        x_label : str
            the label of the x-axis
        y_label : str
            the label of the y-axis
        log_x : bool
            whether the x-axis should be log-scaled (default: False)
        log_y : bool
            whether the y-axis should be log-scaled (default: False)
        """

        idx = slice(None) if element is None else element
        return (self.omega,
                self.A[idx][alpha_index],
                self._add_matrix_structure_to_dict(
                    OrderedDict(
                        label=r'$A_{{\alpha_{}}}(\omega)$'.format(alpha_index),
                        x_label=r'$\omega$',
                        y_label=r'$A(\omega)$',
                        log_x=False,
                        log_y=False,
                        n_alpha_index=len(
                            self.alpha)),
                    check_element_wise=False))

    @plot_function
    def plot_G(self, element=None, **kwargs):
        r""" Plot a particular :math:`G` as a function of the data variable

        This plots the original :math:`G` (see :py:meth:`.G_orig`).

        Parameters
        ----------
        element : tuple
            matrix element
        alpha_index : int
            the index of the alpha value
        label : str
            the label of the curve (for a legend)
        x_label : str
            the label of the x-axis
        y_label : str
            the label of the y-axis
        log_x : bool
            whether the x-axis should be log-scaled (default: False)
        log_y : bool
            whether the y-axis should be log-scaled (default: False)
        """

        idx = slice(None) if element is None else element
        return (self.data_variable[idx],
                self.G_orig[idx],
                self._add_matrix_structure_to_dict(
                OrderedDict(
                    label=r'$G(d)$',
                    x_label=r'$d$',
                    y_label=r'$G(d)$',
                    log_x=False,
                    log_y=False,
                ),
                check_element_wise=False))

    @plot_function
    def plot_G_rec(self, element=None, alpha_index=0, plot_G=True, **kwargs):
        r""" Plot the reconstruction :math:`G_{rec}` as a function of the data variable

        Parameters
        ----------
        element : tuple
            matrix element
        alpha_index : int
            the index of the alpha value
        plot_G : bool
            whether to plot the original data
        label : str
            the label of the curve (for a legend)
        x_label : str
            the label of the x-axis
        y_label : str
            the label of the y-axis
        log_x : bool
            whether the x-axis should be log-scaled (default: False)
        log_y : bool
            whether the y-axis should be log-scaled (default: False)
        """

        ret = []
        if plot_G:
            ret.append(self.plot_G.original(self,
                                            element=element,
                                            **kwargs))
        idx = slice(None) if element is None else element
        ret.append((self.data_variable[idx],
                    self.G_rec[idx][alpha_index],
                    self._add_matrix_structure_to_dict(
            OrderedDict(label=r'$G_{rec}(d)$',
                        x_label=r'$d$',
                        y_label=r'$G(d)$',
                        log_x=False,
                        log_y=False,
                        plot_G=True,
                        n_alpha_index=len(self.alpha)),
            check_element_wise=False)))
        return ret

    @plot_function
    def plot_probability(self, element=None, **kwargs):
        r""" Plot the probability as a function of :math:`\alpha`

        Parameters
        ----------
        element : tuple
            matrix element
        label : str
            the label of the curve (for a legend)
        x_label : str
            the label of the x-axis
        y_label : str
            the label of the y-axis
        log_x : bool
            whether the x-axis should be log-scaled (default: False)
        log_y : bool
            whether the y-axis should be log-scaled (default: False)
        """

        if np.all(np.isnan(self.probability)):
            raise AttributeError('Probability is all NaN')
        idx = slice(None) if element is None else element
        return (self.alpha,
                np.exp(self.probability[idx] -
                       np.nanmax(self.probability[idx])),
                self._add_matrix_structure_to_dict(
                    OrderedDict(label=r'$p$',
                                x_label=r'$\alpha$',
                                y_label=r'$p$',
                                log_x=True,
                                log_y=False)))

    def __reduce_to_dict__(self):
        """ this handles writing to h5 """
        ret = dict()
        ret['all_fields'] = self._all_fields
        for key in self._all_fields:
            if getattr(self, key) is None:
                ret[key] = 'None'
            else:
                ret[key] = getattr(self, key)

        def convert_timedelta(t):
            if isinstance(t, timedelta):
                return dict(days=t.days,
                            seconds=t.seconds,
                            microseconds=t.microseconds)
            elif isinstance(t, float):
                return t
            else:
                ret = [None] * len(t)
                for i in xrange(len(t)):
                    ret[i] = convert_timedelta(t[i])
                return ret
        if 'run_times' in ret:
            ret['run_times'] = [convert_timedelta(t) for t in ret['run_times']]
        if 'run_time_total' in ret:
            ret['run_time_total'] = convert_timedelta(ret['run_time_total'])
        return ret

    @classmethod
    def __factory_from_dict__(cls, name, D):
        """ this handles reading from h5 """
        self = cls()

        def convert_timedelta(t):
            if isinstance(t, dict):
                return timedelta(**t)
            elif isinstance(t, float):
                return t
            else:
                ret = copy.deepcopy(t)
                for i in xrange(len(t)):
                    ret[i] = convert_timedelta(t[i])
                return ret
        if 'run_times' in D:
            D['run_times'] = [convert_timedelta(t) for t in D['run_times']]
        if 'run_time_total' in D:
            D['run_time_total'] = timedelta(**D['run_time_total'])
        if 'omega' in D:
            D['omega'] = DataOmegaMesh(D['omega'])
        if 'alpha' in D:
            D['alpha'] = DataAlphaMesh(D['alpha'])
        if 'all_fields' in D:
            self._all_fields = D['all_fields']
            del D['all_fields']

        def add_maxent_result(x):
            if isinstance(x, dict):
                for key, value in x.iteritems():
                    value.maxent_result = self
            else:
                for y in x:
                    add_maxent_result(y)
        if 'analyzer_results' in D:
            add_maxent_result(D['analyzer_results'])
        for key, val in D.iteritems():
            if isinstance(val, str) and val == 'None':
                self._saved[key] = None
            else:
                self._saved[key] = val
        return self


class MaxEntResult(MaxEntResultData):
    """ Hold the result of a MaxEnt calculation

    For a description of the parameters see :py:class:`MaxEntResultData`.

    Notes
    -----

    In order to write a :py:class:`.MaxEntResult` object to an h5-file, use
    :py:meth:`.MaxEntResult.data`.
    """

    def __init__(self, matrix_structure=None, element_wise=True,
                 complex_elements=False, use_hermiticity=True):
        super(MaxEntResult, self).__init__(matrix_structure,
                                           element_wise,
                                           complex_elements,
                                           use_hermiticity)
        self._results = self._get_empty()
        self._probabilities = self._get_empty()
        self._results_from_analyzers = self._get_empty(fill_with=dict)
        self._start = self._get_empty(fill_with=datetime.now)
        self._end = self._get_empty(fill_with=datetime.now)
        self._last_end = datetime.now()
        self._start_end_alpha = self._get_empty()

    def _get_empty(self, fill_with=list):
        """ Get an empty object with the right matrix structure """
        return _get_empty(matrix_structure=self.effective_matrix_structure,
                          element_wise=self._element_wise,
                          fill_with=fill_with)

    def _forevery(self, func, what=None, dtype=float,
                  hermiticity_conjugate=False):
        """ apply a function to every result matrix element """
        if what is None:
            what = self._results
        if not isinstance(what, Sequence):
            return func(what)
        li = list(recursive_map(what, func))
        if dtype is None:
            dtype = recursive_dtype(li)
        arr = np.empty(_find_shape(li), dtype=dtype)
        _fill_array(arr, li)

        if self._use_hermiticity and hermiticity_conjugate and self.matrix_structure is not None:
            m = map(range, self._matrix_structure)
            for elem in product(*m):
                this_element_is_nan = False
                try:
                    this_element_is_nan = np.all(np.isnan(arr[elem]))
                except TypeError:
                    pass

                if this_element_is_nan:
                    if elem == elem[::-1]:
                        continue
                    arr[elem] = arr[elem[::-1]]
                    if self.complex_elements:
                        arr[elem + (1,)] = -arr[elem + (1,)]

        return arr

    def add_result(self,
                   cost_function,
                   log_probability=None,
                   matrix_element=None,
                   complex_index=None):
        r""" Add the result of one particular MaxEnt optimization

        For every :math:`\alpha` and possibly every matrix element (if a
        matrix structure is given and element_wise is True), the result
        of the MaxEnt optimization is collected by adding it to the result
        object using this method.

        Parameters
        ----------
        cost_function : CostFunction
            the cost function evaluated at the optimum, which gives
            access to quantities like, e.g., :math:`\chi^2`, :math:`Q`, :math:`S`, :math:`A`, etc.
        log_probability : float
            the log of the probability of this particular solution
            or None (if no probability was calculated)
        matrix_element : tuple
            the element that the result represents, if applicable
            (else: None)
        complex_index : int
            if applicable, the complex index (0 for real, 1 for imaginary)
        """

        if self.complex_elements and complex_index is not None and matrix_element is not None:
            matrix_element = matrix_element + (complex_index,)

        # invalidate _saved
        self._saved = dict()

        # add the new values to the corresponding arrays
        self._get_element(self._results, matrix_element).append(cost_function)
        self._get_element(self._probabilities,
                          matrix_element).append(log_probability)
        now = datetime.now()
        self._get_element(self._start_end_alpha,
                          matrix_element).append((self._last_end, now))
        self._last_end = now

    def analyze(self, analyzers, matrix_element=None, complex_index=None):
        """ Analyze the data to find the one true spectral function

        Parameters
        ----------
        analyzers : list of Analyzer
            the data from this result object is handed over to each
            analyzer from this list, which will then produce a spectral
            function, which is saved in analyzer_results
        matrix_element : tuple
            the element that shall be analyzed, if applicable
            (else: None)
        complex_index : int
            if applicable, the complex index (0 for real, 1 for imaginary)
        """

        if self.complex_elements and complex_index is not None and matrix_element is not None:
            matrix_element = matrix_element + (complex_index,)

        self._get_element(self._results_from_analyzers, matrix_element).clear()
        for analyzer in analyzers:
            try:
                res = analyzer.analyze(self, matrix_element)
                res.maxent_result = self
                self._get_element(self._results_from_analyzers,
                                  matrix_element)[res['name']] = res
            except ValueError as e:
                self._get_element(self._results_from_analyzers,
                                  matrix_element)[analyzer.name] = str(e)
                pass

    @property
    def _n_alphas(self):
        """ The number of alpha-values for each matrix element """
        if self._matrix_structure is None:
            return len(self._results)
        ret = np.array(self._get_empty(fill_with=lambda: 0))
        m = map(range, self.effective_matrix_structure)
        for elem in product(*m):
            ret[elem] = len(self._get_element(self._results, elem))
        return ret

    @saved
    def alpha(self):
        r""" The list of :math:`\alpha` values """
        f = self._results
        if self._matrix_structure is not None and self._element_wise:
            n_alphas = self._n_alphas
            # we take the alpha vector for the matrix element with the
            # most alpha values
            idx = np.unravel_index(np.argmax(n_alphas), n_alphas.shape)
            for i in idx:
                f = f[i]
        return np.array([r._alpha for r in f])

    @saved
    def v(self):
        """ The singular space vectors of the optimum

        For a matrix dimension ``M x N``, the number of alphas ``X``
        and the number of singular space values ``S``, this is a
        ``M x N x X x S`` object. For missing values, np.nan is used.
        If no matrix_structure is given, it is a ``X x S`` object.
        """
        return self._forevery(lambda r: r._x)

    @saved
    def chi2(self):
        r""" The :math:`\chi^2` (misfit) values of the optimum

        For a matrix dimension ``M x N`` and the number of alphas ``X``,
        this is a ``M x N x X`` object. For missing values, np.nan is used.
        If no matrix_structure is given, it is a list with ``X`` entries.
        """
        return self._forevery(lambda r: r.chi2.f())

    @saved
    def G(self):
        """ The Green function data (=input data)

        For a matrix dimension ``M x N``, the number of alphas ``X``,
        and the number of data-variables ``T``,
        this is a ``M x N x X x T`` object. For missing values, np.nan is used.

        In the case of an extra transformation (see :py:meth:`.TauMaxEnt.set_cov`)
        this is the transformed G.
        """
        return self._forevery(lambda r: r.chi2.G, dtype=None)[..., 0, :]

    @saved
    def G_orig(self):
        """ The Green function data (=input data)

        For a matrix dimension ``M x N``, the number of alphas ``X``,
        and the number of data-variables ``T``,
        this is a ``M x N x X x T`` object. For missing values, np.nan is used.

        In the case of an extra transformation (see :py:meth:`.TauMaxEnt.set_cov`)
        this is the original G.
        """
        return self._forevery(lambda r: r.G_orig, dtype=None)[..., 0, :]

    @saved
    def data_variable(self):
        r""" The Green function data variable (e.g. tau for :math:`G(\tau)`)

        For a matrix dimension ``M x N``, the number of alphas ``X``,
        and the number of data-variables ``T``,
        this is a ``M x N x X x T`` object. For missing values, np.nan is used.
        """
        return self._forevery(lambda r: r.chi2.data_variable)[..., 0, :]

    @saved
    def G_rec(self):
        """ The reconstructed Green function :math:`G_{rec}`. """
        return self._forevery(lambda r: np.dot(r.chi2.K.K_delta, r.A_of_H.f()))

    @saved
    def S(self):
        r""" The :math:`S` (entropy) values of the optimum

        For a matrix dimension ``M x N`` and the number of alphas ``X``,
        this is a ``M x N x X`` object. For missing values, np.nan is used.
        If no matrix_structure is given, it is a list with ``X`` entries.
        """
        return self._forevery(lambda r: r.S.f())

    @saved
    def H(self):
        r""" The :math:`H` (hidden spectral function) values of the optimum

        For a matrix dimension ``M x N`` and the number of alphas ``X``
        and the number of omega-points ``W``,
        this is a ``M x N x X x W`` object. For missing values, np.nan is used.
        If no matrix_structure is given, it is a ``X x W`` object.
        """
        H = self._forevery(lambda r: r.H_of_v.f(), hermiticity_conjugate=True)
        return H

    @saved
    def A(self):
        r""" The :math:`A` (spectral function) values of the optimum

        For a matrix dimension ``M x N`` and the number of alphas ``X``
        and the number of omega-points ``W``,
        this is a ``M x N x X x W`` object. For missing values, np.nan is used.
        If no matrix_structure is given, it is a ``X x W`` object.
        """
        A = self._forevery(lambda r: r.A_of_H.f(), hermiticity_conjugate=True)
        return A

    @saved
    def Q(self):
        r""" The :math:`Q` (cost function) values of the optimum

        For a matrix dimension ``M x N`` and the number of alphas ``X``,
        this is a ``M x N x X`` object. For missing values, np.nan is used.
        If no matrix_structure is given, it is a list with ``X`` entries.
        """
        return self._forevery(lambda r: r.f())

    @saved
    def omega(self):
        r""" The :math:`\omega` values """

        r = self._results
        if self._matrix_structure is not None and self._element_wise:
            n_alphas = self._n_alphas
            idx = np.unravel_index(np.argmax(n_alphas), n_alphas.shape)
            for i in idx:
                r = r[i]
        # get the first alpha
        r = r[0]
        return r.omega

    @saved
    def probability(self):
        """ The probability values of the optimum

        For a matrix dimension ``M x N`` and the number of alphas ``X``,
        this is a ``M x N x X`` object. For missing values, np.nan is used.
        If no matrix_structure is given, it is a list with ``X`` entries.
        """

        return self._forevery(lambda p: np.nan if p is None else p,
                              what=self._probabilities)

    def start_timing(self, matrix_element=None,
                     complex_index=None, time=None):
        """ Start timing for a particular matrix_element """

        if self.complex_elements and complex_index is not None and matrix_element is not None:
            matrix_element = matrix_element + (complex_index,)

        if time is None:
            time = datetime.now()
        try:
            self._start[matrix_element] = time
        except:
            self._start = time

    def end_timing(self, matrix_element=None,
                   complex_index=None, time=None):
        """ Stop timing for a particular matrix_element """

        if self.complex_elements and complex_index is not None and matrix_element is not None:
            matrix_element = matrix_element + (complex_index,)

        if time is None:
            time = datetime.now()
        try:
            self._end[matrix_element] = time
            st = self._start[matrix_element]
        except:
            self._end = time
            st = self._start
        return time - st

    @saved
    def run_time_total(self):
        """ Total run time of the calculation

        If _start and _end were set, this yields the total run time.
        """
        return self._forevery(lambda x: x, self._end, dtype=object) - \
            self._forevery(lambda x: x, self._start, dtype=object)

    @saved
    def run_times(self):
        """ The run times for the individual alphas (and matrix elements) """
        return self._forevery(lambda t: t[1] - t[0],
                              what=self._start_end_alpha, dtype=object)

    @saved
    def analyzer_results(self):
        """ The results from the alpha analyzers

        For matrix-valued results, this is a list of list of ... with the
        same matrix structure as the result.
        """
        return self._results_from_analyzers

    @saved
    def matrix_structure(self):
        """ The matrix structure """
        return self._matrix_structure

    @saved
    def default_analyzer_name(self):
        """ The name of the default analyzer """
        return self._default_analyzer_name

    @saved
    def effective_matrix_structure(self):
        """ The effective matrix structure including complex elements """
        if self.element_wise and self.complex_elements:
            return self._matrix_structure + (2,)
        else:
            return self._matrix_structure

    @saved
    def element_wise(self):
        return self._element_wise

    @saved
    def zero_elements(self):
        return self._zero_elements

    @saved
    def use_hermiticity(self):
        return self._use_hermiticity

    @saved
    def complex_elements(self):
        return self._complex_elements

    @property
    def data(self):
        """ Get a :py:class:`.MaxEntResultData` data object

        that can be saved to h5-files.
        """
        # in order to make sure that everything is saved
        try:
            d = self.__reduce_to_dict__()
            return MaxEntResultData.__factory_from_dict__(
                "MaxEntResultData", d)
        except AttributeError as e:
            raise Exception(e)

try:
    from pytriqs.archive.hdf_archive_schemes import register_class
    register_class(MaxEntResultData)
except ImportError:  # notriqs
    pass
