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



import matplotlib.pyplot as plt
from functools import wraps
import numpy as np


def plot_function(func):
    """ A decorator for plotting

    A method decorated with this decorator shall return three arguments,
    specifically ``x``, ``y`` and ``default`` or a list of tuples, where
    each tuple consists of ``(x, y, default)`` (in the latter case, more
    than one line will be plotted).
    ``x`` and ``y`` represent the coordinates of the curve that shall be
    plotted and ``default`` is a dictionary of options.
    These options can either by consumed by the decorated function (in
    any way that seems suitable) or they are passed on to the matplotlib
    plotting functions.

    There is a special keyword argument called element which is constructed
    from the keyword arguments ``m``, ``n`` (and ``c``) representing the
    index of the matrix element (with a possible complex index ``c``).

    Entries starting with ``n_``, e.g. ``n_argname``, in ``default``
    mean that there is a keyword argument ``argname`` that can take
    the values ``0`` to ``n_argname-1``.

    After decorating the function, it will not return the arguments
    anymore but rather plot the curves with the corresponding setting.
    The original function which is decorated is still available using
    ``function_name.original``.

    If using this on methods in :py:class:`.MaxEntResultData` or
    :py:class:`.AnalyzerResult`, the plotting GUI and the ``JupyterPlotMaxEnt``
    will automatically detect it and use the entries of ``default`` to
    present GUI elements to the user (see :ref:`visualization`).
    """
    @wraps(func)
    def new_func(self, **kwargs):
        if 'm' in kwargs and 'n' in kwargs and 'c' in kwargs:
            kwargs['element'] = (kwargs['m'], kwargs['n'], kwargs['c'])
            del kwargs['m']
            del kwargs['n']
            del kwargs['c']
        elif 'm' in kwargs and 'n' in kwargs:
            kwargs['element'] = (kwargs['m'], kwargs['n'])
            del kwargs['m']
            del kwargs['n']
        to_plot = func(self, **kwargs)
        if not isinstance(to_plot, list):
            to_plot = [to_plot]
        for qty in to_plot:
            x, y, default = qty
            for key, val in default.items():
                if key not in kwargs:
                    kwargs[key] = val
            _plotter(x, y, **kwargs)
    new_func.original = func
    return new_func


def _plotter(x, y, label=None, x_label=None, y_label=None,
             log_x=False, log_y=False, **kwargs):
    """ actually plotting a curve

    a small wrapper over matplotlib"""

    plot_command = plt.plot
    if log_x and log_y:
        plot_command = plt.loglog
    elif log_x:
        plot_command = plt.semilogx
    elif log_y:
        plot_command = plt.semilogy

    if np.any(np.iscomplex(y)):
        plot_command(x, y.real,
                     label='Re ' + label if label is not None else None)
        plot_command(x, y.imag,
                     label='Im ' + label if label is not None else None)
    else:
        plot_command(x, y, label=label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
