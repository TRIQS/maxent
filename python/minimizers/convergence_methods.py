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



import numpy as np


class ConvergenceMethod(object):
    """ A general convergence method

    When calling the convergence method, it returns a tuple
    ``(conv, is_conv)``, where ``conv`` measures the convergence
    and ``is_conv`` is a ``bool`` that tells whether convergence
    was reached.

    Two convergence methods can be combined using the `&` (and) and
    `|` (or) operators.
    """

    def __and__(self, other):
        return AndConvergenceMethod(self, other)

    def __or__(self, other):
        return OrConvergenceMethod(self, other)

    def __call__(self, function, v, **kwargs):
        raise NotImplementedError(
            'Please use a subclass of ConvergenceMethod.')


class AndConvergenceMethod(ConvergenceMethod):
    """The 'and' conjunction between two convergence methods."""

    def __init__(self, one, two):
        self.one = one
        self.two = two

    def __call__(self, function, v, **kwargs):
        conv1, is_conv1 = self.one(function, v, **kwargs)
        conv2, is_conv2 = self.two(function, v, **kwargs)
        if np.isnan(conv1) or np.isnan(conv2):
            conv = np.nan
        else:
            conv = np.min((conv1, conv2))
        return conv, is_conv1 or is_conv2


class OrConvergenceMethod(ConvergenceMethod):
    """The 'or' conjunction between two convergence methods."""

    def __init__(self, one, two):
        self.one = one
        self.two = two

    def __call__(self, function, v, **kwargs):
        conv1, is_conv1 = self.one(function, v, **kwargs)
        conv2, is_conv2 = self.two(function, v, **kwargs)
        if np.isnan(conv1) or np.isnan(conv2):
            conv = np.nan
        else:
            conv = np.min((conv1, conv2))
        return conv, is_conv1 or is_conv2


class MaxDerivativeConvergenceMethod(ConvergenceMethod):
    """The maximum of the derivative has to be < convergence criterion"""

    def __init__(self, convergence_criterion):
        self.convergence_criterion = convergence_criterion

    def __call__(self, function, v, **kwargs):
        conv = np.max(np.abs(function.d(v)))
        return conv, conv < self.convergence_criterion


class NullConvergenceMethod(ConvergenceMethod):
    """A convergence method that thinks everything is converged."""

    def __call__(self, function, v, **kwargs):
        return 0, True


class FunctionChangeConvergenceMethod(ConvergenceMethod):
    """The function change between two subsequent iterations has to be < convergence criterion"""

    def __init__(self, convergence_criterion):
        self.convergence_criterion = convergence_criterion

    def __call__(self, function, v, **kwargs):
        assert 'Q0' in kwargs, 'argument Q0 missing'
        assert 'Q1' in kwargs, 'argument Q1 missing'
        conv = np.abs(kwargs['Q0'] - kwargs['Q1'])
        return conv, conv < self.convergence_criterion


class RelativeFunctionChangeConvergenceMethod(ConvergenceMethod):
    """The function change between two subsequent iterations divided by the function value has to be < convergence criterion"""

    def __init__(self, convergence_criterion):
        self.convergence_criterion = convergence_criterion

    def __call__(self, function, v, **kwargs):
        assert 'Q0' in kwargs, 'argument Q0 missing'
        assert 'Q1' in kwargs, 'argument Q1 missing'
        conv = np.abs(np.abs(kwargs['Q0'] - kwargs['Q1']) / kwargs['Q1'])
        return conv, conv < self.convergence_criterion
