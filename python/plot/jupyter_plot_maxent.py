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
from triqs_maxent.maxent_result import MaxEntResult
import matplotlib.pyplot as plt
from itertools import product
from collections import OrderedDict

from ipywidgets import interact, Output
try:
    from ipywidgets.widgets.interaction import show_inline_matplotlib_plots
except:
    def show_inline_matplotlib_plots():
        plt.show()
import ipywidgets as widgets
from IPython.display import display, clear_output


class JupyterPlotMaxEnt(object):

    def __init__(self, result):
        self.w = None
        self.result = result
        self.inter = None
        self.quantities = []
        for function in dir(self.result):
            if not function.startswith("plot_"):
                continue
            # if there is an error with getting the data
            # we do not want to offer it in the menu
            try:
                getattr(self.result, function).original(self.result)
                self.quantities.append(function[5:])
            except:
                pass

        if not hasattr(self.result, 'analyzer_results'):
            ar = []
        elif self.result.matrix_structure is not None and self.result.element_wise:
            m = product(*map(range, self.result.effective_matrix_structure))
            ar = [(i, self._get_ar_i(i)) for i in m]
        else:
            ar = [(None, self.result.analyzer_results)]
        for ia, a in ar:
            for key, analyzer in a.iteritems():
                for function in dir(analyzer):
                    if not function.startswith("plot_"):
                        continue
                    # if there is an error with getting the data
                    # we do not want to offer it in the menu
                    try:
                        getattr(analyzer, function).original(analyzer,
                                                             self.result,
                                                             element=ia)
                        ky = key + ': ' + function[5:]
                        if ky not in self.quantities:
                            self.quantities.append(ky)
                    except:
                        pass

        interact(self.quantityfunction,
                 quantity=widgets.Dropdown(options=self.quantities,
                                           value=self.quantities[0]
                                           ))

    def _get_ar_i(self, i):
        ar = self.result.analyzer_results
        j = tuple(i)
        while len(j) > 0:
            ar = ar[j[0]]
            j = j[1:]
        return ar

    def quantity_getattr(self, quantity, elem=None):
        if ':' in quantity:
            analyzer, quantity = quantity.split(': ')
            ar = self.result.analyzer_results
            if elem is not None:
                ar = self._get_ar_i(elem)
            try:
                return getattr(ar[analyzer], "plot_" + quantity), ar[analyzer]
            except KeyError:
                print('Analyzer "{}" not available.'.format(analyzer))
        else:
            return getattr(self.result, "plot_" + quantity), self.result

    def quantityfunction(self, quantity):
        if self.w is not None:
            self.w.close()
        add_mn = False
        add_c = False
        elem = None
        if ':' in quantity and self.result.matrix_structure is not None and self.result.element_wise:
            add_mn = True
            elem = (0, 0)
            if self.result.complex_elements:
                add_c = True
                elem = (0, 0, 0)

        fun, obj = self.quantity_getattr(quantity, elem)

        to_plot = fun.original(obj, maxent_result=self.result, element=elem)
        if isinstance(to_plot, list):
            default = OrderedDict()
            for qty in to_plot[::-1]:
                default.update(qty[2])
        else:
            x, y, default = fun.original(
                obj, maxent_result=self.result, element=elem)
        if add_mn:
            default['n_m'] = self.result.matrix_structure[0]
            default['n_n'] = self.result.matrix_structure[1]
        if add_c:
            default['n_c'] = 2

        if self.inter is not None:
            self.inter.widget.close()
        additional_widgets = OrderedDict()
        for key in default:
            if key in ['x_label', 'y_label', 'label']:
                continue
            if "n_" + key in default:
                continue
            origkey = key
            if key.startswith("n_"):
                key = key[2:]
            if isinstance(default[origkey], bool):
                additional_widgets[key] = \
                    widgets.Checkbox(
                    value=default[origkey],
                    description=key
                )
            elif isinstance(default[origkey], int):
                toval = 0
                if 'n_' + key in default:
                    toval = default['n_' + key] - 1
                additional_widgets[key] = \
                    widgets.IntSlider(value=0,
                                      min=0,
                                      max=toval,
                                      step=1,
                                      description=key)

        def widget_change():
            with self.out:
                clear_output(wait=True)
                arguments = dict()
                for key, widget in additional_widgets.iteritems():
                    arguments[key] = widget.value
                self.plotfunction(quantity, **arguments)
                show_inline_matplotlib_plots()

        for key, widget in additional_widgets.iteritems():
            widget.observe(lambda change: widget_change(),
                           names='value',
                           type='change')
        children = [value for key, value in additional_widgets.iteritems()]
        self.w = widgets.VBox(children=children)
        display(self.w)
        try:
            self.out.clear_output()
        except:
            pass
        self.out = Output()
        display(self.out)
        widget_change()

    def plotfunction(self, quantity, **kwargs):
        elem = None
        if ':' in quantity and 'm' in kwargs and 'n' in kwargs and 'c' in kwargs:
            elem = (kwargs['m'], kwargs['n'], kwargs['c'])
            del kwargs['m']
            del kwargs['n']
            del kwargs['c']
        elif ':' in quantity and 'm' in kwargs and 'n' in kwargs:
            elem = (kwargs['m'], kwargs['n'])
            del kwargs['m']
            del kwargs['n']
        kwargs['element'] = elem
        getat = self.quantity_getattr(quantity, elem)
        if getat is None:
            return
        getat[0](maxent_result=self.result, **kwargs)
