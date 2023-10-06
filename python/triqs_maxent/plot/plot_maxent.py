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




import tkinter as tk

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTk, NavigationToolbar2Tk
from matplotlib.backend_bases import key_press_handler
from itertools import product
from collections import OrderedDict

import argparse
import sys
import os.path

from h5 import HDFArchive
from triqs_maxent.maxent_result import MaxEntResultData
import pickle


def _get_path(path, ar):
    if path.startswith('/'):
        path = path[1:]
    if path in ('', '.'):
        return ar
    pasp = path.split('/', 1)
    if len(pasp) == 1:
        return ar[pasp[0]]
    else:
        return _get_path(pasp[1], ar[pasp[0]])


class PlotMaxEnt(object):

    def __init__(self, h5_file, pickle_mode=False):
        self.pickle_mode = pickle_mode
        self.additional_widget_frame = None
        self.additional_widgets = dict()
        self.additional_variables = dict()
        self.additional_variables_transform = dict()
        self._result = None
        self._locked1 = False
        self._locked2 = False
        self.h5_file = h5_file
        self._init_gui()
        self.load_h5_file()

    def _init_gui(self):
        self.master = tk.Tk()
        # set up variables
        self.dataset = tk.StringVar(self.master)
        self.dataset.set("No datasets loaded.")
        self.dataset.trace("w", self.change_dataset)
        self.quantity = tk.StringVar(self.master)
        self.quantity.set("No quantities loaded.")
        self.quantity.trace("w", self.change_quantity)

        self.canvas = None
        self.toolbar = None
        self.figure = None

        self.top_bar = tk.Frame(self.master)

        self.dataset_menu = tk.OptionMenu(self.top_bar, self.dataset,
                                          self.dataset.get())
        self.dataset_menu.pack(side=tk.LEFT)

        self.quantity_menu = tk.OptionMenu(self.top_bar, self.quantity,
                                           self.quantity.get())
        self.quantity_menu.pack(side=tk.LEFT)

        self.top_bar.pack(fill=tk.X, expand=0)

        self.master.protocol("WM_DELETE_WINDOW", self._quit)

    def _quit(self):
        self.master.quit()
        self.master.destroy()

    def run(self):
        tk.mainloop()

    def load_h5_file(self, filename=None):
        if filename is None:
            filename = self.h5_file
        self.h5_file = filename
        self.datasets = self.get_datasets()
        self.update_dataset_ui()

    def get_datasets(self, ar=None, path=''):
        ret = []
        is_dir = False
        if ar is None:
            if self.pickle_mode:
                with open(self.h5_file, 'r') as fi:
                    return self.get_datasets(pickle.load(fi), path)
            else:
                with HDFArchive(self.h5_file, 'r') as ar:
                    return self.get_datasets(ar, path)
        # we detect whether it is a dataset directory
        if isinstance(ar, MaxEntResultData):
            is_dir = True
            if len(path) == 0:
                path = '/'
            ret.append(path)
            return ret
        for key in ar:
            try:
                ret += self.get_datasets(ar[key], path + '/' + key)
            except:
                pass
        return ret

    def update_dataset_ui(self, *args):
        self.dataset_menu['menu'].delete(0, 'end')
        for dataset in self.datasets:
            cmd = lambda dataset=dataset: self.dataset.set(dataset)
            self.dataset_menu['menu'].add_command(label=dataset,
                                                  command=cmd)
        if not self.dataset.get() in self.datasets:
            if len(self.datasets) > 0:
                self.dataset.set(self.datasets[0])

    def update_quantity_ui(self, *args):
        self.quantity_menu['menu'].delete(0, 'end')
        for quantity in self.quantities:
            cmd = lambda quantity=quantity: self.quantity.set(quantity)
            self.quantity_menu['menu'].add_command(label=quantity, command=cmd)
        if not self.quantity.get() in self.quantities:
            self.quantity.set(self.quantities[0])

    def change_dataset(self, *args):
        self.locked1 = True
        self.quantities = []
        if self.pickle_mode:
            with open(self.h5_file, 'r') as fi:
                self._result = _get_path(self.dataset.get(), pickle.load(fi))
        else:
            with HDFArchive(self.h5_file, 'r') as arx:
                self._result = _get_path(self.dataset.get(), arx)

        for function in dir(self._result):
            if not function.startswith("plot_"):
                continue
            # if there is an error with getting the data
            # we do not want to offer it in the menu
            try:
                getattr(self._result, function).original(self._result)
                self.quantities.append(function[5:])
            except Exception as e:
                print(e)
                pass

        if not hasattr(self._result, 'analyzer_results'):
            ar = []
        elif self._result.matrix_structure is not None and self._result.element_wise:
            m = product(
                *map(range, self._result.effective_matrix_structure))
            ar = [(i, self._get_ar_i(i)) for i in m]
        else:
            ar = [(None, self._result.analyzer_results)]
        for ia, a in ar:
            for key, analyzer in a.items():
                for function in dir(analyzer):
                    if not function.startswith("plot_"):
                        continue
                    # if there is an error with getting the data
                    # we do not want to offer it in the menu
                    try:
                        getattr(analyzer, function).\
                            original(analyzer, self._result, element=ia)
                        ky = key + ': ' + function[5:]
                        if ky not in self.quantities:
                            self.quantities.append(ky)
                    except:
                        pass

        self.update_quantity_ui()
        self.locked1 = False
        self.update_plot()

    def _get_ar_i(self, i):
        ar = self._result.analyzer_results
        j = tuple(i)
        while len(j) > 0:
            ar = ar[j[0]]
            j = j[1:]
        return ar

    def quantity_getattr(self, quantity, elem=None):
        if ':' in quantity:
            analyzer, quantity = quantity.split(': ')
            ar = self._result.analyzer_results
            if elem is not None:
                ar = self._get_ar_i(elem)
            return getattr(ar[analyzer], "plot_" + quantity), ar[analyzer]
        else:
            return getattr(self._result, "plot_" + quantity), self._result

    def change_quantity(self, *args):
        self._locked2 = True
        quantity = self.quantity.get()
        add_mn = False
        add_c = False
        elem = None
        if ':' in quantity and self._result.matrix_structure is not None and self._result.element_wise:
            add_mn = True
            elem = (0, 0)
            if self._result.complex_elements:
                add_c = True
                elem = (0, 0, 0)
        fun, obj = self.quantity_getattr(quantity, elem)
        to_plot = fun.original(obj, maxent_result=self._result, element=elem)
        if isinstance(to_plot, list):
            default = OrderedDict()
            for qty in to_plot[::-1]:
                default.update(qty[2])
        else:
            x, y, default = fun.original(
                obj, maxent_result=self._result, element=elem)
        d = OrderedDict()
        if add_mn:
            d['n_m'] = self._result.matrix_structure[0]
            d['n_n'] = self._result.matrix_structure[1]
        if add_c:
            d['n_c'] = 2
        d.update(default)
        default = d
        if self.additional_widget_frame is not None:
            self.additional_widget_frame.pack_forget()
            self.additional_widget_frame.destroy()
        self.additional_widget_frame = tk.Frame(self.top_bar)
        self.additional_widget_frame.pack(side=tk.LEFT, fill=tk.X, expand=1)
        self.additional_widgets = dict()
        self.additional_variables = dict()
        self.additional_variables_transform = dict()
        for key in default:
            if key in ['x_label', 'y_label', 'label']:
                continue
            if "n_" + key in default:
                continue
            origkey = key
            if key.startswith("n_"):
                key = key[2:]
            if isinstance(default[origkey], bool):
                self.additional_variables[key] = \
                    tk.IntVar(self.additional_widget_frame)
                self.additional_variables[key].trace("w", self.update_plot)
                self.additional_variables_transform[key] = bool
                self.additional_widgets[key] = \
                    tk.Checkbutton(self.additional_widget_frame,
                                   text=key,
                                   variable=self.additional_variables[key])
                self.additional_widgets[key].pack(side=tk.LEFT)
                def_value = 1 if default[origkey] else 0
                self.additional_variables[key].set(def_value)
            elif isinstance(default[origkey], int):
                self.additional_variables[key] = \
                    tk.StringVar(self.additional_widget_frame)
                self.additional_variables[key].trace("w", self.update_plot)
                self.additional_variables_transform[key] = int
                toval = 0
                if 'n_' + key in default:
                    toval = default['n_' + key] - 1
                self.additional_widgets["label_" + key] = \
                    tk.Label(self.additional_widget_frame,
                             text=key[2:] if key.startswith("n_") else key)
                self.additional_widgets["label_" + key].pack(side=tk.LEFT)
                self.additional_widgets[key] = \
                    tk.Spinbox(self.additional_widget_frame,
                               textvariable=self.additional_variables[key],
                               from_=0,
                               to=toval,
                               width=2)
                self.additional_widgets[key].pack(side=tk.LEFT)
                self.additional_variables[key].set('0')
        self._locked2 = False
        self.update_plot()

    def update_plot(self, *args):
        if self._locked1 or self._locked2:
            return
        quantity = self.quantity.get()
        if not self.canvas is None:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
        if not self.toolbar is None:
            self.toolbar.destroy()
            self.toolbar = None
        oldfig = self.figure

        self.figure = plt.figure()
        if not self._result is None:
            kwargs = dict()
            for key in self.additional_variables:
                transform = self.additional_variables_transform[key]
                kwargs[key] = transform(self.additional_variables[key].get())
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
            self.quantity_getattr(quantity, elem)[0](
                maxent_result=self._result, **kwargs)
            plt.tight_layout()
        self.canvas = FigureCanvasTk(self.figure, master=self.master)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.master)
        self.toolbar.update()
        self.canvas._tkcanvas.pack(fill=tk.BOTH, expand=1)

        if not oldfig is None:
            plt.close(oldfig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot results of a Maxent run in a h5-file")
    parser.add_argument(
        'input',
        help='name of the input h5 (or pickle)-file (default: maxent.h5)',
        default='maxent.h5',
        nargs='?')
    parser.add_argument('-p',
                        '--pickle',
                        action='store_true',
                        help='use pickle instead of HDF to load data',
                        default=false)
    args = parser.parse_args()

    pme = PlotMaxEnt(args.input, pickle_mode=args.pickle)
    pme.run()
