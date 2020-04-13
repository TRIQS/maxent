.. _visualization:

Visualization
=============

There are four different ways to visualize :py:class:`.MaxEntResult` (or
:py:class:`.MaxEntResultData`) objects. We assuming here that the
:py:class:`.MaxEntResult` object is named ``res``. The corresponding
:py:class:`.MaxEntResultData` object can be obtained with ``res.data``.

Jupyter Widget
--------------

If you perform you calculation directly in a Jupyter notebook, the most convenient
way to analyze the data is our interactive Jupyter widget `JupyterPlotMaxEnt`:

.. code-block:: python

        from triqs_maxent.plot.jupyter_plot_maxent import JupyterPlotMaxEnt
        JupyterPlotMaxEnt(res)

`JupterPlotMaxEnt` works also with ``res.data``.

MaxEntResultData GUI
--------------------

For those who prefer to work in the shell, or for calculations on machines without web-browsers,
this package also offers an interface to graphically display :py:class:`.MaxEntResultData` objects
contained in h5-files.

To this end, first save ``res`` to a h5-file:

.. code-block:: python

        from h5 import *
        with HDFArchive('maxent_result.h5','w') as ar:
                ar['maxent_result'] = res.data

Then, run the shell command::

        YOUR_TRIQS_PATH/bin/plot_maxent maxent_result.h5

plot_* methods
--------------

All attributes which are shown by the GUI and the Jupyter widget have plot methods implemented.
These methods can be used individually, e.g.:

.. code-block:: python

        # G(tau) and back-transformed G_rec(tau) for alpha index 10
        plt.figure()
        res.plot_G_rec(alpha_index=10)

        # log(chi2) vs log(alpha)
        plt.figure()
        res.plot_chi2()

        # linefit
        plt.figure()
        res.analyzer_results['LineFitAnalyzer'].plot_linefit()

The complete set of plot methods is described in the reference guide for
:py:class:`.MaxEntResultData` and :py:class:`.AnalyzerResult`.

Plot data by hand
-----------------

Of course, it is also possible to access the data and plot/visualize it by hand.
Examples are:

.. code-block:: python

        # G(tau) and back-transformed G_rec(tau) for alpha index 10
        plt.figure()
        plt.plot(res.data_variable, res.G)
        plt.plot(res.data_variable, res.G_rec[10])

        # log(chi2) vs log(alpha)
        plt.figure()
        plt.loglog(res.alpha, res.chi2)
