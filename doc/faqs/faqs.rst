Frequently-Asked Questions
==========================

How can I use this package for a matrix-valued Green function?
--------------------------------------------------------------

Two options to perform a continuation for matrices are available:
:py:class:`.ElementwiseMaxEnt` and :py:class:`.PoormanMaxEnt`. How to use these tools is described in the
:ref:`elementwise example<elementwise>`. Both algorithms are based on the individual continuation
of each matrix element. Treating all elements at the same time is currently
not implemented in this package.

What kernels are available?
---------------------------

Currently we have implemented the base class :py:class:`.Kernel`, a fermionic kernel for :math:`G(\tau)`
(:py:class:`.TauKernel`), a kernel for the preblur formalism (:py:class:`.PreblurKernel`) and a
generic :py:class:`.DataKernel`. Adding further kernels (fermionic and bosonic)
and the corresponding *XMaxEnt* classes is anticipated. However, also contributions
from users in this direction are welcome (e.g., as a :ref:`pull request <pull-requests>`).

Why is there a ! at the end of the output line for some alpha?
--------------------------------------------------------------

The exclamation mark (!) indicates that the minimization did not converge within the maximum
number of steps. The default value is 1000 iterations. To increase the maximum number
of iterations use:

.. code-block:: python

        tm = TauMaxEnt()
        tm.minimizer.maxiter=5000

How can I reduce the size of the singular space?
------------------------------------------------

The default threshold for the singular values is set to 1e-14.
You can change this threshold with:

.. code-block:: python

        tm.reduce_singular_space=1e-8

Warning 'Widget Javascript not detected' when I use ``JupyerPlotMaxEnt``
------------------------------------------------------------------------

Check if you have **ipywidgets** and **widgetsnbextension** installed.
It might also be that you need to enable the **widgetsnbextension** with
**jupyter nbextension enable jupyter-js-widgets/extension**.


Why is my calculation slow?
---------------------------

Usually it is a good idea to run a first MaxEnt calculation with
weaker settings. The following factors can have a strong impact
on the runtime:

    * size of :math:`\omega`-grid: Usually a grid with 100-500 points is enough.
      Instead of a linear grid it is better to choose a grid which is denser around
      zero and has fewer points at higher frequencies (Lorentzian or Hyperbolic mesh).
      (see also :ref:`omega meshes<omegameshes>`)

    * number of :math:`\tau`-grid: Often it is not necessary that 10001 :math:`\tau`-points
      (default value of TRIQS ``GfImTime``) are used for the continuation. Often just a few
      hundred points are sufficient.

    * size of singular space: The default threshold for the singular values is set to 1e-14.
      Reducing the size of the singular space does also speed up the calculation, but can
      also have a strong impact on resulting spectra (less structure).

    * :math:`\alpha`-values: Use a initially a coarse :math:`\alpha`-mesh. Also make sure that
      you start at a rather high value, where the minimization converges quickly. Additionally,
      avoid having too many small :math:`\alpha` where the change in :math:`\chi^2` is already small.

    * initial guess for A: For the first :math:`\alpha` the initial guess is set to the default model.
      This can be changed with ``tm.A_init = myInitA``. For all consequent :math:`\alpha` always
      the spectral function for the previous :math:`\alpha` is used as starting point.

    * matrix-valued Green function: As spectral functions have to be Hermitian, we can
      use this fact and perform the continuation only for one half of
      the off-diagonal elements (``em=ElementwiseMaxEnt(use_hermiticity=True)``).

Always carefully check the validity of your results in terms of the parameters listed above.

How can I save things to file?
------------------------------

With TRIQS, a :py:class:`.MaxEntResult` object can be saved to file using::

    # let res be the result, e.g. res = tau_maxent.run()
    # let file_name_of_h5_file be a str with the file name
    # let name_of_result be the desired name of the group in the file
    from h5 import HDFArchive
    with HDFArchive(file_name_of_h5_file, 'w') as ar:
        ar[name_of_result] = res.data

Loading is similar::

    with HDFArchive(file_name_of_h5_file, 'r') as ar:
        res = ar[name_of_result]

Without TRIQS, there is no platform-independent way of saving and loading implemented.
However, you can always pickle the result::

    # let res be the result, e.g. res = tau_maxent.run()
    # let file_name_of_pickle_file be a str with the file name

    import pickle
    with open(file_name_of_pickle_file,'w') as fi:
        pickle.dump(res.data, fi)

Loading::

    with open(file_name_of_pickle_file,'w') as fi:
        res = pickle.load(fi)

Apart from MaxEntResult and its cousin, :py:class:`.AnalyzerResult`, saving and loading to h5 is
only implemented for :py:class:`.SigmaContinuator` objects.

.. _citations:

Which publications should I cite?
---------------------------------

Please cite the code with its github repo, https://github.com/TRIQS/maxent, and its authors *Gernot J. Kraberger* and *Manuel Zingl*.

Furthermore, the general MaxEnt method for analytical continuation can be cited as `J. E. Gubernatis, M. Jarrell, R. N. Silver, and D. S. Sivia, Phys. Rev. B 44, 6011 (1991) <https://doi.org/10.1103/PhysRevB.44.6011>`_.
If you use the additions in the code concerting matrix-valued Green functions, please consider citing our paper `G. J. Kraberger, R. Triebl, M. Zingl and M. Aichhorn, Phys. Rev. B 96, 155128 (2017) <https://link.aps.org/doi/10.1103/PhysRevB.96.155128>`_.

For a more detailed list of relevant publications that you might want to cite as well, see :ref:`shoulders_giants`.
