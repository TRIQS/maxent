.. highlight:: bash

.. _install:

For the versions of this code, see the :ref:`changelog page <changelog>`.

Packaged Versions of TRIQS/maxent
=================================

A Docker image including the latest version of TRIQS/maxent is available `here <https://hub.docker.com/r/flatironinstitute/triqs>`_. For more information, please see the page on :ref:`TRIQS Docker <triqslibs:triqs_docker>`.

We further provide a Debian package for the Ubuntu LTS Versions 16.04 (xenial) and 18.04 (bionic), which can be installed by following the steps outlined :ref:`here <triqslibs:triqs_debian>`, and the subsequent command::

        sudo apt-get install -y triqs_maxent


Compiling TRIQS/maxent from source
==================================

Prerequisites
-------------

For using this package, in general, you need the :ref:`TRIQS <triqslibs:welcome>` toolbox. In the following, we will suppose that it is installed in the ``path_to_triqs`` directory.
The code is compatible with TRIQS versions ``1.4`` and ``2.1``. Later versions should also work.
Additionally, it is possible to install a subset of features that work without a TRIQS installation.

Apart from the packages in the `python standard library <https://docs.python.org/2/library/index.html>`_, the following python packages are needed:

- `numpy <http://www.numpy.org/>`_,
- `matplotlib <https://matplotlib.org/>`_,
- `decorator <https://pypi.org/project/decorator/>`_,
- ``IPython`` and ``ipywidgets``, optional, for :py:class:`.JupyterPlotMaxEnt`,
- `sphinx <http://www.sphinx-doc.org>`_, optional, including extensions and ``numpydoc`` (if ``USE_TRIQS=OFF``) for the documentation, and
- ``coverage``, optional, for the ``TEST_COVERAGE`` feature.

Installation with TRIQS 2.1
---------------------------

#. Download the sources from github::

     $ git clone https://github.com/TRIQS/maxent.git maxent.src

#. Create an empty build directory where you will prepare the code::

     $ mkdir maxent.build && cd maxent.build

#. Make sure that you have added the TRIQS installation to your environment variables::

     $ source path_to_triqs/share/triqsvarsh.sh

#. Run ``cmake``::

     $ cmake ../maxent.src

   Do not give the option ``TRIQS_PATH`` as it is used to detect whether
   a newer TRIQS version or 1.4 is used.

#. Prepare the code, run the tests and install the application::

     $ make
     $ make test
     $ make install

For running the code, make sure to source::

     $ source path_to_triqs/share/triqsvarsh.sh

and the call your python script with ``python`` or run a Jupyter notebook.

Installation with TRIQS 1.4
---------------------------

#. Download the sources from github::

     $ git clone https://github.com/TRIQS/maxent.git maxent.src

#. Create an empty build directory where you will prepare the code::

     $ mkdir maxent.build && cd maxent.build

#. Run ``cmake``::

     $ cmake -DTRIQS_PATH=path_to_triqs ../maxent.src

#. Prepare the code, run the tests and install the application::

     $ make
     $ make test
     $ make install

.. note::

    In TRIQS 1.4, most TRIQS applications have to be used as (e.g.)::

        from triqs.applications.dft_tools import *

    However, as ``maxent`` uses the same codebase for all TRIQS versions,
    we follow the "new" application format, i.e.::

        from triqs_maxent import *


For running the code, call the ``triqs`` executable from your ``path_to_triqs/bin``
directory or run a Jupyter notebook using ``itriqs_notebook``.

Installation without TRIQS
--------------------------

.. warning::

    Without TRIQS, not all features of the code will work.

#. Download the sources from github::

     $ git clone https://github.com/TRIQS/maxent.git maxent.src

#. Create an empty build directory where you will prepare the code::

     $ mkdir maxent.build && cd maxent.build

#. Run ``cmake``::

     $ cmake -DUSE_TRIQS=OFF ../maxent.src

   Use the option ``USE_TRIQS=OFF`` to use the version that runs without TRIQS.
   You can give the installation path using ``-DCMAKE_INSTALL_PREFIX=installation_path``,
   where you can choose an ``installation_path``.
   If you do not give it, the system will guess a location (something like
   ``/usr/lib/python2.7/dist-packages``).

#. Prepare the code, run the tests and install the application::

     $ make
     $ make test
     $ make install

For running the code, add the ``installation_path`` to your ``PYTHONPATH``::

    $ export PYTHONPATH="installation_path:$PYTHONPATH"

Then, run your code using ``python`` or start a Jupyter notebook.

Custom CMake options
--------------------

The functionality of ``maxent`` can be tweaked using extra CMake options::

    cmake -DOPTION1=value1 -DOPTION2=value2 ... ../maxent.src

+----------------------------------------------------------------+-----------------------------------------------+
| Options                                                        | Syntax                                        |
+================================================================+===============================================+
| Disable testing (not recommended)                              | -DBuild_Tests=OFF                             |
+----------------------------------------------------------------+-----------------------------------------------+
| Build the documentation locally (this requires TRIQS with doc) | -DBuild_Documentation=ON                      |
+----------------------------------------------------------------+-----------------------------------------------+
| Check test coverage when testing                               | -DTEST_COVERAGE=ON                            |
| (run ``make coverage`` to show the results)                    |                                               |
+----------------------------------------------------------------+-----------------------------------------------+
