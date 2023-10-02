.. highlight:: bash

.. _install:

Install triqs_maxent
********************

Packaged Versions of TRIQS/maxent
=================================

.. _ubuntu_debian:
Ubuntu Debian packages
----------------------

We provide a Debian package for the Ubuntu LTS Version 22.04 (jammy), which can be installed by following the steps outlined :ref:`here <triqslibs:ubuntu_debian>`, and the subsequent command::

        sudo apt-get install -y triqs_maxent

.. _docker:
Docker
------

A Docker image including the latest version of triqs_maxent is available `here <https://hub.docker.com/r/flatironinstitute/triqs>`_. For more information, please see the page on :ref:`TRIQS Docker <triqslibs:triqs_docker>`.


Compiling triqs_maxent from source
==================================

.. note:: To guarantee reproducibility in scientific calculations we strongly recommend the use of a stable `release <https://github.com/TRIQS/triqs/releases>`_ of both TRIQS and its applications.

Prerequisites
-------------

For using this package, you need the :ref:`TRIQS <triqslibs:welcome>` toolbox. In the following, we will suppose that it is installed in the ``path_to_triqs`` directory.
The code is compatible with TRIQS versions ``3.2``.

Installation steps
------------------

#. Download the source code of the latest stable version by cloning the ``TRIQS/maxent`` repository from GitHub::

     $ git clone https://github.com/TRIQS/maxent maxent.src

#. Create and move to a new directory where you will compile the code::

     $ mkdir maxent.build && cd maxent.build

#. Ensure that your shell contains the TRIQS environment variables by sourcing the ``triqsvars.sh`` file from your TRIQS installation::

     $ source path_to_triqs/share/triqs/triqsvars.sh

#. In the build directory call cmake, including any additional custom CMake options, see below::

     $ cmake ../maxent.src

#. Compile the code, run the tests and install the application::

     $ make
     $ make test
     $ make install

Version compatibility
---------------------

Keep in mind that the version of ``triqs_maxent`` must be compatible with your TRIQS library version,
see :ref:`TRIQS website <triqslibs:versions>`.
In particular the Major and Minor Version numbers have to be the same.
To use a particular version, go into the directory with the sources, and look at all available versions::

     $ cd triqs_maxent.src && git tag

Checkout the version of the code that you want::

     $ git checkout 2.1.0

and follow steps 2 to 4 above to compile the code.

Custom CMake options
--------------------

The compilation of ``triqs_maxent`` can be configured using CMake-options::

    cmake ../triqs_maxent.src -DOPTION1=value1 -DOPTION2=value2 ...

+-----------------------------------------------------------------+-----------------------------------------------+
| Options                                                         | Syntax                                        |
+=================================================================+===============================================+
| Specify an installation path other than path_to_triqs           | -DCMAKE_INSTALL_PREFIX=path_to_triqs_maxent      |
+-----------------------------------------------------------------+-----------------------------------------------+
| Build in Debugging Mode                                         | -DCMAKE_BUILD_TYPE=Debug                      |
+-----------------------------------------------------------------+-----------------------------------------------+
| Disable testing (not recommended)                               | -DBuild_Tests=OFF                             |
+-----------------------------------------------------------------+-----------------------------------------------+
| Build the documentation                                         | -DBuild_Documentation=ON                      |
+-----------------------------------------------------------------+-----------------------------------------------+
