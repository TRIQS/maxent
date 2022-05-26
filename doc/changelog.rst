.. _changelog:

Changelog
=========

This document describes the main changes in maxent.

The newest version, including changes since the last release, can be obtained using ``git checkout unstable``.

Version 1.1.1
-------------

Fix an import issue for Python 3.10: Sequence should be imported from collections.abc

Version 1.1.0
-------------

Version 1.1.0 is a compatibility release for TRIQS version 3.1.0.


Version 1.0.0
-------------

Version 1.0.0 is a compatibility release for TRIQS version 3.0.0 that
introduces compatibility with Python 3 (Python 2 no longer supported)


Version 0.9.1, 2019-05-22
-------------------------

* :py:class:`.IOmegaKernel` for continuing :math:`G(i\omega)`
  (an object corresponding to :py:class:`.TauMaxEnt` is still missing)
* :py:class:`.ComplexChi2` and :py:class:`.ComplexPlusMinusEntropy` for continuing complex :math:`A(\omega)`
  (Note: with the tau kernel, for single matrix elements and the elementwise formalism, it is usually not
  necessary to continue complex spectral functions in one go. Therefore, this is only interesting for
  using the :py:class:`.IOmegaKernel` or for the full matrix formalism which will be provided.)
* Introduction of the :py:class:`.VerbosityFlags` and a new verbosity management system (as `suggested <https://github.com/TRIQS/maxent/issues/3>`_ by Marcel Klett)
* Make package compatible with TRIQS release 2.1.0. The package might work with TRIQS 2.0, but we do not provide support for it.

This version can be obtained using ``git checkout 0.9.1``.
It has been checked to work with the following TRIQS releases:

- TRIQS 1.4.2 release, hash `e7217b45e1e96fcf2820fd655346a057f22233d4 <https://github.com/TRIQS/triqs/tree/816aff2882e581b7fe0ae071842b53fc27d31346>`_
- TRIQS 2.1.0 release, hash `e96f1d26217af753191482398dc568ac43aa920a <https://github.com/TRIQS/triqs/tree/e96f1d26217af753191482398dc568ac43aa920a>`_

but it probably also works with other commits.


Version 0.9.0, 2018-08-24
-------------------------

This is the initial version that was published.

It includes (among other features)

* analytic continuation with a :math:`G(\tau)` kernel,
* the Bryan and MaxEnt cost function,
* entropies for diagonal and off-diagonal elements,
* elementwise and poor man's matrix MaxEnt,
* several analyzers for determining :math:`\alpha`,
* the sigma continuator,
* tools for getting :math:`G(\omega)`,
* plotting h5 files, and
* preblur.

This version can be obtained using ``git checkout 0.9``.
It has been checked to work with the following TRIQS hashes:

- TRIQS 1.4.x, hash `816aff2882e581b7fe0ae071842b53fc27d31346 <https://github.com/TRIQS/triqs/tree/816aff2882e581b7fe0ae071842b53fc27d31346>`_
- TRIQS 2.0.x, hash `dc2cbc572478bcd5c35ba4777f6bbdeacd3f2262 <https://github.com/TRIQS/triqs/tree/dc2cbc572478bcd5c35ba4777f6bbdeacd3f2262>`_

but it probably also works with other commits.
