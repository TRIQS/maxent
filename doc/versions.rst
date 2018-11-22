.. _versions:

Versions
========

Changes since the last version
------------------------------

The newest version, including changes since the last release, can be obtained using ``git checkout master``.

.. Right now, there are no changes beyond the last release.
   Maybe there are some feature branches waiting to be explored.

Changes with respect to the last release:

- :py:class:`.IOmegaKernel` for continuing :math:`G(i\omega)`
  (an object corresponding to :py:class:`.TauMaxEnt` is still missing)
- :py:class:`.ComplexChi2` and :py:class:`.ComplexPlusMinusEntropy` for continuing complex :math:`A(\omega)`
  (Note: with the tau kernel, for single matrix elements and the elementwise formalism, it is usually not
  necessary to continue complex spectral functions in one go. Therefore, this is only interesting for
  using the :py:class:`.IOmegaKernel` or for the full matrix formalism which will be provided.)
- Introduction of the :py:class:`.VerbosityFlags` and a new verbosity management system (as `suggested <https://github.com/TRIQS/maxent/issues/3>`_ by Marcel Klett)


Version 0.9, 2018-08-24
-----------------------

This version can be obtained using ``git checkout 0.9``, as there is a tag called ``0.9``.
It has been checked to work with the following TRIQS hashes:

- TRIQS 1.4.x, hash `816aff2882e581b7fe0ae071842b53fc27d31346 <https://github.com/TRIQS/triqs/tree/816aff2882e581b7fe0ae071842b53fc27d31346>`_
- TRIQS 2.0.x, hash `dc2cbc572478bcd5c35ba4777f6bbdeacd3f2262 <https://github.com/TRIQS/triqs/tree/dc2cbc572478bcd5c35ba4777f6bbdeacd3f2262>`_

but it probably also works with other commits.


This is the initial version that was published.

It includes (among other features)

- analytic continuation with a :math:`G(\tau)` kernel,
- the Bryan and MaxEnt cost function,
- entropies for diagonal and off-diagonal elements,
- elementwise and poor man's matrix MaxEnt,
- several analyzers for determining :math:`\alpha`,
- the sigma continuator,
- tools for getting :math:`G(\omega)`,
- plotting h5 files, and
- preblur.
