.. _maxent-flavors:

Ways of choosing :math:`\alpha`
===============================

The regularization of the misfit :math:`\chi^2` with an entropy :math:`S`
introduces the ad-hoc parameter :math:`\alpha`. The way to choose :math:`\alpha`
marks various varieties of the MaxEnt approach:

        * Historic MaxEnt: :math:`\chi^2` equal to number of data points
        * Probability :math:`p(\alpha | G, D)` based:
                * Classic: determine maximum of :math:`p`
                * Bryan: average over :math:`A(\alpha)` with weights :math:`p`
        * Kink in :math:`\log(\chi^2)` vs. :math:`\log(\alpha)`
                * :math:`\Omega`-MaxEnt: use :math:`\alpha` at maximum curvature
                * Line fit: fit two lines and use intersection for optimal :math:`\alpha`

A disadvantage of the Historic MaxEnt and the probabilistic methods is that
the resulting :math:`A` is strongly dependent on the provided covariance matrix.
If the  statistical  error  of  Monte  Carlo measurements, for example, is not
estimated accurately, the data could be over- or under-fitted.

Methods analyzing the dependence of :math:`\log(\chi^2)` on :math:`\log(\alpha)`
can overcome this problem by searching for the cross-over point
from the noise-fitting (small :math:`\alpha`) to the
information-fitting (intermediate :math:`\alpha`) regime.
In the noise-fitting regime :math:`\chi^2` is essentially constant, while in the
information-fitting region it behaves linearly.

One can either select the point of maximum curvature (:math:`\Omega`-MaxEnt),
or fit two straight lines and select the :math:`\alpha` at the intersection.

In this package, :ref:`different ways <analyzersref>` of determining :math:`\alpha` are implemented, and
with one run of the code the solutions of different MaxEnt flavors can be obtained.
