Mathematical and Physical Background
====================================

Many of the methods used for solving problems for interacting quantum
systems yield results on the imaginary-frequency (or, equivalently,
imaginary-time) axis.
But often it is necessary to get results on the real-frequency axis to
be able to physically interpret them.
Obtaining real-frequency results from imaginary-frequency data can be
achieved using *analytic continuation*.

Mathematical Formulation of the Problem
---------------------------------------

The retarded fermionic one electron Green function :math:`G(\omega+i0^+)` and
the Matsubara Green function :math:`G(i \omega_n)` are related through
the analyticity of :math:`G(z)` in the whole complex plane with the
exception of the poles below the real axis. This connection is explicit
by writing the Green function :math:`G(z)` in terms of the spectral
function \ :math:`A(\omega)` as

.. math::
   :label: eq:spectral-representation

   G_{ab}(z) = \int d\omega \frac{A_{ab}(\omega)}{z - \omega}.

In general, both :math:`G(z)` and :math:`A(\omega)` are matrix-valued
(with indices :math:`a`, :math:`b`), but Eq. :eq:`eq:spectral-representation`
is valid for each matrix element separately. The reversed expression,
i.e. the matrix-valued :math:`A_{ab}(\omega)` given the real-frequency Green function
:math:`G_{ab}(\omega)`, is

.. math:: A_{ab}(\omega) = \frac{1}{2\pi} i (G_{ab}(\omega) - G_{ba}^*(\omega)).

Note that for matrices, the spectral function is not proportional to
the element-wise imaginary part of the Green function. A drawback of
the expression in Eq. :eq:`eq:spectral-representation` is that the real and
imaginary parts of :math:`G` and :math:`A` are coupled due to the fact that
:math:`z` is complex-valued. This is avoided by Fourier-transforming
:math:`G(i\omega_n)` to the imaginary time Green function
:math:`G(\tau)` at inverse temperature :math:`\beta`;

.. math::
   :label: eq:Gtau-from-Aw

   G_{ab}(\tau) = \int d\omega  \; \frac{-e^{-\omega\tau}}{1+e^{-\omega\beta}}A_{ab}(\omega).

The real part of the spectral function is only connected to the real
part of :math:`G(\tau)`, and analogous for the imaginary part. In the
following, we will first recapitulate the maximum entropy theory for a
real-valued single-orbital problem as presented in
Ref. [#gubernatis]_ and later generalize to
matrix-valued problems.

In order to handle this problem numerically, the functions
:math:`G(\tau)` and :math:`A(\omega)` in Eq. :eq:`eq:Gtau-from-Aw` can be
discretized to vectors :math:`G_n= G(\tau_n)` and :math:`A_m = A(\omega_m)`; then,
Eq. :eq:`eq:Gtau-from-Aw` can be formulated as

.. math::
   :label: eq:Gtau-from-Aw-matrix

   \mathbf{G} = K \mathbf{A},

where the matrix

.. math:: K_{nm} = \frac{-e^{-\omega_m\tau_n}}{1+e^{-\omega_m\beta}} \Delta \omega_m

is the kernel of the transformation [#kernel]_. It seems that inverting Eq. :eq:`eq:Gtau-from-Aw-matrix`,
i.e. calculating :math:`\mathbf{A}` via :math:`\mathbf{A} = K^{-1}\mathbf{G}`, is just
as straightforward as calculating :math:`G(\tau)` from
:math:`A(\omega)`. However, the inverse is an *ill-posed* problem. To be
more specific, the condition number of :math:`K` is very large due to
the exponential decay of :math:`K_{nm}` with :math:`\omega_m` and
:math:`\tau_n`, so that the direct inversion of :math:`K` is numerically
not feasible by standard techniques. Therefore, a bare minimization of
the misfit
:math:`\chi^2 (\mathbf{A}) = (K \mathbf{A} - \mathbf{G})^TC^{-1} (K \mathbf{A} - \mathbf{G})`, with
the covariance matrix :math:`C`, leads to an uncontrollable
error [#beach]_.

Entropy Regularization
----------------------

One efficient way to regularize this ill-posed problem is to add an
entropic term :math:`S(A)`. This leads to the maximum entropy method
(MEM), where one does not minimize :math:`\chi^2 (A)`, but

.. math::

   \label{eq:Q}
    Q_\alpha(A) = \frac12 \chi^2 (A) - \alpha S(A).

The pre-factor of the entropy, usually denoted :math:`\alpha`, is a
hyper-parameter that is introduced *ad hoc* and needs to be specified.
There are :ref:`different ways <maxent-flavors>` to choose :math:`\alpha` that have been employed in the literature.
This regularization with an entropy has been put on a rigorous probabilistic
footing by John Skilling in 1989, using Bayesian
methods [#skilling]_. He showed that the only
consistent way to choose the entropy for a non-negative function
:math:`A(\omega)` is

.. math::

   S(A) = \int d\omega \left[A(\omega) - D(\omega) - A(\omega) \log\frac{A(\omega)}{D(\omega)} \right],
       \label{eq:entropy-conventional}

where :math:`D(\omega)` is the default model. The default model
influences the result in two ways:
First, it defines the maximum of the prior distribution, which means
that in the limit of large :math:`\alpha` one has
:math:`A(\omega) \rightarrow D(\omega)`. Second, it is also related to
the width of the distribution, since the variance of the prior
distribution is proportional to :math:`D(\omega)`.
Often, a flat default model is used, corresponding to no prior knowledge.

Off-diagonal elements
---------------------

For off-diagonal matrix elements, the spectral function :math:`A(\omega)` is not
non-negative anymore (the spectral function as a matrix is Hermitian and positive definite).
Therefore, the entropy given above cannot be used anymore.
However, the off-diagonal spectral function :math:`A(\omega)` can be regarded
as the difference of two non-negative functions, :math:`A(\omega) = A^+(\omega) - A^-(\omega)`.
Then, for both :math:`A^+(\omega)` and :math:`A^-(\omega)`, the expression of the normal
entropy can be used and the total entropy is just the sum of the two [#kraberger]_.

.. rubric:: Footnotes

.. [#gubernatis] J\. E. Gubernatis, M. Jarrell, R. N. Silver, and D. S. Sivia, `Phys. Rev. B 44, 6011 (1991) <https://doi.org/10.1103/PhysRevB.44.6011>`_.
.. [#beach] K\. S. D. Beach, R. J. Gooding, and F. Marsiglio, `Phys. Rev. B 61, 5147 (2000) <https://doi.org/10.1103/PhysRevB.61.5147>`_.
.. [#kernel] Note that the kernel depends on the target Green function. Here we use the kernel for :math:`G(\tau)`; it would be different, e.g., for :math:`G(i\omega_n)`, a Legendre representations or even bosonic Green functions.
.. [#skilling] J\. Skilling, "`Classic Maximum Entropy <https://doi.org/10.1007/978-94-015-7860-8_3>`__," in Maximum Entropy and Bayesian Methods, edited by J. Skilling (Kluwer Academic Publishers, Dortrecht, 1989) pp. 45–52.
.. [#kraberger] G\. J. Kraberger, R. Triebl, M. Zingl, and M. Aichhorn, `Phys. Rev. B 96, 155128 (2017) <https://doi.org/10.1103/PhysRevB.96.155128>`_.

