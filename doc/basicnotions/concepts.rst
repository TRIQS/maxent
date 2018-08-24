Conceptual ideas of this package
================================

Flexible implementation
-----------------------

In most MaxEnt codes (and all publicly available codes that we
are aware of), the expressions for :math:`\chi^2` and :math:`S`
are fixed and hard-coded in the program. Often, simplifications
are performed that are only possible for the usual expressions
for these quantities.
In this code, in principle any (doubly derivable) expression
for :math:`\chi^2` and :math:`S` can be used.
The usual choices as well as choices suited for off-diagonal
elements of matrix-valued spectral functions are already implemented,
and it is very easy to swap out the functions that are used.

The implementation of the whole framework is so that it is highly
*flexible*, allowing the user to change the individual building
blocks (e.g., but not at all limited to, as mentioned above,
the expressions for :math:`\chi^2` and :math:`S`).
But even implementing new functions is possible for the user.

Nevertheless, there are sensible defaults for everything and the
most common tasks can be carried out in a user-friendly way in a
few lines of python.

Different ways of choosing alpha
--------------------------------

Furthermore, the TRIQS/maxent provides the spectral functions for different
ways of choosing :math:`\alpha` at the same time, which provides the user valuable
information when assessing the quality of the continuation.

The procedure is done in two steps:

 * The first step is to perform the analytic continuation for a range of :math:`\alpha` values.
   If selected by the user also the probability for each :math:`\alpha`
   is calculated. This step is the computationally more demanding
   part.

 * In the second step the solution for each way of choosing :math:`\alpha` is then obtained
   by analyzing the full data set of the first step. The code ships with a variety
   of :ref:`Analyzers<analyzersref>`, which perform this task. Of course,
   the user can also write their own analyzer for their preferred way of
   selecting the optimal :math:`\alpha`.

Continuation of off-diagonal elements
-------------------------------------

A main feature of this package is the continuation of off-diagonal elements
which correspond to spectral functions which are not non-negative. The normal entropy term
is not defined for negative spectral functions. To circumvent this we use the so-called
PosNeg entropy to continue these elements of matrix-valued spectral functions.
The flexibility of the implementation allows us to just swap out the expression for the entropy.

In principle, it is necessary to ensure a hermitian and positive semi-definite spectral function.
However, this is only possible when all matrix elements are treated at the same time.
The code only supports an elementwise continuation, but the full matrix-version will be released eventually.

Quality control
---------------

Along with the desired output (i.e., the spectral function), other quantities are returned by
the program. Using these, it is possible to assess the quality and correctness of the result.
Due to the ill-posed nature of the problem, it is not always straightforward to decide whether
the features of the obtained spectral function are real or artefacts. Investigating this extra
information helps to come to a conclusion.

The package also offers tools to visualize these quantities to encourage the user to actually
have a look at them.
