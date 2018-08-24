In-depth discussion of the program structure
============================================

Structure of the code
---------------------

The code is set up in different layers.

The lowest layer are input quantities and parameters, such as the
:py:mod:`omega meshes <.omega_meshes>`, the :py:mod:`values of alpha <.alpha_meshes>`,
the input data (i.e., a Green function :math:`G(\tau)` and its error).
Furthermore, the kernel (e.g., the :py:class:`TauKernel`) that linearly relates the
spectral function to the input Green function has to be defined.

The next layer is the choice of the :py:class:`doubly derivable functions <.DoublyDerivableFunction>`
for :math:`\chi^2` (it will usually be the :py:class:`.NormalChi2`) and
for :math:`S` (e.g., the :py:class:`.NormalEntropy` or the :py:class:`.PlusMinusEntropy`).
The function for :math:`\chi^2` has as parameters the kernel (which itself depends on :math:`\tau` and :math:`\omega`),
the Green function data and the error and evaluates the value of the misfit for a given spectral function.
The function for :math:`S` has as a parameter the default model (which depends on :math:`\omega`)
and evaluates the value of the entropy for a given spectral function.
Typically, the solution of the optimization of the cost function :math:`Q = \frac12 \chi^2 - \alpha S`
is not searched in the space of :math:`A(\omega)`, but :math:`A(\omega)` is parametrized using the
singular value decomposition of the kernel, using a parameter vector :math:`v` in singular space.
As a slight complication, for non-uniform :math:`\omega` meshes,
the cost function has to be optimized with respect to :math:`H = A(\omega) \Delta\omega`.
Therefore, we split the parametrization into :math:`A(v) = A(H(v))`.
Different parametrizations are possible, we provide, e.g., the :py:class:`.NormalH_of_v` and
:py:class:`.PlusMinusH_of_v`, which are typically used together with :py:class:`.IdentityA_of_H`.

The next layer is the cost function, which represents the :py:class:`doubly derivable functions <.DoublyDerivableFunction>`
:math:`Q(v)`.
It, of course, depends on the choices made in the previous layers.
Here, for the first time, the :math:`\alpha` dependence comes in; the cost function can only
be evaluated once this parameter is chosen.
For this, we offer the :py:class:`.MaxEntCostFunction`.
There is a different choice, the :py:class:`.BryanCostFunction`, which uses optimizations for speed
that imply the choice of the normal expressions for :math:`\chi^2` and :math:`S`.
Therefore, the latter only works for diagonal elements of the spectral function.

Given the :math:`\alpha` dependence of the cost function, typically a loop over different values
of :math:`\alpha` has to be performed.
This is the next layer, which consists of the :py:class:`.MaxEntLoop` class.
It connects the cost function, a minimizer (which does the actual minimization, e.g. the :py:class:`.LevenbergMinimizer`),
an :py:mod:`alpha mesh <.alpha_meshes>`, an expression for the :py:mod:`probability <.probabilities>` of a given
:math:`\alpha` given the solution and :ref:`analyzers <analyzersref>` that pick a particular :math:`\alpha`.
The :py:class:`.MaxEntLoop` class is the lowest layer that can perform an analytic continuation automatically
and give back meaningful results with little effort.
Of course, it has sensible defaults for all its dependencies.
When :py:meth:`running <.MaxEntLoop.run>` the MaxEnt loop, it returns a :py:class:`.MaxEntResult` that
contains the output spectral function as well as other quantities for diagnostic purposes.
The data of the :py:class:`.MaxEntResult` can be written to an h5-file.

In order to simplify the use for common cases, there is the :py:class:`.TauMaxEnt` class that provides
a simplification layer for :py:class:`.MaxEntLoop`.
It uses the :py:class:`.TauKernel` and provides methods to set the Green function and the error.

For matrix-valued spectral functions, the continuation can be performed element-wise by using the :py:class:`.ElementwiseMaxEnt` layer
on top of the :py:class:`.TauMaxEnt` class.

We want to note that from each level downwards, typically the values of all the quantities are shadowed.
This means, if we want to access the :math:`omega`-mesh of an :py:class:`.ElementwiseMaxEnt` class,
all the following calls are equivalent::

    elementwise_maxent.omega
    elementwise_maxent.maxent_diagonal.omega # maxent_diagonal is a TauMaxEnt instance
    elementwise_maxent.maxent_diagonal.maxent_loop.omega
    elementwise_maxent.maxent_diagonal.maxent_loop.cost_function.omega
    elementwise_maxent.maxent_diagonal.maxent_loop.cost_function.chi2.omega
    elementwise_maxent.maxent_diagonal.maxent_loop.cost_function.chi2.K.omega


The concept of caching the evaluation
-------------------------------------

Whenever a :py:class:`doubly derivable function <.DoublyDerivableFunction>` is called with a certain input,
we want to save the output value if we need it later on.
This leads to a considerable speed increase of the program.
This is implemented for all py:class:`doubly derivable functions <.DoublyDerivableFunction>`.
For instance, the :py:class:`.NormalChi2`::

    chi2.f(A)
    chi2.f(A)

will only evaluated once, the second time the cached value will be returned.
The same holds for the first derivative, ``chi2.d(A)``, and the
second derivative, ``chi2.dd(A)``.

There is another way of supplying the argument to the function::

    chi2_A = chi2(A)
    chi2_A.f()
    chi2_A.d()
    chi2_A.dd()

That way, the supplied argument is saved and does not have to be repeated.
