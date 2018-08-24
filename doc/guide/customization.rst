Custom functions
================

The flexible implementation of the whole framework, allows
the user to change individual building blocks or implement
new functions.

We take as first example the :ref:`omega meshes<omegameshes>`.
A range of commonly used meshes are implemented and additionally
it is possible to use your own meshes with :py:class:`.DataOmegaMesh`
by supplying the mesh points as an array. However, you can also create your
own mesh class which inherits from :py:class:`.BaseOmegaMesh`:

This is an example creating a logarithmic mesh:

.. code-block:: python

        class MyLogOmegaMesh(BaseOmegaMesh):
           def __init__(self, order_min=-5, order_max=1, n_points=100):
               super(MyLogOmegaMesh, self).__init__(omega_min=-10^(order_max),
                                                    omega_max=10^(order_max),
                                                    n_points=n_points)
               if (n_points % 2 != 0):
                   raise Exception('MyLogOmegaMesh needs an even number of n_points.')
               mesh_p = -np.logspace(order_min, order_max, n_points/2.0)
               self[:] = np.append(mesh_p[::-1], -mesh_p)

In the same way, the next code block shows how you can write your own default model class,
here we implement a Gaussian, by inheriting from :py:class:`.BaseDefaultModel`:

.. code-block:: python

        class MyGaussianDefaultModel(BaseDefaultModel):
           def __init__(self, omega, sigma=0.5):
               super(MyGaussianDefaultModel, self).__init__(omega)
               self.sigma = sigma
               self._fill_values()

           def _fill_values(self):
               self._D = 1.0/(np.sqrt(2.0*np.pi*self.sigma**2)) \
                         * np.exp(-self.omega**2/(2.0*self.sigma**2))

You can use your new mesh and default model, e.g. with :py:class:`.TauMaxEnt`:

.. code-block:: python

        tm = TauMaxEnt()
        tm.omega = MyLogOmegaMesh(order_max=1, n_points=400)
        tm.D = MyGaussianDefaultModel(tm.omega)

.. note::

    If you implement a custom function that extends the capabilities of
    the maxent code, please consider sharing it with us and the world by
    means of a :ref:`pull request <pull-requests>`. Thank you!
