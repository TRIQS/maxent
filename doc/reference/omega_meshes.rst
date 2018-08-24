.. _omegameshes:

Omega Meshes
============

The following plot shows how a subset of those behaves.
Typically, all other meshes aim to be denser around 0 than
a linear mesh.

.. plot::

   import matplotlib.pyplot as plt
   import numpy as np
   from triqs_maxent.omega_meshes import *

   for mesh in [LinearOmegaMesh, LorentzianOmegaMesh,
        LorentzianSmallerOmegaMesh, HyperbolicOmegaMesh]:

        m = mesh(omega_min=-10,omega_max=10,n_points=100)
        plt.plot(list(m),label=mesh.__name__)
   plt.xlabel('Data point number $i$')
   plt.ylabel('Value of $\\omega_i$')
   plt.legend()

.. plot::

   import matplotlib.pyplot as plt
   import numpy as np
   from triqs_maxent.omega_meshes import *

   N = 100000
   gauge = None
   for mesh in [LinearOmegaMesh, LorentzianOmegaMesh,
        LorentzianSmallerOmegaMesh, HyperbolicOmegaMesh]:

        m = mesh(omega_min=-10,omega_max=10,n_points=N)
        h,b=np.histogram(list(m),bins=np.linspace(-10,10,100))
        if gauge is None:
            gauge = h[0]
        plt.semilogy((b[:-1]+b[1:])/2.0,np.array(h)/float(gauge),label=mesh.__name__)
   plt.xlabel('$\\omega$')
   plt.ylabel('density of points (arb. u.)')
   plt.legend(loc='lower right')



.. automodule:: triqs_maxent.omega_meshes
   :members:
   :show-inheritance:
   :member-order: bysource
