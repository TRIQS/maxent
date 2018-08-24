Continuation of a TRIQS BlockGf
===============================

The continuation of a BlockGf can be easily performed by
looping over all blocks. The following example is for
a BlockGf with individual blocks of size 1, thus we can use
:py:class:`.TauMaxEnt` for each Green function.

.. code-block:: python

    results = {}
    for name, gtau in G_tau:
         tm = TauMaxEnt()
         tm.set_G_tau(gtau)
         tm.set_error(1.e-3)
         results[name] = tm.run()
         # for saving to h5, better use
         # results[name] = tm.run().data

Should your BlockGf also contain matrix-valued Blocks, :py:class:`.ElementwiseMaxEnt`
or :py:class:`.PoormanMaxEnt` can be used for these entries.
