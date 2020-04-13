
from triqs_maxent import *
from pytriqs.archive import *

# Load self-energy from h5-file
with HDFArchive('Sr2RuO4_b37.h5', 'r') as ar:
    S_iw = ar['S_iw']
    dc = ar['dc']

# Initialize SigmaContinuator, we use the
# double counting (dc) as constant shift c.
isc = InversionSigmaContinuator(S_iw, dc)

# run TauMaxEnt for each block and collect results
# in dict res.
res = {}
for name, gaux_iw in isc.Gaux_iw:
    tm = TauMaxEnt()
    tm.set_G_iw(gaux_iw)
    tm.set_error(1e-4)
    tm.omega = HyperbolicOmegaMesh(omega_min=-10, omega_max=10, n_points=500)
    tm.alpha_mesh = LogAlphaMesh(alpha_min=1e-2, alpha_max=1e2, n_points=30)
    res[name] = tm.run()

# save results and the SigmaContinuator to h5-file
with HDFArchive('Sr2RuO4_b37.h5', 'a') as ar:
    for key in res:
        ar['maxent_result_' + key] = res[key].data
    ar['isc'] = isc
