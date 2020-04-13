
from triqs_maxent import *
import numpy as np

# initialize TauMaxEnt, set G_tau from file, set omega and alpha mesh
# set the probability; then add Bryan and Classic analyzers are added by the program
# LineFit and Chi2Curvature analyzers are set set by default
tm = TauMaxEnt(probability='normal')
tm.set_G_tau_file(filename='srvo3_G_tau.dat', tau_col=0, G_col=1, err_col=2)
tm.omega = HyperbolicOmegaMesh(omega_min=-10, omega_max=10, n_points=400)
tm.alpha_mesh = LogAlphaMesh(alpha_min=1e-4, alpha_max=100, n_points=50)
# run MaxEnt
res = tm.run()

# run the same calculation with preblur
# Here we guess the preblur parameter b, but it is better to determine
# b with a chi2(b) linefit or curvature analysis!
# set preblur parameter b, PreblurA_of_H and PreblurKernel
b = 0.1
tm.A_of_H = PreblurA_of_H(b=b, omega=tm.omega)
K_orig = tm.K
tm.K = PreblurKernel(K=K_orig, b=b)
# run Maxent with preblur
res_pb = tm.run()
