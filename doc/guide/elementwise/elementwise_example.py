
from triqs_maxent import *

# Generate a 2x2 model Green function
#####################################
# TRIQS Green functions
G_iw = GfImFreq(beta=300, indices=[0, 1], n_points=200)
G_w = GfReFreq(indices=[0, 1], window=(-10, 10), n_points=5000)

# Initialize Green functions with a set of rectangles
G_iw[1, 1] << (Flat(0.6) - Flat(0.3) / 2.0) * 2.0
G_iw[0, 0] << ((Flat(0.3) - Flat(0.1) / 3.0) * 6.0 / 2.0 +
               (Flat(1.0) - Flat(0.8) / (10.0 / 8.0)) * 2.0 * 10.0 / 2.0) / 4.0
G_w[1, 1] << (Flat(0.6) - Flat(0.3) / 2.0) * 2.0
G_w[0, 0] << ((Flat(0.3) - Flat(0.1) / 3.0) * 6.0 / 2.0 +
              (Flat(1.0) - Flat(0.8) / (10.0 / 8.0)) * 2.0 * 10.0 / 2.0) / 4.0

# Rotation to generate off-diagonals
theta = np.pi / 7
R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
G_iw_rot = copy.deepcopy(G_iw)
G_iw_rot.from_L_G_R(R, G_iw, R.transpose())
G_w_rot = copy.deepcopy(G_w)
G_w_rot.from_L_G_R(R, G_w, R.transpose())

# Calculate G(tau) and add some noise
G_tau_rot = GfImTime(beta=G_iw_rot.beta, indices=G_iw_rot.indices)
G_tau_rot.set_from_fourier(G_iw_rot)
G_tau_rot.data[:, :, :] = G_tau_rot.data.real + 1.e-4 * np.reshape(
    np.random.randn(np.size(G_tau_rot.data)),
    np.shape(G_tau_rot.data))

# Run Elementwise and Poorman MaxEnt
####################################

# Elementwise (all elements individually)
ew = ElementwiseMaxEnt(use_hermiticity=True)
ew.set_G_tau(G_tau_rot, tau_new=np.linspace(0, G_iw_rot.beta, 200))
ew.set_error(1e-4)
result_ew = ew.run()

# Poorman (construct default model for off-diagonals from diagonal solution)
pm = PoormanMaxEnt(use_hermiticity=True)
pm.set_G_tau(G_tau_rot, tau_new=np.linspace(0, G_iw_rot.beta, 200))
pm.set_error(1.e-4)
result_pm = pm.run()


# Plot resulting spectra
########################
import matplotlib
matplotlib.rcParams['figure.figsize'] = (12, 12)
matplotlib.rcParams.update({'font.size': 16})
from triqs.plot.mpl_interface import oplot

fig = plt.figure()
for i in range(G_iw_rot.N1):
    for j in range(G_iw_rot.N2):
        plt.subplot(2, 2, i * 2 + j + 1)
        oplot(G_w_rot[i, j], mode='S', color='b', lw=3, label='Model')
        plt.plot(result_ew.omega,
                 result_ew.get_A_out('LineFitAnalyzer')[i][j],
                 'k', lw=3, label='Elementwise')
        plt.plot(result_pm.omega,
                 result_pm.get_A_out('LineFitAnalyzer')[i][j],
                 'r', lw=3, label='Poorman')
        plt.xlim(-3, 3)
        plt.ylabel(r'A$_{%s%s}$($\omega$)' % (i, j))
        plt.xlabel(r'$\omega$ (eV)')
        plt.grid()
        plt.legend(loc='upper right', prop={'size': 12})

# Interactive Visualization in Jupyter Notebook
###############################################

from triqs_maxent.plot.jupyter_plot_maxent import JupyterPlotMaxEnt
JupyterPlotMaxEnt(result_pm)
