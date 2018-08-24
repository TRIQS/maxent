from pytriqs.plot.mpl_interface import oplot

fig1 = plt.figure()
# chi2(alpha) and linefit
plt.subplot(2, 2, 1)
res.analyzer_results['LineFitAnalyzer'].plot_linefit()
res.plot_chi2()
plt.ylim(1e1, 1e3)
# curvature(alpha)
plt.subplot(2, 2, 3)
res.analyzer_results['Chi2CurvatureAnalyzer'].plot_curvature()
# probablity(alpha)
plt.subplot(2, 2, 2)
res.plot_probability()
# backtransformed G_rec(tau) and original G(tau)
# by default (plot_G=True) also original G(tau) is plotted
plt.subplot(2, 2, 4)
res.plot_G_rec(alpha_index=5)
plt.tight_layout()

# spectral function A
fig2 = plt.figure()
oplot(G_w, mode='S', color='k', lw=3, label='Original Model')
plt.plot(res.omega, res.analyzer_results['LineFitAnalyzer']['A_out'],
         '-', lw=3, label='LineFit')
plt.plot(res.omega, res.analyzer_results['Chi2CurvatureAnalyzer']['A_out'],
         '--', lw=3, label='Chi2Curvature')
plt.plot(res.omega, res.analyzer_results['BryanAnalyzer']['A_out'],
         '-', lw=3, label='Bryan')
plt.plot(res.omega, res.analyzer_results['ClassicAnalyzer']['A_out'],
         '--', lw=3, label='Classic')

plt.legend()
plt.xlim(-3, 3)
plt.ylim(0, 0.6)
plt.ylabel('A')
plt.xlabel(r'$\omega$')
plt.tight_layout()

# print the optimal alpha-values
print('Curvature: ',
      res.analyzer_results['Chi2CurvatureAnalyzer']['alpha_index'])
print('LineFit: ', res.analyzer_results['LineFitAnalyzer']['alpha_index'])
print('Classic: ', res.analyzer_results['ClassicAnalyzer']['alpha_index'])
