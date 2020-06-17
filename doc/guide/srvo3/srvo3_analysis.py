# analysis of the output (here we only look at the normal run without preblur)
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (8, 8)
plt.rcParams.update({'font.size': 16})

fig1 = plt.figure()
# chi2(alpha)
plt.subplot(2, 2, 1)
res.plot_chi2()
# curvature(alpha)
plt.subplot(2, 2, 2)
res.analyzer_results['Chi2CurvatureAnalyzer'].plot_curvature()
# linefit of chi2(alpha)
plt.subplot(2, 2, 3)
res.analyzer_results['LineFitAnalyzer'].plot_linefit()
plt.ylim(1e1, 1e4)
# probablity(alpha)
plt.subplot(2, 2, 4)
res.plot_probability()
plt.tight_layout()
fig1.savefig('srvo3_analysis.png')

# spectral function A
matplotlib.rcParams['figure.figsize'] = (8, 6)
fig2 = plt.figure()
plt.plot(res.omega, res.get_A_out('LineFitAnalyzer'),
         '-', lw=3, label='LineFit')
plt.plot(res.omega, res.get_A_out('Chi2CurvatureAnalyzer'),
         '--', lw=3, label='Chi2Curvature')
plt.plot(res.omega, res.get_A_out('BryanAnalyzer'),
         '-', lw=3, label='Bryan')
plt.plot(res.omega, res.get_A_out('ClassicAnalyzer'),
         '--', lw=3, label='Classic')
plt.plot(res_pb.omega, res_pb.get_A_out('LineFitAnalyzer'),
         'k-', lw=3, label='Preblur LineFit')
plt.legend()
plt.xlim(-6, 6)
plt.ylim(0, 1)
plt.ylabel('A')
plt.xlabel(r'$\omega$')
plt.tight_layout()
fig2.savefig('srvo3_A.png')

# print the optimal alpha-values
print(('Curvature: ', res.analyzer_results[
      'Chi2CurvatureAnalyzer']['alpha_index']))
print(('LineFit: ', res.analyzer_results['LineFitAnalyzer']['alpha_index']))
print(('Classic: ', res.analyzer_results['ClassicAnalyzer']['alpha_index']))
print(('Preblur LineFit: ', res_pb.analyzer_results[
      'LineFitAnalyzer']['alpha_index']))
