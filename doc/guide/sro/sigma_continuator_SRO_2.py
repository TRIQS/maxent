
from triqs_maxent import *
from h5 import *
from pytriqs.plot.mpl_interface import oplot

# load res and SigmaContinuator from h5-file
res = {}
with HDFArchive('Sr2RuO4_b37.h5', 'r') as ar:
    S_iw = ar['S_iw']
    for key in S_iw.indices:
        res[key] = ar['maxent_result_' + key]
    isc = ar['isc']

# get S_w from the auxilliary spectral function Aaux_w
Aaux_w = {}
w = res['up_0'].omega
for key in res:
    Aaux_w[key] = res[key].analyzer_results['LineFitAnalyzer']['A_out']

isc.set_Gaux_w_from_Aaux_w(Aaux_w, w, np_interp_A=10000,
                           np_omega=4000, w_min=-1.0, w_max=1.0)

# save SigmaContinuator again (now it contains S_w)
with HDFArchive('Sr2RuO4_b37.h5', 'a') as ar:
    ar['isc'] = isc

# check linfit and plot S_w
plt.figure()
plt.subplot(1, 2, 1)
for key in res:
    res[key].analyzer_results['LineFitAnalyzer'].plot_linefit()
plt.ylim(1e1, 1e4)
plt.subplot(1, 2, 2)
oplot(isc.S_w['up_0'], mode='I', label='maxent xy', lw=3)
oplot(isc.S_w['up_1'], mode='I', label='maxent xz', lw=3)
plt.ylabel(r'$\Sigma(\omega)$')
plt.xlim(-0.75, 0.75)
plt.ylim(-0.4, 0.0)
