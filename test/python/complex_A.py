# TRIQS application maxent
# Copyright (C) 2018 Gernot J. Kraberger
# Copyright (C) 2018 Simons Foundation
# Authors: Gernot J. Kraberger and Manuel Zingl
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


from __future__ import absolute_import, print_function
import numpy as np
from triqs_maxent import *
from triqs_maxent.triqs_support import *

# to make it reproducible
np.random.seed(658436166)

beta = 40
tau = np.linspace(0, beta, 100)
omega = HyperbolicOmegaMesh(omega_min=-10, omega_max=10, n_points=100)
K = TauKernel(tau=tau, omega=omega, beta=beta)
# here we construct the G(tau)
A = np.exp(-(omega - 1)**2) - np.exp(-(omega + 1)**2) + \
    (-1.0j) * (np.exp(-(omega - 1)**2) - np.exp(-(omega + 1)**2))
assert np.abs(np.trapz(A, omega)) < 1.e-14, "A not normalized to 0"

G = np.dot(K.K, A)
G += 1.e-4 * np.random.randn(len(G))
G += 1.e-4j * np.random.randn(len(G))
err = 1.e-4 * np.ones(len(G))

chi2_rp = NormalChi2(K=K, G=G.real, err=err)
chi2_ip = NormalChi2(K=K, G=G.imag, err=err)
chi2_complex = ComplexChi2(K=K, G=G, err=err)
# this should be around the number of data points, here 200, because we
# have real and imag part for 100 values
c2_rp = chi2_rp(A.real)
c2_ip = chi2_ip(A.imag)
c2_complex = chi2_complex(A.view(float).reshape(A.shape + (2,)))
assert (np.abs(c2_rp.f() + c2_ip.f() - c2_complex.f())) < 1.e-13
assert (np.max(np.abs(c2_complex.d()[:, 0] - c2_rp.d()))) < 1.e-10
assert (np.max(np.abs(c2_complex.d()[:, 1] - c2_ip.d()))) < 1.e-10
assert (np.max(np.abs(c2_complex.dd()[:, 0, :, 0] - c2_rp.dd()))) < 1.e-14
assert (np.max(np.abs(c2_complex.dd()[:, 1, :, 1] - c2_ip.dd()))) < 1.e-14
assert (np.max(np.abs(c2_complex.dd()[:, 0, :, 1]))) < 1.e-14
assert (np.max(np.abs(c2_complex.dd()[:, 1, :, 0]))) < 1.e-14

D = 0.3 - 0.1j + 0 * A
assert chi2_rp.check_derivatives(D.real, chi2_rp.f(D.real))
assert chi2_ip.check_derivatives(D.imag, chi2_ip.f(D.imag))
D_view = D.view(float).reshape(A.shape + (2,))
assert chi2_complex.check_derivatives(D_view, chi2_complex.f(D_view))

iomega = np.linspace(-(2 * 48 + 1) * np.pi / beta,
                     (2 * 48 + 1) * np.pi / beta, len(tau) - 2)
K_iw = IOmegaKernel(iomega, omega, beta=beta)
G_iw_K = np.dot(K_iw.K, A)
G_iw_K += 1.e-4 * np.random.randn(len(G_iw_K))
G_iw_K += 1.e-4j * np.random.randn(len(G_iw_K))
err_iw = 1.e-4 * np.ones(len(G_iw_K))

chi2_iw_rp = NormalChi2(K=K_iw, G=np.dot(K_iw.K, A.real), err=err_iw)
chi2_iw_ip = NormalChi2(K=K_iw, G=np.dot(K_iw.K, A.imag), err=err_iw)
chi2_iw_complex = ComplexChi2(K=K_iw, G=G_iw_K, err=err_iw)
assert chi2_iw_rp.check_derivatives(D.real, chi2_rp.f(D.real))
assert chi2_iw_ip.check_derivatives(D.imag, chi2_ip.f(D.imag))
assert chi2_iw_complex.check_derivatives(D_view, chi2_complex.f(D_view))

Def = FlatDefaultModel(omega=omega)
entropy_normal = PlusMinusEntropy(D=Def)
entropy_complex = ComplexPlusMinusEntropy(D=Def)
S_rp = entropy_normal(A.real)
S_ip = entropy_normal(A.imag)
S_complex = entropy_complex(A.view(float).reshape(A.shape + (2,)))
assert (np.abs(S_rp.f() + S_ip.f() - S_complex.f())) < 1.e-13
assert (np.max(np.abs(S_complex.d()[:, 0] - S_rp.d()))) < 1.e-10
assert (np.max(np.abs(S_complex.d()[:, 1] - S_ip.d()))) < 1.e-10
assert (S_complex.d().shape == c2_complex.d().shape)
assert (S_complex.dd().shape == c2_complex.dd().shape)
assert (np.max(np.abs(S_complex.dd()[:, 0, :, 0] - S_rp.dd()))) < 1.e-14
assert (np.max(np.abs(S_complex.dd()[:, 1, :, 1] - S_ip.dd()))) < 1.e-14
assert (np.max(np.abs(S_complex.dd()[:, 0, :, 1]))) < 1.e-14
assert (np.max(np.abs(S_complex.dd()[:, 1, :, 0]))) < 1.e-14

assert S_rp.check_derivatives(D.real, S_rp.f(D.real))
assert S_ip.check_derivatives(D.imag, S_ip.f(D.imag))
assert S_complex.check_derivatives(D_view, S_complex.f(D_view))
