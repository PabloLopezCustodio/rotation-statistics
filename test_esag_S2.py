# This script tests the ESAG distribution for S^2 (d=3)
# For details, see the paper:
# Lopez-Custodio PC, 2025, "A cheatsheet for probability distributions of orientational data", preprint: https://arxiv.org/abs/2412.08934
# coded by: Pablo Lopez-Custodio, pablo.lopez-custodio@ntu.ac.uk

import numpy as np
from numpy import pi as PI
import rotstats.esag as esag
from rotstats.utils import exp_SO3, plot_data_S2
import matplotlib.pyplot as plt

N_SAMPLES = 30

######################################################################################################################
if __name__ == '__main__':
	mu = 10 * np.array([np.sin(PI/4)*np.cos(PI/4),
						np.sin(PI/4)*np.sin(PI/4),
						np.cos(PI/4)])
	rho = 0.5
	psi = PI/4
	D_esag = esag.ESAG(mu=mu, rho=rho, psi=psi)
	print("ORIGINAL DISTRIBUTION:")
	print("mu =", mu)
	print("rho =", rho)
	print("psi =", psi)
	print(f"\nSimulating {N_SAMPLES} samples from the original distribution:")
	samples = D_esag.r_ESAG(N_SAMPLES)
	print(samples)
	print("\nFitting an ESAG distribution to these samples.")
	D_esag_fitted = esag.fit_ESAG(samples)
	print("FITTED DISTRIBUTION:")
	print("mu =", D_esag_fitted.mu)
	print("rho =", D_esag_fitted.rho)
	print("psi =", D_esag_fitted.psi)
	plot_data_S2(samples, hold_show=True, title="samples from original distribution")
	D_esag.view_ESAG(n_points=200, hold_show=True, renorm_den=0.3, title="original ESAG distribution")
	D_esag_fitted.view_ESAG(n_points=200, hold_show=True, renorm_den=0.3, title="fitted ESAG distribution")
	plt.show()
