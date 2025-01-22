# This script tests the Gaussian in the tangent space of S^2 (d=3)
# For details, see the paper:
# Lopez-Custodio PC, 2025, "A cheatsheet for probability distributions of orientational data", preprint: https://arxiv.org/abs/2412.08934
# author: Pablo Lopez-Custodio, pablo.lopez-custodio@ntu.ac.uk

import numpy as np
from numpy import pi as PI
import rotstats.tangent_space_gaussian as tsg
from rotstats.utils import exp_SO3, plot_data_S2
import matplotlib.pyplot as plt

N_SAMPLES = 30

######################################################################################################################
if __name__ == '__main__':
	Q = np.matmul(exp_SO3([0,0,1], PI/4), np.matmul(exp_SO3([1,0,0], PI/4), exp_SO3([0,0,1], PI/3)))
	mu = Q[:,2]
	B = Q[:,:2]
	Sigma = np.diag([0.6 ** 2, 0.2 ** 2])
	print("ORIGINAL DISTRIBUTION:")
	print("mu =", mu)
	print("B =\n", B)
	print("Sigma =\n", Sigma)
	D_tsg = tsg.TS_Gaussian(b=mu, Sigma=Sigma, Tb=B, antipodal_sym=False)
	print(f"\nSimulating {N_SAMPLES} samples from the original distribution:")
	samples = D_tsg.r_TSG(N_SAMPLES)
	print(samples)
	print("\nFitting a Gaussian in the tangent space of S^2 to these samples.")
	D_tsg_fitted = tsg.fit_TSG(samples, antipodal_sym=False)
	print("FITTED DISTRIBUTION:")
	print("mu =", D_tsg_fitted.b)
	print("B =\n", D_tsg_fitted.Tb)
	print("Sigma =\n", D_tsg_fitted.Sigma)
	plot_data_S2(samples, hold_show=True, title="samples from original distribution")
	D_tsg.view_TSG(n_points=200, hold_show=True, title="original distribution")
	D_tsg_fitted.view_TSG(n_points=200, hold_show=True, title="fitted distribution")
	plt.show()