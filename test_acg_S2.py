# This script tests the ACG distribution for S^2 (d=3)

import numpy as np
from numpy import pi as PI
import rotstats.acg as acg
from rotstats.utils import exp_SO3, plot_data_S2
import matplotlib.pyplot as plt

N_SAMPLES = 30

######################################################################################################################
if __name__ == '__main__':
	Q = np.matmul(exp_SO3([0,0,1], PI/4), exp_SO3([1,0,0], PI/3))
	a = [1, 1e-3, 1e-5]
	print("ORIGINAL DISTRIBUTION:")
	print("Lambda has spectral decomposition Q*A*Q^T where:")
	print("Q =")
	print(Q)
	print("A =")
	print(np.diag(a))
	Lambda = np.matmul(Q, np.matmul(np.diag(a), Q.T))
	Lambda = Lambda * 3 / np.trace(Lambda)
	D_acg = acg.ACG(Lambda)
	print(f"\nSimulating {N_SAMPLES} samples from the original distribution:")
	samples = D_acg.r_ACG(N_SAMPLES)
	print(samples)
	print("\nFitting an ACG distribution to these samples.")
	D_acg_fitted = acg.fit_ACG(samples)
	print("Fitted distribution has Lambda:")
	print(D_acg_fitted.Lambda)
	print("Lambda from the original distribution:")
	print(Lambda)
	plot_data_S2(samples, hold_show=True, title="samples from original distribution")
	D_acg.view_ACG(n_points=200, hold_show=True, title="original distribution")
	D_acg_fitted.view_ACG(n_points=200, hold_show=True, title="fitted distribution")
	plt.show()