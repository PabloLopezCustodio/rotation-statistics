# This script tests the ACG distribution for S^3 (d=4)
# For details, see the paper:
# Lopez-Custodio PC, 2025, "A cheatsheet for probability distributions of orientational data", preprint: https://arxiv.org/abs/2412.08934
# coded by: Pablo Lopez-Custodio, pablo.lopez-custodio@ntu.ac.uk

import numpy as np
import matplotlib.pyplot as plt
import rotstats.acg as acg
from rotstats.utils import plot_frames

N_SAMPLES = 30

######################################################################################################################
if __name__ == '__main__':
	Q = np.array([[-0.63768237, -0.64554305, -0.00865933, -0.42019088],
				  [-0.53153255,  0.64578973,  0.51186171, -0.19602642],
				  [ 0.45761155,  0.12141668, -0.01526373, -0.88069102],
				  [ 0.3184745,  -0.38921432,  0.8588886,   0.09693599]])
	a = [1, 2e-3, 4e-4, 2e-4]
	print("ORIGINAL DISTRIBUTION:")
	print("Lambda has spectral decomposition Q*A*Q^T where:")
	print("Q =")
	print(Q)
	print("A =")
	print(np.diag(a))
	Lambda = np.matmul(Q, np.matmul(np.diag(a), Q.T))
	Lambda = Lambda*4/np.trace(Lambda)
	D_acg = acg.ACG(Lambda)
	samples = D_acg.r_ACG(N_SAMPLES)
	print(f"\nSimulating {N_SAMPLES} samples from the original distribution:")
	print(samples)
	print("\nFitting an ACG distribution to these samples. \nFitted distribution has Lambda:")
	D_acg_fitted = acg.fit_ACG(samples)
	print(D_acg_fitted.Lambda)
	print("Lambda from the original distribution:")
	print(Lambda)
	plot_frames(samples, hold_show=True, title="samples from original distribution")
	D_acg.view_ACG(hold_show=True, title="original distribution")
	D_acg_fitted.view_ACG(hold_show=True, title="fitted distribution")
	plt.show()