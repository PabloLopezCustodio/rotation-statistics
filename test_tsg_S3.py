# This script tests the Gaussian in the tangent space of S^3 (d=4)
# For details, see the paper:
# Lopez-Custodio PC, 2025, "A cheatsheet for probability distributions of orientational data", preprint: https://arxiv.org/abs/2412.08934
# author: Pablo Lopez-Custodio, pablo.lopez-custodio@ntu.ac.uk

import numpy as np
import matplotlib.pyplot as plt
import rotstats.tangent_space_gaussian as tsg
from rotstats.utils import plot_frames

N_SAMPLES = 30

######################################################################################################################
if __name__ == '__main__':
	mu = np.array([-0.64222603, -0.53427126,  0.45441167,  0.30920863])
	B = np.array([[ 0.6372863,  -0.10895546, -0.41175309],
				  [-0.66510798, -0.51837609, -0.05892216],
				  [-0.13076859, -0.20310188, -0.85741427],
				  [ 0.36660214, -0.82350747,  0.30303191]])
	Sigma = np.diag([0.0045128, 0.0011362, 0.00061019])
	print("ORIGINAL DISTRIBUTION:")
	print("mu =", mu)
	print("B =\n", B)
	print("Sigma =\n", Sigma)
	D_tsg = tsg.TS_Gaussian(b=mu, Sigma=Sigma, Tb=B)
	print(f"\nSimulating {N_SAMPLES} samples from the original distribution:")
	samples = D_tsg.r_TSG(N_SAMPLES)
	print(samples)
	print("\nFitting a Gaussian in the tangent space of S^2 to these samples.")
	D_tsg_fitted = tsg.fit_TSG(samples)
	print("FITTED DISTRIBUTION:")
	print("mu =", D_tsg_fitted.b)
	print("B =\n", D_tsg_fitted.Tb)
	print("Sigma =\n", D_tsg_fitted.Sigma)
	plot_frames(samples, hold_show=True, title="samples from original distribution")
	D_tsg.view_TSG(hold_show=True, title="original distribution")
	D_tsg_fitted.view_TSG(hold_show=True, title="fitted distribution")
	plt.show()

