# This script tests the ACG distribution for S^3 (d=4)

import numpy as np
from numpy import pi as PI
import rotstats.bingham as bingham

N_SAMPLES = 30

######################################################################################################################
if __name__ == '__main__':
	kappa = [-870, -450, -120, 0.]
	V = np.array([[-0.41185276,  0.10900555,  0.63721332,  0.64222603],
				  [-0.05880556,  0.51833354, -0.66515147,  0.53427126],
				  [-0.85738932,  0.20311033, -0.13091896, -0.45441167],
				  [0.3029897,    0.82352555,  0.36659642, -0.30920863]])
	B = np.matmul(V, np.matmul(np.diag(kappa), V.T))
	D_bingham = bingham.Bingham(B)
	print("ORIGINAL DISTRIBUTION:")
	print("B has spectral decomposition V*diag(kappa)*V^T where:")
	print("V =")
	print(V)
	print("kappa =", kappa)
	print(f"\nSimulating {N_SAMPLES} samples from the original distribution:")
	samples = D_bingham.r_bingham(N_SAMPLES)
	print("\nFitting a Bingham distribution to these samples.")
	D_bingham_fitted = bingham.fit_Bingham(samples)
	print("fitted distribution has parameter B:")
	print(D_bingham_fitted.B)
	print("original distribution has parameter B:")
	print(D_bingham.B)
	print("\nPlotting original distribution")
	D_bingham.view_bingham()
