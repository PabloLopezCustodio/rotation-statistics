# This script tests the Matrix Fisher distribution for SO(3)
# For details, see the paper:
# Lopez-Custodio PC, 2025, "A cheatsheet for probability distributions of orientational data", preprint: https://arxiv.org/abs/2412.08934
# coded by: Pablo Lopez-Custodio, pablo.lopez-custodio@ntu.ac.uk

import numpy as np
import matplotlib.pyplot as plt
from rotstats.utils import plot_frames
import rotstats.mat_fisher as mf

N_SAMPLES = 100

######################################################################################################################
if __name__ == '__main__':
	U = np.array([[0.80483186, -0.21854443,  0.5518007],
				  [0.19554222, -0.78018154, -0.59420536],
				  [0.56036499,  0.58613573, -0.5851803]])
	V = np.array([[0.06020042, -0.90661097, -0.41765112],
				  [-0.21502139, -0.42036528,  0.88150941],
				  [-0.97475213, 0.03673669, -0.22024692]])
	s = np.array([300, 125, -50])
	D_mf = mf.M_Fisher(U=U, V=V, s=s)
	print("ORIGINAL DISTRIBUTION:")
	print("F has proper SVD:")
	print("U:\n", D_mf.U)
	print("V:\n", D_mf.V)
	print("s:\n", D_mf.s)
	print(f"\nSimulating {N_SAMPLES} samples from the original distribution:")
	samples = D_mf.r_MFisher(N_SAMPLES)
	#for i,R in enumerate(samples):
	#	print("sample", i, ":\n", R)
	print("\nFitting a Matrix Fisher distribution to these samples.")
	D_mf_fitted, error = mf.fit_MFisher(samples, return_error=True)
	print("MLE error:", error)
	print("fitted distribution has parameters:")
	print("U =")
	print(D_mf_fitted.U)
	print("V =")
	print(D_mf_fitted.V)
	print("s = ",D_mf_fitted.s)
	plot_frames(samples, hold_show=True, title="samples from the original distribution")
	D_mf.view_MFisher(hold_show=True, title="original distribution")
	D_mf_fitted.view_MFisher(hold_show=True, title="recovered distribution")
	plt.show()
