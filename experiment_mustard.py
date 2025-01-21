# Operations related to the experiment in Section 7.1 of:
# Lopez-Custodio PC, 2025, "A cheatsheet for probability distributions of orientational data", preprint: https://arxiv.org/abs/2412.08934
# author: Pablo Lopez-Custodio, pablo.lopez-custodio@ntu.ac.uk

import json
import os
import numpy as np
import matplotlib.pyplot as plt

import rotstats.acg as acg
import rotstats.bingham as bingham
import rotstats.mat_fisher as mf
import rotstats.tangent_space_gaussian as tsg
from rotstats.utils import mat_2_quat, plot_frames

DATA_PATH = os.path.join("data", "mustard_data.json")

def load_data():
	f = open(DATA_PATH)
	data_dict = json.load(f)
	TOEs = [np.asarray(T) for T in data_dict["TOEs"]]
	return TOEs

######################################################################################################################
if __name__ == '__main__':
	TOEs = load_data()
	ROEs = []
	QOEs = []
	for TOE in TOEs:
		ROEs.append(TOE[:3, :3])
		QOEs.append(mat_2_quat(TOE[:3, :3]))
	QOEs = np.array(QOEs)

	n_points_plot = 100
	elevation = -60
	azimuth = -90

	# print("===============================================")
	# print("\nACG distribution:\n")
	# D_acg = acg.fit_ACG(QOEs)
	# print("Lambda:\n", D_acg.Lambda)
	# print("Eigenvalues:\n", D_acg.a)
	# D_acg.view_ACG(n_points=n_points_plot, combine=True, el=elevation, az=azimuth, hold_show=True, title="ACG distribution for the mustard experiment")

	# print("\n\n===============================================")
	# print("\nMat Fisher distribution:\n")
	# D_mf, error = mf.fit_MFisher(ROEs, return_error=True)
	# print("U_hat:\n", D_mf.U)
	# print("V_hat:\n", D_mf.V)
	# print("s_hat:", D_mf.s)
	# print("error:", error)
	# D_mf.view_MFisher(n_points=n_points_plot, combine=True, el=elevation, az=azimuth, hold_show=True, title="Matrix Fisher distribution for the mustard experiment", renorm_den=1.0)

	# print("\n\n===============================================")
	# print("\nBingham distribution:\n")
	# D_bingham = bingham.fit_Bingham(QOEs)
	# print('kappa:', D_bingham.kappa)
	# print('V:\n', D_bingham.V)
	# D_bingham.view_bingham(n_points=n_points_plot, combine=True, el=elevation, az=azimuth, hold_show=True, title="Bingham distribution for the mustard experiment")
	#
	print("\n\n===============================================")
	print("\nGaussian in the tangent space:\n")
	D_tsg = tsg.fit_TSG(QOEs)
	print("base point:", D_tsg.b)
	print("covariance:\n", D_tsg.Sigma)
	print("R^3 basis:\n", D_tsg.Tb)
	D_tsg.view_TSG(n_points=n_points_plot, combine=True, el=elevation, az=azimuth, hold_show=True, title="Gaussian in the tangent space for the mustard experiment")


	plot_frames(ROEs, hold_show=True, title="Recordings of frame E in the mustard experiment")
	plt.show()