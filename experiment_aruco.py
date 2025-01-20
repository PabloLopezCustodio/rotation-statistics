import numpy as np
import json
import os

import rotstats.acg as acg
import rotstats.bingham as bingham
import rotstats.mat_fisher as mf
import rotstats.tangent_space_gaussian as tsg
from rotstats.utils import *

DATA_PATH = os.path.join("data", "arucos_data.json")

def load_data():
	f = open(DATA_PATH)
	data_dict = json.load(f)
	TOAs_cad = [np.asarray(T) for T in data_dict["TOAs_cad"]]
	TOAs_ehc = [np.asarray(T) for T in data_dict["TOAs_ehc"]]
	TCAs = [np.asarray(T) for T in data_dict["TCAs"]]
	TOHs = [np.asarray(T) for T in data_dict["TOHs"]]
	THC_cad = np.asarray(data_dict["THC_cad"])
	THC_ehc = np.asarray(data_dict["THC_ehc"])
	return TOAs_cad, TOAs_ehc, TCAs, TOHs, THC_cad, THC_ehc


######################################################################################################################
if __name__ == '__main__':
	TOAs_cad, TOAs_ehc, TCAs, TOHs, THC_cad, THC_ehc = load_data()
	ROAs_cad = [T[:3,:3] for T in TOAs_cad]
	ROAs_ehc = [T[:3, :3] for T in TOAs_ehc]
	QOAs_cad = np.asarray([mat_2_quat(R) for R in ROAs_cad])
	QOAs_ehc = np.asarray([mat_2_quat(R) for R in ROAs_ehc])
	N = len(ROAs_cad)
	print("dataset contains", N, "points")

	n_points_plot = 150
	elevation = 30
	azimuth = -135

	# print("===============================================")
	# D_acg_cad = acg.fit_ACG(QOAs_cad)
	# D_acg_ehc = acg.fit_ACG(QOAs_ehc)
	# print("\nACG distribution:\n")
	# print("CAD Lambda:")
	# print(D_acg_cad.Lambda)
	# print("EHC Lambda:")
	# print(D_acg_ehc.Lambda)
	# print("CAD eigenvalues:", D_acg_cad.a)
	# print("EHC eigenvalues:", D_acg_ehc.a)
	# D_acg_cad.view_ACG(n_points=n_points_plot, combine=True, el=elevation, az=azimuth, hold_show=False)
	# D_acg_ehc.view_ACG(n_points=n_points_plot, combine=True, el=elevation, az=azimuth, hold_show=False)


	# print("\n\n===============================================")
	# D_mf_cad, error_cad = mf.fit_MFisher(ROAs_cad, return_error=True)
	# D_mf_ehc, error_ehc = mf.fit_MFisher(ROAs_ehc, return_error=True)
	# print("\nMat Fisher distribution:\n")
	# print("CAD U_hat:\n", D_mf_cad.U)
	# print("CAD V_hat:\n", D_mf_cad.V)
	# print("CAD s_hat:", D_mf_cad.s)
	# print("CAD dispersion around u_1:", D_mf_cad.s[1]+D_mf_cad.s[2])
	# print("CAD dispersion around u_2:", D_mf_cad.s[2] + D_mf_cad.s[0])
	# print("CAD dispersion around u_3:", D_mf_cad.s[0] + D_mf_cad.s[1])
	# print("CAD error:", error_cad)
	# print("EHC U_hat:\n", D_mf_ehc.U)
	# print("EHC V_hat:\n", D_mf_ehc.V)
	# print("EHC s_hat:", D_mf_ehc.s)
	# print("EHC dispersion around u_1:", D_mf_ehc.s[1] + D_mf_ehc.s[2])
	# print("EHC dispersion around u_2:", D_mf_ehc.s[2] + D_mf_ehc.s[0])
	# print("EHC dispersion around u_3:", D_mf_ehc.s[0] + D_mf_ehc.s[1])
	# print("EHC error:", error_ehc)
	#
	# D_mf_cad.view_MFisher(n_points=n_points_plot, combine=True, hold_show=False, el=elevation,az=azimuth)
	# D_mf_ehc.view_MFisher(n_points=n_points_plot, combine=True, hold_show=False, el=elevation,az=azimuth)

	#
	# print("\n\n===============================================")
	# S = np.matmul(QOAs_cad.T, QOAs_cad) / N
	# D_bingham_cad = bingham.fit_Bingham(S, scatter_matrix=True)
	# S = np.matmul(QOAs_ehc.T, QOAs_ehc) / N
	# D_bingham_ehc = bingham.fit_Bingham(S, scatter_matrix=True)
	# print("\nBingham distribution:\n")
	# print('CAD kappa:', D_bingham_cad.kappa)
	# print('CAD V:\n', D_bingham_cad.V)
	# print('EHC kappa:', D_bingham_ehc.kappa)
	# print('EHC V:\n', D_bingham_ehc.V)
	# D_bingham_cad.view_bingham(n_points=n_points_plot, combine=True, hold_show=False, el=elevation, az=azimuth)
	# D_bingham_ehc.view_bingham(n_points=n_points_plot, combine=True, hold_show=False, el=elevation, az=azimuth)
	#
	# print("\n\n===============================================")
	# D_tsg_cad = tsg.fit_TSG(QOAs_cad)
	# D_tsg_ehc = tsg.fit_TSG(QOAs_ehc)
	# print("\nGaussian in the tangent space:\n")
	# print("CAD base point:", D_tsg_cad.b)
	# print("CAD covariance:\n", D_tsg_cad.Sigma)
	# print("CAD R^3 basis:\n", D_tsg_cad.Tb)
	# print("EHC base point:", D_tsg_ehc.b)
	# print("EHC covariance:\n", D_tsg_ehc.Sigma)
	# print("EHC R^3 basis:\n", D_tsg_ehc.Tb)
	# D_tsg_cad.view_TSG(n_points=n_points_plot, combine=True, hold_show=True, el=elevation, az=azimuth)
	#
	plot_frames(QOAs_cad, hold_show=False, elev=elevation, azim=azimuth)
	plot_frames(QOAs_ehc, hold_show=False, elev=elevation, azim=azimuth)