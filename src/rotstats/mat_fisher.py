# Class and functions for the Matrix Fisher distribution
# For details, see the paper:
# Lopez-Custodio PC, 2025, "A cheatsheet for probability distributions of orientational data", preprint: https://arxiv.org/abs/2412.08934
# author: Pablo Lopez-Custodio, pablo.lopez-custodio@ntu.ac.uk
# Some of the functions of this script are based on the Matlab library: https://github.com/tylee-fdcl/Matrix-Fisher-Distribution.git

import numpy as np
from numpy import linalg as LA
from numpy import pi as PI
from scipy import integrate, LowLevelCallable
from scipy.optimize import fsolve
import rotstats.func as func
import warnings
from scipy.integrate import IntegrationWarning
from rotstats.utils import *
from rotstats.bingham import Bingham

HIDE_INTEGRATION_WARNINGS = True
USE_CYTHON_FUNCTIONS = True

if HIDE_INTEGRATION_WARNINGS:
    warnings.simplefilter("ignore", IntegrationWarning)

integrand_c = LowLevelCallable.from_cython(func,'integrand')
integrand_dci = LowLevelCallable.from_cython(func,'integrand2i')
integrand_dcj = LowLevelCallable.from_cython(func,'integrand2j')
integrand_dck = LowLevelCallable.from_cython(func,'integrand2k')

def normalising_c(s):
    res = integrate.quad(integrand_c, -1, 1, args=(s[0], s[1], s[2]))
    return res[0]

def dc_i(s):
    res = integrate.quad(integrand_dci, -1, 1, args=(s[0], s[1], s[2]))
    return res[0]

def dc_j(s):
    res = integrate.quad(integrand_dcj, -1, 1, args=(s[0], s[1], s[2]))
    return res[0]

def dc_k(s):
    res = integrate.quad(integrand_dck, -1, 1, args=(s[0], s[1], s[2]))
    return res[0]

class M_Fisher:
    # Class for the Matrix-Fisher distribution
    # F = concentration matrix R^(3x3)
    # Up = proper U O(3)
    # Vp = proper V O(3)
    # s = eigenvalues, descending order
    # c = normalising constant
    def __init__(self, F=None, U=None, V=None, s=None):
        self.F = None
        self.U = None
        self.V = None
        self.s = None
        self.c = None
        if F is not None:
            self.set_F(F)
        elif U is not None and V is not None and s is not None:
            self.set_F_from_SVD(U, V, s)


    def set_F_from_SVD(self, U, V, s):
        if U.shape != (3, 3) or V.shape != (3, 3):
            raise Exception("U and V must be 3x3")
        if (not np.allclose(np.matmul(U,U.T), np.eye(3))) or (not np.allclose(np.matmul(V,V.T), np.eye(3))):
            raise Exception("U and V must be orthogonal matrices")
        s = np.asarray(s)
        if s.shape not in [(3,), (3,1), (1,3)]:
            raise Exception("is must contain 3 elements")
        s.reshape((3,))
        ind = np.flip(np.argsort(s))
        self.s = s[ind]
        self.U = U[:,ind]
        self.V = V[:,ind]
        detUp = det_3_by_3(U)
        detVp = det_3_by_3(V)
        self.U = np.matmul(self.U, np.diag([1, 1, detUp]))
        self.V = np.matmul(self.V, np.diag([1, 1, detVp]))
        self.s[2] = detUp * detVp * self.s[2]
        self.F = np.matmul(self.U, np.matmul(np.diag(self.s), self.V.T))

    def set_F(self, F):
        if np.shape(np.shape(F))[0] != 2:
            raise Exception("concentration matrix must be 2-dimensional")
        if F.shape != (3, 3):
            raise Exception("concentration matrix must be 3-by-3")
        self.F = F
        Up, sp, Vpt = LA.svd(self.F)
        ind = np.flip(np.argsort(sp)) # descending order, all non-negative
        sp = sp[ind]
        Up = Up[:, ind]
        Vpt = Vpt[ind, :]
        detUp = det_3_by_3(Up)
        detVp = det_3_by_3(Vpt) # det(Vpt) = det(Vp)
        self.U = np.matmul(Up, np.diag([1,1,detUp]))
        self.V = np.matmul(Vpt.T, np.diag([1, 1, detVp]))
        self.s = np.array([sp[0], sp[1], detUp*detVp*sp[2]])

    def set_c(self):
        self.c = self.c_MFisher()

    def c_MFisher(self):
        # normalising constant of the Matrix Fisher distribution
        if self.F is None:
            raise Exception("concentration matrix not set")
        return normalising_c(self.s)

    def d_MFisher(self, R):
        # density at R
        # INPUT:
        # R: Rotation matrix, (3,3), or list of rotation matices
        # OUTPUT:
        # (n,) or scalar
        if self.F is None:
            raise Exception("concentration matrix not set")
        if self.c is None:
            self.set_c()
        if np.size(R) == 9:
            R_ = np.reshape(R,(3,3))
            den = np.exp(np.trace(np.matmul(self.F.T, R_))) / self.c
        else:
            den = np.zeros(len(R))
            for i, R_ in enumerate(R):
                den[i] = np.exp(np.trace(np.matmul(self.F.T, R_))) / self.c
        return den

    def d_MFisher_unprot(self, R):
        return np.exp(np.trace(np.matmul(self.F.T, R))) / self.c

    def r_MFisher(self, n):
        # draws samples from the ESAG distribution
        # INPUT:
        # n: number of samples
        # OUTPUT:
        # list of n rotation matrices
        tr_S = np.sum(self.s)
        B = np.diag([2*self.s[0]-tr_S, 2*self.s[1]-tr_S, 2*self.s[2]-tr_S, tr_S])
        D_bingham = Bingham(B)
        Qs = D_bingham.r_bingham(n)
        Rs = []
        for i in range(n):
            Rs.append(np.matmul(self.U, np.matmul(quat_2_mat(Qs[i,:]),self.V.T)))
        return Rs

    def view_MFisher(self, n_points=100, combine=True, hold_show=False, el=30, az=45, renorm_den=None, title="Matrix Fisher density map"):
        # plots a visualisation of the Matrix-Fisher distribution.
        # INPUT:
        # n_points: number of points to create grid
        # combine: combine the density map of the three axes in one single plot
        # hold_show: do not run 'plt.show()'
        # el, az: elevation and azimuth of view
        # renorm_den: renormalise density by this factor of the maximum density
        # title: plot title
        print('preparing visualisation plot for', title, '....')
        if self.F is None:
            raise Exception("concentration matrix not set")
        self.view_orientation_den(n_points, combine, hold_show, el=el,az=az,renorm_den=renorm_den, title=title)

    def view_orientation_den(self, n_points, combine, hold_show, el,az, renorm_den, title):
        n = int(n_points/4)
        vv, uu = np.meshgrid(np.linspace(0, PI, 2*n+1), np.linspace(0, 2*PI, 4*n+1))
        xx_ = np.sin(vv) * np.cos(uu)
        yy_ = np.sin(vv) * np.sin(uu)
        zz_ = np.cos(vv)
        R = np.matmul(self.U, self.V.T)
        xx = R[0,0]*xx_ + R[0,1]*yy_ + R[0,2]*zz_
        yy = R[1,0]*xx_ + R[1,1]*yy_ + R[1,2]*zz_
        zz = R[2,0]*xx_ + R[2,1]*yy_ + R[2,2]*zz_
        face_den_e1 = np.zeros_like(xx)
        face_den_e2 = np.zeros_like(xx)
        face_den_e3 = np.zeros_like(xx)
        for i in range(len(xx)):
            for j in range(len(xx[i])):
                u = np.array([xx[i, j], yy[i, j], zz[i, j]])
                u = u / LA.norm(u)
                face_den_e1[i, j] = self.marginal_den_ei(u, ei=1)
                face_den_e2[i, j] = self.marginal_den_ei(u, ei=2)
                face_den_e3[i, j] = self.marginal_den_ei(u, ei=3)
        fig = plt.figure(figsize=plt.figaspect(1))
        ax_colour = ['r', 'g', 'b']
        if combine:
            face_den = face_den_e1 + face_den_e2 + face_den_e3
            if renorm_den is not None:
                face_den = face_den / (np.max(face_den) * renorm_den)
            ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
            ax.plot_surface(xx, yy, zz, facecolors=plt.cm.spring(face_den), rstride=1, cstride=1, antialiased=False,
                            zorder=5)
            for i in range(3):
                ax.plot3D([R[0, i], 1.5 * R[0, i]], [R[1, i], 1.5 * R[1, i]], [R[2, i], 1.5 * R[2, i]], ax_colour[i],
                          zorder=10)
            arrx, arry, arrz = arrow('x', offset=1.5)
            ax.plot_surface(arrx, arry, arrz, color='k', zorder=1)
            ax.plot3D([1, 1.5], [0, 0], [0, 0], 'k', zorder=1, linewidth=3)
            arrx, arry, arrz = arrow('y', offset=1.5)
            ax.plot_surface(arrx, arry, arrz, color='k', zorder=1)
            ax.plot3D([0, 0], [1, 1.5], [0, 0], 'k', zorder=1, linewidth=3)
            arrx, arry, arrz = arrow('z', offset=1.5)
            ax.plot_surface(arrx, arry, arrz, color='k', zorder=1)
            ax.plot3D([0, 0], [0, 0], [1, 1.5], 'k', zorder=1, linewidth=3)
            ax.set_title(title)
            ax.set_box_aspect((np.ptp([1, -1]), np.ptp([1, -1]), np.ptp([1, -1])))
            ax.axes.set_xlim3d(left=-1.5, right=1.5)
            ax.axes.set_ylim3d(bottom=-1.5, top=1.5)
            ax.axes.set_zlim3d(bottom=-1.5, top=1.5)
            ax.view_init(elev=el, azim=az)
            ax.set_axis_off()
            ax.grid(False)
            if not hold_show:
                plt.show()
        else:
            for i, face_den in enumerate([face_den_e1, face_den_e2, face_den_e3]):
                if renorm_den is not None:
                    face_den = face_den / (np.max(face_den) * renorm_den)
                ax = fig.add_subplot(1, 3, i+1, projection='3d',computed_zorder=False)
                ax.plot_surface(xx, yy, zz, facecolors=plt.cm.spring(face_den), rstride=1, cstride=1, antialiased=False, zorder=1)
                mu = R[:,i]
                ax.plot3D([mu[0], 1.5*mu[0]], [mu[1], 1.5*mu[1]], [mu[2], 1.5*mu[2]], ax_colour[i], zorder=5)
                ax.set_title(title + f": e_{i+1}")
                ax.view_init(elev=30, azim=45)
                ax.set_box_aspect((np.ptp([1, -1]), np.ptp([1, -1]), np.ptp([1, -1])))
                ax.axes.set_xlim3d(left=-1.5, right=1.5)
                ax.axes.set_ylim3d(bottom=-1.5, top=1.5)
                ax.axes.set_zlim3d(bottom=-1.5, top=1.5)
                ax.set_axis_off()
                ax.grid(False)
            if not hold_show:
                plt.show()

    def marginal_den_ei(self, r, ei):
        # r is already normalised
        ei_ = np.zeros(3)
        ei_[ei-1] = 1
        s = np.cross(ei_, r)
        tmp = LA.norm(s)
        if tmp < 1e-4:
            R_O_i = np.eye(3)
        else:
            R_O_i = exp_SO3(s/tmp, np.arccos(np.dot(ei_, r)))
        if self.c is None:
            self.set_c()

        def integrand_ei(theta):
            return self.d_MFisher_unprot(np.matmul(R_O_i, exp_SO3(ei_, theta)))

        res = integrate.quad(integrand_ei, 0, 2*PI)
        return res[0]/(2*PI)

def fit_MFisher(Rs, return_error=False):
    # fits a Matrix-Fisher distribution to data
    # INPUT:
    # Rs: list of rotation matrices
    # OUTPUT:
    # Mat_Fisher object with estimated parameter
    R_bar = emp_mean(Rs)
    U, d, Vt = np.linalg.svd(R_bar)
    ind = np.flip(np.argsort(d))
    d = d[ind]
    U = U[:, ind]
    Vt = Vt[ind, :]

    def H(s):
        c = normalising_c(s)
        return [dc_i(s)/c - d[0], dc_j(s)/c - d[1], dc_k(s)/c - d[2]]

    s_hat = fsolve(H, d)
    if return_error:
        return M_Fisher(U=U, V=Vt.T, s=s_hat), LA.norm(H(s_hat))
    return M_Fisher(U=U, V=Vt.T, s=s_hat)




def emp_mean(Rs):
    am_R = np.zeros((3,3))
    for R in Rs:
        am_R = am_R + R
    return am_R/len(Rs)


