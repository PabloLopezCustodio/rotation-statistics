import numpy as np
from numpy import linalg as LA
from numpy import pi as PI
import sys
import os
import matplotlib.pyplot as plt
from rotstats.utils import *
from scipy.stats import multivariate_normal
from scipy import integrate

class TS_Gaussian:
    # d: dimension of ambient space
    # b: base point of tangent space, in S^{d-1}. (d,), (d,1), (1,d)
    # Sigma: covariance matrix. (d-1, d-1)
    # Tb: basis transformation from R^{d} to R^{d-1}. (d, d-1)
    # D_mvn: scipy.stats.multivariate_normal object for the Gaussian in the tangent space
    def __init__(self, b=None, Sigma=None, Tb=None):
        self.d = None
        self.b = None
        self.Sigma = None
        self.det = None
        self.Tb = Tb
        self.D_mvn = None
        if Sigma is not None and b is not None and Tb is not None:
            self.set_param(b, Sigma, Tb)

    def set_param(self, b, Sigma, Tb):
        if Sigma.shape[0] != np.size(b) - 1:
            raise Exception('Sigma must have shape (d-1,d-1)')
        if np.shape(Sigma)[0] != np.shape(Sigma)[1]:
            raise Exception("Sigma must be square")
        if not np.allclose(Sigma, np.transpose(Sigma)):
            raise Exception("Sigma must be symmetric positive definite")
        self.d = np.size(b)
        if Tb.shape != (self.d, self.d - 1):
            raise Exception("Tb must have shape (d,d-1)")
        self.Sigma = Sigma
        self.b = np.asarray(b)
        self.D_mvn = multivariate_normal(np.zeros(self.d-1), self.Sigma)
        self.Tb = Tb
        if self.d == 3:
            self.det = det_3_by_3
        elif self.d == 4:
            self.det = det_4_by_4
        else:
            self.det = LA.det

    def d_TSG(self, X):
        if self.D_mvn is None or self.Tb is None:
            raise Exception('Parameters are not defined')
        if np.ndim(X) == 1:
            X = [X]
        if X.shape[1] != self.d:
            raise Exception(f'x must be a {self.d}-dimensional vector')
        Z = np.zeros_like(X)
        for i, x in enumerate(X):
            Z[i] = log_S(self.b, x if np.dot(self.b,x) > 0 else -x)
        return self.D_mvn.pdf(np.matmul(Z, self.Tb))

    def d_TSG_unprot(self, x):
        z = np.matmul(log_S(self.b, x if np.dot(self.b,x) > 0 else -x), self.Tb)
        return self.D_mvn.pdf(z)

    def r_TSG(self, n):
        return np.matmul(self.D_mvn.rvs(n), self.Tb.T)

    def view_TSG(self, n_points=100, combine=False, hold_show=False, el=30,az=45):
        print('preparing visualisation plot...')
        if self.D_mvn is None:
            raise Exception("distribution parameters are not defined")
        if self.d == 4:
            self.view_orientation_den(n_points, combine, el=el,az=az)
        elif self.d == 3:
            vv, uu = np.meshgrid(np.linspace(0, PI, int(n_points/2)), np.linspace(0, 2 * PI, int(n_points/2)*2))
            xx = np.sin(vv) * np.cos(uu)
            yy = np.sin(vv) * np.sin(uu)
            zz = np.cos(vv)
            face_den = np.zeros_like(xx)
            for i in range(len(xx)):
                for j in range(len(xx[i])):
                    u = np.array([xx[i, j], yy[i, j], zz[i, j]])
                    u = u / LA.norm(u)
                    face_den[i,j] = self.d_TSG(u)
            max_den = np.max(face_den)
            face_den = face_den / (max_den / 2)
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(xx, yy, zz, facecolors=plt.cm.spring(face_den), rstride=1, cstride=1, antialiased=False)
            #ax.set_title("ACG density map")
            #ax.set_xlabel('X')
            #ax.set_ylabel('Y')
            #ax.set_zlabel('Z')
            ax.plot([1, 1.5], [0, 0], [0, 0], color='red', zorder=1, linewidth=4)
            ax.plot([0, 0], [1, 1.5], [0, 0], color='green', zorder=10, linewidth=4)
            ax.plot([0, 0], [0, 0], [1, 1.5], color='blue', zorder=10, linewidth=4)
            ax.view_init(elev=-15, azim=100)
            #ax.view_init(elev=30, azim=120)
            ax.set_axis_off()
            ax.grid(False)
            if not hold_show:
                plt.show()
        else:
            raise Exception("function only implemented for d=3 and d=4")

    def view_orientation_den(self, n_points=100, combine=False, hold_show=False, el=30,az=45):
        n = int(n_points/4)
        vv, uu = np.meshgrid(np.linspace(0, PI, 2*n+1), np.linspace(0, 2*PI, 4*n+1))
        xx_ = np.sin(vv) * np.cos(uu)
        yy_ = np.sin(vv) * np.sin(uu)
        zz_ = np.cos(vv)
        R = quat_2_mat(self.b)
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
            face_den = face_den / (np.max(face_den))
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
            ax.set_title("Tangent Gaussian density map")
            ax.view_init(elev=el, azim=az)
            ax.set_axis_off()
            ax.grid(False)
            if not hold_show:
                plt.show()
        else:
            for i, face_den in enumerate([face_den_e1, face_den_e2, face_den_e3]):
                ax = fig.add_subplot(1, 3, i+1, projection='3d',computed_zorder=False)
                ax.plot_surface(xx, yy, zz, facecolors=plt.cm.spring(face_den), rstride=1, cstride=1, antialiased=False, zorder=1)
                ax.plot3D([R[0, i], 1.5 * R[0, i]], [R[1, i], 1.5 * R[1, i]], [R[2, i], 1.5 * R[2, i]], ax_colour[i], zorder=5)
                ax.set_title(f"ACG density for e_{i+1}")
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
            if not hold_show:
                plt.show()

    def marginal_den_ei(self, r, ei):
        # r is already normalised
        ei_ = np.zeros(3)
        ei_[ei-1] = 1
        s = np.cross(ei_,r)
        tmp = LA.norm(s)
        if tmp < 1e-4:
            q_O_i = np.array([0,0,0,1])
        else:
            q_O_i = quat_from_axis_angle(s/tmp, np.arccos(np.dot(ei_, r)))

        def integrand_ei(theta):
            return self.d_TSG_unprot(quat_mul(q_O_i, quat_from_axis_angle(ei_, theta)))

        res = integrate.quad(integrand_ei, 0, 2 * PI)
        return res[0]/(2*PI)


def fit_TSG(X):
    # X has shape (n,d)
    n, d = X.shape
    S = np.matmul(X.T, X) / n
    e, V = LA.eigh(S)
    b = V[:,np.argmax(e)]
    if np.sum(np.where(b < 0, 1, 0)) > d/2:
        b = -b
    X = correct_sign(b, X)
    Y = np.zeros_like(X)
    for i in range(n):
        Y[i,:] = log_S(b, X[i,:])
    _, _, Vh = np.linalg.svd(Y) # descending order
    Tb = Vh[:-1,:]
    Tb = np.transpose(Tb)
    Z = np.matmul(Y, Tb)
    #Sigma = np.cov(Z.T)
    Sigma = np.diag(np.var(Z, axis=0, ddof=1))
    return TS_Gaussian(b, Sigma, Tb)

