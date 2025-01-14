import numpy as np
from numpy import linalg as LA
from numpy import pi as PI
from scipy.special import gamma as Gamma
from scipy import integrate
from scipy.integrate import IntegrationWarning
import math
from rotstats.utils import *
import matplotlib.pyplot as plt
from matplotlib import cm
import warnings

warnings.simplefilter("ignore", IntegrationWarning)

class ACG:
    # Lambda = concentration matrix
    # d = ambient space dimension, q in Tyler's paper. However, note that he calls S^(d-1) "S^q"
    # Q = eigenvectors matrix, descending order
    # a = eigenvalues, descending order
    # n = normalising constant
    # det = determinant function (avoid numpy for d=3,4)
    def __init__(self, Lambda=None):
        self.d = None
        self.Q = None
        self.a = None
        self.Lambda = None
        self.c = None
        self.log_c = None
        self.det = None
        if Lambda is not None:
            self.set_Lambda(Lambda)

    def set_Lambda(self, Lambda):
        if np.shape(np.shape(Lambda))[0] != 2:
            raise Exception("concentration matrix must be 2-dimensional")
        if np.shape(Lambda)[0] != np.shape(Lambda)[1]:
            raise Exception("concentration matrix must be square")
        if not np.allclose(Lambda, np.transpose(Lambda)):
            raise Exception("concentration matrix must be symmetric positive definite")
        self.Lambda = Lambda
        self.d = np.shape(Lambda)[0]
        self.a, self.Q = LA.eigh(Lambda)
        self.Q = self.Q[:, np.flip(np.argsort(self.a))] #descending order
        self.a = np.flip(np.sort(self.a))
        if self.d == 3:
            self.det = det_3_by_3
        elif self.d == 4:
            self.det = det_4_by_4
        else:
            self.det = LA.det
        #self.c = self.c_ACG()

    def r_ACG(self, n):
        # draws n random samples from the ACG distribution
        # output: (n,d)
        if self.Lambda is None:
            raise Exception("concentration matrix not set")
        N_samples = np.random.multivariate_normal(np.zeros(self.d), self.Lambda, size=n)
        ACG_samples = np.zeros((n, self.d))
        for i in range(n):
            ACG_samples[i, :] = N_samples[i, :] / LA.norm(N_samples[i, :])
        return ACG_samples

    def c_ACG(self):
        # normalising constant of the ACG distribution
        if self.Lambda is None:
            raise Exception("concentration matrix not set")
        alpha_d = 2*pow(PI, self.d/2)/Gamma(self.d/2)
        return alpha_d*pow(self.det(self.Lambda), 0.5)

    def log_c_ACG(self):
        if self.Lambda is None:
            raise Exception("concentration matrix not set")
        log_det = 2 * sum(np.log(np.diag(LA.cholesky(self.Lambda))))
        log_w_p = np.log(2) + 0.5*self.d*np.log(PI) - math.lgamma(0.5*self.d)
        return - (log_w_p + 0.5*log_det)

    def d_ACG(self, x):
        # density at x
        # x: (d,), (d,1), (1,d)
        if self.Lambda is None:
            raise Exception("concentration matrix not set")
        x = np.reshape(x, (self.d,1))
        if self.log_c is None:
            self.log_c = self.log_c_ACG()
        log_dens = self.log_c - 0.5 * self.d * np.log(np.matmul(x.T, LA.solve(self.Lambda, x)))
        return np.exp(log_dens)

    def d_ACG_unprot(self, x):
        return np.exp(self.log_c - 0.5 * self.d * np.log(np.matmul(x.T, LA.solve(self.Lambda, x))))

    def view_ACG(self, n_points=100, combine=False, hold_show=False, el=30, az=45):
        print('preparing visualisation plot...')
        if self.Lambda is None:
            raise Exception("concentration matrix not set")
        if self.d == 4:
            self.view_orientation_den(n_points, combine, el=el, az=az)
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
                    face_den[i,j] = self.d_ACG(u)
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(xx, yy, zz, facecolors=plt.cm.spring(face_den), rstride=1, cstride=1, antialiased=False)
            ax.set_title("ACG density map")
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            if not hold_show:
                plt.show()
        else:
            raise Exception("function only implemented for d=3 and d=4")

    def view_orientation_den(self, n_points=100, combine=False, hold_show=False, el=30, az=45):
        n = int(n_points/4)
        vv, uu = np.meshgrid(np.linspace(0, PI, 2*n+1), np.linspace(0, 2*PI, 4*n+1))
        xx_ = np.sin(vv) * np.cos(uu)
        yy_ = np.sin(vv) * np.sin(uu)
        zz_ = np.cos(vv)
        R = quat_2_mat(self.Q[:, 0])
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
            #face_den = face_den/(np.max(face_den)/3)
            ax = fig.add_subplot(111, projection='3d',computed_zorder=False)
            ax.plot_surface(xx, yy, zz, facecolors=plt.cm.spring(face_den), rstride=1, cstride=1, antialiased=False, zorder=5)
            for i in range(3):
                ax.plot3D([R[0,i], 1.5 * R[0,i]], [R[1,i], 1.5 * R[1,i]], [R[2,i], 1.5 * R[2,i]], ax_colour[i], zorder=10)
            arrx, arry, arrz = arrow('x', offset=1.5)
            ax.plot_surface(arrx, arry, arrz, color='k', zorder=1)
            ax.plot3D([1, 1.5], [0, 0], [0, 0], 'k', zorder=1, linewidth=3)
            arrx, arry, arrz = arrow('y', offset=1.5)
            ax.plot_surface(arrx, arry, arrz, color='k', zorder=1)
            ax.plot3D([0, 0], [1, 1.5], [0, 0], 'k', zorder=1, linewidth=3)
            arrx, arry, arrz = arrow('z', offset=1.5)
            ax.plot_surface(arrx, arry, arrz, color='k', zorder=1)
            ax.plot3D([0, 0], [0, 0], [1, 1.5], 'k', zorder=1, linewidth=3)
            ax.set_title("ACG density map")
            ax.view_init(elev=el, azim=az)
            ax.set_axis_off()
            ax.grid(False)
            if not hold_show:
                plt.show()
        else:
            for i, face_den in enumerate([face_den_e1, face_den_e2, face_den_e3]):
                ax = fig.add_subplot(1, 3, i+1, projection='3d',computed_zorder=False)
                ax.plot_surface(xx, yy, zz, facecolors=plt.cm.spring(face_den), rstride=1, cstride=1, antialiased=False, zorder=1)
                mu = R[:,i]
                ax.plot3D([mu[0], 1.5*mu[0]], [mu[1], 1.5*mu[1]], [mu[2], 1.5*mu[2]], ax_colour[i], zorder=5)
                ax.set_title(f"ACG density for e_{i+1}")
                ax.view_init(elev=30, azim=45)
                ax.set_axis_off()
                ax.grid(False)
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
        if self.log_c is None:
            self.log_c = self.log_c_ACG()

        def integrand_ei(theta):
            return self.d_ACG_unprot(quat_mul(q_O_i, quat_from_axis_angle(ei_, theta)))

        res = integrate.quad(integrand_ei, 0, 2 * PI)
        return res[0]/(2*PI)


def fit_ACG(axial_data):
    # Tyler's iterative algorithm for maximum-likelihood estimation of the ACG parameters
    # axial_data: (number of samples, dimension of ambient space)
    n, d = np.shape(axial_data)
    if d == 3:
        det = det_3_by_3
    elif d == 4:
        det = det_4_by_4
    else:
        det = LA.det
    Lambda_ml = np.eye(d)
    divergence = 1
    k = 1
    gram_matrices = []
    for i in range(n):
        gram_matrices.append(np.matmul(np.transpose([axial_data[i, :]]), [axial_data[i, :]]))
    while divergence>1e-10:
        C = 0.0
        M = np.zeros((d, d))
        for i in range(n):
            w = 1.0 / (np.dot(axial_data[i, :], LA.solve(Lambda_ml, axial_data[i, :])))
            C = C + w
            M = M + w*gram_matrices[i]
        Lambda_ml_new = (d / C) * M
        divergence = (np.trace(LA.solve(Lambda_ml_new, Lambda_ml)) - d +
                      np.log(det(Lambda_ml_new)/det(Lambda_ml)))/ 2
        Lambda_ml = Lambda_ml_new
        k+=1
    print('Maximum likelihood estimate of Lambda in', k-1, 'iterations.')
    return ACG(Lambda_ml * d / np.trace(Lambda_ml))

