import numpy as np
from numpy import linalg as LA
from numpy import pi as PI
import json
from scipy.spatial import KDTree
from scipy.optimize import fsolve
from scipy import optimize
import sys
import os
import matplotlib.pyplot as plt
from rotstats.utils import *
from scipy import integrate
from scipy.integrate import IntegrationWarning
import importlib.resources
import warnings

warnings.simplefilter("ignore", IntegrationWarning)

import rotstats.acg as acg

TABLE_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'json','Bingham_table.json')

table = None

class Bingham:
    # d = ambient space dimension
    # B = concentration matrix
    # kappa = eigenvalues, ascending order
    # Q = corresponding eigenvectors matrix
    # c = normalising constant
    # det = determinant function (avoid numpy for d=3,4)
    def __init__(self, B=None):
        self.d = None
        self.Q = None
        self.kappa = None
        self.B = None
        self.c = None
        self.det = None
        if B is not None:
            self.set_B(B)

    def set_B(self, B):
        if np.shape(np.shape(B))[0] != 2:
            raise Exception("concentration matrix must be 2-dimensional")
        if np.shape(B)[0] != np.shape(B)[1]:
            raise Exception("concentration matrix must be square")
        if not np.allclose(B, np.transpose(B)):
            raise Exception("concentration matrix must be symmetric positive definite")
        self.B = B
        self.d = np.shape(B)[0]
        self.kappa, self.V = LA.eigh(B)
        self.V = self.V[:, np.argsort(self.kappa)] # ascending order of kappa
        self.kappa = np.sort(self.kappa)
        self.kappa = self.kappa - self.kappa[-1]
        if self.d == 3:
            self.det = det_3_by_3
        elif self.d == 4:
            self.det = det_4_by_4
        else:
            self.det = LA.det
    def c_bingham(self):
        # normalising constant of the Bingham distribution
        if self.B is None:
            raise Exception("concentration matrix not set")
        if self.d != 4:
            raise Exception("this function is only available for an ambient space of dimension d = 4")
        return find_in_table('normalising_term',self.kappa)

    def set_c(self):
        self.c = self.c_bingham()

    def d_bingham(self, x):
        if self.B is None:
            raise Exception("concentration matrix not set")
        if self.c is None:
            self.set_c()
        if np.size(x) != self.d:
            raise Exception(f"x must be {self.d}-dimensional")
        x = np.reshape(np.asarray(x), (self.d,1))
        x = x/LA.norm(x)
        tmp = np.matmul(x.T, np.matmul(self.B, x))
        return np.exp(tmp[0,0])/self.c

    def d_bingham_unprot(self, x):
        # x has shape (d,)
        return np.exp(np.matmul(x.T, np.matmul(self.B, x)))/self.c

    def r_bingham(self, n):
        # draws n random samples from the Bingham distribution
        # output: (n,d)
        if self.B is None:
            raise Exception("concentration matrix not set")
        lamb = -self.kappa # in Kent and Mardia, the concentration matrix is multilpied by -1 compared to Glover (used for MLE here)
        lamb = lamb - np.min(lamb)
        A = np.matmul(self.V, np.matmul(np.diag(lamb), self.V.T))

        def fun_b(b):
            return np.sum(1/(b+2*lamb)) - 1

        def dfun_b(b):
            return -np.sum(1/(b+2*lamb)**2)

        sol = optimize.root_scalar(fun_b, x0=1, fprime=dfun_b, method='newton')
        b = sol.root

        if fun_b(b)*fun_b(b) > 1e-5:
            raise Exception("unable to compute bound for rejection/acceptance")

        M_star = np.exp(-(4-b)/2)*((4/b)**2)
        Omega = np.eye(self.d) + 2*A/b
        D_acg = acg.ACG(LA.inv(Omega)) # in Kent and Mardia, Omega = inv(Lambda) compared to Tyler
        samples = []
        while True:
            x = D_acg.r_ACG(1)
            w = np.random.uniform(0, 1, 1)[0]
            log_fB_star = -np.matmul(x, np.matmul(A, x.T))[0,0]
            fACG_star = np.matmul(x, np.matmul(Omega,x.T))[0,0]**(-self.d/2)
            if np.log(w*M_star*fACG_star) < log_fB_star:
                #print('accept')
                samples.append(x[0])
            #else:
                #print('reject')
            if len(samples) == n:
                break
        data_arr = np.asarray(samples)
        return data_arr

    def view_bingham(self, n_points=100, combine=True, hold_show=False, el=30,az=45, renorm_den=None, title="Bingham density map"):
        print('preparing visualisation plot for', title, '....')
        if self.B is None:
            raise Exception("concentration matrix not set")
        if self.d == 4:
            self.view_orientation_den(n_points, combine, hold_show, el, az, renorm_den, title)
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
                    face_den[i,j] = self.d_bingham(u)
            if renorm_den is not None:
                face_den = face_den/(np.max(face_den)*renorm_den)
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(xx, yy, zz, facecolors=plt.cm.spring(face_den), rstride=1, cstride=1, antialiased=False)
            ax.set_title(title)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_box_aspect((np.ptp([1, -1]), np.ptp([1, -1]), np.ptp([1, -1])))
            ax.axes.set_xlim3d(left=-1.5, right=1.5)
            ax.axes.set_ylim3d(bottom=-1.5, top=1.5)
            ax.axes.set_zlim3d(bottom=-1.5, top=1.5)
            if not hold_show:
                plt.show()
        else:
            raise Exception("function only implemented for d=3 and d=4")

    def view_orientation_den(self, n_points, combine, hold_show, el, az, renorm_den, title):
        n = int(n_points/4)
        vv, uu = np.meshgrid(np.linspace(0, PI, 2*n+1), np.linspace(0, 2*PI, 4*n+1))
        xx_ = np.sin(vv) * np.cos(uu)
        yy_ = np.sin(vv) * np.sin(uu)
        zz_ = np.cos(vv)
        R = quat_2_mat(self.V[:, -1])
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
                face_den = face_den/(np.max(face_den)*renorm_den)
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
            ax.view_init(elev=el, azim=az)
            ax.set_axis_off()
            ax.set_box_aspect((np.ptp([1, -1]), np.ptp([1, -1]), np.ptp([1, -1])))
            ax.axes.set_xlim3d(left=-1.5, right=1.5)
            ax.axes.set_ylim3d(bottom=-1.5, top=1.5)
            ax.axes.set_zlim3d(bottom=-1.5, top=1.5)
            ax.grid(False)
            if not hold_show:
                plt.show()
        else:
            for i, face_den in enumerate([face_den_e1, face_den_e2, face_den_e3]):
                if renorm_den is not None:
                    face_den = face_den / (np.max(face_den) * renorm_den)
                ax = fig.add_subplot(1, 3, i+1, projection='3d',computed_zorder=False)
                ax.plot_surface(xx, yy, zz, facecolors=plt.cm.spring(face_den), rstride=1, cstride=1, antialiased=False, zorder=1)
                ax.plot3D([R[0,i], 1.5 * R[0,i]], [R[1,i], 1.5 * R[1,i]], [R[2,i], 1.5 * R[2,i]], ax_colour[i], zorder=5)
                ax.set_title(title + f": e_{i+1}")
                ax.view_init(elev=30, azim=45)
                ax.set_axis_off()
                ax.set_box_aspect((np.ptp([1, -1]), np.ptp([1, -1]), np.ptp([1, -1])))
                ax.axes.set_xlim3d(left=-1.5, right=1.5)
                ax.axes.set_ylim3d(bottom=-1.5, top=1.5)
                ax.axes.set_zlim3d(bottom=-1.5, top=1.5)
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
        if self.c is None:
            self.set_c()

        def integrand_ei(theta):
            return self.d_bingham_unprot(quat_mul(q_O_i, quat_from_axis_angle(ei_, theta)))

        res = integrate.quad(integrand_ei, 0, 2 * PI)
        return res[0]/(2*PI)


def fit_Bingham(M, scatter_matrix=False):
    # is either the data matrix (n,d) or the intertia matrix (d,d)
    if not scatter_matrix:
        n = np.shape(M)[0]
        S = gram(M)/n
    else:
        S = M
    e, U = LA.eigh(S)
    U = U[:, np.argsort(e)]  # ascending order of e
    e = np.sort(e)/np.sum(e)
    kappa = find_in_table("mle", e[:-1])
    B = np.matmul(U, np.matmul(np.diag(kappa), U.T))
    return Bingham(B)


def load_table():
    # returns table as dictionary
    with importlib.resources.open_text("rotstats.data", "Bingham_table.json") as f:
        data_dict = json.load(f)
    return data_dict

def find_in_table(case, request):
    global table
    if table is None:
        table = load_table()
    sqrt_z = np.asarray(table['sqrt_z_input'])
    K = 4
    if case == 'mle':
        dF_over_F = np.asarray(table['dF_over_F'])
        tree = KDTree(dF_over_F)
        distances, indexes = tree.query(request, k=K)
        w = np.exp(-(distances/0.05)**2)
        sqrt_z_res = np.matmul(w.reshape((1,K)), np.asarray(sqrt_z[indexes,:]))/np.sum(w)
        return np.array([-sqrt_z_res[0, 0] ** 2, -sqrt_z_res[0, 1] ** 2, -sqrt_z_res[0, 2] ** 2, 0])
    elif case == 'normalising_term':
        F = np.asarray(table['F'])
        tree = KDTree(sqrt_z)
        distances, indexes = tree.query([np.sqrt(-request[0]), np.sqrt(-request[1]), np.sqrt(-request[2])], k=K)
        w = np.exp(-(distances / 0.05) ** 2)
        F_res = np.dot(F[indexes],w) / np.sum(w)
        return F_res
