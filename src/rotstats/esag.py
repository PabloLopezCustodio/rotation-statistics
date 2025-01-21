# Class and functions for the ESAG distribution
# For details, see the paper:
# Lopez-Custodio PC, 2025, "A cheatsheet for probability distributions of orientational data", preprint: https://arxiv.org/abs/2412.08934
# author: Pablo Lopez-Custodio, pablo.lopez-custodio@ntu.ac.uk
# Some of the functions of this script are based on the Matlab library: https://www.maths.nottingham.ac.uk/plp/pmzspp/files.php

import numpy as np
from numpy import linalg as LA
from numpy import pi as PI
from scipy.stats import norm as normal
from scipy.stats import multivariate_normal as mvn
from scipy.optimize import fmin
from rotstats.utils import *

class ESAG:
    # only functional for d=3. Initialise with parametrisation mu and gamma, or parametrisation mu, rho and psi.
    # mu: mu parameter, array (d,)
    # gamma: geometry parameters (gamma_1, gamma_2), array (d,)
    # rho: concentration parameter, scalar
    # psi: rotation of principal axes, scalar
    # xis: matrix with columns being the vectors xi_1 and xi_2
    # invV: inverse of matrix V, array (d,d)
    # det: determinant function to avoid numpy
    def __init__(self, mu=None, gamma=None, rho=None, psi=None, fast_definition=False):
        self.mu = None
        self.gamma = None
        self.invV = None
        self.psi = None
        self.rho = None
        self.xis = None
        self.det = None
        if fast_definition:
            self.d = 3
            self.mu = mu
            self.gamma = gamma
            self.det = det_3_by_3
        else:
            if mu is not None and rho is not None and psi is not None:
                # parametrisation with mu, rho and psi
                self.d = 3
                self.rho = rho
                self.psi = psi
                self.gamma = np.array([0.5 * (1 / rho - rho) * np.cos(2 * psi),
                                       0.5 * (1 / rho - rho) * np.sin(2 * psi)])
                if np.size(mu) != 3:
                    raise Exception(f"Parametrisation with psi and gamma only works for d=3. Entered mu is in R^{np.size(mu)}")
                self.mu = mu
                self.set_xis()
            elif mu is not None and gamma is not None:
                # parametrisation with mu and gamma
                self.d = np.size(mu)
                if len(gamma) != self.d-1:
                    raise Exception("wrong number of geometry parameters")
                self.gamma = gamma
                self.mu = mu
                if self.d == 2:
                    self.det = det_2_by_2
                elif self.d == 3:
                    self.det = det_3_by_3
                    self.reparametrise()
                elif self.d == 4:
                    self.det = det_4_by_4
                else:
                    self.det = LA.det


    def comp_invV(self):
        # computes the inverse of V
        if self.d != 3:
            raise Exception("function only available for d=3")
        mu = self.mu
        gamma = self.gamma
        mu0 = np.sqrt(mu[1]**2 + mu[2]**2)
        ksi1 = 1/(mu0 * LA.norm(mu))*np.array([-mu0**2, mu[0]*mu[1], mu[0]*mu[2]]).reshape((3,1))
        ksi2 = 1/mu0*np.array([0, -mu[2], mu[1]]).reshape((3,1))
        self.invV = np.eye(3) + gamma[0]*(np.matmul(ksi1, ksi1.T) - np.matmul(ksi2, ksi2.T)) + \
                           gamma[1]*(np.matmul(ksi1, ksi2.T) + np.matmul(ksi2, ksi1.T)) + \
                           (np.sqrt(1 + gamma[0]**2 + gamma[1]**2) - 1)*(np.matmul(ksi1, ksi1.T) + np.matmul(ksi2, ksi2.T))

    def reparametrise(self):
        # if the model was defined by gamma and mu, reparametrise it in terms of rho, psi and mu, also sets xis
        if self.d != 3:
            raise Exception("function only available for d=3")
        self.psi = np.arctan2(self.gamma[1], self.gamma[0]) / 2
        self.rho = -self.gamma[0]/np.cos(2*self.psi) + np.sqrt(self.gamma[0]**2/(np.cos(2*self.psi)**2)+1)
        self.set_xis()

    def set_xis(self):
        # calculates xi1 and xi2 from mu and psi
        if self.d!= 3:
            raise Exception("function only available for d=3")
        mu0 = np.sqrt(self.mu[1] ** 2 + self.mu[2] ** 2)
        xi2_ = np.array([0, -self.mu[2], self.mu[1]]) / mu0
        xi1_ = np.array([-mu0 ** 2, self.mu[0] * self.mu[1], self.mu[0] * self.mu[2]]) / (mu0 * LA.norm(self.mu))
        xi1 = np.cos(self.psi) * xi1_ + np.sin(self.psi) * xi2_
        xi2 = -np.sin(self.psi) * xi1_ + np.cos(self.psi) * xi2_
        self.xis = np.stack((xi1, xi2)).T

    def d_ESAG(self, X):
        # density at X
        # INPUT:
        # X: n points in S^{d-1}, (d,) or (n, d)
        # OUTPUT:
        # (n,) or scalar
        if self.invV is None:
            self.comp_invV()
        if np.size(X) == self.d:
            X = np.reshape(X,(1, self.d))
        n = len(X)
        X = np.transpose(X) # (d,n)
        C1 = np.diag(np.matmul(X.T, np.matmul(self.invV, X))) # (n,)
        C2 = np.matmul(X.T, self.mu.reshape(3,1)).reshape(n) # (n,)
        C3 = np.dot(self.mu, self.mu)
        f_x = np.zeros(n)
        for i in range(n):
            if C1[i] < 0:
                f_x[i] = -np.inf
                continue
            alpha = C2[i]/np.sqrt(C1[i])
            M2 = (1+alpha**2)*normal.cdf(alpha) + alpha*normal.pdf(alpha)
            f_x[i] = 1/(2*PI)*C1[i]**(-3/2)*np.exp(1/2*(alpha**2 - C3))*M2
        if np.size(f_x) == 1:
            return f_x[0]
        return f_x

    def loglik_ESAG(self, X):
        # X: data (n,3)
        n = len(X)
        f_x = self.d_ESAG(X)
        loglik = 0
        for i in range(n):
            loglik = loglik + np.log(f_x[i])
        return loglik

    def r_ESAG(self, n):
        # draws samples from the ESAG distribution
        # INPUT:
        # n: number of samples
        # OUTPUT:
        # samples in S^{d-1}, (n,d)
        if self.invV is None:
            self.comp_invV()
        samples = mvn.rvs(mean=self.mu, cov=LA.inv(self.invV), size=n)
        for i in range(n):
            samples[i] = samples[i]/LA.norm(samples[i])
        return samples

    def view_ESAG(self, n_points=100, hold_show=False, el=30, az=45, renorm_den=None, title="ESAG density map"):
        # plots a visualisation of the ESAG distribution for d=3.
        # INPUT:
        # n_points: number of points to create grid
        # hold_show: do not run 'plt.show()'
        # el, az: elevation and azimuth of view
        # renorm_den: renormalise density by this factor of the maximum density
        # title: plot title
        print('preparing visualisation plot for', title, '....')
        if self.d != 3:
            raise Exception("function only available for d=3")
        vv, uu = np.meshgrid(np.linspace(0, PI, int(n_points / 2)), np.linspace(0, 2 * PI, int(n_points / 2) * 2))
        xx = np.sin(vv) * np.cos(uu)
        yy = np.sin(vv) * np.sin(uu)
        zz = np.cos(vv)
        face_den = np.zeros_like(xx)
        for i in range(len(xx)):
            for j in range(len(xx[i])):
                u = np.array([xx[i, j], yy[i, j], zz[i, j]])
                u = u / LA.norm(u)
                face_den[i, j] = self.d_ESAG(u)
        if renorm_den is not None:
            face_den = face_den / (np.max(face_den) * renorm_den)
        fig = plt.figure(figsize=plt.figaspect(1))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(xx, yy, zz, facecolors=plt.cm.spring(face_den), rstride=1, cstride=1, antialiased=False)
        ax.set_box_aspect((np.ptp([1, -1]), np.ptp([1, -1]), np.ptp([1, -1])))
        ax.axes.set_xlim3d(left=-1.5, right=1.5)
        ax.axes.set_ylim3d(bottom=-1.5, top=1.5)
        ax.axes.set_zlim3d(bottom=-1.5, top=1.5)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(title)
        ax.view_init(elev=el, azim=az)
        if not hold_show:
            plt.show()


def fit_ESAG(X):
    # fits an ESAG distribution to data
    # INPUT:
    # X: data array (n, d)
    # OUTPUT:
    # ESAG object with estimated parameter
    def f(a):
        mu = a[:3]
        gamma = a[3:]
        D_esag = ESAG(mu=mu, gamma=gamma, fast_definition=True)
        return -D_esag.loglik_ESAG(X)
    a0 = [1, 1, 1, 1, 1]
    ahat = fmin(f,a0, disp=1)
    muhat = ahat[:3]
    gammahat = ahat[3:]
    return ESAG(mu=muhat, gamma=gammahat)

