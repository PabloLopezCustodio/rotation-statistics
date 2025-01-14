import numpy as np
from numpy import linalg as LA
from numpy import pi as PI
from scipy.stats import norm as normal
from scipy.optimize import fmin
from rotstats.utils import *

class ESAG:
    # mu: mean, (d,) or (d,1)
    # gamma: geometry parameters in R^(d-1), (d-1,) or (d-1,1)
    def __init__(self, mu=None, gamma=None):
        self.mu = None
        self.gamma = None
        self.invV = None
        self.psi = None
        self.rho = None
        self.eigvecs = None
        self.det = None
        if mu is not None and gamma is not None:
            self.d = np.size(mu)
            if len(gamma) != self.d-1:
                raise Exception("wrong number of geometry parameters")
            self.gamma = gamma
            self.mu = mu
            if self.d == 2:
                self.det = det_2_by_2
            elif self.d == 3:
                self.det = det_3_by_3
            elif self.d == 4:
                self.det = det_4_by_4
            else:
                self.det = LA.det

    def comp_invK(self):
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
        if self.d != 3:
            raise Exception("function only available for d=3")
        self.psi = np.arctan2(self.gamma[1], self.gamma[0]) / 2
        rho1 = -self.gamma[0]/np.cos(2*self.psi) + np.sqrt(self.gamma[0]**2/(np.cos(2*self.psi)**2)+1)
        # rho1 = LA.norm(self.gamma)+np.sqrt(LA.norm(self.gamma)**2+1)
        rho2 = 1/rho1
        self.rho = np.array([rho1,rho2])
        mu0 = np.sqrt(self.mu[1]**2+self.mu[2]**2)
        #xi2_ = np.cross(np.array([1,0,0]), self.mu)/LA.norm(np.cross(np.array([1,0,0]), self.mu))
        xi2_ = np.array([0, -self.mu[2], self.mu[1]])/mu0
        #xi1_ = np.cross(self.mu, xi2_)/LA.norm(np.cross(self.mu, xi2_))
        xi1_ = np.array([-mu0**2, self.mu[0]*self.mu[1], self.mu[0]*self.mu[2]])/(mu0*LA.norm(self.mu))
        xi1 = np.cos(self.psi)*xi1_ + np.sin(self.psi)*xi2_
        xi2 = -np.sin(self.psi)*xi1_ + np.cos(self.psi)*xi2_
        #print('unit mu =', self.mu/LA.norm(self.mu))
        #print('xi1_ X xi2_ =', np.cross(xi1_,xi2_))
        #print('xi1 X xi2 =', np.cross(xi1, xi2))
        self.eigvecs = np.vstack((xi1, xi2)).T

    def d_ESAG(self, X):
        # density at X
        # X: (n, d)
        # output: (n,)
        self.comp_invK()
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
        return f_x
    def loglik_ESAG(self, X):
        # X: data (n,3)
        n = len(X)
        f_x = self.d_ESAG(X)
        loglik = 0
        for i in range(n):
            loglik = loglik + np.log(f_x[i])
        return loglik

def fit_ESAG(X):
    # X: (n,3)
    def f(a):
        mu = a[:3]
        gamma = a[3:]
        esag_pdf = ESAG(mu, gamma)
        return -esag_pdf.loglik_ESAG(X)
    a0 = [1, 1, 1, 1, 1]
    ahat = fmin(f,a0,disp=1)
    muhat = ahat[:3]
    gammahat = ahat[3:]
    return ESAG(muhat, gammahat)

def create_ellipse(R, alpha, a_2, a_3, num_points):
    times = np.linspace(0.0, 2*np.pi, num=num_points)
    r = np.stack((np.sqrt(a_2)*np.cos(times)/alpha,
                  np.sqrt(a_3)*np.sin(times)/alpha,
                  np.zeros(num_points)))
    v = np.matmul(R, r)  # (3, num_points)
    k = np.transpose([R[:,2]])
    p = np.tile(k, num_points) # (3, num_points)
    theta = LA.norm(v, axis=0)
    exp_v = np.tile(np.cos(theta), (3,1))*p + np.tile(np.sin(theta)/theta, (3,1))*v
    return exp_v # (3, num_points)
