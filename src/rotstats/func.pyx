#cython: boundscheck=False, wraparound=False, nonecheck=False

import cython
from scipy.special.cython_special cimport i0, i1
from libc.math cimport exp

@cython.cdivision(True)
cdef double integrand(int n, double[4] args):
    x = args[0]
    si = args[1]
    sj = args[2]
    sk = args[3]
    return 0.5*i0(0.5*(si-sj)*(1-x))*i0(0.5*(si+sj)*(1+x))*exp(sk*x)

cdef double integrand2i(int n, double[4] args):
    x = args[0]
    si = args[1]
    sj = args[2]
    sk = args[3]
    return (0.25*(1-x)*i1(0.5*(si-sj)*(1-x))*i0(0.5*(si+sj)*(1+x))+
            0.25*(1+x)*i0(0.5*(si-sj)*(1-x))*i1(0.5*(si+sj)*(1+x)))*exp(sk*x)

cdef double integrand2j(int n, double[4] args):
    x = args[0]
    si = args[1]
    sj = args[2]
    sk = args[3]
    return (-0.25*(1-x)*i1(0.5*(si-sj)*(1-x))*i0(0.5*(si+sj)*(1+x))+
            0.25*(1+x)*i0(0.5*(si-sj)*(1-x))*i1(0.5*(si+sj)*(1+x)))*exp(sk*x)

cdef double integrand2k(int n, double[4] args):
    x = args[0]
    si = args[1]
    sj = args[2]
    sk = args[3]
    return 0.5*i0(0.5*(si-sj)*(1-x))*i0(0.5*(si+sj)*(1+x))*x*exp(sk*x)

# SCALED VERSIONS OF THE INTEGRANDS
cdef double integrand_sc(int n, double[5] args):
    x = args[0]
    si = args[1]
    sj = args[2]
    sk = args[3]
    return args[4]*0.5*i0(0.5*(si-sj)*(1-x))*i0(0.5*(si+sj)*(1+x))*exp(sk*x)

cdef double integrand2i_sc(int n, double[5] args):
    x = args[0]
    si = args[1]
    sj = args[2]
    sk = args[3]
    return args[4]*(0.25*(1-x)*i1(0.5*(si-sj)*(1-x))*i0(0.5*(si+sj)*(1+x))+
            0.25*(1+x)*i0(0.5*(si-sj)*(1-x))*i1(0.5*(si+sj)*(1+x)))*exp(sk*x)

cdef double integrand2j_sc(int n, double[5] args):
    x = args[0]
    si = args[1]
    sj = args[2]
    sk = args[3]
    return args[4]*(-0.25*(1-x)*i1(0.5*(si-sj)*(1-x))*i0(0.5*(si+sj)*(1+x))+
            0.25*(1+x)*i0(0.5*(si-sj)*(1-x))*i1(0.5*(si+sj)*(1+x)))*exp(sk*x)

cdef double integrand2k_sc(int n, double[5] args):
    x = args[0]
    si = args[1]
    sj = args[2]
    sk = args[3]
    return args[4]*0.5*i0(0.5*(si-sj)*(1-x))*i0(0.5*(si+sj)*(1+x))*x*exp(sk*x)





# SINGLE-INTEGRAND METHOD:

cdef double Gi(int n, double[5] args):
    # u = args[0], si = args[1], sj = args[2], sk = args[3], di = args[4]
    u = args[0]
    v1 = (args[1]+args[2])*(1+u)/2
    v2 = (args[1]-args[2])*(1-u)/2
    g1 = i0(v1)
    g2 = i0(v2)
    g3 = i1(v1)
    g4 = i1(v2)
    return 0.25*((1-u)*g1*g4 + (1+u)*g3*g2 - 2*args[4]*g1*g2)*exp(u*args[3])

cdef double Gj(int n, double[5] args):
    # u = args[0], si = args[1], sj = args[2], sk = args[3], di = args[4]
    u = args[0]
    v1 = (args[1]+args[2])*(1+u)/2
    v2 = (args[1]-args[2])*(1-u)/2
    g1 = i0(v1)
    g2 = i0(v2)
    g3 = i1(v1)
    g4 = i1(v2)
    return 0.25*(-(1-u)*g1*g4 + (1+u)*g3*g2 - 2*args[4]*g1*g2)*exp(u*args[3])

cdef double Gk(int n, double[5] args):
    # u = args[0], si = args[1], sj = args[2], sk = args[3], di = args[4]
    u = args[0]
    v1 = (args[1]+args[2])*(1+u)/2
    v2 = (args[1]-args[2])*(1-u)/2
    g1 = i0(v1)
    g2 = i0(v2)
    return 0.5*g2*g1*exp(u*args[3])*(u-args[4])

def inte(double x, double si, double sj, double sk):
    return 0.5*i0(0.5*(si-sj)*(1-x))*i0(0.5*(si+sj)*(1+x))*exp(sk*x)