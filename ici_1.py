#! /usr/bin/env python

import autograd.numpy as np
from autograd import grad  
#import numpy as np
import scipy
import scipy.linalg
import scipy.io
import copy as cp
import argparse
import scipy.sparse
import scipy.sparse.linalg
#import sys
#sys.path.insert(0, '../')

from hdvv import *
from block import *


from scipy.optimize import minimize





n_sites = 4
j12 = np.zeros((8,8))
for i in range(n_sites-1):
    j12[i,i+1] -= 1
    j12[i+1,i] -= 1

lattice = np.ones((n_sites))
lattice_guess = np.ones((4))
j12_guess = j12[0:4][0:4] 

np.random.seed(2)

    
H, tmp, S2, Sz = form_hdvv_H(lattice,j12)
l_full, v_full = np.linalg.eigh(H)
s2 = v_full.T.dot(S2).dot(v_full)
print " Exact Energies:"
for s in range(10):
    print "   %12.8f %12.8f " %(l_full[s],s2[s,s])



a = (np.random.rand(n_sites,n_sites)-.5)/100
b = (np.random.rand(n_sites,n_sites)-.5)/100

v0 = np.random.rand(v_full.shape[0])
v0 = v0/np.linalg.norm(v0)

# Add component along the exact solution to get better starting point
#v0 += cp.deepcopy(v_full[:,0])*3
#v0 = v0/np.linalg.norm(v0)

aa = np.zeros((n_sites,n_sites))
bb = np.zeros((n_sites,n_sites))
Uold = np.zeros(H.shape)
   


def form_energy(inp):

    #aa = inp[0:n_sites*n_sites]
    #bb = inp[n_sites*n_sites::]
    #aa.resize(n_sites,n_sites) 
    #bb.shape = (n_sites,n_sites) 
    
    #U = form_hdvv_U(lattice,aa,bb)
    U = form_hdvv_U_1v(lattice,inp)
   
    E = v0.T.dot(U.T).dot(H).dot(U).dot(v0) / v0.T.dot(U.T).dot(U).dot(v0)
    Uold = cp.deepcopy(U)

    return E

input_data = [a,b]

#print "grad: ", grad_E(a)

#result = minimize(form_energy, a, jac=grad_E, options={'gtol': 1e-6, 'disp': True})
#
#
##options={'xtol': 1e-8, 'disp': True})
#if result.success:
#    fitted_params = result.x
#    print(fitted_params)
#else:
#    raise ValueError(result.message)


def form_energy(inp):

    U = form_hdvv_U_1v(lattice,inp)
    
    E = v0.T.dot(U.T).dot(H).dot(U).dot(v0) / v0.T.dot(U.T).dot(U).dot(v0)

    return E
grad_E = grad(form_energy)

prev_E = 0
for it in range(100):

    if it>0:
        a = result.x
        U = form_hdvv_U_1v(lattice,a)
        v0 = U.dot(v0)

    result = minimize(form_energy, a, method='BFGS', jac=grad_E, options={'gtol': 1e-10, 'disp': True})
    #result = minimize(form_energy, a, method='L-BFGS-B', jac=grad_E, options={'gtol': 1e-2, 'disp': True})
    if abs(result.fun - prev_E) < 1e-10:
        break
    
    print " Iteration: %4i  Current Energy: %16.12f  Delta Energy: %16.12f" %(it, result.fun, result.fun-prev_E)
    
    prev_E = result.fun



print
print " Exact Energies:"
for s in range(10):
    print "   %16.12f %12.8f " %(l_full[s],s2[s,s])
