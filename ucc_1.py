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
v0 += cp.deepcopy(v_full[:,0])*3
v0 = v0/np.linalg.norm(v0)

aa = np.zeros((n_sites,n_sites))
bb = np.zeros((n_sites,n_sites))
Uold = np.zeros(H.shape)
   


def form_energy(inp):
    U = form_hdvv_U_1v(lattice,inp)
   
    E = v0.T.dot(U.T).dot(H).dot(U).dot(v0) / v0.T.dot(U.T).dot(U).dot(v0)
    Uold = cp.deepcopy(U)

    return E

input_data = [a,b]
grad_E = grad(form_energy)

#print "grad: ", grad_E(a)

#result = minimize(form_energy, a, options={'gtol': 1e-6, 'disp': True})
#result = minimize(form_energy, a, jac=grad_E, options={'gtol': 1e-6, 'disp': True})
#result = minimize(form_energy, input_data,jac=grad_E, options={'gtol': 1e-6, 'disp': True})



for it in range(10):

    if it>0:
        a = result.x
        U = form_hdvv_U_1v(lattice,a)
        v0 = U.dot(v0)

    #result = minimize(form_energy, a, jac=grad_E, method = 'Nelder-Mead', options={'gtol': 1e-6, 'disp': True})
    result = minimize(form_energy, a, jac=grad_E, method='BFGS', options={'gtol': 1e-6, 'disp': True})
    
    if result.success:
        fitted_params = result.x
        #print(fitted_params)
    else:
        raise ValueError(result.message)






print
print " Exact Energies:"
for s in range(10):
    print "   %12.8f %12.8f " %(l_full[s],s2[s,s])



