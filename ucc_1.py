#! /usr/bin/env python

#import autograd.numpy as np
#from autograd import grad  
#import numpy as np
import sys
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
from multiprocessing import Pool





n_sites = 10 
j12 = np.zeros((n_sites,n_sites))
for i in range(n_sites-1):
    j12[i,i+1] -= 1
    j12[i+1,i] -= 1

lattice = np.ones((n_sites))
lattice_guess = np.ones((n_sites))
j12_guess = j12[0:4][0:4] 

np.random.seed(2)

    
H, tmp, S2, Sz = form_hdvv_H(lattice,j12)

l_full, v_full = np.linalg.eigh(H)
s2 = v_full.T.dot(S2).dot(v_full)

#sort_ind = np.argsort(s2.diagonal())
#l_full = l_full[sort_ind]
#v_full = v_full[:,sort_ind]

#s2 = v_full.T.dot(S2).dot(v_full)
print " Exact Energies:"
for s in range(10):
    print "   %12.8f %12.8f " %(l_full[s],s2[s,s])



a = (np.random.rand(n_sites*(n_sites-1)/2)-.5)/100
b = (np.random.rand(n_sites*(n_sites-1)/2)-.5)/100



v0 = np.random.rand(v_full.shape[0])
v0 = v_full[:,0] + v_full[:,7]

#v0 = np.zeros((v_full.shape[0]))
#for i in range(10):
#    v0 += v_full[:,i]
v0 = v0/np.linalg.norm(v0)

# Add component along the exact solution to get better starting point
#v0 += cp.deepcopy(v_full[:,0])*3
#v0 = v0/np.linalg.norm(v0)

aa = np.zeros((n_sites,n_sites))
bb = np.zeros((n_sites,n_sites))
Uold = np.zeros(H.shape)


U0 = cp.deepcopy(Uold)

curr_grad_norm = 0
curr_energy = 0

operators = []
form_hdvv_operators(lattice,operators)
niter = 0

def callback(inp):
    global niter
    print(" Iter: %5i  Energy: %16.12f Norm of gradient %12.1e"%(niter, curr_energy, curr_grad_norm))
    niter += 1
    sys.stdout.flush()
    return

def form_energy(inp):
    #U0 = form_hdvv_U_1v(lattice,inp)
    global U0,v0
    U0 = np.zeros(H.shape)
    for ui in range(len(operators)):
        U0 += inp[ui] * operators[ui]
    Uv = scipy.sparse.linalg.expm_multiply(1j*U0,v0)

    E = np.vdot(Uv,H.dot(Uv)) / np.vdot(Uv,Uv)
    global curr_energy 
    curr_energy += np.vdot(Uv,H.dot(Uv)) 
    if np.isclose(E.imag, 0):
        curr_energy = E.real
        return E.real
    else:
        curr_energy = E
        return E



def form_gradient_param(pi):
   
    step_size = 1e-6
    global U0,v0
    Ucur = U0 + operators[pi]*step_size
    Uv = scipy.sparse.linalg.expm_multiply(1j*Ucur,v0)

    ef = np.vdot(Uv,H.dot(Uv)) / np.vdot(Uv,Uv)  
    if np.isclose(ef.imag, 0):
        ef = ef.real
    
    Ucur = U0 - operators[pi]*step_size
    Uv = scipy.sparse.linalg.expm_multiply(1j*Ucur,v0)

    er = np.vdot(Uv,H.dot(Uv)) / np.vdot(Uv,Uv)  
    if np.isclose(er.imag, 0):
        er = er.real
    return np.real(ef - er) / (2*step_size)

def form_gradient(inp):
   
    """ 
    pool = Pool(1) 
    results = pool.map(form_gradient_param, range(len(inp)))
    pool.close() 
    pool.join() 
    der = np.array(results)
    """ 
    der = np.zeros((len(inp)))
    for pi in range(len(inp)):
        der[pi] = form_gradient_param(pi)
    global curr_grad_norm
    curr_grad_norm = np.linalg.norm(der)
    return der 


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


#def form_energy(inp):
#
#    U = form_hdvv_U_1v(lattice,inp)
#    
#    E = v0.T.dot(U.T).dot(H).dot(U).dot(v0) / v0.T.dot(U.T).dot(U).dot(v0)
#
#    return E
#grad_E = grad(form_energy)

prev_E = 0
for it in range(100):

    sys.stdout.flush()

    if it>0:
        U0 = np.zeros(H.shape)
        global U0,v0
        for ui in range(len(operators)):
            U0 += a[ui] * operators[ui]
        v0 = scipy.sparse.linalg.expm_multiply(1j*U0,v0)
        a = (np.random.rand(n_sites*(n_sites-1)/2)-.5)/100
        print(" Initial energy of new iterate: %16.12f" %form_energy(a))
    #result = minimize(form_energy, a, method='BFGS', options={'gtol': 1e-10, 'disp': True}, callback=callback)
    #result = minimize(form_energy, a, method='L-BFGS-B', jac=form_gradient, options={'ftol': 1e-12,'gtol': 1e-6, 'disp': True}, callback=callback)
    result = minimize(form_energy, a, method='BFGS', jac=form_gradient, options={'gtol': 1e-8, 'disp': True}, callback=callback)
    #result = minimize(form_energy, a, method='L-BFGS-B', jac=grad_E, options={'gtol': 1e-2, 'disp': True})
    
    print " Iteration: %4i  Current Energy: %16.12f  Delta Energy: %16.12f" %(it, result.fun, result.fun-prev_E)
    a = cp.deepcopy(result.x)
    print(result.x)
    if abs(result.fun - prev_E) < 1e-10:
        break
    
    
    prev_E = result.fun


print
print " Exact Energies:"
for s in range(10):
    print "   %16.12f %12.8f " %(l_full[s],s2[s,s])
