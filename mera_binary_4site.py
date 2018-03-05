#! /usr/bin/env python
import numpy as np
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

#   Setup input arguments
parser = argparse.ArgumentParser(description='Finds eigenstates of a spin lattice',
formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#parser.add_argument('-d','--dry_run', default=False, action="store_true", help='Run but don\'t submit.', required=False)
parser.add_argument('-ju','--j_unit', type=str, default="ev", help='What units are the J values in', choices=['cm','ev'],required=False)
parser.add_argument('-l','--lattice', type=str, default="heis_lattice.m", help='File containing vector of sizes number of electrons per lattice site', required=False)
parser.add_argument('-j','--j12', type=str, default="heis_j12.m", help='File containing matrix of exchange constants', required=False)
parser.add_argument('-b','--blocks', type=str, default="heis_blocks.m", help='File containing vector of block sizes', required=False)
#parser.add_argument('-s','--save', default=False, action="store_true", help='Save the Hamiltonian and S2 matrices', required=False)
#parser.add_argument('-r','--read', default=False, action="store_true", help='Read the Hamiltonian and S2 matrices', required=False)
#parser.add_argument('-hdvv','--hamiltonian', type=str, default="heis_hamiltonian.npy", help='File containing matrix of Hamiltonian', required=False)
#parser.add_argument('-s2','--s2', type=str, default="heis_s2.npy", help='File containing matrix of s2', required=False)
#parser.add_argument('--eigvals', type=str, default="heis_eigvals.npy", help='File of Hamiltonian eigvals', required=False)
#parser.add_argument('--eigvecs', type=str, default="heis_eigvecs.npy", help='File of Hamiltonian eigvecs', required=False)
parser.add_argument('-np','--n_p_space', type=int, nargs="+", help='Number of vectors in block P space', required=False)
parser.add_argument('-nq','--n_q_space', type=int, nargs="+", help='Number of vectors in block Q space', required=False)
parser.add_argument('-nb','--n_body_order', type=int, default="0", help='n_body spaces', required=False)
parser.add_argument('-nr','--n_roots', type=int, default="10", help='Number of eigenvectors to find in compressed space', required=False)
parser.add_argument('--n_print', type=int, default="10", help='number of states to print', required=False)
parser.add_argument('--use_exact_tucker_factors', action="store_true", default=False, help='Use compression vectors from tucker decomposition of exact ground states', required=False)
parser.add_argument('-ts','--target_state', type=int, default="0", nargs='+', help='state(s) to target during (possibly state-averaged) optimization', required=False)
parser.add_argument('-mit', '--max_iter', type=int, default=10, help='Max iterations for solving for the compression vectors', required=False)
parser.add_argument('--thresh', type=int, default=8, help='Threshold for pspace iterations', required=False)
parser.add_argument('-pt','--pt_order', type=int, default=2, help='PT correction order ?', required=False)
parser.add_argument('-pt_type','--pt_type', type=str, default='mp', choices=['mp','en'], help='PT correction denominator type', required=False)
parser.add_argument('-ms','--target_ms', type=float, default=0, help='Target ms space', required=False)
parser.add_argument('-opt','--optimization', type=str, default="None", help='Optimization algorithm for Tucker factors',choices=["none", "diis"], required=False)
args = vars(parser.parse_args())
    


j12 = np.loadtxt(args['j12'])
lattice = np.loadtxt(args['lattice']).astype(int)
blocks = np.loadtxt(args['blocks']).astype(int)
n_sites = len(lattice)
n_blocks = len(blocks)

np.random.seed(2)

chi = 2
    
H, tmp, S2, Sz = form_hdvv_H(lattice,j12)
H.shape = (2,2,2,2,2,2,2,2)
w1 = scipy.linalg.orth(np.random.rand(4,4))[:,0:chi]
w1 = w1.T
w2 = w1
w3 = scipy.linalg.orth(np.random.rand(chi*chi,chi*chi))[:,0:chi]

if chi == 2:
    w1 = np.array([[0,1,-1,0],[0,1,1,0]])/np.sqrt(2)
    w1 = w1.T
    w2 = w1
    w3 = w1

w1.shape = (2,2,chi)
w2.shape = (2,2,chi)
w3.shape = (chi,chi,chi)

U = np.eye(4)
U += .01*np.random.rand(4,4)
U,s,V = np.linalg.svd(U)
U = U.dot(V)
U.shape = (2,2,2,2)
    
v = np.zeros((chi,1))
E_prev = 0

U0_errv = []
U0_diis = []
w1_diis = []
w2_diis = []
w3_diis = []
w1_errv = []
w2_errv = []
w3_errv = []

mera = "jkmn,imo,nlp,opq->ijklq" # with u, w1, w2, w3
for it in range(1000):
    m = np.einsum(mera, U, w1, w2, w3)
    H_curr = np.einsum("ijklq,ijklmnop,mnopr->qr",m,H,m)
    l_curr, v_curr = np.linalg.eigh(H_curr)
    v = v_curr[:,0]
  
    sigma = np.einsum("ijklmnop,mnopq,q->ijkl",H,m,v)
    
    E_curr = np.einsum('i,jki,lmj,nok,pqmn,lpqo->',v,w3,w1,w2,U,sigma)
    #E_curr = np.einsum("ijkl,jkmn,imo,nlp,opq,q->",sigma,U,w1,w2,w3,v)

    if abs(E_curr-E_prev)<1e-8 and it > 2 :
        pass
        break
    
    E_prev = E_curr
    for it2 in range(10):
        Gu_curr = np.einsum('i,jki,lmj,nok,lpqo->pqmn',v,w3,w1,w2,sigma)
       
        U_old = U
        Gu_curr = Gu_curr.reshape(4,4)
        u0,s0,v0 = np.linalg.svd(Gu_curr)
        U_new = -v0.T.dot(u0.T)
        
        U_new.shape = (2,2,2,2)
        U = U_new 

    w1_old = w1
    w2_old = w2
    w3_old = w3
    for it2 in range(10):
        G1_curr = np.einsum('i,jki,nok,pqmn,lpqo->lmj',v,w3,w2,U,sigma)
        G2_curr = np.einsum('i,jki,lmj,pqmn,lpqo->nok',v,w3,w1,U,sigma)
      
        G1_curr = G1_curr.reshape(4,chi)
        u1,s1,v1 = np.linalg.svd(G1_curr)
        u1 = u1[:,0:chi]
        v1 = v1[0:chi,:]
        w1 = -u1.dot(v1)
        w1.shape = (2,2,chi)
        
        G2_curr = G2_curr.reshape(4,chi)
        u2,s2,v2 = np.linalg.svd(G2_curr)
        u2 = u2[:,0:chi]
        v2 = v2[0:chi,:]
        w2 = -u2.dot(v2)
        w2.shape = (2,2,chi)

        for it3 in range(1):
            G3_curr = np.einsum('i,lmj,nok,pqmn,lpqo->jki',v,w1,w2,U,sigma)
            G3_curr = G3_curr.reshape(chi*chi,chi)
            u3,s3,v3 = np.linalg.svd(G3_curr)
            u3 = u3[:,0:chi]
            v3 = v3[0:chi,:]
            w3 = -u3.dot(v3)
            w3.shape = (chi,chi,chi)
   

    diis = 0
    if diis:
        # U error vector
        e_u = U - U_old 
        e_u.shape = (16,1)
           
        # w1 error vector
        p = cp.deepcopy(w1_old)
        d = cp.deepcopy(G1_curr)
        p.shape = (4,chi)
        d.shape = (4,chi)
        p = p.dot(p.T)
        d = d.dot(d.T)
        e_w1 = p.dot(d) - d.dot(p)
        e_w1.shape = (16,1)
 
        # w2 error vector
        p = cp.deepcopy(w2_old)
        d = cp.deepcopy(G2_curr)
        p.shape = (4,chi)
        d.shape = (4,chi)
        p = p.dot(p.T)
        d = d.dot(d.T)
        e_w2 = p.dot(d) - d.dot(p)
        e_w2.shape = (16,1)
 
        # w3 error vector
        p = cp.deepcopy(w3_old)
        d = cp.deepcopy(G3_curr)
        p.shape = (4,chi)
        d.shape = (4,chi)
        p = p.dot(p.T)
        d = d.dot(d.T)
        e_w3 = p.dot(d) - d.dot(p)
        e_w3.shape = (16,1)
 
        if it == 0:
            U0_errv = e_u
            w1_errv = e_w1
            w2_errv = e_w2
            w3_errv = e_w3
        else:
            U0_errv = np.hstack((U0_errv,e_u))
            w1_errv = np.hstack((w1_errv,e_w1))
            w2_errv = np.hstack((w1_errv,e_w2))
            w3_errv = np.hstack((w1_errv,e_w3))
 
        U0_diis.append(Gu_curr) 
        if it>0:
            errv = cp.deepcopy(U0_errv)
            n_evecs = errv.shape[1]
            d = Gu_curr
            envs = U0_diis
 
            S = U0_errv.T.dot(errv)
            B = np.ones( (n_evecs+1, n_evecs+1) )
            B[-1,-1] = 0
            B[0:-1,0:-1] = cp.deepcopy(S) 
            r = np.zeros( (n_evecs+1,1) )
            r[-1] = 1
            if n_evecs > 0: 
                x = np.linalg.pinv(B).dot(r)
                
                extrap_err_vec = np.zeros((errv.shape[0]))
                extrap_err_vec.shape = (extrap_err_vec.shape[0])
                
 
                for i in range(0,x.shape[0]-1):
                    d += x[i]*envs[i]
                    extrap_err_vec += x[i]*errv[:,i]
                
                print " DIIS Coeffs"
                for i in x:
                    print "  %12.8f" %i
            
            u0,s0,v0 = np.linalg.svd(d)
            U_new = -v0.T.dot(u0.T)
            U_new.shape = (2,2,2,2)
            #U = U_new 
        print " Norms: %8.3e %8.3e %8.3e %8.3e"%(np.linalg.norm(e_u),np.linalg.norm(e_w1),np.linalg.norm(e_w2),np.linalg.norm(e_w3)),

    U_prev = U
    w1_prev = w1
    w2_prev = w2
    w3_prev = w3

    print " %5i E_curr: %12.8f " %(it, E_curr)
    

