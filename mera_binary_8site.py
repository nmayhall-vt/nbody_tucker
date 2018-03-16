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
    

n_sites = 8
j12 = np.zeros((8,8))
for i in range(n_sites-1):
    j12[i,i+1] -= 1
    j12[i+1,i] -= 1

lattice = np.ones((n_sites))

np.random.seed(2)

chi = 2
    
H, tmp, S2, Sz = form_hdvv_H(lattice,j12)
l_full, v_full = np.linalg.eigh(H)
s2 = v_full.T.dot(S2).dot(v_full)
print " Exact Energies:"
for s in range(10):
    print "   %12.8f %12.8f " %(l_full[s],s2[s,s])

H.shape = ([2]*(2*n_sites))
S2.shape = ([2]*(2*n_sites))
Sz.shape = ([2]*(2*n_sites))

w1 = scipy.linalg.orth(np.random.rand(4,4))[:,0:chi]
w2 = w1
w3 = w1
w4 = w1
w5 = scipy.linalg.orth(np.random.rand(chi*chi,chi*chi))[:,0:chi]
w6 = scipy.linalg.orth(np.random.rand(chi*chi,chi*chi))[:,0:chi]
w7 = scipy.linalg.orth(np.random.rand(chi*chi,chi*chi))[:,0:chi]

if chi == -2:
    w1 = np.array([[0,1,-1,0],[0,1,1,0]])/np.sqrt(2)
    w1 = w1.T
    w2 = w1
    w3 = w1
    w4 = w1
    w5 = w1
    w6 = w1
    w7 = w1

if chi == 1:
    w1 = np.array([0,1,-1,0])/np.sqrt(2)
    w1 = w1.T
    w2 = w1
    w3 = w1
    w4 = w1

w1.shape = (2,2,chi)
w2.shape = (2,2,chi)
w3.shape = (2,2,chi)
w4.shape = (2,2,chi)
w5.shape = (chi,chi,chi)
w6.shape = (chi,chi,chi)
w7.shape = (chi,chi,chi)

u1 = np.eye(4)
u1 = .01*np.random.rand(4,4)
U,s,V = np.linalg.svd(u1)
u1 = U.dot(V)
u1.shape = (2,2,2,2)
 
u2 = u1
u3 = u1

u4 = np.eye(chi*chi)
u4 = .01*np.random.rand(chi*chi,chi*chi)
U,s,V = np.linalg.svd(u4)
u4 = U.dot(V)
u4.shape = (chi,chi,chi,chi)

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

for it in range(10000):
    #sigma_s = np.einsum("abcdefghijklmnop,jkqr,lmst,nouv->abcdefghiqrstuv",H,u1,u2,u3)
    #sigma_s = np.einsum("abcdefghiqrstuv,iqw,rsx,tuy,vpz->abcdefghwxyz",sigma_s,w1,w2,w3,w4)
    #sigma_s = np.einsum("abcdefghwxyz,xyjk->abcdefghwjkz",sigma_s,u3)
    #sigma_s = np.einsum("abcdefghwjkz,wjl,kzm->abcdefghlm",sigma_s,w5,w6)
    #sigma_s = np.einsum("abcdefghlm,lmn->abcdefghn",sigma_s,w7)
   
    mera_s = np.einsum("wjl,kzm,lma->wjkza",w5,w6,w7)
    mera_s = np.einsum("wjkza,xyjk->wxyza",mera_s,u4)
    mera_s = np.einsum("wxyza,iqw,rsx,tuy,vpz->iqrstuvpa",mera_s,w1,w2,w3,w4)
    mera_s = np.einsum("iqrstuvpa,jkqr,lmst,nouv->ijklmnopa",mera_s,u1,u2,u3)

    H_curr = np.einsum("abcdefghs,abcdefghijklmnop,ijklmnopt->st",mera_s,H,mera_s)
    l_curr, v_curr = np.linalg.eigh(H_curr)
    v = v_curr[:,0]
    
    E_curr = l_curr[0]
    E_curr  = np.einsum("s,abcdefghs,abcdefghijklmnop,ijklmnopt,t->",v,mera_s,H,mera_s,v)
    S2_curr = np.einsum("s,abcdefghs,abcdefghijklmnop,ijklmnopt,t->",v,mera_s,S2,mera_s,v)
    Sz_curr = np.einsum("s,abcdefghs,abcdefghijklmnop,ijklmnopt,t->",v,mera_s,Sz,mera_s,v)
    
    print " %5i E_curr: %12.8f  S2_curr: %12.8f  Sz_curr: %12.8f " %(it, E_curr, S2_curr, Sz_curr)

    if abs(E_curr-E_prev)<1e-16 and it > 2 :
        pass
        break
   
    E_prev = E_curr

    #
    sigma = np.einsum("abcdefghijklmnop,ijklmnopq,q->abcdefgh",H,mera_s,v)
    #E_curr = np.einsum("ijklmnop,jkqr,lmst,nouv,iqw,rsx,tuy,vpz,xyjk,wjl,kzm,lma,a->",sigma,u1,u2,u3,w1,w2,w3,w4,u4,w5,w6,w7,v)
    E_curr = np.einsum("abcdefgh,bcij,dekl,fgmn,aio,jkp,lmq,nhr,pqst,osu,trv,uvw,w->",sigma,u1,u2,u3,w1,w2,w3,w4,u4,w5,w6,w7,v)
    
    Gu1_curr = np.einsum("abcdefgh,dekl,fgmn,aio,jkp,lmq,nhr,pqst,osu,trv,uvw,w->bcij",sigma,u2,u3,w1,w2,w3,w4,u4,w5,w6,w7,v)
    Gu2_curr = np.einsum("abcdefgh,bcij,fgmn,aio,jkp,lmq,nhr,pqst,osu,trv,uvw,w->dekl",sigma,u1,u3,w1,w2,w3,w4,u4,w5,w6,w7,v)
    Gu3_curr = np.einsum("abcdefgh,bcij,dekl,aio,jkp,lmq,nhr,pqst,osu,trv,uvw,w->fgmn",sigma,u1,u2,w1,w2,w3,w4,u4,w5,w6,w7,v)
    
    Gu1_curr = Gu1_curr.reshape(4,4)
    u0,s0,v0 = np.linalg.svd(Gu1_curr)
    u1 = -u0.dot(v0)
    u1.shape = (2,2,2,2)
    
    Gu2_curr = Gu2_curr.reshape(4,4)
    u0,s0,v0 = np.linalg.svd(Gu2_curr)
    u2 = -u0.dot(v0)
    u2.shape = (2,2,2,2)
    
    Gu3_curr = Gu3_curr.reshape(4,4)
    u0,s0,v0 = np.linalg.svd(Gu3_curr)
    u3 = -u0.dot(v0)
    u3.shape = (2,2,2,2)
    
    Gw1_curr = np.einsum("abcdefgh,bcij,dekl,fgmn,jkp,lmq,nhr,pqst,osu,trv,uvw,w->aio",sigma,u1,u2,u3,w2,w3,w4,u4,w5,w6,w7,v)
    Gw1_curr = Gw1_curr.reshape(4,chi)
    u0,s0,v0 = np.linalg.svd(Gw1_curr)
    u0 = u0[:,0:chi]
    v0 = v0[0:chi,:]
    w1 = -u0.dot(v0)
    w1.shape = (2,2,chi)
    
    Gw2_curr = np.einsum("abcdefgh,bcij,dekl,fgmn,aio,lmq,nhr,pqst,osu,trv,uvw,w->jkp",sigma,u1,u2,u3,w1,w3,w4,u4,w5,w6,w7,v)
    Gw2_curr = Gw2_curr.reshape(4,chi)
    u0,s0,v0 = np.linalg.svd(Gw2_curr)
    u0 = u0[:,0:chi]
    v0 = v0[0:chi,:]
    w2 = -u0.dot(v0)
    w2.shape = (2,2,chi)
    
    Gw3_curr = np.einsum("abcdefgh,bcij,dekl,fgmn,aio,jkp,nhr,pqst,osu,trv,uvw,w->lmq",sigma,u1,u2,u3,w1,w2,w4,u4,w5,w6,w7,v)
    Gw3_curr = Gw3_curr.reshape(4,chi)
    u0,s0,v0 = np.linalg.svd(Gw3_curr)
    u0 = u0[:,0:chi]
    v0 = v0[0:chi,:]
    w3 = -u0.dot(v0)
    w3.shape = (2,2,chi)
            
    Gw4_curr = np.einsum("abcdefgh,bcij,dekl,fgmn,aio,jkp,lmq,pqst,osu,trv,uvw,w->nhr",sigma,u1,u2,u3,w1,w2,w3,u4,w5,w6,w7,v)
    Gw4_curr = Gw4_curr.reshape(4,chi)
    u0,s0,v0 = np.linalg.svd(Gw4_curr)
    u0 = u0[:,0:chi]
    v0 = v0[0:chi,:]
    w4 = -u0.dot(v0)
    w4.shape = (2,2,chi)
    
    
    
    Gu4_curr = np.einsum("abcdefgh,bcij,dekl,fgmn,aio,jkp,lmq,nhr,osu,trv,uvw,w->pqst",sigma,u1,u2,u3,w1,w2,w3,w4,w5,w6,w7,v)
    Gu4_curr = Gu4_curr.reshape(chi*chi,chi*chi)
    u0,s0,v0 = np.linalg.svd(Gu4_curr)
    u4 = -u0.dot(v0)
    u4.shape = (chi,chi,chi,chi)
    
    
    Gw5_curr = np.einsum("abcdefgh,bcij,dekl,fgmn,aio,jkp,lmq,nhr,pqst,trv,uvw,w->osu",sigma,u1,u2,u3,w1,w2,w3,w4,u4,w6,w7,v)
    Gw5_curr = Gw5_curr.reshape(chi*chi,chi)
    u0,s0,v0 = np.linalg.svd(Gw5_curr)
    u0 = u0[:,0:chi]
    v0 = v0[0:chi,:]
    w5 = -u0.dot(v0)
    w5.shape = (chi,chi,chi)
    
    Gw6_curr = np.einsum("abcdefgh,bcij,dekl,fgmn,aio,jkp,lmq,nhr,pqst,osu,uvw,w->trv",sigma,u1,u2,u3,w1,w2,w3,w4,u4,w5,w7,v)
    Gw6_curr = Gw6_curr.reshape(chi*chi,chi)
    u0,s0,v0 = np.linalg.svd(Gw6_curr)
    u0 = u0[:,0:chi]
    v0 = v0[0:chi,:]
    w6 = -u0.dot(v0)
    w6.shape = (chi,chi,chi)
    
    Gw7_curr = np.einsum("abcdefgh,bcij,dekl,fgmn,aio,jkp,lmq,nhr,pqst,osu,trv,w->uvw",sigma,u1,u2,u3,w1,w2,w3,w4,u4,w5,w6,v)
    Gw7_curr = Gw7_curr.reshape(chi*chi,chi)
    u0,s0,v0 = np.linalg.svd(Gw7_curr)
    u0 = u0[:,0:chi]
    v0 = v0[0:chi,:]
    w7 = -u0.dot(v0)
    w7.shape = (chi,chi,chi)
   
    continue
    

