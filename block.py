import numpy as np
import scipy
import scipy.linalg
import scipy.io
import copy as cp
import argparse
import scipy.sparse
import scipy.sparse.linalg

from hdvv import *

class Block:
    def __init__(self):
        self.index = 0
        self.n_sites = 0
        self.sites = []
        self.lattice = [] 
        self.vectors = np.array([]) # local eigenvector matrix for block [P|Q]
        self.np      = 0            # number of p-space vectors
        self.nq      = 0            # number of q-space vectors
        
        #op_Sp = np.array([])        # matrix_rep of S^+@i in local basis
        #op_Sm = np.array([])        # matrix_rep of S^-@i in local basis
        #op_Sz = np.array([])        # matrix_rep of S^z@i in local basis

        self.Spi = {}                # matrix_rep of i'th S^+ in local basis
        self.Smi = {}                # matrix_rep of i'th S^- in local basis
        self.Szi = {}                # matrix_rep of i'th S^z in local basis

        self.H     = np.array([])    # Hamiltonian on block sublattice
        self.S2    = np.array([])    # S2 on block sublattice
        self.Sz    = np.array([])    # Sz on block sublattice

    def init(self,_index,_sites):
        self.index = _index
        self.sites = _sites
        self.n_sites = len(self.sites)
        return 
    def __str__(self):
        out = " Block %-4i:" %(self.index)
        for si in range(0,self.n_sites):
            if si < self.n_sites-1:
                out += "%5i," %(self.sites[si])
            else:
                out += "%5i" %(self.sites[si])
        return out
    def extract_j12(self,j12):
        if self.n_sites == 0:
            print " No sites yet!"
        self.j12 = j12[:,self.sites][self.sites]

    def extract_lattice(self,lattice):
        if self.n_sites == 0:
            print " No sites yet!"
        self.lattice = lattice[self.sites]
    
    def form_H(self):
        self.H, tmp, self.S2, self.Sz = form_hdvv_H(self.lattice, self.j12)  # rewrite this
        self.H = self.vecs.T.dot(self.H).dot(self.vecs)
        self.S2 = self.vecs.T.dot(self.S2).dot(self.vecs)
        self.Sz = self.vecs.T.dot(self.Sz).dot(self.vecs)

    def form_site_operators(self):
        sx = .5*np.array([[0,1.],[1.,0]])
        sy = .5*np.array([[0,(0-1j)],[(0+1j),0]])
        sz = .5*np.array([[1,0],[0,-1]])
        s2 = .75*np.array([[1,0],[0,1]])
        s1 = sx + sy + sz
        I1 = np.eye(2)
 
        sp = sx + sy*(0+1j)
        sm = sx - sy*(0+1j)
 
        sp = sp.real
        sm = sm.real
        for i in range(0,self.n_sites):
            i1 = np.eye(np.power(2,i))
            i2 = np.eye(np.power(2,self.n_sites-i-1))
            self.Spi[i] = np.kron(i1,np.kron(sp,i2))
            self.Smi[i] = np.kron(i1,np.kron(sm,i2))
            self.Szi[i] = np.kron(i1,np.kron(sz,i2))
            self.Spi[i] = self.vecs.T.dot(self.Spi[i]).dot(self.vecs)
            self.Smi[i] = self.vecs.T.dot(self.Smi[i]).dot(self.vecs)
            self.Szi[i] = self.vecs.T.dot(self.Szi[i]).dot(self.vecs)








