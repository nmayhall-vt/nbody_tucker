import numpy as np
import scipy
import scipy.linalg
import scipy.io
import copy as cp
import argparse
import scipy.sparse
import scipy.sparse.linalg

from hdvv import *

class Lattice_Block:
    def __init__(self):
        self.index      = 0             # Lattice_Block index
        self.n_sites    = 0             # n_sites
        self.sites      = []            # sites in this Lattice_Block
        self.j12        = np.array([])  # exchange constant matrix
        self.full_dim   = 1             # number of spin configurations on current block 
    
    def init(self, _index, _sites, _j12):
        """
        _index  = index of block
        _sites  = list of lattice sites contained in block
        _j12    = full system J12 matrix
        """
        self.index = _index
        self.sites = _sites
        self.n_sites = len(self.sites)
        for si in range(0,self.n_sites):
            self.full_dim *= 2
        if self.n_sites > 0:
            self.j12 = _j12[:,self.sites][self.sites]
    
    def __str__(self):
        out = " Lattice_Block %-4i:" %(self.index)
        for si in range(0,self.n_sites):
            if si < self.n_sites-1:
                out += "%5i," %(self.sites[si])
            else:
                out += "%5i" %(self.sites[si])
        return out


class Block_Basis:
    def __init__(self,lb,name):
        """
        This is a set of cluster states for a give Lattice_Block

        A Block_Basis object is essentially just a vector subspace of the 
        full Hilbert space associated with Lattice_Block lb. There can be 
        many Block_Basis objects assocated with a single Lattice_Block. 
        
        A Block_Basis object can simply be the P or Q or R space of a 
        Lattice_Block, or it could be (NYI) a quantum number (M_s) subblock 
        of P or Q or S.
        """
        self.lb = cp.deepcopy(lb)   # Lattice_Block object

        self.ms = None  # quantum number of this block. None means don't conserve 
        self.name = cp.deepcopy(name)
        self.vecs = np.array([])
        self.n_vecs = 0

    def __str__(self):
        out = ""
        if self.ms != None:
            out = " Block_Basis:  LB=%-4i  Ms=%-4i "%(self.lb.index,self.ms)
        else:
            out = " Block_Basis:  LB=%-4i "%(self.lb.index)
        out += " Dim=%-8i"%self.n_vecs 
        out += str(self.name)
        return out

    def set_vecs(self,v):
        self.vecs = cp.deepcopy(v)
        self.n_vecs = self.vecs.shape[1]

