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

    def append(self,other):
        if self.lb.index != other.lb.index:
            print(" Can't add Block_Basis objects from different Lattice_Blocks")
            exit(-1)
        self.name = self.name + "|" + other.name

        self.vecs = np.hstack((self.vecs,other.vecs))
        self.n_vecs = self.vecs.shape[1]

    def orthogonalize(self):
   
        if len(self.vecs.shape) > 1:
            if self.vecs.shape[1] > 0:
                self.vecs = scipy.linalg.orth(self.vecs)
        else:
            self.vecs.shape = (self.vecs.shape[0],1)


class Tucker_Block:
    """
    Essentially a collection of Block_Basis objects, which defines a particular subspace of the full systems subspace
        which is given as the direct product space of each Block_Basis.
    """
    def __init__(self,_id):
        self.id = cp.deepcopy(_id)  # (-1):PPP, (1):PQP, (1,3):QPQ, etc 
        self.start = 0              # starting point in super ci 
        self.stop = 0               # stopping point in super ci
        self.address = []           # list of block spaces. e.g.: [0,0,0,1,0,0,1,1,0] means that Blocks 3,6,7 are in their
                                    #       Q-spaces. This doesn't specify which subspace of their Q-space, just that the local
                                    #       states in 3,6,7 are orthogonal to their P spaces
        self.blocks = []            # list of references to Block_Basis objects
        
        self.n_blocks = 0
        self.full_dim = 1
        self.block_dims = [] 
   

    def add_block(self,_block):
        self.blocks.append(_block)
        self.full_dim = self.full_dim * _block.n_vecs
        self.stop = self.start + self.full_dim
        self.address.append(_block.name)

    def set_start(self,start):
        self.start = cp.deepcopy(start)
        self.stop = self.start + self.full_dim

    def set_block(self,_block):
        bi = _block.lb.index   # cluster id
        self.blocks[bi] = _block
        self.full_dim = 1
        for bj in self.blocks:
            self.full_dim = self.full_dim * bj.n_vecs
        self.address[bi] = (_block.name)
        self.stop = self.start + self.full_dim


    def __str__(self):
        out = ""
        #for a in self.address:
        #    out += "%4s"%a[0]
        #out += " :: "
        for b in self.blocks:
            out += "%4i"%b.n_vecs
        out += " :: "+ "%9i"%self.full_dim
        return out
