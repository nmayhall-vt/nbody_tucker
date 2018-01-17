import numpy as np
import scipy
import scipy.linalg
import scipy.io
import copy as cp
import argparse
import scipy.sparse
import scipy.sparse.linalg

from hdvv import *

class Quantum_Number:
    def __init__(self,_label):
        self.label  = cp.deepcopy(_label)
        self.value  = 0
    def set(self,_value):
        self.value = 1*_value


class Lattice_Block:
    def __init__(self):
        self.index      = 0
        self.n_sites    = 0
        self.sites      = []
        self.vecs       = np.array([])  # local basis vectors: matrix for block [P or Q]
        self.n_vecs     = 0             # number of local basis vectors 
        self.quant_nums = []            # list of Quantum_Number objects associated with Lattice_block 

        self.full_dim   = 0             # number of spin configurations on current block 

        self.j12 = np.array([])
        self.Spi = {}                   # matrix_rep of i'th S^+ in local basis
        self.Smi = {}                   # matrix_rep of i'th S^- in local basis
        self.Szi = {}                   # matrix_rep of i'th S^z in local basis
                                        
        #   in tucker basis             
        self.H     = np.array([])       # Hamiltonian on block sublattice
        self.S2    = np.array([])       # S2 on block sublattice
        self.Sz    = np.array([])       # Sz on block sublattice

        self.diis_vecs = np.array([])   # DIIS error vectors for converging tucker basis

    def init(self, _index, _sites, _quant_nums, _p_states, _j12):
        """
        _index = index of block
        _sites = list of lattice sites contained in block

        _quant_nums = list of Quantum_Number objects: NYI
        _p_states   = matrix of p vectors for lattice block
        _j12        = full system J12 matrix

        """
        for qn in _quant_nums:
            self.quant_nums.append(qn)

        self.index = _index
        self.sites = _sites
        self.n_sites = len(self.sites)
        for si in range(0,self.n_sites):
            self.full_dim *= 2
        self.extract_j12(_j12)
    
        self.vecs_p = cp.deepcopy(_p_states)
        self.n_vecs = self.vecs_p.shape[1]
       
    def update_q_space(self,n_keep):
        self.form_H_full()
        E, self.vecs = np.linalg.eigh(self.full_H)

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
    
    def form_H_full(self):
        lattice = [1]*self.n_sites  # assume spin-1/2 lattice for now 
        self.full_H, tmp, self.full_S2, self.full_Sz = form_hdvv_H(lattice, self.j12)  # rewrite this


    def form_H(self):
        lattice = [1]*self.n_sites  # assume spin-1/2 lattice for now 
        self.full_H, tmp, self.full_S2, self.full_Sz = form_hdvv_H(lattice, self.j12)  # rewrite this
        self.H = self.vecs.T.dot(self.full_H).dot(self.vecs)
        self.S2 = self.vecs.T.dot(self.full_S2).dot(self.vecs)
        self.Sz = self.vecs.T.dot(self.full_Sz).dot(self.vecs)

    def form_site_operators(self):
        """
            Form the spin operators in the basis of local block states.

            todo: include logic to only compute the portions needed - not very important
        """
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
        for i,s in enumerate(self.sites):
            i1 = np.eye(np.power(2,i))
            i2 = np.eye(np.power(2,self.n_sites-i-1))
            self.Spi[s] = np.kron(i1,np.kron(sp,i2))
            self.Smi[s] = np.kron(i1,np.kron(sm,i2))
            self.Szi[s] = np.kron(i1,np.kron(sz,i2))
            # Transform to P|Q basis
            self.Spi[s] = self.vecs.T.dot(self.Spi[s]).dot(self.vecs)
            self.Smi[s] = self.vecs.T.dot(self.Smi[s]).dot(self.vecs)
            self.Szi[s] = self.vecs.T.dot(self.Szi[s]).dot(self.vecs)


    def H_pp(self):
        # get view of PP block of H
        return self.H[0:self.np, 0:self.np]

    def vec(self):
        """ 
        Get view of space1,space2 block of H for whole block 
        """
        return self.vecs[:,0:self.np+self.nq]

    def v_ss(self,i):
        """ 
        Get view of space1,space2 block of H for whole block 
        where space1, space2 are the spaces of the bra and ket respectively
            i.e., H_ss(0,1) would return <P|H|Q>
        """
        if   i==0:
            assert(self.np>0)
            return self.vecs[:, 0:self.np]
        elif i==1:
            assert(self.nq>0)
            return self.vecs[:,self.np:self.np+self.nq]
        else:
            print " vectors for i>1 NYI" 
            exit(-1)

    def H_ss(self,i,j):
        """ 
        Get view of space1,space2 block of H for whole block 
        where space1, space2 are the spaces of the bra and ket respectively
            i.e., H_ss(0,1) would return <P|H|Q>
        """
        if   i==0 and j==0:
            assert(self.np>0)
            return self.H[0:self.np, 0:self.np]
        elif i==1 and j==0:
            assert(self.np>0)
            assert(self.nq>0)
            return self.H[self.np:self.np+self.nq , 0:self.np]
        elif i==0 and j==1:
            assert(self.np>0)
            assert(self.nq>0)
            return self.H[0:self.np , self.np:self.np+self.nq]
        elif i==1 and j==1:
            assert(self.nq>0)
            return self.H[self.np:self.np+self.np+self.nq, self.np:self.np+self.nq]

    def S2_ss(self,i,j):
        """ 
        Get view of space1,space2 block of S2 for whole block 
        where space1, space2 are the spaces of the bra and ket respectively
            i.e., H_ss(0,1) would return <P|H|Q>
        """
        if   i==0 and j==0:
            assert(self.np>0)
            return self.S2[0:self.np, 0:self.np]
        elif i==1 and j==0:
            assert(self.np>0)
            assert(self.nq>0)
            return self.S2[self.np:self.np+self.nq , 0:self.np]
        elif i==0 and j==1:
            assert(self.np>0)
            assert(self.nq>0)
            return self.S2[0:self.np , self.np:self.np+self.nq]
        elif i==1 and j==1:
            assert(self.nq>0)
            return self.S2[self.np:self.np+self.np+self.nq, self.np:self.np+self.nq]

    def S2_pp(self):
        # get view of PP block of S2 
        return self.S2[0:self.np, 0:self.np]

    def Sz_pp(self):
        # get view of PP block of Sz
        return self.Sz[0:self.np, 0:self.np]

    def Spi_pp(self,site):
        # get view of PP block of S^+ operator on site
        return self.Spi[site][0:self.np, 0:self.np]

    def Smi_pp(self,site):
        # get view of PP block of S^- operator on site
        return self.Smi[site][0:self.np, 0:self.np]

    def Szi_pp(self,site):
        # get view of PP block of S^z operator on site
        return self.Szi[site][0:self.np, 0:self.np]

    def Spi_ss(self,site,i,j):
        """ 
        Get view of space1,space2 block of S^+ operator on site
        where space1, space2 are the spaces of the bra and ket respectively
            i.e., Spi(3,0,1) would return S+ at site 3, between P and Q
            <P|S^+_i|Q>
        """
        if   i==0 and j==0:
            assert(self.np>0)
            return self.Spi[site][0:self.np, 0:self.np]
        elif i==1 and j==0:
            assert(self.np>0)
            assert(self.nq>0)
            return self.Spi[site][self.np:self.np+self.nq , 0:self.np]
        elif i==0 and j==1:
            assert(self.np>0)
            assert(self.nq>0)
            return self.Spi[site][0:self.np , self.np:self.np+self.nq]
        elif i==1 and j==1:
            assert(self.nq>0)
            return self.Spi[site][self.np:self.np+self.np+self.nq, self.np:self.np+self.nq]
    def Smi_ss(self,site,i,j):
        """ 
        Get view of space1,space2 block of S^z operator on site
        where space1, space2 are the spaces of the bra and ket respectively
            i.e., Smi(3,0,1) would return S- at site 3, between P and Q
            <P|S^-_i|Q>
        """
        if   i==0 and j==0:
            assert(self.np>0)
            return self.Smi[site][0:self.np, 0:self.np]
        elif i==1 and j==0:
            assert(self.np>0)
            assert(self.nq>0)
            return self.Smi[site][self.np:self.np+self.nq , 0:self.np]
        elif i==0 and j==1:
            assert(self.np>0)
            assert(self.nq>0)
            return self.Smi[site][0:self.np , self.np:self.np+self.nq]
        elif i==1 and j==1:
            assert(self.nq>0)
            return self.Smi[site][self.np:self.np+self.np+self.nq, self.np:self.np+self.nq]
    def Szi_ss(self,site,i,j):
        """ 
        Get view of space1,space2 block of S^z operator on site
        where space1, space2 are the spaces of the bra and ket respectively
            i.e., Szi(3,0,1) would return Sz at site 3, between P and Q
            <P|S^z_i|Q>
        """
        if   i==0 and j==0:
            assert(self.np>0)
            return self.Szi[site][0:self.np, 0:self.np]
        elif i==1 and j==0:
            assert(self.np>0)
            assert(self.nq>0)
            return self.Szi[site][self.np:self.np+self.nq , 0:self.np]
        elif i==0 and j==1:
            assert(self.np>0)
            assert(self.nq>0)
            return self.Szi[site][0:self.np , self.np:self.np+self.nq]
        elif i==1 and j==1:
            assert(self.nq>0)
            return self.Szi[site][self.np:self.np+self.np+self.nq, self.np:self.np+self.nq]

class Tucker_Block:
    def __init__(self):
        self.id = 0                 # (-1):PPP, (1):PQP, (1,3):QPQ, etc 
        self.start = 0              # starting point in super ci 
        self.stop = 0               # stopping point in super ci
        self.address = []           # tells us which space each cluster is in
        self.blocks = []
        self.n_blocks = 0
        self.full_dim = 0
        self.block_dims = [] 
        self.initialized = False
        self.vecs = []              # List of local vectors [Ai,Bj,Ck,...] defining current tucker block 
        self.j12 = np.array([])     # J12 matrix


    def init(self,_id,blocks,add,_start,_j12):
        if self.initialized == True:
            print " Tucker Block already initialized:", self
            exit(-1)
        self.address = cp.deepcopy(add)
        self.blocks = blocks 
        self.n_blocks = len(blocks)
        self.id = _id
        self.start = _start


        print("Form current Hamiltonian")
        
    

        for bi in range(0,self.n_blocks):
            self.full_dim *= self.blocks[bi].ss_dims[self.address[bi]]
            self.block_dims.append( self.blocks[bi].ss_dims[self.address[bi]])
        
        self.stop = self.start + self.full_dim
        self.initialized = True


    def __str__(self):
        out = ""
        for a in self.address:
            out += "%4s"%a
        out += " :: "
        for a in self.block_dims:
            out += "%4s"%a
        out += " :: "+ "%6i"%self.full_dim
        return out

