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
        self.index = 0
        self.n_sites = 0
        self.sites = []
        self.lattice = [] 
        self.vectors = np.array([]) # local eigenvector matrix for block [P|Q]
        self.np     = 0            # number of p-space vectors
        self.nq     = 0            # number of q-space vectors
        self.ss_dims= []            # number of vectors in each subspace [P,Q,...]
        self.n_ss   = 0             # number of subspaces, usually 2 or 3
        self.full_dim= 1             # dimension of full space in block  

        self.Spi = {}                # matrix_rep of i'th S^+ in local basis
        self.Smi = {}                # matrix_rep of i'th S^- in local basis
        self.Szi = {}                # matrix_rep of i'th S^z in local basis

        #   in tucker basis
        self.H     = np.array([])    # Hamiltonian on block sublattice
        self.S2    = np.array([])    # S2 on block sublattice
        self.Sz    = np.array([])    # Sz on block sublattice

        #   in configuration basis
        self.full_H     = np.array([])    # Hamiltonian on block sublattice
        self.full_S2    = np.array([])    # S2 on block sublattice
        self.full_Sz    = np.array([])    # Sz on block sublattice

        self.diis_vecs = np.array([]) 

    def init(self,_index,_sites,_ss):
        """
        _index = index of block
        _sites = list of lattice sites contained in block
        _ss    = list of dimensions of vectors per subspace
        """
        self.index = _index
        self.sites = _sites
        self.n_sites = len(self.sites)
        for si in range(0,self.n_sites):
            self.full_dim *= 2
        
        vec_count = 0
        for ss in _ss:
            self.ss_dims.append(ss)
            vec_count += ss
        if (self.full_dim-vec_count) < 0:
            print "Problem setting block dimensions", self
            exit(-1)
        self.ss_dims.append(self.full_dim-vec_count)
        return 

    def __str__(self):
        out = " Block %-4i:" %(self.index)
        for si in range(0,self.n_sites):
            if si < self.n_sites-1:
                out += "%5i," %(self.sites[si])
            else:
                out += "%5i" %(self.sites[si])
        out += " : " + str(self.ss_dims)
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
        self.full_H, tmp, self.full_S2, self.full_Sz = form_hdvv_H(self.lattice, self.j12)  # rewrite this
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
        self.id = 0         # (-1):PPP, (1):PQP, (1,3):QPQ, etc 
        self.start = 0      # starting point in super ci 
        self.stop = 0       # stopping point in super ci
        self.address = []
        self.blocks = []
        self.n_blocks = 0
        self.full_dim = 1
        self.block_dims = [] 
        self.initialized = False

    def init(self,_id,blocks,add,_start):
        if self.initialized == True:
            print " Tucker Block already initialized:", self
            exit(-1)
        self.address = cp.deepcopy(add)
        self.blocks = blocks 
        self.n_blocks = len(blocks)
        self.id = _id
        self.start = _start
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

##############################################################################################
#                   Functions
##############################################################################################




def build_dimer_H(tb_l, tb_r, Bi, Bj,j12):
# {{{
    bi = Bi.index
    bj = Bj.index
    
    h12 = np.zeros((tb_l.block_dims[bi]*tb_l.block_dims[bj],tb_r.block_dims[bi]*tb_r.block_dims[bj]))
    s2 = np.zeros((tb_l.block_dims[bi]*tb_l.block_dims[bj],tb_r.block_dims[bi]*tb_r.block_dims[bj]))
    sz = np.zeros((tb_l.block_dims[bi]*tb_l.block_dims[bj],tb_r.block_dims[bi]*tb_r.block_dims[bj]))

    h12.shape = (tb_l.block_dims[bi], tb_r.block_dims[bi], tb_l.block_dims[bj], tb_r.block_dims[bj])
    s2.shape = (tb_l.block_dims[bi], tb_r.block_dims[bi], tb_l.block_dims[bj], tb_r.block_dims[bj])
    sz.shape = (tb_l.block_dims[bi], tb_r.block_dims[bi], tb_l.block_dims[bj], tb_r.block_dims[bj])
    for si in Bi.sites:
        space_i_l = tb_l.address[Bi.index]
        space_i_r = tb_r.address[Bi.index]
        spi = Bi.Spi_ss(si,space_i_l,space_i_r)
        smi = Bi.Smi_ss(si,space_i_l,space_i_r)
        szi = Bi.Szi_ss(si,space_i_l,space_i_r)
        
        for sj in Bj.sites:
            space_j_l = tb_l.address[Bj.index]
            space_j_r = tb_r.address[Bj.index]
            spj = Bj.Spi_ss(sj,space_j_l,space_j_r)
            smj = Bj.Smi_ss(sj,space_j_l,space_j_r)
            szj = Bj.Szi_ss(sj,space_j_l,space_j_r)
           
            #h12  -= j12[si,sj] * np.kron(spi, smj)
            #h12  -= j12[si,sj] * np.kron(smi, spj)
            #h12  -= j12[si,sj] * np.kron(szi, szj) * 2
           
            s1s2  = np.tensordot(spi,smj, axes=0)
            s1s2 += np.tensordot(smi,spj, axes=0)
            s1s2 += 2 * np.tensordot(szi,szj, axes=0)
            #h12 -= j12[si,sj] * np.tensordot(spi,smj, axes=0)
            #h12 -= j12[si,sj] * np.tensordot(smi,spj, axes=0)
            #h12 -= j12[si,sj] * 2 * np.tensordot(szi,szj, axes=0)

            h12 -= j12[si,sj] * s1s2
            s2  += s1s2

    sort_ind = [0,2,1,3]
    h12 = h12.transpose(sort_ind)
    h12 = h12.reshape(tb_l.block_dims[bi]*tb_l.block_dims[bj],tb_r.block_dims[bi]*tb_r.block_dims[bj])
    s2 = s2.transpose(sort_ind)
    s2 = s2.reshape(tb_l.block_dims[bi]*tb_l.block_dims[bj],tb_r.block_dims[bi]*tb_r.block_dims[bj])
    return h12, s2
# }}}


def build_tucker_blocked_H(n_blocks,tucker_blocks, lattice_blocks, n_body_order, j12):
    #{{{
    dim_tot = 0
    for ti in sorted(tucker_blocks):
        tbi = tucker_blocks[ti]
        dim_tot += tbi.full_dim


    H = np.zeros((dim_tot, dim_tot))
    S2 = np.zeros((dim_tot, dim_tot))

        
    for t_l in tucker_blocks:
        for t_r in tucker_blocks:
            tb_l = tucker_blocks[t_l]
            tb_r = tucker_blocks[t_r]
            if tb_r.id >= tb_l.id:
                h,s2 = build_H(lattice_blocks, tb_l, tb_r, j12)
                H[tb_l.start:tb_l.stop, tb_r.start:tb_r.stop] = h 
                H[tb_r.start:tb_r.stop, tb_l.start:tb_l.stop] = h.T
                S2[tb_l.start:tb_l.stop, tb_r.start:tb_r.stop] = s2 
                S2[tb_r.start:tb_r.stop, tb_l.start:tb_l.stop] = s2.T
                #H[tb_l.start:tb_l.stop, tb_r.start:tb_r.stop] = build_H(lattice_blocks, tb_l, tb_r, j12)
                #H[tb_r.start:tb_r.stop, tb_l.start:tb_l.stop] = H[tb_l.start:tb_l.stop, tb_r.start:tb_r.stop].T

    return H, S2
    #}}}

def build_H(blocks,tb_l, tb_r,j12):
  # {{{
    """
    Build the Hamiltonian between two tensor blocks, tb_l and tb_r, without ever constructing a full hilbert space
    """

    n_blocks = len(blocks)
    assert(n_blocks == tb_l.n_blocks)
    assert(n_blocks == tb_r.n_blocks)
    H_dim_layout = []  # dimensions of Ham block as a tensor (d1,d2,..,d1',d2',...)
    H_dim_layout = np.append(tb_l.block_dims,tb_r.block_dims)
   
    """
    form one-block and two-block terms of H separately
        1-body

            for each block, form Hamiltonian in subspace, and combine
            with identity on other blocks

        2-body
            
            for each block-dimer, form Hamiltonian in subspace, and combine
            with identity on other blocks
    """
    # How many blocks are different between left and right?
    different = []
    for bi in range(0,n_blocks):
        if tb_l.address[bi] != tb_r.address[bi]:
            different.append(bi)
    #if len(different) > 2:
    #    print " Nothing to do, why are we here?"
    #    exit(-1)
    
    H  = np.zeros((tb_l.full_dim,tb_r.full_dim))
    S2 = np.zeros((tb_l.full_dim,tb_r.full_dim))
   
    #print " Ham block size", H.shape, H_dim_layout
    H.shape = H_dim_layout
    S2.shape = H_dim_layout
    #   Add up all the one-body contributions, making sure that the results is properly dimensioned for the 
    #   target subspace

    if len(different) == 0:

        assert(tb_l.full_dim == tb_r.full_dim)
        full_dim = tb_l.full_dim
        #<abcd|H1+H2+H3+H4|abcd>
        #
        #   <a|H1|a> Ib Ic Id
        # + Ia <b|H1|b> Ic Id + etc

        for bi in range(0,n_blocks):
            Bi = blocks[bi]
            dim_e = full_dim / tb_l.block_dims[bi] 
            
            h1 = Bi.H_ss(tb_l.address[bi],tb_r.address[bi])
            s1 = Bi.S2_ss(tb_l.address[bi],tb_r.address[bi])
            #h = np.kron(h1,np.eye(dim_e))   
            h1.shape = (tb_l.block_dims[bi],tb_r.block_dims[bi])
            s1.shape = (tb_l.block_dims[bi],tb_r.block_dims[bi])
        
            tens_inds    = []
            tens_inds.extend([bi])
            tens_inds.extend([bi+n_blocks])
            for bj in range(0,n_blocks):
                if (bi != bj):
                    tens_inds.extend([bj])
                    tens_inds.extend([bj+n_blocks])
                    assert(tb_l.block_dims[bj] == tb_r.block_dims[bj] )
                    h1 = np.tensordot(h1,np.eye(tb_l.block_dims[bj]),axes=0)
                    s1 = np.tensordot(s1,np.eye(tb_l.block_dims[bj]),axes=0)

            sort_ind = np.argsort(tens_inds)

            H  += h1.transpose(sort_ind)
            S2 += s1.transpose(sort_ind)
        
        
        #   <ab|H12|ab> Ic Id
        # + <ac|H13|ac> Ib Id
        # + Ia <bc|H23|bc> Id + etc
        
        for bi in range(0,n_blocks):
            for bj in range(bi+1,n_blocks):
                Bi = blocks[bi]
                Bj = blocks[bj]
                dim_e = full_dim / tb_l.block_dims[bi] / tb_l.block_dims[bj]

                #build full Hamiltonian on sublattice
                h2,s2 = build_dimer_H(tb_l, tb_r, Bi, Bj, j12)
                h2.shape = (tb_l.block_dims[bi],tb_l.block_dims[bj],tb_r.block_dims[bi],tb_r.block_dims[bj])
                s2.shape = (tb_l.block_dims[bi],tb_l.block_dims[bj],tb_r.block_dims[bi],tb_r.block_dims[bj])

                #h = np.kron(h12,np.eye(dim_e))   
            
                tens_inds    = []
                tens_inds.extend([bi,bj])
                tens_inds.extend([bi+n_blocks, bj+n_blocks])
                for bk in range(0,n_blocks):
                    if (bk != bi) and (bk != bj):
                        tens_inds.extend([bk])
                        tens_inds.extend([bk+n_blocks])
                        assert(tb_l.block_dims[bk] == tb_r.block_dims[bk] )
                        h2 = np.tensordot(h2,np.eye(tb_l.block_dims[bk]),axes=0)
                        s2 = np.tensordot(s2,np.eye(tb_l.block_dims[bk]),axes=0)
               
                sort_ind = np.argsort(tens_inds)
               
                H  += h2.transpose(sort_ind)
                S2 += s2.transpose(sort_ind)
    
    
    
    
    
    
    elif len(different) == 1:

        full_dim_l = tb_l.full_dim
        full_dim_r = tb_r.full_dim
        #<abcd|H1+H2+H3+H4|abcd>
        #
        #   <a|H1|a> Ib Ic Id  , for block 1 being different


        bi = different[0] 

        Bi = blocks[bi]
        dim_e_l = full_dim_l / tb_l.block_dims[bi] 
        dim_e_r = full_dim_r / tb_r.block_dims[bi] 
        h1 = Bi.H_ss(tb_l.address[bi],tb_r.address[bi])
        s1 = Bi.S2_ss(tb_l.address[bi],tb_r.address[bi])

        h1.shape = (tb_l.block_dims[bi],tb_r.block_dims[bi])
        s1.shape = (tb_l.block_dims[bi],tb_r.block_dims[bi])

        assert(dim_e_l == dim_e_r)
        dim_e = dim_e_l
       
        
        tens_inds    = []
        tens_inds.extend([bi])
        tens_inds.extend([bi+n_blocks])
        for bj in range(0,n_blocks):
            if (bi != bj):
                tens_inds.extend([bj])
                tens_inds.extend([bj+n_blocks])
                assert(tb_l.block_dims[bj] == tb_r.block_dims[bj] )
                h1 = np.tensordot(h1,np.eye(tb_l.block_dims[bj]),axes=0)
                s1 = np.tensordot(s1,np.eye(tb_l.block_dims[bj]),axes=0)

        sort_ind = np.argsort(tens_inds)

        H  += h1.transpose(sort_ind)
        S2 += s1.transpose(sort_ind)
        
        
        #   <ab|H12|Ab> Ic Id
        # + <ac|H13|Ac> Ib Id
        # + <ad|H13|Ad> Ib Id
        
        for bj in range(0,bi):
            Bj = blocks[bj]
            dim_e_l = full_dim_l / tb_l.block_dims[bi] / tb_l.block_dims[bj]
            dim_e_r = full_dim_r / tb_r.block_dims[bi] / tb_r.block_dims[bj]
         
            assert(dim_e_l == dim_e_r)
            dim_e = dim_e_l
            
            #build full Hamiltonian on sublattice
            #h12 = build_dimer_H(tb_l, tb_r, Bi, Bj, j12)
            h2,s2 = build_dimer_H(tb_l, tb_r, Bj, Bi, j12)
          
            h2.shape = (tb_l.block_dims[bj],tb_l.block_dims[bi],tb_r.block_dims[bj],tb_r.block_dims[bi])
            s2.shape = (tb_l.block_dims[bj],tb_l.block_dims[bi],tb_r.block_dims[bj],tb_r.block_dims[bi])
         
            
            #h = np.kron(h12,np.eye(dim_e))   
            
            tens_dims    = []
            tens_inds    = []
            tens_inds.extend([bj,bi])
            tens_inds.extend([bj+n_blocks, bi+n_blocks])
            for bk in range(0,n_blocks):
                if (bk != bi) and (bk != bj):
                    tens_inds.extend([bk])
                    tens_inds.extend([bk+n_blocks])
                    assert(tb_l.block_dims[bk] == tb_r.block_dims[bk] )
                    h2 = np.tensordot(h2,np.eye(tb_l.block_dims[bk]),axes=0)
                    s2 = np.tensordot(s2,np.eye(tb_l.block_dims[bk]),axes=0)
            
            sort_ind = np.argsort(tens_inds)
            H  += h2.transpose(sort_ind)
            S2 += s2.transpose(sort_ind)
        
        for bj in range(bi+1, n_blocks):
            Bj = blocks[bj]
            dim_e_l = full_dim_l / tb_l.block_dims[bi] / tb_l.block_dims[bj]
            dim_e_r = full_dim_r / tb_r.block_dims[bi] / tb_r.block_dims[bj]
         
            assert(dim_e_l == dim_e_r)
            dim_e = dim_e_l
            
            #build full Hamiltonian on sublattice
            #h12 = build_dimer_H(tb_l, tb_r, Bi, Bj, j12)
            h2,s2 = build_dimer_H(tb_l, tb_r, Bi, Bj, j12)
          
            h2.shape = (tb_l.block_dims[bi],tb_l.block_dims[bj],tb_r.block_dims[bi],tb_r.block_dims[bj])
            s2.shape = (tb_l.block_dims[bi],tb_l.block_dims[bj],tb_r.block_dims[bi],tb_r.block_dims[bj])
         
            
            #h = np.kron(h12,np.eye(dim_e))   
            
            tens_dims    = []
            tens_inds    = []
            tens_inds.extend([bi,bj])
            tens_inds.extend([bi+n_blocks, bj+n_blocks])
            for bk in range(0,n_blocks):
                if (bk != bi) and (bk != bj):
                    tens_inds.extend([bk])
                    tens_inds.extend([bk+n_blocks])
                    assert(tb_l.block_dims[bk] == tb_r.block_dims[bk] )
                    h2 = np.tensordot(h2,np.eye(tb_l.block_dims[bk]),axes=0)
                    s2 = np.tensordot(s2,np.eye(tb_l.block_dims[bk]),axes=0)
            
            sort_ind = np.argsort(tens_inds)
            H  += h2.transpose(sort_ind)
            S2 += s2.transpose(sort_ind)
    
    
    
    
    elif len(different) == 2:
    
        full_dim_l = tb_l.full_dim
        full_dim_r = tb_r.full_dim
        #<abcd|H1+H2+H3+H4|abcd> = 0


        bi = different[0] 
        bj = different[1] 

        Bi = blocks[bi]
        Bj = blocks[bj]

        dim_e_l = full_dim_l / tb_l.block_dims[bi] / tb_l.block_dims[bj] 
        dim_e_r = full_dim_r / tb_r.block_dims[bi] / tb_r.block_dims[bj] 

        assert(dim_e_l == dim_e_r)
        dim_e = dim_e_l
        
        
        #  <ac|H13|Ac> Ib Id  for 1 3 different
        
        #build full Hamiltonian on sublattice
        #h12 = build_dimer_H(tb_l, tb_r, Bi, Bj, j12)
        h2,s2 = build_dimer_H(tb_l, tb_r, Bi, Bj, j12)
       
        h2.shape = (tb_l.block_dims[bi],tb_l.block_dims[bj],tb_r.block_dims[bi],tb_r.block_dims[bj])
        s2.shape = (tb_l.block_dims[bi],tb_l.block_dims[bj],tb_r.block_dims[bi],tb_r.block_dims[bj])
        #h2 = np.kron(h12,np.eye(dim_e))   
        
        tens_dims    = []
        tens_inds    = []
        tens_inds.extend([bi,bj])
        tens_inds.extend([bi+n_blocks, bj+n_blocks])
        for bk in range(0,n_blocks):
            if (bk != bi) and (bk != bj):
                tens_inds.extend([bk])
                tens_inds.extend([bk+n_blocks])
                tens_dims.extend([tb_l.block_dims[bk]])
                tens_dims.extend([tb_r.block_dims[bk]])
                h2 = np.tensordot(h2,np.eye(tb_l.block_dims[bk]),axes=0)
                s2 = np.tensordot(s2,np.eye(tb_l.block_dims[bk]),axes=0)
        
        sort_ind = np.argsort(tens_inds)
        #H += h2.reshape(tens_dims).transpose(sort_ind)
        H += h2.transpose(sort_ind)
        S2 += s2.transpose(sort_ind)

    H = H.reshape(tb_l.full_dim,tb_r.full_dim)
    S2 = S2.reshape(tb_l.full_dim,tb_r.full_dim)
    return H,S2
# }}}



def build_Hv(blocks,tb_l, tb_r,j12,v):
  # {{{
    """
    Build the Hamiltonian vector product between two tensor blocks, tb_l and tb_r, without ever constructing a full hilbert space
    """

    n_blocks = len(blocks)
    assert(n_blocks == tb_l.n_blocks)
    assert(n_blocks == tb_r.n_blocks)
    H_dim_layout = []  # dimensions of Ham block as a tensor (d1,d2,..,d1',d2',...)
    H_dim_layout = np.append(tb_l.block_dims,tb_r.block_dims)
   
    """
    form one-block and two-block terms of H separately
        1-body

            for each block, form Hamiltonian in subspace, and combine
            with identity on other blocks

        2-body
            
            for each block-dimer, form Hamiltonian in subspace, and combine
            with identity on other blocks
    """
    # How many blocks are different between left and right?
    different = []
    for bi in range(0,n_blocks):
        if tb_l.address[bi] != tb_r.address[bi]:
            different.append(bi)
    #if len(different) > 2:
    #    print " Nothing to do, why are we here?"
    #    exit(-1)
    
    H  = np.zeros((tb_l.full_dim,tb_r.full_dim))
    S2 = np.zeros((tb_l.full_dim,tb_r.full_dim))
   
    n_sig = v.shape[1]  # number of sigma vectors 

    Hv  = np.zeros((tb_l.full_dim,n_sig))
    S2v = np.zeros((tb_l.full_dim,n_sig))


    H.shape = H_dim_layout
    S2.shape = H_dim_layout
    #   Add up all the one-body contributions, making sure that the results is properly dimensioned for the 
    #   target subspace

    if len(different) == 0:

        assert(tb_l.full_dim == tb_r.full_dim)
        full_dim = tb_l.full_dim
        #<abcd|H1+H2+H3+H4|abcd>
        #
        #   <a|H1|a> Ib Ic Id
        # + Ia <b|H1|b> Ic Id + etc
        

        for bi in range(0,n_blocks):
            Bi = blocks[bi]
            dim_e = full_dim / tb_l.block_dims[bi] 
            
            h1 = Bi.H_ss(tb_l.address[bi],tb_r.address[bi])
            s1 = Bi.S2_ss(tb_l.address[bi],tb_r.address[bi])

            #h = np.kron(h1,np.eye(dim_e))   
            h1.shape = (tb_l.block_dims[bi],tb_r.block_dims[bi])
            s1.shape = (tb_l.block_dims[bi],tb_r.block_dims[bi])

            
            #print 
            #print tb_l, " ||| ", tb_r
            # 
            # restructure incoming trial vectors as a tensor
            v_ind = cp.deepcopy(tb_r.block_dims)
            v_ind.extend([n_sig])
            v_tens = v.reshape(v_ind)
            v_tens = v_tens.swapaxes(bi,0)

            h1v = np.tensordot(h1,v_tens,axes=(0,0) )
            s1v = np.tensordot(s1,v_tens,axes=(0,0) )
            h1v = h1v.swapaxes(bi,0)
            s1v = s1v.swapaxes(bi,0)
            
            Hv  += h1v.reshape(tb_l.full_dim, n_sig)
            S2v += s1v.reshape(tb_l.full_dim, n_sig)
            """
            tens_ind = []
            for bj in range(0,n_blocks):
                if bi!=bj:
                    tens_ind.extend([bj])
            tens_ind.extend([bi])
            h1v = h1v.transpose(tens_ind)
            print h1v.shape
            """
            

        #   <ab|H12|ab> Ic Id
        # + <ac|H13|ac> Ib Id
        # + Ia <bc|H23|bc> Id + etc
        
        for bi in range(0,n_blocks):
            for bj in range(bi+1,n_blocks):
                Bi = blocks[bi]
                Bj = blocks[bj]
                dim_e = full_dim / tb_l.block_dims[bi] / tb_l.block_dims[bj]

                #build full Hamiltonian on sublattice
                h2,s2 = build_dimer_H(tb_l, tb_r, Bi, Bj, j12)
                h2.shape = (tb_l.block_dims[bi],tb_l.block_dims[bj],tb_r.block_dims[bi],tb_r.block_dims[bj])
                s2.shape = (tb_l.block_dims[bi],tb_l.block_dims[bj],tb_r.block_dims[bi],tb_r.block_dims[bj])

                # 
                # restructure incoming trial vectors as a tensor
                #   
                #   <abcdef| h24 |abcdef> = <ce|h24|ce> I0 I1 I3 I4 I5
                #
                #   v(0,1,2,3,4,5) => v(2,1,0,3,4,5) 
                #                  => v(2,4,0,3,1,5) * h(2,4,2,4) =  sig(2,4,0,3,1,5)
                #                                                 => sig(0,4,2,3,1,5)
                #                                                 => sig(0,1,2,3,4,5)
                #
                v_ind = cp.deepcopy(tb_r.block_dims)
                v_ind.extend([n_sig])
                v_tens = v.reshape(v_ind)
                
                sort_ind = [bi,bj]
                for bk in range(0,n_blocks+1):
                    if bk != bi and bk != bj:
                        sort_ind.extend([bk])
                v_tens = v_tens.transpose(sort_ind)
                
                sort_ind = np.argsort(sort_ind)

                h2v = np.tensordot(h2,v_tens,axes=([0,1],[0,1]) )
                s2v = np.tensordot(s2,v_tens,axes=([0,1],[0,1]) )
                
                h2v = h2v.transpose(sort_ind)
                s2v = s2v.transpose(sort_ind)

                Hv  += h2v.reshape(tb_l.full_dim, n_sig)
                S2v += s2v.reshape(tb_l.full_dim, n_sig)
            
    
    
    elif len(different) == 1:

        full_dim_l = tb_l.full_dim
        full_dim_r = tb_r.full_dim
        #<abcd|H1+H2+H3+H4|abcd>
        #
        #   <a|H1|a> Ib Ic Id  , for block 1 being different


        bi = different[0] 

        Bi = blocks[bi]
        dim_e_l = full_dim_l / tb_l.block_dims[bi] 
        dim_e_r = full_dim_r / tb_r.block_dims[bi] 
        h1 = Bi.H_ss(tb_l.address[bi],tb_r.address[bi])
        s1 = Bi.S2_ss(tb_l.address[bi],tb_r.address[bi])

        h1.shape = (tb_l.block_dims[bi],tb_r.block_dims[bi])
        s1.shape = (tb_l.block_dims[bi],tb_r.block_dims[bi])

        assert(dim_e_l == dim_e_r)
        dim_e = dim_e_l
                
       
        v_ind = cp.deepcopy(tb_r.block_dims)
        v_ind.extend([n_sig])
        v_tens = v.reshape(v_ind)
        
        sort_ind = [bi]
        for bk in range(0,n_blocks+1):
            if bk != bi:
                sort_ind.extend([bk])
        v_tens = v_tens.transpose(sort_ind)
        
        h1v = np.tensordot(h1,v_tens,axes=([1],[0]) )
        s1v = np.tensordot(s1,v_tens,axes=([1],[0]) )
        sort_ind = np.argsort(sort_ind)

        h1v = h1v.transpose(sort_ind)
        s1v = s1v.transpose(sort_ind)
        Hv  += h1v.reshape(tb_l.full_dim, n_sig)
        S2v += s1v.reshape(tb_l.full_dim, n_sig)
        
        #   <ab|H12|Ab> Ic Id
        # + <ac|H13|Ac> Ib Id
        # + <ad|H13|Ad> Ib Id
        
        for bj in range(0,bi):
            Bj = blocks[bj]
            dim_e_l = full_dim_l / tb_l.block_dims[bi] / tb_l.block_dims[bj]
            dim_e_r = full_dim_r / tb_r.block_dims[bi] / tb_r.block_dims[bj]
         
            assert(dim_e_l == dim_e_r)
            dim_e = dim_e_l
            
            #build full Hamiltonian on sublattice
            #h12 = build_dimer_H(tb_l, tb_r, Bi, Bj, j12)
            h2,s2 = build_dimer_H(tb_l, tb_r, Bj, Bi, j12)
          
            h2.shape = (tb_l.block_dims[bj],tb_l.block_dims[bi],tb_r.block_dims[bj],tb_r.block_dims[bi])
            s2.shape = (tb_l.block_dims[bj],tb_l.block_dims[bi],tb_r.block_dims[bj],tb_r.block_dims[bi])
         
            v_ind = cp.deepcopy(tb_r.block_dims)
            v_ind.extend([n_sig])
            v_tens = v.reshape(v_ind)
            
            sort_ind = [bj,bi]
            for bk in range(0,n_blocks+1):
                if bk != bi and bk != bj:
                    sort_ind.extend([bk])
            v_tens = v_tens.transpose(sort_ind)
            
            h2v = np.tensordot(h2,v_tens,axes=([2,3],[0,1]) )
            s2v = np.tensordot(s2,v_tens,axes=([2,3],[0,1]) )

            sort_ind = np.argsort(sort_ind)
            
            h2v = h2v.transpose(sort_ind)
            s2v = s2v.transpose(sort_ind)

            Hv += h2v.reshape(tb_l.full_dim, n_sig)
            S2v += s2v.reshape(tb_l.full_dim, n_sig)
        
        for bj in range(bi+1, n_blocks):
            Bj = blocks[bj]
            dim_e_l = full_dim_l / tb_l.block_dims[bi] / tb_l.block_dims[bj]
            dim_e_r = full_dim_r / tb_r.block_dims[bi] / tb_r.block_dims[bj]
         
            assert(dim_e_l == dim_e_r)
            dim_e = dim_e_l
            
            #build full Hamiltonian on sublattice
            #h12 = build_dimer_H(tb_l, tb_r, Bi, Bj, j12)
            h2,s2 = build_dimer_H(tb_l, tb_r, Bi, Bj, j12)
          
            h2.shape = (tb_l.block_dims[bi],tb_l.block_dims[bj],tb_r.block_dims[bi],tb_r.block_dims[bj])
            s2.shape = (tb_l.block_dims[bi],tb_l.block_dims[bj],tb_r.block_dims[bi],tb_r.block_dims[bj])
         
            v_ind = cp.deepcopy(tb_r.block_dims)
            v_ind.extend([n_sig])
            v_tens = v.reshape(v_ind)
            
            sort_ind = [bi,bj]
            for bk in range(0,n_blocks+1):
                if bk != bi and bk != bj:
                    sort_ind.extend([bk])
            v_tens = v_tens.transpose(sort_ind)
            
            h2v = np.tensordot(h2,v_tens,axes=([2,3],[0,1]) )
            s2v = np.tensordot(s2,v_tens,axes=([2,3],[0,1]) )

            sort_ind = np.argsort(sort_ind)
            
            h2v = h2v.transpose(sort_ind)
            s2v = s2v.transpose(sort_ind)

            Hv  += h2v.reshape(tb_l.full_dim, n_sig)
            S2v += s2v.reshape(tb_l.full_dim, n_sig)
    
    elif len(different) == 2:
    
        full_dim_l = tb_l.full_dim
        full_dim_r = tb_r.full_dim
        #<abcd|H1+H2+H3+H4|abcd> = 0


        bi = different[0] 
        bj = different[1] 

        Bi = blocks[bi]
        Bj = blocks[bj]

        dim_e_l = full_dim_l / tb_l.block_dims[bi] / tb_l.block_dims[bj] 
        dim_e_r = full_dim_r / tb_r.block_dims[bi] / tb_r.block_dims[bj] 

        assert(dim_e_l == dim_e_r)
        dim_e = dim_e_l
        
        
        #  <ac|H13|Ac> Ib Id  for 1 3 different
        
        #build full Hamiltonian on sublattice
        #h12 = build_dimer_H(tb_l, tb_r, Bi, Bj, j12)
        h2,s2 = build_dimer_H(tb_l, tb_r, Bi, Bj, j12)
       
        h2.shape = (tb_l.block_dims[bi],tb_l.block_dims[bj],tb_r.block_dims[bi],tb_r.block_dims[bj])
        s2.shape = (tb_l.block_dims[bi],tb_l.block_dims[bj],tb_r.block_dims[bi],tb_r.block_dims[bj])
            
        v_ind = cp.deepcopy(tb_r.block_dims)
        v_ind.extend([n_sig])
        v_tens = v.reshape(v_ind)
        
        sort_ind = [bi,bj]
        for bk in range(0,n_blocks+1):
            if bk != bi and bk != bj:
                sort_ind.extend([bk])
        v_tens = v_tens.transpose(sort_ind)
        
        h2v = np.tensordot(h2,v_tens,axes=([2,3],[0,1]) )
        s2v = np.tensordot(s2,v_tens,axes=([2,3],[0,1]) )
        
        sort_ind = np.argsort(sort_ind)
        
        h2v = h2v.transpose(sort_ind)
        s2v = s2v.transpose(sort_ind)

        Hv  += h2v.reshape(tb_l.full_dim, n_sig)
        S2v += s2v.reshape(tb_l.full_dim, n_sig)

    return Hv,S2v
# }}}

def build_tucker_blocked_sigma(n_blocks,tucker_blocks, lattice_blocks, n_body_order, j12,v):
    #{{{

    """
        s   = <abcd| H | aBcD> v_aBcD
        
            = <bd|H24|BD>v_BD I1*v_a I3v_c
            
    """
    dim_tot = 0
    for ti in sorted(tucker_blocks):
        tbi = tucker_blocks[ti]
        dim_tot += tbi.full_dim


    Hv = np.zeros((dim_tot, v.shape[1]))
    S2v = np.zeros((dim_tot, v.shape[1]))

        
    for t_l in sorted(tucker_blocks):
        for t_r in sorted(tucker_blocks):
            tb_l = tucker_blocks[t_l]
            tb_r = tucker_blocks[t_r]
            v_r = cp.deepcopy( v[tb_r.start:tb_r.stop,:])

            hv,s2v = build_Hv(lattice_blocks, tb_l, tb_r, j12,v_r)
            #h,s2 = build_H(lattice_blocks, tb_l, tb_r, j12)
            
            Hv[tb_l.start:tb_l.stop,:] += hv
            S2v[tb_l.start:tb_l.stop,:] += s2v
            #H[tb_l.start:tb_l.stop, tb_r.start:tb_r.stop] = build_H(lattice_blocks, tb_l, tb_r, j12)
            #H[tb_r.start:tb_r.stop, tb_l.start:tb_l.stop] = H[tb_l.start:tb_l.stop, tb_r.start:tb_r.stop].T

    return Hv, S2v
    #}}}



