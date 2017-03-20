#! /usr/bin/env python
import numpy as np
import scipy
import scipy.linalg
import scipy.io
import copy as cp
import argparse
import scipy.sparse
import scipy.sparse.linalg
import sys
sys.path.insert(0, '../')

from hdvv import *


def printm(m):
    # {{{
    """ print matrix """
    for r in m:
        for ri in r:
            print "%10.1e" %ri,
        print
    # }}}

def get_guess_vectors(lattice, j12, blocks, n_p_states):
    # {{{
    print " Generate initial guess vectors"
    p_states = []
    q_states = []
    for bi,b in enumerate(blocks):
        # form block hamiltonian
        j_b = np.zeros([len(b),len(b)])
        j_b = j12[:,b][b]
        lat_b = lattice[b]
        H_b, tmp, S2_b, Sz_b = form_hdvv_H(lat_b,j_b)
    
        l_b,v_b = np.linalg.eigh(H_b)
        print " Guess eigenstates"
        for l in l_b:
            print "%12.8f" %l
        p_states.extend([v_b[:,0:n_p_states[bi]]])
        q_states.extend([v_b[:,n_p_states[bi]::]])
    return p_states, q_states
    # }}}

def form_superblock_hamiltonian(lattice, j12, blocks, block_list):
    # {{{
    print " Generate sub-block Hamiltonian"
    super_block = []
    for bi,b in enumerate(block_list):
	# form block hamiltonian
        super_block.extend(blocks[b])
    
    j_b = np.zeros([len(super_block),len(super_block)])
    j_b = j12[:,super_block][super_block]
    lat_b = lattice[super_block]
    H_b, tmp, S2_b, Sz_b = form_hdvv_H(lat_b,j_b)
    return H_b, S2_b, Sz_b
    # }}}

def form_compressed_hamiltonian_diag(vecs,Hi,Hij):
    # {{{
    dim = 1 # dimension of subspace
    dims = [] # list of mode dimensions (size of hilbert space on each fragment) 
    for vi,v in enumerate(vecs):
        dim = dim*v.shape[1]
        dims.extend([v.shape[1]])
    H = np.zeros((dim,dim))
    Htest = np.zeros((dim,dim))
    n_dims = len(dims)


    H1 = cp.deepcopy(Hi)
    H2 = cp.deepcopy(Hij)
    for vi,v in enumerate(vecs):
        H1[vi] = np.dot(v.transpose(),np.dot(Hi[vi],v))

    for vi,v in enumerate(vecs):
        for wi,w in enumerate(vecs):
            if wi>vi:
                vw = np.kron(v,w)
                H2[(vi,wi)] = np.dot(vw.transpose(),np.dot(Hij[(vi,wi)],vw))
    
    dimsdims = dims
    dimsdims = np.append(dims,dims)

    #Htest = Htest.reshape(dimsdims)
    #   Add up all the one-body contributions, making sure that the results is properly dimensioned for the 
    #   target subspace
    dim_i1=1 #   dimension of space to the left
    dim_i2=dim #   dimension of space to the right
    
    for vi,v in enumerate(vecs):
        i1 = np.eye(dim_i1)
        dim_i2 = dim_i2/v.shape[1]
        i2 = np.eye(dim_i2)
        
        #print "dim_i1  :  dim_i2", dim_i1, dim_i2, dim
        H += np.kron(i1,np.kron(H1[vi],i2))
        
        #nv = v.shape[1]
        #test = np.ones(len(dimsdims)).astype(int)
        #test[vi] = nv
        #test[vi+len(dims)] = nv
        
        #h = cp.deepcopy(H1[vi])
        #h = h.reshape(test)
        #Htest = np.einsum('ijkljk->ijklmn',Htest,h)
        
        dim_i1 = dim_i1 * v.shape[1]

     
#    print H.reshape(dimsdims)[:,:,:,:,:,:]
#    print "a"
#    print "a"
#    print "a"
#    print H.reshape(dimsdims)
#    Htest = Htest.reshape(dim,dim) 
#    exit(-1)
#    printm(Htest)
#    print
#    printm(H)


    #   Add up all the two-body contributions, making sure that the results is properly dimensioned for the 
    #   target subspace
    dim_i1=1 #   dimension of space to the left
    dim_i2=1 #   dimension of space in the middle 
    dim_i2=dim #   dimension of space to the right
  
    H = H.reshape(dimsdims)

    #print H.shape
    #print H[tuple([slice(0,3)])*len(H.shape)].shape
    ##print H[tuple([slice(0,3)])*len(H.shape)] - H
    #print np.diagonal(np.diagonal(H)).shape
    #print H[np.ones(len(H.shape)).astype(int)].shape
    
    #sliceij = []
    #for d in dimsdims:
    #    sliceij.extend([slice(0,d)])
    #print sliceij

    for vi,v in enumerate(vecs):
        for wi,w in enumerate(vecs):
            if wi>vi:

                nv = v.shape[1]
                nw = w.shape[1]
                dim_env = dim / nv / nw
                #print ": ", nv, nw, dim_env, dim
                
    
                i1 = np.eye(dim_env)
                h = np.kron(H2[(vi,wi)],i1)
                #print ": ", H2[(vi,wi)].shape, i1.shape, h.shape, H.shape
                
                #print H2[(vi,wi)].shape, " x ", i1.shape, " = ", h.shape
                
                tens_dims    = []
                tens_inds    = []
                tens_inds.extend([vi])
                tens_inds.extend([wi])
                tens_dims.extend([nv])
                tens_dims.extend([nw])
                for ti,t in enumerate(vecs):
                    if (ti != vi) and (ti != wi):
                        tens_dims.extend([t.shape[1]])
                        tens_inds.extend([ti])
                tens_dims = np.append(tens_dims, tens_dims) 
                #tens_inds = np.append(tens_inds, tens_inds) 

                sort_ind = np.argsort(tens_inds)
                
    
                #print "sort: ", sort_ind, np.array(tens_inds)[sort_ind]
                #print ":",vi,wi, tens_inds, tens_dims
                #swap indices since we have done kronecker product as H2xI
                #tens_dims[vi], tens_dims[] = tens_dims[0], tens_dims[vi] 
                #tens_dims[vi+n_dims], tens_dims[0+n_dims] = tens_dims[0+n_dims], tens_dims[vi+n_dims] 
                swap = np.append(sort_ind,sort_ind+n_dims) 

                #todo and check
                #h.shape = (tens_dims)
                #H += h.transpose(swap)
                H += h.reshape(tens_dims).transpose(swap)
                
                #print "swap ", swap
                #h = h.reshape(tens_dims)
                #h = h.transpose(swap)
                #print h.shape
                #print h.shape, dimsdims
                
  
                #sl = cp.deepcopy(sliceij)
                #sl
                #print H[sl].shape
                #H[] = Hij[(vi,wi)]
                #dim_i2 = 1
                #for i in range(vi,wi):
                #    dim_i2 = dim_i2 * vecs[i].shape[1]
                #dim_i2 = dim_i2/v.shape[1]
                #i2 = np.eye(dim_i2)
                
                #print "dim_i1  :  dim_i2", dim_i1, dim_i2, dim
                #H += np.kron(i1,np.kron(H1[vi],i2))
             
                #dim_i1 = dim_i1 * v.shape[1]

    H = H.reshape(dim,dim)
    print "     Size of Hamitonian block: ", H.shape
    #printm(H)
    return H
    # }}}

def form_compressed_hamiltonian_offdiag_1block_diff(vecs_l,vecs_r,Hi,Hij,differences):
    # {{{
    """
        Find (1-site) Hamiltonian matrix in between differently compressed spaces:
            i.e., 
                <Abcd| H(1) |ab'c'd'> = <A|Ha|a> * del(bb') * del(cc')
            or like,
                <aBcd| H(1) |abcd> = <B|Hb|b> * del(aa') * del(cc')
        
        Notice that the full Hamiltonian will also have the two-body part:
            <Abcd| H(2) |abcd> = <Ab|Hab|ab'> del(cc') + <Ac|Hac|ac'> del(bb')

       
        differences = [blocks which are not diagonal]
            i.e., for <Abcd|H(1)|abcd>, differences = [1]
                  for <Abcd|H(1)|aBcd>, differences = [1,2], and all values are zero for H(1)
            
        vecs_l  [vecs_A, vecs_B, vecs_C, ...]
        vecs_r  [vecs_A, vecs_B, vecs_C, ...]
    """
    dim_l = 1 # dimension of left subspace
    dim_r = 1 # dimension of right subspace
    dim_same = 1 # dimension of space spanned by all those fragments is the same space (i.e., not the blocks inside differences)
    dims_l = [] # list of mode dimensions (size of hilbert space on each fragment) 
    dims_r = [] # list of mode dimensions (size of hilbert space on each fragment) 
  
    
    block_curr = differences[0] # current block which is offdiagonal

    #   vecs_l correspond to the bra states
    #   vecs_r correspond to the ket states


    assert(len(vecs_l) == len(vecs_r))
    

    for bi,b in enumerate(vecs_l):
        dim_l = dim_l*b.shape[1]
        dims_l.extend([b.shape[1]])
        if bi !=block_curr:
            dim_same *= b.shape[1]

    dim_same_check = 1
    for bi,b in enumerate(vecs_r):
        dim_r = dim_r*b.shape[1]
        dims_r.extend([b.shape[1]])
        if bi !=block_curr:
            dim_same_check *= b.shape[1]

    assert(dim_same == dim_same_check)

    H = np.zeros((dim_l,dim_r))
    print "     Size of Hamitonian block: ", H.shape

    assert(len(dims_l) == len(dims_r))
    n_dims = len(dims_l)


    H1 = cp.deepcopy(Hi)
    H2 = cp.deepcopy(Hij)
    
    ## Rotate the single block Hamiltonians into the requested single site basis 
    #for vi in range(0,n_dims):
    #    l = vecs_l[vi]
    #    r = vecs_r[vi]
    #    H1[vi] = l.T.dot(Hi[vi]).dot(r)

    # Rotate the double block Hamiltonians into the appropriate single site basis 
#    for vi in range(0,n_dims):
#        for wi in range(0,n_dims):
#            if wi>vi:
#                vw_l = np.kron(vecs_l[vi],vecs_l[wi])
#                vw_r = np.kron(vecs_r[vi],vecs_r[wi])
#                H2[(vi,wi)] = vw_l.T.dot(Hij[(vi,wi)]).dot(vw_r)
   

    dimsdims = np.append(dims_l,dims_r) # this is the tensor layout for the many-body Hamiltonian in the current subspace
    
    vecs = vecs_l
    dims = dims_l
    dim = dim_l
    
    #   Add up all the one-body contributions, making sure that the results is properly dimensioned for the 
    #   target subspace
    dim_i1=1 # dimension of space for fragments to the left of the current 'different' fragment
    dim_i2=1 # dimension of space for fragments to the right of the current 'different' fragment

   
    # <abCdef|H1|abcdef> = eye(a) x eye(b) x <C|H1|c> x eye(d) x eye(e) x eye(f)
    for vi in range(n_dims):
        if vi<block_curr:
            dim_i1 *= dims_l[vi]
            assert(dims_l[vi]==dims_r[vi])
        elif vi>block_curr:
            dim_i2 *= dims_l[vi]
            assert(dims_l[vi]==dims_r[vi])
   
    # Rotate the current single block Hamiltonian into the requested single site basis 
    l = vecs_l[block_curr]
    r = vecs_r[block_curr]
    
    h1_block  = l.T.dot(Hi[block_curr]).dot(r)

    i1 = np.eye(dim_i1)
    i2 = np.eye(dim_i2)
    H += np.kron(i1,np.kron(h1_block,i2))

    # <abCdef|H2(0,2)|abcdef> = <aC|H2|ac> x eye(b) x eye(d) x eye(e) x eye(f) = <aCbdef|H2|acbdef>
    #
    #   then transpose:
    #       flip Cb and cb
    H.shape = (dims_l+dims_r)

    for bi in range(0,block_curr):
        #print "block_curr, bi", block_curr, bi
        vw_l = np.kron(vecs_l[bi],vecs_l[block_curr])
        vw_r = np.kron(vecs_r[bi],vecs_r[block_curr])
        h2 = vw_l.T.dot(Hij[(bi,block_curr)]).dot(vw_r) # i.e.  get reference to <aC|Hij[0,2]|a'c>, where block_curr = 2 and bi = 0

        dim_i = dim_same / dims[bi]
        #i1 = np.eye(dim_i)

        #tmp_h2 = np.kron( h2, i1)
        h2.shape = (dims_l[bi],dims_l[block_curr], dims_r[bi],dims_r[block_curr])
        
        tens_inds = []
        tens_inds.extend([bi,block_curr])
        tens_inds.extend([bi+n_dims,block_curr+n_dims])
        for bbi in range(0,n_dims):
            if bbi != bi and bbi != block_curr:
                tens_inds.extend([bbi])
                tens_inds.extend([bbi+n_dims])
                #print "h2.shape", h2.shape,
                h2 = np.tensordot(h2,np.eye(dims[bbi]),axes=0)
   
        #print "h2.shape", h2.shape,
        sort_ind = np.argsort(tens_inds)
        H += h2.transpose(sort_ind)
        #print "tens_inds", tens_inds
        #print "sort_ind", sort_ind 
        #print "h2", h2.transpose(sort_ind).shape
        #H += h2
    
    for bi in range(block_curr, n_dims):
        if bi == block_curr:
            continue
        #print "block_curr, bi", block_curr, bi
        vw_l = np.kron(vecs_l[block_curr],vecs_l[bi])
        vw_r = np.kron(vecs_r[block_curr],vecs_r[bi])
        h2 = vw_l.T.dot(Hij[(block_curr,bi)]).dot(vw_r) # i.e.  get reference to <aC|Hij[0,2]|a'c>, where block_curr = 2 and bi = 0

        h2.shape = (dims_l[block_curr],dims_l[bi], dims_r[block_curr],dims_r[bi])
        
        tens_inds = []
        tens_inds.extend([block_curr,bi])
        tens_inds.extend([block_curr+n_dims,bi+n_dims])
        for bbi in range(0,n_dims):
            if bbi != bi and bbi != block_curr:
                tens_inds.extend([bbi])
                tens_inds.extend([bbi+n_dims])
                #print "h2.shape", h2.shape,
                h2 = np.tensordot(h2,np.eye(dims[bbi]),axes=0)
   
        #print "h2.shape", h2.shape,
        sort_ind = np.argsort(tens_inds)
        #print "h2", h2.transpose(sort_ind).shape
        #print "tens_inds", tens_inds
        #print "sort_ind", sort_ind 
        H += h2.transpose(sort_ind)
        #print "h2.shape", h2.shape
    
    H.shape = (dim_l,dim_r)

    
    return H
    # }}}

def form_compressed_hamiltonian_offdiag_2block_diff(vecs_l,vecs_r,Hi,Hij,differences):
    # {{{
    """
        Find Hamiltonian matrix in between differently compressed spaces:
            i.e., 
                <Abcd| H(0,2) |a'b'C'd'> = <Ac|Ha|a'C'> * del(bb')
        
       
        differences = [blocks which are not diagonal]
            i.e., for <Abcd|H(1)|abCd>, differences = [0,2]
            
        vecs_l  [vecs_A, vecs_B, vecs_C, ...]
        vecs_r  [vecs_A, vecs_B, vecs_C, ...]
    """
    dim_l = 1 # dimension of left subspace
    dim_r = 1 # dimension of right subspace
    dim_same = 1 # dimension of space spanned by all those fragments is the same space (i.e., not the blocks inside differences)
    dims_l = [] # list of mode dimensions (size of hilbert space on each fragment) 
    dims_r = [] # list of mode dimensions (size of hilbert space on each fragment) 
  
    assert( len(differences) == 2) # make sure we are not trying to get H(1) between states with multiple fragments orthogonal 
    
    block_curr1 = differences[0] # current block which is offdiagonal
    block_curr2 = differences[1] # current block which is offdiagonal

    #   vecs_l correspond to the bra states
    #   vecs_r correspond to the ket states


    assert(len(vecs_l) == len(vecs_r))
    

    for bi,b in enumerate(vecs_l):
        dim_l = dim_l*b.shape[1]
        dims_l.extend([b.shape[1]])
        if bi !=block_curr1 and bi != block_curr2:
            dim_same *= b.shape[1]

    dim_same_check = 1
    for bi,b in enumerate(vecs_r):
        dim_r = dim_r*b.shape[1]
        dims_r.extend([b.shape[1]])
        if bi !=block_curr1 and bi != block_curr2:
            dim_same_check *= b.shape[1]

    assert(dim_same == dim_same_check)

    H = np.zeros((dim_l,dim_r))
    print "     Size of Hamitonian block: ", H.shape

    assert(len(dims_l) == len(dims_r))
    n_dims = len(dims_l)



    dimsdims = np.append(dims_l,dims_r) # this is the tensor layout for the many-body Hamiltonian in the current subspace
    
    #   Add up all the one-body contributions, making sure that the results is properly dimensioned for the 
    #   target subspace
    dim_i1=1 # dimension of space for fragments to the left of the current 'different' fragment
    dim_i2=1 # dimension of space for fragments to the right of the current 'different' fragment

   

    # <Abcdef|H2(0,2)|abCdef> = <Ac|H2|aC> x eye(b) x eye(d) x eye(e) x eye(f) = <aCbdef|H2|acbdef>
    #
    #   then transpose:
    #       flip Cb and cb
    H.shape = (dims_l+dims_r)

    assert(block_curr1 < block_curr2)

    #print " block_curr1, block_curr2", block_curr1, block_curr2
    vw_l = np.kron(vecs_l[block_curr1],vecs_l[block_curr2])
    vw_r = np.kron(vecs_r[block_curr1],vecs_r[block_curr2])
    h2 = vw_l.T.dot(Hij[(block_curr1,block_curr2)]).dot(vw_r) # i.e.  get reference to <aC|Hij[0,2]|a'c>, where block_curr = 2 and bi = 0

    h2.shape = (dims_l[block_curr1],dims_l[block_curr2], dims_r[block_curr1],dims_r[block_curr2])
    
    tens_inds = []
    tens_inds.extend([block_curr1,block_curr2])
    tens_inds.extend([block_curr1+n_dims,block_curr2+n_dims])
    for bbi in range(0,n_dims):
        if bbi != block_curr1 and bbi != block_curr2:
            tens_inds.extend([bbi])
            tens_inds.extend([bbi+n_dims])
            #print "h2.shape", h2.shape,

            assert(dims_l[bbi] == dims_r[bbi])
            dims = dims_l
            h2 = np.tensordot(h2,np.eye(dims[bbi]),axes=0)
   
    #print "h2.shape", h2.shape,
    sort_ind = np.argsort(tens_inds)
    H += h2.transpose(sort_ind)
    #print "tens_inds", tens_inds
    #print "sort_ind", sort_ind 
    #print "h2", h2.transpose(sort_ind).shape
    #H += h2
    
    H.shape = (dim_l,dim_r)

    
    return H
    # }}}

def assemble_blocked_matrix(H_sectors,n_blocks,n_body_order):
    # {{{
    Htest = H_sectors[0,0]
    if n_body_order == 1:
        # Singles
        for bi in range(n_blocks+1):
            row_i = np.array([])
            
            for bj in range(n_blocks+1):
                if bj == 0:
                    row_i = H_sectors[bi,bj]
                else:
                    row_i = np.hstack((row_i,H_sectors[bi,bj]))
            
            if bi == 0:
                Htest = row_i
            else:
                Htest = np.vstack((Htest,row_i))
    
    
    if n_body_order == 2:
        # 0,0 
        row_0 = H_sectors[0,0]
        # 0,S
        for bi in range(1,n_blocks+1):
            row_0 = np.hstack( ( row_0, H_sectors[0,bi] ) )
        # 0,D
        for bi in range(1,n_blocks+1):
            for bj in range(bi+1, n_blocks+1):
                bij = (bi,bj)
                print row_0.shape, H_sectors[0,bij].shape
                
                row_0 = np.hstack( ( row_0, H_sectors[0,(bi,bj)] ) )
    
        Htest = row_0
    
        
        # Singles
        for bi in range(1,n_blocks+1):
            # bi,0
            row_i = H_sectors[bi,0] 
            # bi,bj 
            for bj in range(1,n_blocks+1):
                row_i = np.hstack((row_i,H_sectors[bi,bj]))
            # bi,bjk 
            for bj in range(1,n_blocks+1):
                for bk in range(bj+1,n_blocks+1):
                    row_i = np.hstack((row_i,H_sectors[bi,(bj,bk)]))
            
            Htest = np.vstack((Htest,row_i))
    
    
        #Doubles 
        for bi in range(1,n_blocks+1):
            for bj in range(bi+1,n_blocks+1):
                # bij,0
                row_ij = H_sectors[(bi,bj),0]                                       #<ij|H|0>
                # bij,bk
                for bk in range(1,n_blocks+1):
                    row_ij = np.hstack( (row_ij, H_sectors[(bi,bj),bk]) )           #<ij|H|k>
                # bij,bkl
                for bk in range(1,n_blocks+1):
                    for bl in range(bk+1,n_blocks+1):
                        row_ij = np.hstack( (row_ij, H_sectors[(bi,bj),(bk,bl)]) )  #<ij|H|kl>
                
                Htest = np.vstack((Htest,row_ij))
    return Htest
    # }}}




"""
Test forming HDVV Hamiltonian and projecting onto "many-body tucker basis"
"""
#   Setup input arguments
parser = argparse.ArgumentParser(description='Finds eigenstates of a spin lattice',
formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#parser.add_argument('-d','--dry_run', default=False, action="store_true", help='Run but don\'t submit.', required=False)
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
parser.add_argument('-nb','--n_body_order', type=int, default="0", help='n_body spaces', required=False)
parser.add_argument('--n_print', type=int, default="10", help='number of states to print', required=False)
parser.add_argument('--use_exact_tucker_factors', action="store_true", default=False, help='Use compression vectors from tucker decomposition of exact ground states', required=False)
parser.add_argument('-ts','--target_state', type=int, default="0", nargs='+', help='state(s) to target during (possibly state-averaged) optimization', required=False)
parser.add_argument('--max_iter', type=int, default=10, help='Max iterations for solving for the compression vectors', required=False)
parser.add_argument('--thresh', type=int, default=8, help='Threshold for pspace iterations', required=False)
args = vars(parser.parse_args())
#
#   Let minute specification of walltime override hour specification

j12 = np.loadtxt(args['j12'])
lattice = np.loadtxt(args['lattice']).astype(int)
blocks = np.loadtxt(args['blocks']).astype(int)
n_sites = len(lattice)
n_blocks = len(blocks)

n_p_states = args['n_p_space'] 


if args['n_p_space'] == None:
    n_p_states = []
    for bi in range(n_blocks):
        n_p_states.extend([1])
    args['n_p_space'] = n_p_states

assert(len(args['n_p_space']) == n_blocks)

np.random.seed(2)

print " j12:\n", j12
print " lattice:\n", lattice 
print " blocks:\n", blocks
print " n_blocks:\n", n_blocks

H_tot = np.array([])
S2_tot = np.array([])
H_dict = {}

#get Hamiltonian and eigenstates 
#if args['read']:
#    print "Reading Hamiltonian and S2 from disk"
#    H_tot = np.load(args['hamiltonian'])
#    S2_tot = np.load(args['s2'])
#    v = np.load(args['eigvecs'])
#    l = np.load(args['eigvals'])
#else:
#    print "Building Hamiltonian"
#    H_tot, H_dict, S2_tot, Sz_tot = form_hdvv_H(lattice,j12)


    #print " Diagonalize Hamiltonian (%ix%i):\n" %(H_tot.shape[0],H_tot.shape[0]), H_tot.shape
    #l,v = np.linalg.eigh(H_tot)
    #l,v = scipy.sparse.linalg.eigsh(H_tot, k=min(100,H_tot.shape[0]))



#if args['save']==True:
#    np.save("heis_hamiltonian",H_tot)
#    np.save("heis_s2",S2_tot)
#


#print v.shape
#print S2_tot.shape
au2ev = 27.21165;
au2cm = 219474.63;

convert = au2ev/au2cm;		# convert from wavenumbers to eV
convert = 1;			# 1 for wavenumbers
#S2_eig = np.dot(v.transpose(),np.dot(S2_tot,v))
#print " %5s    %12s  %12s  %12s" %("State","Energy","Relative","<S2>")
#for si,i in enumerate(l):
#    print " %5i =  %12.8f  %12.8f  %12.8f" %(si,i*convert,(i-l[0])*convert,S2_eig[si,si])
#    if si>10:
#        break

#v0 = v[:,0]

#if args['save']==True:
#    np.save("heis_eigvecs",v)
#    np.save("heis_eigvals",l)
#




# reshape eigenvector into tensor
dims_tot = []
dim_tot = 1
for bi,b in enumerate(blocks):
    print b
    block_dim = np.power(2,b.shape[0])
    dims_tot.extend([block_dim])
    dim_tot *= block_dim

#v0 = np.reshape(v0,dims_tot)


# Get initial compression vectors 
p_states, q_states = get_guess_vectors(lattice, j12, blocks, n_p_states)

if args['use_exact_tucker_factors']:
    p_states = []
    q_states = []
    Acore, Atfac = tucker_decompose(v0,0,0)
    for bi,b in enumerate(Atfac):
        #p_states.extend([scipy.linalg.orth(np.random.rand(b.shape[0],n_p_states))])
        p_states.extend([b[:,0:n_p_states[bi]]])
        q_states.extend([b[:,n_p_states[bi]::]])

if 0:
    # do random guess
    p_states = []
    q_states = []
    for bi,b in enumerate(blocks):
        block_dim = np.power(2,b.shape[0])
        r = scipy.linalg.orth(np.random.rand(block_dim,block_dim))
        p_states.extend([r[:,0:n_p_states[bi]]])
        q_states.extend([r[:,n_p_states[bi]::]])

dims_0 = n_p_states

#
# |Ia,Ib,Ic> P(Ia,a) P(Ib,b) P(Ic,c) = |abc>    : |PPP>
#
# |Ia,Ib,Ic> Q(Ia,A) P(Ib,b) P(Ic,c) = |Abc>    : |QPP>
# |Ia,Ib,Ic> P(Ia,a) Q(Ib,B) P(Ic,c) = |aBc>    : |PQP>
#
# |Ia,Ib,Ic> Q(Ia,A) Q(Ib,B) P(Ic,c) = |ABc>    : |QQP>
#
#<abc|Ha+Hb+Hc+Hab+Hac+Hbc|abc>
#
#<a|Ha|a><bc|bc> = <a|Ha|a>
#<ab|Hab|ab><c|c> = <ab|Hab|ab>
#<Abc|Hab|Abc> = <Ab|Hab|Ab>


Hi = {}
Hij = {}
S2i = {}
S2ij = {}
Szi = {}
Szij = {}
#1 body operators
for bi,b in enumerate(blocks):
    Hi[bi], S2i[bi], Szi[bi] = form_superblock_hamiltonian(lattice, j12, blocks, [bi])

#2 body operators
for bi,b in enumerate(blocks):
    for bj,bb in enumerate(blocks):
        if bj>bi:
            hi = Hi[bi]
            hj = Hi[bj]
            s2i = S2i[bi]
            s2j = S2i[bj]
            
            Hij[(bi,bj)], S2ij[(bi,bj)], Szij[(bi,bj)] = form_superblock_hamiltonian(lattice, j12, blocks, [bi,bj])
            Hij[(bi,bj)] -= np.kron(hi,np.eye(hj.shape[0])) 
            Hij[(bi,bj)] -= np.kron(np.eye(hi.shape[0]),hj) 
            S2ij[(bi,bj)] -= np.kron(s2i,np.eye(s2j.shape[0])) 
            S2ij[(bi,bj)] -= np.kron(np.eye(s2i.shape[0]),s2j) 





# loop over compression vector iterations
energy_per_iter = []
maxiter = args['max_iter'] 
for it in range(0,maxiter):

    print " Tucker optimization: Iteration %4i" %it

    vecs0 = []
    # get vecs for PPP class
    for bi,b in enumerate(blocks):
        vecs0.extend([p_states[bi]])
    
    vecsQ = []
    for bi in range(n_blocks):
        v = cp.deepcopy(vecs0)
        v[bi] = q_states[bi]
        vecsQ.extend([v])
    
    vecsQQ = {}
    for bi in range(n_blocks):
        for bj in range(bi+1,n_blocks):
            v = cp.deepcopy(vecs0)
            v[bi] = q_states[bi]
            v[bj] = q_states[bj]
            vecsQQ[bi,bj] = v
    
    H0_0 = form_compressed_hamiltonian_diag(vecs0,Hi,Hij)   # <PPP|H|PPP>
    S20_0 = form_compressed_hamiltonian_diag(vecs0,S2i,S2ij)# <PPP|S^2|PPP>
    #
    
    H_sectors = {}
    H_sectors[0,0] = H0_0
    
    S2_sectors = {}
    S2_sectors[0,0] = S20_0
    
    n_body_order = args['n_body_order'] 
    
    if n_body_order >= 1:
        for bi in range(n_blocks):
            H_sectors[bi+1,bi+1]    = form_compressed_hamiltonian_diag(vecsQ[bi],Hi,Hij) # <QPP|H|QPP>
            
            H_sectors[0,bi+1]       = form_compressed_hamiltonian_offdiag_1block_diff(vecs0,vecsQ[bi],Hi,Hij,[bi]) # <PPP|H|QPP>
            H_sectors[bi+1,0]       = H_sectors[0,bi+1].T
            
            S2_sectors[bi+1,bi+1]    = form_compressed_hamiltonian_diag(vecsQ[bi],S2i,S2ij) # <QPP|H|QPP>
            S2_sectors[0,bi+1]       = form_compressed_hamiltonian_offdiag_1block_diff(vecs0,vecsQ[bi],S2i,S2ij,[bi]) # <PPP|H|QPP>
            S2_sectors[bi+1,0]       = S2_sectors[0,bi+1].T
            for bj in range(bi+1,n_blocks):
                H_sectors[bi+1,bj+1] = form_compressed_hamiltonian_offdiag_2block_diff(vecsQ[bi],vecsQ[bj],Hi,Hij,[bi,bj]) # <QPP|H|PQP>
                H_sectors[bj+1,bi+1] = H_sectors[bi+1,bj+1].T
    
                S2_sectors[bi+1,bj+1] = form_compressed_hamiltonian_offdiag_2block_diff(vecsQ[bi],vecsQ[bj],S2i,S2ij,[bi,bj]) # <QPP|H|PQP>
                S2_sectors[bj+1,bi+1] = S2_sectors[bi+1,bj+1].T
    
    
    if n_body_order >= 2:
        for bi in range(n_blocks):
            for bj in range(bi+1,n_blocks):
                bij = (bi+1,bj+1)
                print 
                print " Form Hamiltonian for <%s|H|%s>" %(bij, bij)
                H_sectors[bij,bij]  = form_compressed_hamiltonian_diag(vecsQQ[bi,bj],Hi,Hij) # <QPQ|H|QPQ>
                
                print 
                print " Form Hamiltonian for <%s|H|%s>" %(0, bij)
                H_sectors[0,bij]    = form_compressed_hamiltonian_offdiag_2block_diff(vecs0,vecsQQ[bi,bj],Hi,Hij,[bi,bj]) # <PPP|H|QQP>
                H_sectors[bij,0]    = H_sectors[0,bij].T
                
                S2_sectors[bij,bij]  = form_compressed_hamiltonian_diag(vecsQQ[bi,bj],S2i,S2ij) # <QPQ|H|QPQ>
                S2_sectors[0,bij]    = form_compressed_hamiltonian_offdiag_2block_diff(vecs0,vecsQQ[bi,bj],S2i,S2ij,[bi,bj]) # <PPP|H|QQP>
                S2_sectors[bij,0]    = S2_sectors[0,bij].T
                
        for bi in range(n_blocks):
            for bj in range(bi+1,n_blocks):
                bij = (bi+1,bj+1)
                for bk in range(n_blocks):
                    if bk == bi:
                        print 
                        print " Form Hamiltonian for <%s|H|%s>" %(bij, bk+1)
                        H_sectors[bk+1,bij]     = form_compressed_hamiltonian_offdiag_1block_diff(vecsQ[bk],vecsQQ[bi,bj],Hi,Hij,[bj]) # <PPQ|H|PQQ>
                        H_sectors[bij,bk+1]     = H_sectors[bk+1,bij].T
                        
                        S2_sectors[bk+1,bij]    = form_compressed_hamiltonian_offdiag_1block_diff(vecsQ[bk],vecsQQ[bi,bj],S2i,S2ij,[bj]) # <PPQ|H|PQQ>
                        S2_sectors[bij,bk+1]    = S2_sectors[bk+1,bij].T
                    elif bk == bj:
                        print 
                        print " Form Hamiltonian for <%s|H|%s>" %(bij, bk+1)
                        H_sectors[bk+1,bij]     = form_compressed_hamiltonian_offdiag_1block_diff(vecsQ[bk],vecsQQ[bi,bj],Hi,Hij,[bi]) # <PQP|H|PQQ>
                        H_sectors[bij,bk+1]     = H_sectors[bk+1,bij].T
                        
                        S2_sectors[bk+1,bij]    = form_compressed_hamiltonian_offdiag_1block_diff(vecsQ[bk],vecsQQ[bi,bj],S2i,S2ij,[bi]) # <PQP|H|PQQ>
                        S2_sectors[bij,bk+1]    = S2_sectors[bk+1,bij].T
                    else:
                        H_sectors[bk+1,bij]     = np.zeros( (H_sectors[bk+1,bk+1].shape[1] , H_sectors[bij,bij].shape[1] ) ) # <PQP|H|QPQ>
                        H_sectors[bij,bk+1]     = H_sectors[bk+1,bij].T
                
                        S2_sectors[bk+1,bij]    = np.zeros( (H_sectors[bk+1,bk+1].shape[1] , H_sectors[bij,bij].shape[1] ) ) # <PQP|H|QPQ>
                        S2_sectors[bij,bk+1]    = S2_sectors[bk+1,bij].T
                
                    for bl in range(bk+1,n_blocks):
                        bkl = (bk+1,bl+1)
    
                        #only compute upper triangular blocks
                        if bk < bi:
                            continue
                        if bk == bi and bl <= bj:
                            continue
                        
                        diff = {}
                        diff[bi] = 1
                        diff[bj] = 1
                        diff[bk] = 1
                        diff[bl] = 1
                        for bbi in (bi,bj):
                            for bbj in (bk,bl):
                                if bbi == bbj:
                                    diff[bbi] = 0
                        diff2 = []
                        for bbi in diff.keys():
                            if diff[bbi] == 1:
                                diff2.extend([bbi])
                       
                        if len(diff2) == 2:
                            print 
                            print " Form Hamiltonian for <%s|H|%s>" %(bij, bkl)
                            H_sectors[bij,bkl]  = form_compressed_hamiltonian_offdiag_2block_diff(vecsQQ[bi,bj],vecsQQ[bk,bl],Hi,Hij,diff2) # <QPQ|H|QQP>
                            H_sectors[bkl,bij]  = H_sectors[bij,bkl].T
                            
                            S2_sectors[bij,bkl] = form_compressed_hamiltonian_offdiag_2block_diff(vecsQQ[bi,bj],vecsQQ[bk,bl],S2i,S2ij,diff2) # <QPQ|H|QQP>
                            S2_sectors[bkl,bij] = S2_sectors[bij,bkl].T
                        if len(diff2) > 2:
                            H_sectors[bij,bkl]  = np.zeros( (H_sectors[0,bij].shape[1] , H_sectors[0,bkl].shape[1] ) )
                            H_sectors[bkl,bij]  = H_sectors[bij,bkl].T
                            
                            S2_sectors[bij,bkl] = np.zeros( (H_sectors[0,bij].shape[1] , H_sectors[0,bkl].shape[1] ) )
                            S2_sectors[bkl,bij] = S2_sectors[bij,bkl].T
    
    
    Htest = assemble_blocked_matrix(H_sectors, n_blocks, n_body_order) 
    S2test = assemble_blocked_matrix(S2_sectors, n_blocks, n_body_order) 
    
    if 0:
        dims_0
        Htest = cp.deepcopy(H_tot)
        Htest.shape = dims_tot + dims_tot
        v0v0 = vecs0+vecs0
        Htest   = transform_tensor(Htest,vecs0+vecs0,trans=1)
        dim0 = 1
        for d in dims_0:
            dim0 *= d
        print dim0,dim0
        print Htest.shape
        Htest.shape = [dim0,dim0] 
        print Htest
        print H0_0
    
   
    print " Dimensions of Full Hamiltonian      ", H_tot.shape
    print " Dimensions of Subspace Hamiltonian  ", Htest.shape
    lp,vp = np.linalg.eigh(Htest)

    s2 = vp.T.dot(S2test).dot(vp)
    print s2
    print 
    print " Eigenvectors of compressed Hamiltonian"
    print " %5s    %12s  %12s  %12s" %("State","Energy","Relative","<S2>")
    for si,i in enumerate(lp):
        print " %5i =  %12.8f  %12.8f  %12.8f" %(si,i*convert,(i-lp[0])*convert,s2[si,si])
        if si>10:
            break
    #print 
    #print
    #print " Energy  Error due to compression    :  %12.8f - %12.8f = %12.8f" %(lp[0],l[0],lp[0]-l[0])

    if 1:
        print " Recompose target state (SLOW)"
        n0 = H_sectors[0,0].shape[0] 
       
        P_dim = n0
        Q_dims = [] # dimension of Q space for each block i.e., Q_dims[3] is the dimension of this space |abcDef...> 
        QQ_dims = [] # dimension of Q space for each block-dimer i.e., QQ_dims[3] is the dimension of this space |AbcdEf...> 
        QQQ_dims = []

        for bi,b in enumerate(blocks):
            q_dim = n0 / p_states[bi].shape[1] * q_states[bi].shape[1]
            Q_dims.extend([q_dim])

        for bi,b in enumerate(blocks):
            for bbi,bb in enumerate(blocks):
                if bbi > bi:
                    q_dim = n0 / p_states[bi].shape[1] / p_states[bbi].shape[1] * q_states[bi].shape[1] * q_states[bbi].shape[1]
                    QQ_dims.extend([q_dim])

        for bi,b in enumerate(blocks):
            for bbi,bb in enumerate(blocks):
                if bbi > bi:
                    for bbbi,bbb in enumerate(blocks):
                        if bbbi > bbi:
                            q_dim = n0 
                            q_dim = q_dim / p_states[bi].shape[1]   * q_states[bi].shape[1]
                            q_dim = q_dim / p_states[bbi].shape[1]  * q_states[bbi].shape[1]
                            q_dim = q_dim / p_states[bbbi].shape[1] * q_states[bbbi].shape[1]
                            QQQ_dims.extend([q_dim])
        
        print P_dim
        print Q_dims
        print QQ_dims
        print QQQ_dims
   
        target_state = args['target_state'] 
        v = cp.deepcopy(vp[:,target_state])


        
        

        v_0 = v[0:P_dim]
        
        print " norm of PPP component %12.8f" %np.dot(v_0.T,v_0)
        
        v_0.shape = n_p_states
        vec_curr = transform_tensor(v_0, vecs0)
        
        if n_body_order >= 1:
            
            start = P_dim
            for bi,b in enumerate(blocks):
                print bi
                stop = start + Q_dims[bi]
                v_tmp = cp.deepcopy(vp[start:stop,target_state])
        
                # copy all P space vectors, and replace current block with Q vectors
                vecs_b = cp.deepcopy(vecs0)
                vecs_b[bi] = q_states[bi]
                
                dim_b = cp.deepcopy(n_p_states)
                dim_b[bi] = q_states[bi].shape[1]
                v_tmp = v_tmp.reshape(dim_b)
               
                # add this recomposed portion of the CI vector
                vec_curr += transform_tensor(v_tmp, vecs_b)
                start = stop
        
        if n_body_order >= 2:
            
            block_dimer_index = 0
            for bi,b in enumerate(blocks):
                for bbi,bb in enumerate(blocks):
                    if bbi > bi:

                        stop = start + QQ_dims[block_dimer_index]
                        v_tmp = cp.deepcopy(vp[start:stop,target_state])
                     
                        # copy all P space vectors, and replace current block with Q vectors
                        vecs_b = cp.deepcopy(vecs0)
                        vecs_b[bi] = q_states[bi]
                        vecs_b[bbi] = q_states[bbi]
                        
                        dim_b       = cp.deepcopy(n_p_states)
                        dim_b[bi]   = q_states[bi].shape[1]
                        dim_b[bbi]  = q_states[bbi].shape[1]
                        
                        v_tmp = v_tmp.reshape(dim_b)
                        
                        # add this recomposed portion of the CI vector
                        vec_curr -= transform_tensor(v_tmp, vecs_b)

                        block_dimer_index += 1
                        print "QQ_dims, start, stop", start, stop
                        start = stop
        
        
        Acore, Atfac = tucker_decompose(vec_curr,0,0)
        p_states = []
        q_states = []
        for bi,b in enumerate(Atfac):
            p_states.extend([b[:,0:n_p_states[bi]]])
            q_states.extend([b[:,n_p_states[bi]::]])
      
        vec_curr = vec_curr.reshape(dim_tot)
        #H_tot = H_tot.reshape(dim_tot, dim_tot)
        

    print " %5s    %12s  %12s  %12s" %("State","Energy","Relative","<S2>")
    for si,i in enumerate(lp):
        print " %5i =  %12.8f  %12.8f  %12.8f" %(si,i*convert,(i-lp[0])*convert,abs(s2[si,si]))
        if si>args['n_print']:
            break
    
    #print
    #print " Energy  Error due to compression    :  %12.8f - %12.8f = %12.8f" %(lp[0],l[0],lp[0]-l[0])


    energy_per_iter += [lp[target_state]]

    thresh = 1.0*np.power(10.0,-float(args['thresh']))
    if it > 0:
        if abs(lp[target_state]-energy_per_iter[it-1]) < thresh:
            break

    
    
    
    
    
    
print " %10s  %12s  %12s" %("Iteration", "Energy", "Delta")
for ei,e in enumerate(energy_per_iter):
    if ei>0:
        print " %10i  %12.8f  %12.1e" %(ei,e,e-energy_per_iter[ei-1])
    else:
        print " %10i  %12.8f  %12s" %(ei,e,"")



