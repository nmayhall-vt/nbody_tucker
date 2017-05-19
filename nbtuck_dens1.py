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


def printm(m):
    # {{{
    """ print matrix """
    for r in m:
        for ri in r:
            print "%10.1e" %ri,
        print
    # }}}

def get_guess_vectors(lattice, j12, blocks, n_p_states, n_q_states):
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
    
        # Diagonalize an arbitrary linear combination of the quantum numbers we insist on preserving
        l_b,v_b = np.linalg.eigh(H_b + Sz_b + S2_b) 
        l_b = v_b.transpose().dot(H_b).dot(v_b).diagonal()
        
        sort_ind = np.argsort(l_b)
        l_b = l_b[sort_ind]
        v_b = v_b[:,sort_ind]
            

        print " Guess eigenstates"
        for l in l_b:
            print "%12.8f" %l
        p_states.extend([v_b[:,0:n_p_states[bi]]])
        #q_states.extend([v_b[:,n_p_states[bi]::]])
        q_states.extend([v_b[:,n_p_states[bi]: n_p_states[bi]+n_q_states[bi]]])
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

def form_compressed_zero_order_hamiltonian_diag(vecs,Hi):
    # {{{
    dim = 1 # dimension of subspace
    dims = [] # list of mode dimensions (size of hilbert space on each fragment) 
    for vi,v in enumerate(vecs):
        dim = dim*v.shape[1]
        dims.extend([v.shape[1]])
    H = np.zeros((dim,dim))
    n_dims = len(dims)


    H1 = cp.deepcopy(Hi)
    for vi,v in enumerate(vecs):
        H1[vi] = np.dot(v.transpose(),np.dot(Hi[vi],v))

    
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

     

    #H = H.reshape(dim,dim)
    print "     Size of Hamitonian block: ", H.shape
    return H.diagonal()
    # }}}

def form_compressed_hamiltonian_diag(vecs,Hi,Hij):
    # {{{
    dim = 1 # dimension of subspace
    dims = [] # list of mode dimensions (size of hilbert space on each fragment) 
    for vi,v in enumerate(vecs):
        dim = dim*v.shape[1]
        dims.extend([v.shape[1]])
    H = np.zeros((dim,dim))
    #Htest = np.zeros((dim,dim))
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

    #print block_curr1, block_curr2
    #assert(block_curr1 < block_curr2)

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
    #{{{
    Htest = np.array([])


    if n_body_order == 0:
        Htest = H_sectors[0,0]

    if n_body_order == 1:
        Htest = H_sectors[0,0]
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
        
        #
        #   Get dimensionality
        #
        n0 = 0
        n1 = 0
        n2 = 0

        n0 = H_sectors[0,0].shape[0]
        for bi in range(1,n_blocks+1):
            n1 += H_sectors[bi,bi].shape[0]
        for bi in range(1,n_blocks+1):
            for bj in range(bi+1,n_blocks+1):
                n2 += H_sectors[(bi,bj),(bi,bj)].shape[0]

        nd = n0 + n1 + n2

        print "Dimensions: ", n0, n1, n2, " = ", nd

        Htest = np.empty([nd,nd])

        row = np.empty([n0,nd])
        
        # 0,0 


        col_start = 0
        
        col_stop = col_start + n0
        row[0::,col_start:col_stop] = H_sectors[0,0]
        col_start = col_stop

        #row_0 = H_sectors[0,0]
        # 0,S
        for bi in range(1,n_blocks+1):
            #row = np.hstack( ( row_0, H_sectors[0,bi] ) )

            col_stop = col_start + H_sectors[0,bi].shape[1]
            
            row[0::,col_start:col_stop] = H_sectors[0,bi]

            col_start = col_stop
            #row_0 = np.hstack( ( row_0, H_sectors[0,bi] ) )
        # 0,D
        for bi in range(1,n_blocks+1):
            for bj in range(bi+1, n_blocks+1):
                bij = (bi,bj)
                #print row_0.shape, H_sectors[0,bij].shape
                
                #row_0 = np.hstack( ( row_0, H_sectors[0,(bi,bj)] ) )
                
                col_stop = col_start + H_sectors[0,bij].shape[1]
                row[0::,col_start:col_stop] = H_sectors[0,bij]
                col_start = col_stop
    
    
        row_start = 0
        row_stop  = row_start + n0
        
        Htest[row_start:row_stop,0::] = row
        #Htest[row_start:row_stop,0::] = row_0 
       
        row_start = row_stop
        
        
        
        # Singles
        for bi in range(1,n_blocks+1):
            
            col_start = 0
            
            row_stop = row_start + H_sectors[bi,0].shape[0]
            
            row = np.empty([row_stop-row_start,nd])

            # bi,0
            col_stop = col_start + H_sectors[bi,0].shape[1]
            row[0::,col_start:col_stop] = H_sectors[bi,0]
            col_start = col_stop
            
            #row_i = H_sectors[bi,0] 
            
            # bi,bj 
            for bj in range(1,n_blocks+1):
                col_stop = col_start + H_sectors[bi,bj].shape[1]
                row[0::,col_start:col_stop] = H_sectors[bi,bj]
                col_start = col_stop
                
                #row_i = np.hstack((row_i,H_sectors[bi,bj]))
            # bi,bjk 
            for bj in range(1,n_blocks+1):
                for bk in range(bj+1,n_blocks+1):
                    col_stop = col_start + H_sectors[bi,(bj,bk)].shape[1]
                    row[0::,col_start:col_stop] = H_sectors[bi,(bj,bk)]
                    col_start = col_stop
                
                    #row_i = np.hstack((row_i,H_sectors[bi,(bj,bk)]))
            
            #Htest = np.vstack((Htest,row_i))
           
                
            Htest[row_start:row_stop,0::] = row 
            #Htest[row_start:row_stop,0::] = row_i 
    
            row_start = row_stop
    
        
        
    
        #Doubles 
        for bi in range(1,n_blocks+1):
            for bj in range(bi+1,n_blocks+1):
            
                col_start = 0
        
            
                bij = (bi,bj)

                row_stop = row_start + H_sectors[bij,0].shape[0]
            
                row = np.empty([row_stop-row_start,nd])

                # bij,0
                col_stop = col_start + H_sectors[bij,0].shape[1]
                row[0::,col_start:col_stop] = H_sectors[bij,0]
                col_start = col_stop
            
                #row_ij = H_sectors[(bi,bj),0]                                       #<ij|H|0>
                # bij,bk
                for bk in range(1,n_blocks+1):
                    col_stop = col_start + H_sectors[bij,bk].shape[1]
                    row[0::,col_start:col_stop] = H_sectors[bij,bk]
                    col_start = col_stop
            
                    #row_ij = np.hstack( (row_ij, H_sectors[(bi,bj),bk]) )           #<ij|H|k>
                # bij,bkl
                for bk in range(1,n_blocks+1):
                    for bl in range(bk+1,n_blocks+1):
                        bkl = (bk,bl)

                        col_stop = col_start + H_sectors[bij,bkl].shape[1]
                        row[0::,col_start:col_stop] = H_sectors[bij,bkl]
                        col_start = col_stop
            
                        #row_ij = np.hstack( (row_ij, H_sectors[(bi,bj),(bk,bl)]) )  #<ij|H|kl>
                
                #Htest = np.vstack((Htest,row_ij))
                
                Htest[row_start:row_stop,0::] = row
                #Htest[row_start:row_stop,0::] = row_ij 
    
                row_start = row_stop

    return Htest
    #}}}

def build_tucker_blocked_H(n_blocks,tucker_blocks, n_body_order):
    #{{{
    dim_tot = 0
    for ti in sorted(tucker_blocks):
        tbi = tucker_blocks[ti]
        dim_tot += tbi.full_dim


    Htest = np.zeros((dim_tot, dim_tot))

        
    # Fill diagonals
    tb_l = tucker_blocks[-1]
    tb_r = tucker_blocks[-1]
    #Htest[tb_l.start:tb_l.stop, tb_r.start:tb_r.stop] = H_sectors[tb_l.id,tb_r.id] 
    Htest[tb_l.start:tb_l.stop, tb_r.start:tb_r.stop] = build_H(blocks, tb_l, tb_r)
    # <0|H|Q>
    for t_l in tucker_blocks:
        for t_r in tucker_blocks:
            tb_l = tucker_blocks[t_l]
            tb_r = tucker_blocks[t_r]
            Htest[tb_l.start:tb_l.stop, tb_r.start:tb_r.stop] = build_H(blocks, tb_l, tb_r)
            Htest[tb_r.start:tb_r.stop, tb_l.start:tb_l.stop] = Htest[tb_l.start:tb_l.stop, tb_r.start:tb_r.stop].T

    return Htest
    #}}}

def build_H(blocks,tb_l, tb_r):
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
    
    H = np.zeros((tb_l.full_dim,tb_r.full_dim))
   
    #print " Ham block size", H.shape, H_dim_layout
    H.shape = H_dim_layout
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
            h = np.kron(h1,np.eye(dim_e))   
            
            tens_dims    = []
            tens_inds    = []
            tens_inds.extend([bi])
            tens_dims.extend([tb_l.block_dims[bi]])
            for bj in range(0,n_blocks):
                if (bi != bj):
                    tens_inds.extend([bj])
                    tens_dims.extend([tb_l.block_dims[bj]])
            tens_dims = np.append(tens_dims, tens_dims) 
            #tens_inds = np.append(tens_inds, tens_inds) 

            sort_ind = np.argsort(tens_inds)
            swap = np.append(sort_ind,sort_ind+n_blocks) 

            H += h.reshape(tens_dims).transpose(swap)
        
        
        #   <ab|H12|ab> Ic Id
        # + <ac|H13|ac> Ib Id
        # + Ia <bc|H23|bc> Id + etc
        
        for bi in range(0,n_blocks):
            for bj in range(bi+1,n_blocks):
                Bi = blocks[bi]
                Bj = blocks[bj]
                dim_e = full_dim / tb_l.block_dims[bi] / tb_l.block_dims[bj]

                #build full Hamiltonian on sublattice
                h12 = build_dimer_H(tb_l, tb_r, Bi, Bj, j12)

                h = np.kron(h12,np.eye(dim_e))   
            
                tens_dims    = []
                tens_inds    = []
                tens_inds.extend([bi])
                tens_inds.extend([bj])
                tens_dims.extend([tb_l.block_dims[bi]])
                tens_dims.extend([tb_l.block_dims[bj]])
                for bk in range(0,n_blocks):
                    if (bk != bi) and (bk != bj):
                        tens_inds.extend([bk])
                        tens_dims.extend([tb_l.block_dims[bk]])
                tens_dims = np.append(tens_dims, tens_dims) 
                #tens_inds = np.append(tens_inds, tens_inds) 
               
                sort_ind = np.argsort(tens_inds)
                swap = np.append(sort_ind,sort_ind+n_blocks) 
               
                H += h.reshape(tens_dims).transpose(swap)
    
    
    
    
    
    
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

        h1.shape = (tb_l.block_dims[bi],tb_r.block_dims[bi])

        assert(dim_e_l == dim_e_r)
        dim_e = dim_e_l
       
        
        #HERE NICK!!
        
        tens_dims    = []
        tens_inds    = []
        tens_inds.extend([bi])
        tens_inds.extend([bi+n_blocks])
        for bj in range(0,n_blocks):
            if (bi != bj):
                tens_inds.extend([bj])
                tens_inds.extend([bj+n_blocks])
                assert(tb_l.block_dims[bj] == tb_r.block_dims[bj] )
                h1 = np.tensordot(h1,np.eye(tb_l.block_dims[bj]),axes=0)

        sort_ind = np.argsort(tens_inds)

        H += h1.transpose(sort_ind)
        
        
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
            h2 = build_dimer_H(tb_l, tb_r, Bj, Bi, j12)
          
            h2.shape = (tb_l.block_dims[bj],tb_l.block_dims[bi],tb_r.block_dims[bj],tb_r.block_dims[bi])
         
            
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
            
            sort_ind = np.argsort(tens_inds)
            H += h2.transpose(sort_ind)
        
        for bj in range(bi+1, n_blocks):
            Bj = blocks[bj]
            dim_e_l = full_dim_l / tb_l.block_dims[bi] / tb_l.block_dims[bj]
            dim_e_r = full_dim_r / tb_r.block_dims[bi] / tb_r.block_dims[bj]
         
            assert(dim_e_l == dim_e_r)
            dim_e = dim_e_l
            
            #build full Hamiltonian on sublattice
            #h12 = build_dimer_H(tb_l, tb_r, Bi, Bj, j12)
            h2 = build_dimer_H(tb_l, tb_r, Bi, Bj, j12)
          
            h2.shape = (tb_l.block_dims[bi],tb_l.block_dims[bj],tb_r.block_dims[bi],tb_r.block_dims[bj])
         
            
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
            
            sort_ind = np.argsort(tens_inds)
            H += h2.transpose(sort_ind)
    
    
    
    
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
        h2 = build_dimer_H(tb_l, tb_r, Bi, Bj, j12)
       
        h2.shape = (tb_l.block_dims[bi],tb_l.block_dims[bj],tb_r.block_dims[bi],tb_r.block_dims[bj])
        #h2 = np.kron(h12,np.eye(dim_e))   
        
        tens_dims    = []
        tens_inds    = []
        tens_inds.extend([bi,bj])
        tens_inds.extend([bi+n_blocks, bj+n_blocks])
        tens_dims.extend([tb_l.block_dims[bi]])
        tens_dims.extend([tb_l.block_dims[bj]])
        tens_dims.extend([tb_r.block_dims[bi]])
        tens_dims.extend([tb_r.block_dims[bj]])
        for bk in range(0,n_blocks):
            if (bk != bi) and (bk != bj):
                tens_inds.extend([bk])
                tens_inds.extend([bk+n_blocks])
                tens_dims.extend([tb_l.block_dims[bk]])
                tens_dims.extend([tb_r.block_dims[bk]])
                h2 = np.tensordot(h2,np.eye(tb_l.block_dims[bk]),axes=0)
        
        sort_ind = np.argsort(tens_inds)
        #H += h2.reshape(tens_dims).transpose(sort_ind)
        H += h2.transpose(sort_ind)

    H = H.reshape(tb_l.full_dim,tb_r.full_dim)
    return H
# }}}






       


"""
Test forming HDVV Hamiltonian and projecting onto "many-body tucker basis"
"""
#   Setup input arguments
parser = argparse.ArgumentParser(description='Finds eigenstates of a spin lattice',
formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#parser.add_argument('-d','--dry_run', default=False, action="store_true", help='Run but don\'t submit.', required=False)
parser.add_argument('-ju','--j_unit', type=str, default="cm", help='What units are the J values in', choices=['cm','ev'],required=False)
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
parser.add_argument('-pt_type','--pt_type', type=str, default='en', choices=['mp','en'], help='PT correction denominator type', required=False)
args = vars(parser.parse_args())
#
#   Let minute specification of walltime override hour specification

j12 = np.loadtxt(args['j12'])
lattice = np.loadtxt(args['lattice']).astype(int)
blocks = np.loadtxt(args['blocks']).astype(int)
n_sites = len(lattice)
n_blocks = len(blocks)
    
if len(blocks.shape) == 1:
    print 'blocks',blocks
    
    blocks.shape = (1,len(blocks.transpose()))
    n_blocks = len(blocks)


n_p_states = args['n_p_space'] 
n_q_states = args['n_q_space'] 


if args['n_p_space'] == None:
    n_p_states = []
    for bi in range(n_blocks):
        n_p_states.extend([1])
    args['n_p_space'] = n_p_states

assert(len(args['n_p_space']) == n_blocks)

np.random.seed(2)


au2ev = 27.21165;
au2cm = 219474.63;
convert = au2ev/au2cm;	# convert from wavenumbers to eV
convert = 1;			# 1 for wavenumbers

if args['j_unit'] == 'cm':
    j12 = j12 * au2ev/au2cm

print " j12:\n", j12
print " lattice:\n", lattice 
print " blocks:\n", blocks
print " n_blocks:\n", n_blocks

H_tot = np.array([])
S2_tot = np.array([])
H_dict = {}




# calculate problem dimensions 
dims_tot = []
dim_tot = 1
for bi,b in enumerate(blocks):
    block_dim = np.power(2,b.shape[0])
    dims_tot.extend([block_dim])
    dim_tot *= block_dim

if args['n_q_space'] == None:
    n_q_states = []
    for bi in range(n_blocks):
        n_q_states.extend([dims_tot[bi]-n_p_states[bi]])
    args['n_q_space'] = n_q_states


# Get initial compression vectors 
p_states, q_states = get_guess_vectors(lattice, j12, blocks, n_p_states, n_q_states)



"""
H = -2 \sum_{ij} J_{ij} S_i\cdotS_j
  = - \sum_{ij} S^+_i S^-_j  +  S^-_i S^+_j  + 2 S^z_i S^z_j

<abc| H12 |def> = - \sum_{ij}  <a|S+i|d><b|S-j|e> <c|f> 
                             + <a|S-i|d><b|S+j|e> <c|f>
                             +2<a|Szi|d><b|Szj|e> <c|f> 

Form matrix representations of S+, S-, Sz in local basis

Sp = {}
Sp[bi] = Op(i,a,b)    # bi = block, i=site, a=bra, b=ket

<Abc| H12 | aBd> = -\sum_{ij\in 12}   <A|S+i|a><b|S-j|B><c|d>
                                    + <A|S-i|a><b|S+j|B><c|d>
                                    +2<A|Szi|a><b|Szj|B><c|d>


"""

#build operators
op_Sp = {}
op_Sm = {}
op_Sz = {}
local_states = {}   
                    

blocks_in = cp.deepcopy(blocks)
blocks = {}         # dictionary of block objects


#
#   Initialize Block objects
print 
print " Prepare Lattice Blocks:"
print n_p_states, n_q_states
for bi in range(0,n_blocks):
    blocks[bi] = Block()
    blocks[bi].init(bi,blocks_in[bi,:],[n_p_states[bi], n_q_states[bi]])

    blocks[bi].np = n_p_states[bi] 
    blocks[bi].nq = n_q_states[bi] 
    blocks[bi].vecs = np.hstack((p_states[bi],q_states[bi]))
    
    blocks[bi].extract_lattice(lattice)
    blocks[bi].extract_j12(j12)

    blocks[bi].form_H()
    blocks[bi].form_site_operators()

    print blocks[bi]

n_body_order = args['n_body_order'] 
    
dim_tot = 0

tb_0 = Tucker_Block()
address_0 = np.zeros(n_blocks,dtype=int)
tb_0.init((-1), blocks,address_0, dim_tot)

dim_tot += tb_0.full_dim

tucker_blocks = {}
tucker_blocks[-1] = tb_0 

print 
print " Prepare Tucker blocks:"
if n_body_order > 0:
    for bi in range(0,n_blocks):
        tb = Tucker_Block()
        address = np.zeros(n_blocks,dtype=int)
        address[bi] = 1
        tb.init((bi), blocks,address, dim_tot)
        tucker_blocks[bi] = tb
        dim_tot += tb.full_dim
        if n_body_order > 1:
            for bj in range(bi+1,n_blocks):
                tb = Tucker_Block()
                address = np.zeros(n_blocks,dtype=int)
                address[bi] = 1
                address[bj] = 1
                tb.init((bi,bj), blocks,address,dim_tot)
                tucker_blocks[bi,bj] = tb
                dim_tot += tb.full_dim
                if n_body_order > 2:
                    for bk in range(bj+1,n_blocks):
                        tk = Tucker_Block()
                        address = np.zeros(n_blocks,dtype=int)
                        address[bi] = 1
                        address[bj] = 1
                        address[bk] = 1
                        tk.init((bi,bj,bk), blocks,address,dim_tot)
                        tucker_blocks[bi,bj,bk] = tb
                        dim_tot += tb.full_dim

for tb in sorted(tucker_blocks):
    print tucker_blocks[tb]
print 
print " Build Hamiltonian:"
Htest = build_tucker_blocked_H(n_blocks, tucker_blocks, n_body_order) 
#tb_l = tucker_blocks[0]
#tb_r = tucker_blocks[0,2]
#Htest = Htest[tb_l.start:tb_l.stop, tb_r.start:tb_r.stop]
#Htest = build_H(blocks, tb_l, tb_r)
#l,s,v = np.linalg.svd(Htest)
#print s
#exit(-1)

print " Size of H: ", Htest.shape

l = np.array([])
v = np.array([])

if Htest.shape[0] > 3000:
    l,v = scipy.sparse.linalg.eigsh(Htest, k=args["n_roots"] )
else:
    l,v = np.linalg.eigh(Htest)


print " %5s    %16s  %16s  %12s" %("State","Energy","Relative","<S2>")
for si,i in enumerate(l):
    print " %5i =  %16.8f  %16.8f  %12.8s" %(si,i*convert,(i-l[0])*convert,"--")
    if si>args['n_print']:
        break


