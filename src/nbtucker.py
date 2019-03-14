#! /usr/bin/env python
import numpy as np
import scipy
import scipy.linalg
import scipy.io
import copy as cp
import argparse
import scipy.sparse
import scipy.sparse.linalg

from hdvv import *
from block import *
from davidson import *

import hamiltonian_generator 

np.set_printoptions(suppress = True, precision = 4, linewidth=800)

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
        l_b,v_b = np.linalg.eigh(H_b + .9*Sz_b + .8*S2_b) 
        h_b = v_b.transpose().dot(H_b).dot(v_b).diagonal()
        
        sort_ind = np.argsort(h_b)
        h_b = h_b[sort_ind]
        v_b = v_b[:,sort_ind]
        
        s2_b = v_b.transpose().dot(S2_b).dot(v_b).diagonal()
        sz_b = v_b.transpose().dot(Sz_b).dot(v_b).diagonal()
            

        print " Guess eigenstates"
        print "   %-12s   %16s  %12s  %12s "%("Local State", "<H>", "<S2>", "<Sz>")
        for si,i in enumerate(l_b):
            print "   %-12i   %16.8f  %12.4f  %12.4f "%(si,h_b[si],abs(s2_b[si]),sz_b[si])
        p_states.extend([v_b[:,0:n_p_states[bi]]])
        #q_states.extend([v_b[:,n_p_states[bi]::]])
        q_states.extend([v_b[:,n_p_states[bi]: n_p_states[bi]+n_q_states[bi]]])
    return p_states, q_states
    # }}}

def do_variational_microiteration_update(args,n_blocks, tucker_blocks, lattice_blocks, n_body_order, j12, dav_thresh,it,last_vectors):
# {{{

    """
    varialtionally solve the blocks and output E and v

    """   
    
    dim_tot = 0
    
    for tb in tucker_blocks:
        dim_tot += tucker_blocks[tb].full_dim
   
    assert(args['n_roots'] <= dim_tot) 
    # 
    #   Loop over davidson micro-iterations
    print 
    print " Solve for supersystem eigenvalues: Dimension = ", dim_tot
    dav = Davidson(dim_tot, args['n_roots'])
    dav.thresh = dav_thresh 
    dav.max_vecs = args['dav_max_ss']
    s2v = np.array([])
    if it == 0:
        if args['dav_guess'] == 'rand':
            dav.form_rand_guess()
        else:
            dav.form_p_guess()
    else:
        dav.vec_curr = last_vectors 
    dav.max_iter = args['dav_max_iter']

    for dit in range(0,dav.max_iter):
        #dav.form_sigma()
       
        if args['direct'] == 0:
            dav.sig_curr = H.dot(dav.vec_curr)
            hv = H.dot(dav.vec_curr)
            s2v = S2.dot(dav.vec_curr)
        else:
            hv, s2v = build_tucker_blocked_sigma(n_blocks, tucker_blocks, lattice_blocks, n_body_order, j12, dav.vec_curr) 
            dav.sig_curr = hv
    
        if args['dav_precond']:
            hv_diag = build_tucker_blocked_diagonal(n_blocks, tucker_blocks, lattice_blocks, n_body_order, j12, 0) 
            dav.set_preconditioner(hv_diag)
        #dav.set_preconditioner(H.diagonal())
        
        dav.update()
        dav.print_iteration()
        if dav.converged():
            break
    if dav.converged():
        print " Davidson Converged"
    else:
        print " Davidson Not Converged"
    print 

    l = np.array([])
    v = np.array([])
    if dav.max_iter == -1 and direct == 0:
        print 
        print " Diagonalizing explicitly:"

        if H.shape[0] > 3000:
            l,v = scipy.sparse.linalg.eigsh(H, k=n_roots )
        else:
            l,v = np.linalg.eigh(H)
    else:
        # get eigen stuff from davidson
        l = dav.eigenvalues()
        v = dav.eigenvectors()

    return l,v
# }}}

def form_brdm(n_blocks, tucker_blocks, lattice_blocks, j12,v,ts):
    # {{{
    brdms = {}   # block reduced density matrix
    for bi in range(0,n_blocks):
        Bi = lattice_blocks[bi]
        brdms[bi] = np.zeros(( Bi.full_dim, Bi.full_dim )) 
      
    """"
    changing v to v_pt
    """
    print
    print " Compute Block Reduced Density Matrices (BRDM):"
    for tb1 in sorted(tucker_blocks):
        Tb1 = tucker_blocks[tb1]
        vb1 = cp.deepcopy(v[Tb1.start:Tb1.stop, ts])
        vb1.shape = Tb1.block_dims
        for tb2 in sorted(tucker_blocks):
            Tb2 = tucker_blocks[tb2]
            
            if Tb1.id <= Tb2.id:
                # How many blocks are different between left and right?
                different = []
                for bi in range(0,n_blocks):
                    if Tb2.address[bi] != Tb1.address[bi]:
                        different.append(bi)
                
                if len(different) == 0:
                    vb2 = cp.deepcopy(v[Tb2.start:Tb2.stop, ts])
                    vb2.shape = Tb2.block_dims
                    for bi in range(0,n_blocks):
                        brdm_tmp = form_1fdm(vb1,vb2,[bi])
                        Bi = lattice_blocks[bi]
                        u1 = Bi.v_ss(Tb1.address[bi])
                        u2 = Bi.v_ss(Tb2.address[bi])
                        brdms[bi] += u1.dot(brdm_tmp).dot(u2.T)
                if len(different) == 1:
                    vb2 = cp.deepcopy(v[Tb2.start:Tb2.stop, ts])
                    vb2.shape = Tb2.block_dims
                    bi = different[0]
                    brdm_tmp = form_1fdm(vb1,vb2,[bi])
                    Bi = lattice_blocks[bi]
                    u1 = Bi.v_ss(Tb1.address[bi])
                    u2 = Bi.v_ss(Tb2.address[bi])
                    brdm_tmp = u1.dot(brdm_tmp).dot(u2.T)
                    brdms[bi] += brdm_tmp + brdm_tmp.T 
    # }}}
    return brdms



def nbody_tucker(   j12 = hamiltonian_generator.make_2d_lattice(), 
                    blocks = [[0,1],[2,3]],
                    n_p_states = None,
                    n_q_states = None,
                    n_body_order = 2,
                    pt_order = 0,
                    pt_type = 'mp',
                    pt_mit = 1,
                    # cluster_state optimization variables
                    max_iter = 100,     #max_iter
                    diis_thresh = 1e-7, #thresh
                    opt = 'diis',       #what kind of solver
                    diis_start=0,       #which iter starts diis
                    n_diis_vecs=8,      #max diis subspace size
                    # davidson optimization variables
                    dav_thresh  = 1e-7, #Threshold for supersystem davidson iterations
                    dav_max_iter = 20,  #maxiter for supersystem davidson iterations
                    n_roots = 1,        #How many roots to solve for
                    dav_max_ss = 20,    #Max number of vectors in davidson subspace
                    dav_guess='pspace', #Initial guess for davidson
                    #static args
                    direct = 1,
                    dav_precond = 1,
                    n_print=10          #num states to print
                ):


    #args = dict(j12, blocks)
    args = {'j12':j12,
            'blocks' : blocks,
            'n_p_states'    :   n_p_states,
            'n_q_states'    :   n_q_states,
            'n_body_order'  :   n_body_order,
            'pt_order'      :   pt_order,
            'pt_type'       :   pt_type,
            'pt_mit'        :   pt_mit,
            'max_iter'      :   max_iter,    
            'diis_thresh'   :   diis_thresh,
            'opt'           :   opt,      
            'diis_start'    :   diis_start,      
            'n_diis_vecs'   :   n_diis_vecs,     
            'dav_thresh'    :   dav_thresh,
            'dav_max_iter'  :   dav_max_iter, 
            'n_roots'       :   n_roots,       
            'dav_max_ss'    :   dav_max_ss,   
            'dav_guess'     :   dav_guess,
            'direct'        :   direct,
            'dav_precond'   :   dav_precond,
            'n_print'       :   n_print         
            }
    print(args)
    print()


    lattice = np.ones((j12.shape[0],1))
    n_blocks = len(blocks)

    np.random.seed(2)
    
    H_tot = np.array([])
    S2_tot = np.array([])
    H_dict = {}
    # calculate problem dimensions 
    dims_tot = []
    dim_tot = 1
    for bi,b in enumerate(blocks):
        block_dim = np.power(2,len(b))
        dims_tot.extend([block_dim])
        dim_tot *= block_dim
    
    if n_p_states == None:
        n_p_states = []
        for bi in range(n_blocks):
            n_p_states.append(1)
    
    if n_q_states == None:
        n_q_states = []
        for bi in range(n_blocks):
            n_q_states.extend([dims_tot[bi]-n_p_states[bi]])
        n_q_space = n_q_states
    
    
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
    lattice_blocks = {}         # dictionary of block objects


    #
    #   Initialize Block objects
    print 
    print " Prepare Lattice Blocks:"
    print n_p_states, n_q_states
    for bi in range(0,n_blocks):
        lattice_blocks[bi] = Lattice_Block()
        lattice_blocks[bi].init(bi,blocks_in[bi],[n_p_states[bi], n_q_states[bi]])
    
        lattice_blocks[bi].np = n_p_states[bi] 
        lattice_blocks[bi].nq = n_q_states[bi] 
        lattice_blocks[bi].vecs = np.hstack((p_states[bi],q_states[bi]))
        
        lattice_blocks[bi].extract_lattice(lattice)
        lattice_blocks[bi].extract_j12(j12)
    
        lattice_blocks[bi].form_H()
        lattice_blocks[bi].form_site_operators()
    
        print lattice_blocks[bi]



    dim_tot = 0
    
    tb_0 = Tucker_Block()
    address_0 = np.zeros(n_blocks,dtype=int)
    tb_0.init((-1), lattice_blocks,address_0, dim_tot)
    
    dim_tot += tb_0.full_dim
    
    tucker_blocks = {}
    tucker_blocks_pt = {}
    tucker_blocks[0,-1] = tb_0 
    
    print 
    print " Prepare Tucker blocks:"
    if n_body_order >= 1:
        for bi in range(0,n_blocks):
            tb = Tucker_Block()
            address = np.zeros(n_blocks,dtype=int)
            address[bi] = 1
            tb.init((bi), lattice_blocks,address, dim_tot)
            if tb.start < tb.stop:
                tucker_blocks[1,bi] = tb
                dim_tot += tb.full_dim
    if n_body_order >= 2:
        for bi in range(0,n_blocks):
            for bj in range(bi+1,n_blocks):
                tb = Tucker_Block()
                address = np.zeros(n_blocks,dtype=int)
                address[bi] = 1
                address[bj] = 1
                tb.init((bi,bj), lattice_blocks,address,dim_tot)
                if tb.start < tb.stop:
                    tucker_blocks[2,bi,bj] = tb
                    dim_tot += tb.full_dim
    if n_body_order >= 3:
        for bi in range(0,n_blocks):
            for bj in range(bi+1,n_blocks):
                for bk in range(bj+1,n_blocks):
                    tb = Tucker_Block()
                    address = np.zeros(n_blocks,dtype=int)
                    address[bi] = 1
                    address[bj] = 1
                    address[bk] = 1
                    tb.init((bi,bj,bk), lattice_blocks,address,dim_tot)
                    if tb.start < tb.stop:
                        tucker_blocks[3,bi,bj,bk] = tb
                        dim_tot += tb.full_dim
    if n_body_order >= 4:
        for bi in range(0,n_blocks):
            for bj in range(bi+1,n_blocks):
                for bk in range(bj+1,n_blocks):
                    for bl in range(bk+1,n_blocks):
                        tb = Tucker_Block()
                        address = np.zeros(n_blocks,dtype=int)
                        address[bi] = 1
                        address[bj] = 1
                        address[bk] = 1
                        address[bl] = 1
                        tb.init((bi,bj,bk,bl), lattice_blocks,address,dim_tot)
                        if tb.start < tb.stop:
                            tucker_blocks[4,bi,bj,bk,bl] = tb
                            dim_tot += tb.full_dim
    if n_body_order >= 5:
        for bi in range(0,n_blocks):
            for bj in range(bi+1,n_blocks):
                for bk in range(bj+1,n_blocks):
                    for bl in range(bk+1,n_blocks):
                        for bm in range(bl+1,n_blocks):
                            tb = Tucker_Block()
                            address = np.zeros(n_blocks,dtype=int)
                            address[bi] = 1
                            address[bj] = 1
                            address[bk] = 1
                            address[bl] = 1
                            address[bm] = 1
                            tb.init((bi,bj,bk,bl,bm), lattice_blocks,address,dim_tot)
                            if tb.start < tb.stop:
                                tucker_blocks[5,bi,bj,bk,bl,bm] = tb
                                dim_tot += tb.full_dim
    if n_body_order >= 6:
        for bi in range(0,n_blocks):
            for bj in range(bi+1,n_blocks):
                for bk in range(bj+1,n_blocks):
                    for bl in range(bk+1,n_blocks):
                        for bm in range(bl+1,n_blocks):
                            for bn in range(bm+1,n_blocks):
                                tb = Tucker_Block()
                                address = np.zeros(n_blocks,dtype=int)
                                address[bi] = 1
                                address[bj] = 1
                                address[bk] = 1
                                address[bl] = 1
                                address[bm] = 1
                                address[bn] = 1
                                tb.init((bi,bj,bk,bl,bm,bn), lattice_blocks,address,dim_tot)
                                if tb.start < tb.stop:
                                    tucker_blocks[6,bi,bj,bk,bl,bm,bn] = tb
                                    dim_tot += tb.full_dim
    
    if n_body_order >= 7:
        for bi in range(0,n_blocks):
            for bj in range(bi+1,n_blocks):
                for bk in range(bj+1,n_blocks):
                    for bl in range(bk+1,n_blocks):
                        for bm in range(bl+1,n_blocks):
                            for bn in range(bm+1,n_blocks):
                                for bo in range(bn+1,n_blocks):
                                    tb = Tucker_Block()
                                    address = np.zeros(n_blocks,dtype=int)
                                    address[bi] = 1
                                    address[bj] = 1
                                    address[bk] = 1
                                    address[bl] = 1
                                    address[bm] = 1
                                    address[bn] = 1
                                    address[bo] = 1
                                    tb.init((bi,bj,bk,bl,bm,bn,bo), lattice_blocks,address,dim_tot)
                                    if tb.start < tb.stop:
                                        tucker_blocks[7,bi,bj,bk,bl,bm,bn,bo] = tb
                                        dim_tot += tb.full_dim
    
    if n_body_order >= 8:
        for bi in range(0,n_blocks):
            for bj in range(bi+1,n_blocks):
                for bk in range(bj+1,n_blocks):
                    for bl in range(bk+1,n_blocks):
                        for bm in range(bl+1,n_blocks):
                            for bn in range(bm+1,n_blocks):
                                for bo in range(bn+1,n_blocks):
                                    for bp in range(bo+1,n_blocks):
                                        tb = Tucker_Block()
                                        address = np.zeros(n_blocks,dtype=int)
                                        address[bi] = 1
                                        address[bj] = 1
                                        address[bk] = 1
                                        address[bl] = 1
                                        address[bm] = 1
                                        address[bn] = 1
                                        address[bo] = 1
                                        address[bp] = 1
                                        tb.init((bi,bj,bk,bl,bm,bn,bo,bp), lattice_blocks,address,dim_tot)
                                        if tb.start < tb.stop:
                                            tucker_blocks[8,bi,bj,bk,bl,bm,bn,bo,bp] = tb
                                            dim_tot += tb.full_dim
    
    #
    #   Prepare tucker_blocks for perturbation
    dim_tot_pt = 0
    if pt_order > 1:
        if n_body_order == 0:
            for bi in range(0,n_blocks):
                tb = Tucker_Block()
                address = np.zeros(n_blocks,dtype=int)
                address[bi] = 1
                tb.init((bi), lattice_blocks,address, dim_tot_pt)
                if tb.start < tb.stop:
                    tucker_blocks_pt[1,bi] = tb
                    dim_tot_pt += tb.full_dim
        if n_body_order == 0 or n_body_order == 1:
            for bi in range(0,n_blocks):
                for bj in range(bi+1,n_blocks):
                    tb = Tucker_Block()
                    address = np.zeros(n_blocks,dtype=int)
                    address[bi] = 1
                    address[bj] = 1
                    tb.init((bi,bj), lattice_blocks,address,dim_tot_pt)
                    if tb.start < tb.stop:
                        tucker_blocks_pt[2,bi,bj] = tb
                        dim_tot_pt += tb.full_dim
        if n_body_order == 1 or n_body_order == 2:
            for bi in range(0,n_blocks):
                for bj in range(bi+1,n_blocks):
                    for bk in range(bj+1,n_blocks):
                        tb = Tucker_Block()
                        address = np.zeros(n_blocks,dtype=int)
                        address[bi] = 1
                        address[bj] = 1
                        address[bk] = 1
                        tb.init((bi,bj,bk), lattice_blocks,address,dim_tot_pt)
                        if tb.start < tb.stop:
                            tucker_blocks_pt[3,bi,bj,bk] = tb
                            dim_tot_pt += tb.full_dim
        if n_body_order == 2 or n_body_order == 3:
            for bi in range(0,n_blocks):
                for bj in range(bi+1,n_blocks):
                    for bk in range(bj+1,n_blocks):
                        for bl in range(bk+1,n_blocks):
                            tb = Tucker_Block()
                            address = np.zeros(n_blocks,dtype=int)
                            address[bi] = 1
                            address[bj] = 1
                            address[bk] = 1
                            address[bl] = 1
                            tb.init((bi,bj,bk,bl), lattice_blocks,address,dim_tot_pt)
                            if tb.start < tb.stop:
                                tucker_blocks_pt[4,bi,bj,bk,bl] = tb
                                dim_tot_pt += tb.full_dim
        if n_body_order == 3 or n_body_order == 4:
            for bi in range(0,n_blocks):
                for bj in range(bi+1,n_blocks):
                    for bk in range(bj+1,n_blocks):
                        for bl in range(bk+1,n_blocks):
                            for bm in range(bl+1,n_blocks):
                                tb = Tucker_Block()
                                address = np.zeros(n_blocks,dtype=int)
                                address[bi] = 1
                                address[bj] = 1
                                address[bk] = 1
                                address[bl] = 1
                                address[bm] = 1
                                tb.init((bi,bj,bk,bl,bm), lattice_blocks,address,dim_tot_pt)
                                if tb.start < tb.stop:
                                    tucker_blocks_pt[5,bi,bj,bk,bl,bm] = tb
                                    dim_tot_pt += tb.full_dim
        if n_body_order == 4 or n_body_order == 5:
            for bi in range(0,n_blocks):
                for bj in range(bi+1,n_blocks):
                    for bk in range(bj+1,n_blocks):
                        for bl in range(bk+1,n_blocks):
                            for bm in range(bl+1,n_blocks):
                                for bn in range(bm+1,n_blocks):
                                    tb = Tucker_Block()
                                    address = np.zeros(n_blocks,dtype=int)
                                    address[bi] = 1
                                    address[bj] = 1
                                    address[bk] = 1
                                    address[bl] = 1
                                    address[bm] = 1
                                    address[bn] = 1
                                    tb.init((bi,bj,bk,bl,bm,bn), lattice_blocks,address,dim_tot_pt)
                                    if tb.start < tb.stop:
                                        tucker_blocks_pt[6,bi,bj,bk,bl,bm,bn] = tb
                                        dim_tot_pt += tb.full_dim
        if n_body_order >= 5:
            print "n_body_order > 4 NYI for PT2"
            exit(-1)
    
    print
    print " Configurations defining the variational space"
    for tb in sorted(tucker_blocks):
        print tucker_blocks[tb], " Range= %8i:%-8i" %( tucker_blocks[tb].start, tucker_blocks[tb].stop)



    #print 
    #print " Configurations defining the perturbative space"
    #for tb in sorted(tucker_blocks_pt):
    #    print tucker_blocks_pt[tb], " Range= %8i:%-8i" %( tucker_blocks_pt[tb].start, tucker_blocks_pt[tb].stop)
    
    
    
    
    
    
    
    # loop over compression vector iterations
    energy_per_iter = []
    energy_per_iter_lcc = []

    last_vectors = np.array([])  # used to detect root flipping
    last_vectors_0 = np.array([])  # used to detect root flipping
    
    n_roots = min(n_roots, dim_tot)
    
    diis_err_vecs = {}
    diis_frag_grams = {}
   

    # static parameters
    ts = 0          # find ground state 
    direct = 1      # avoid full matrix formation
    dav_precond = 1 # do preconditioner? 

    for bi in range(0,n_blocks):
        diis_frag_grams[bi] = []
    
    for it in range(0,max_iter):

        if pt_order == 0:
            l,v = do_variational_microiteration_update(args,n_blocks, tucker_blocks, lattice_blocks, n_body_order, j12,dav_thresh,it,last_vectors)  
            last_vectors = cp.deepcopy(v)
            hv, s2v = build_tucker_blocked_sigma(n_blocks, tucker_blocks, lattice_blocks, n_body_order, j12, v) 
            S2 = v.T.dot(s2v)
            l = v.T.dot(hv).diagonal()

            energy_per_iter.append(l[ts]) 

            if it > 0:
                if abs(l[ts]-energy_per_iter[it-1]) < diis_thresh:
                    break



        if pt_order >= 2:

            tucker_blocks_0 = {}
            for tb in tucker_blocks:
                if tb[0] <= n_body_order - pt_order:
                    tucker_blocks_0[tb] = cp.deepcopy(tucker_blocks[tb])
                    #tucker_blocks_0[tb] = tucker_blocks[tb]
                    
     
            l,v = do_variational_microiteration_update(args, n_blocks, tucker_blocks_0, lattice_blocks, n_body_order - pt_order, j12,dav_thresh,it,last_vectors_0)  
            hv, s2v = build_tucker_blocked_sigma(n_blocks, tucker_blocks_0, lattice_blocks, n_body_order - pt_order, j12, v) 
            S2 = v.T.dot(s2v)
            l = v.T.dot(hv).diagonal()
            last_vectors_0 = cp.deepcopy(v)
            l_lcc = np.zeros(v.shape[1])


            print
            print "-----------------------------------------------------------------------------"   
            print "                         DMBPT-infinity Calculation"
            print "-----------------------------------------------------------------------------"   
            print
            n_roots = args['n_roots']
            pt_type = args['pt_type']
            #PT_nth_vector(n_blocks,lattice_blocks, tucker_blocks, tucker_blocks_pt,n_body_order,pt_order, l, v, j12, pt_type)
            #e_lcc =PT_lcc(n_blocks,lattice_blocks, tucker_blocks, tucker_blocks_pt,n_body_order,pt_order, l, v, j12, pt_type)
            #v_pt = PT_lcc_2(n_blocks,lattice_blocks, tucker_blocks, tucker_blocks_pt,n_body_order,pt_order, l, v, j12, pt_type)
            if n_body_order ==0:
                l,v = do_variational_microiteration_update(args,n_blocks, tucker_blocks, lattice_blocks, n_body_order, j12,dav_thresh,it,last_vectors)  
                last_vectors = cp.deepcopy(v)
                hv, s2v = build_tucker_blocked_sigma(n_blocks, tucker_blocks, lattice_blocks, n_body_order, j12, v) 
                S2 = v.T.dot(s2v)
                l = v.T.dot(hv).diagonal()

                energy_per_iter.append(l[ts]) 

                if it > 0:
                    if abs(l[ts]-energy_per_iter[it-1]) < diis_thresh:
                        break

                #v_pt = PT_lcc_2(n_blocks,lattice_blocks, tucker_blocks, tucker_blocks_pt,n_body_order,pt_order, l, v, j12, pt_type)
                vibin_pt2(n_blocks,lattice_blocks, tucker_blocks, tucker_blocks_pt,n_body_order,pt_order, l, v, j12, pt_type)
            elif pt_type == 'lcc':
                
                e2, v_pt = PT_lcc_3(n_blocks,lattice_blocks, tucker_blocks, tucker_blocks_pt,n_body_order,pt_order, l, v, j12, pt_type, pt_mit)
                print "PT type : LCC"
                print
                print " %5s    %16s  %16s  %12s" %("State","Energy LCC","Relative","<S2>")
                for i in range(0,n_roots):
                    e = l[i] + e2[i]
                    e0 = l[0] + e2[0]
                    l_lcc[i] = l[i] +  e2[i]
                    print " %5i =  %16.8f  %16.8f  %12.8f" %(i,e,(e-e0),abs(S2[i,i]))

            elif pt_type == 'mp':
                print "PT type : MP"
                print 
                if pt_order != n_body_order:
                    print "WARNING: Excitation order not same as PT order (The method might not be size extensive)"
                e2, v_pt = PT_mp(n_blocks,lattice_blocks, tucker_blocks, tucker_blocks_pt,n_body_order,pt_order, l, v, j12, pt_type)
                
                ##For checking each renormalised and normal terms, use this function.
                #e2 = eqn_pt(n_blocks,lattice_blocks, tucker_blocks, tucker_blocks_pt,n_body_order,pt_order, l, v, j12, pt_type)

                print
                print " %5s    %16s  %16s  %12s" %("State","Energy LCC","Relative","<S2>")
                for i in range(0,n_roots):
                    e = l[i] + e2[i]
                    e0 = l[0] + e2[0]
                    l_lcc[i] = l[i] +  e2[i]
                    print " %5i =  %16.8f  %16.8f  %12.8f" %(i,e,(e-e0),abs(S2[i,i]))


            v = v_pt
            last_vectors = cp.deepcopy(v)
            energy_per_iter_lcc.append(l_lcc[ts])
            if it > 0:
                if abs(l_lcc[ts]-energy_per_iter_lcc[it-1]) < diis_thresh:
                    break

        if pt_order == 0:
            # compute S2 for converged states    
            print
            print " %5s    %16s  %16s  %12s" %("State","Energy","Relative","<S2>")
            for si,i in enumerate(l):
                if si<args['n_print']:
                    print " %5i =  %16.8f  %16.8f  %12.8f" %(si,i,(i-l[0]),abs(S2[si,si]))

        if pt_order >= 2:
            # compute S2 for converged states    
            print
            print " Variational Part"
            print " %5s    %16s  %16s  %12s" %("State","Energy","Relative","<S2>")
            for si,i in enumerate(l):
                if si<args['n_print']:
                    print " %5i =  %16.8f  %16.8f  %12.8f" %(si,i,(i-l[0]),abs(S2[si,si]))

            print pt_type," Correction"
            for si,i in enumerate(energy_per_iter_lcc):
                if si<args['n_print']:
                    print " %5i =  %16.8f  %16.8f" %(si,i,(i-energy_per_iter_lcc[0]))

    
    
        brdms = form_brdm(n_blocks, tucker_blocks, lattice_blocks, j12,v,ts)
    
        if 0:
            print
            print " Compute Block Reduced Density Matrices (BRDM) FULL!:"
    # {{{
            dim_tot_list = []
            full_dim = 1
            sum_1 = 0
            for bi in range(0,n_blocks):
                dim_tot_list.append(lattice_blocks[bi].full_dim)
                full_dim *= lattice_blocks[bi].full_dim
            
            vec_curr = np.zeros(dim_tot_list)
            for tb1 in sorted(tucker_blocks):
                Tb1 = tucker_blocks[tb1]
                vb1 = cp.deepcopy(v[Tb1.start:Tb1.stop, ts])
                sum_1 += vb1.T.dot(vb1)
                vb1.shape = Tb1.block_dims
     
                vec = []
                for bi in range(0,n_blocks):
                    Bi = lattice_blocks[bi]
                    vec.append((Bi.v_ss(Tb1.address[bi])))
    
                #sign = 1
                #print " tb1: ", tb1
                #if tb2[0] != 0:
                #    sign = -1
                #vec_curr += sign * transform_tensor(vb1,vec)
                vec_curr += transform_tensor(vb1,vec)
            Acore, Atfac = tucker_decompose(vec_curr,0,0)
            vec_curr.shape = (full_dim,1)
            print "v'v:   %12.8f" % vec_curr.T.dot(vec_curr)
            print "sum_1: %12.8f" % sum_1 
     # }}}
    
    
        print
        print " Compute Eigenvalues of BRDMs:"
        overlaps = []
        diis_start = diis_start
        for bi in range(0,n_blocks):
            Bi = lattice_blocks[bi]
    
            # TODO: do this with a proper projection, where the user specifies the number of spin states for p space
            """
            lx,vx = np.linalg.eigh(Bi.full_S2)
            spin_proj = np.empty((vx.shape[0],0));
            for si in range(0,lx.shape[0]):
                if abs(lx[si]) < 1.0:
                    spin_proj = np.hstack((spin_proj, vx[:,si:si+1]))
            
            spin_proj = spin_proj.dot(spin_proj.T)
            brdm_curr = spin_proj.dot(brdms[bi]).dot(spin_proj)
            """
    
            brdm_curr = brdms[bi] + Bi.full_S2
            if opt == "diis":
                n_diis_vecs = n_diis_vecs 
                proj_p = Bi.v_ss(0).dot(Bi.v_ss(0).T)
                error_vector = proj_p.dot(brdm_curr) - (brdm_curr).dot(proj_p)
                error_vector.shape = (error_vector.shape[0]*error_vector.shape[1],1)
               
                #print "   Dimension of Error Vector matrix: ", Bi.diis_vecs.shape
                if Bi.diis_vecs.shape[0] == 0:
                    Bi.diis_vecs = error_vector
                else:
                    Bi.diis_vecs = np.hstack( (Bi.diis_vecs, error_vector) ) 
    
                diis_frag_grams[bi].append( brdm_curr )
                
                n_evecs = Bi.diis_vecs.shape[1]
                
                if it>diis_start:
                    S = Bi.diis_vecs.T.dot(Bi.diis_vecs )
                    
                    collapse = 1
                    if collapse:
                        sort_ind = np.argsort(S.diagonal())
                        sort_ind = [sort_ind[i] for i in range(0,min(len(sort_ind),n_diis_vecs))]
                        Bi.diis_vecs = Bi.diis_vecs[:,sort_ind]
                        tmp = []
                        for i in sort_ind:
                            tmp.append(diis_frag_grams[bi][i])
                        diis_frag_grams[bi] = cp.deepcopy(tmp)
                        print " Vector errors", S.diagonal()[sort_ind] 
                        n_evecs = Bi.diis_vecs.shape[1]
                        S = Bi.diis_vecs.T.dot(Bi.diis_vecs )
                    print " Number of error vectors: %4i " %n_evecs
                    B = np.ones( (n_evecs+1, n_evecs+1) )
                    B[-1,-1] = 0
                    B[0:-1,0:-1] = cp.deepcopy(S) 
                    r = np.zeros( (n_evecs+1,1) )
                    r[-1] = 1
                    if n_evecs > 0: 
                        x = np.linalg.pinv(B).dot(r)
                        
                        extrap_err_vec = np.zeros((Bi.diis_vecs.shape[0]))
                        extrap_err_vec.shape = (extrap_err_vec.shape[0])
    
                        for i in range(0,x.shape[0]-1):
                            brdm_curr += x[i]*diis_frag_grams[bi][i]
                            extrap_err_vec += x[i]*Bi.diis_vecs[:,i]
                        
                        print " DIIS Coeffs"
                        for i in x:
                            print "  %12.8f" %i
                        #print x.T
                        print " CURRENT           error vector %12.2e " % error_vector.T.dot(error_vector)
    
            if it ==0:
                lxold,vx_old = np.linalg.eigh(brdms[bi] + 0.0022 * Bi.full_S2 )
                print(lxold)
                sort_ind = np.argsort(lxold)[::-1]
                lxold = lxold[sort_ind]
                vx_old = vx_old[:,sort_ind]

            lx,vx = np.linalg.eigh(brdm_curr + Bi.full_S2)  #have to use it while doing PT correcrions and not the next one. why??
            #lx,vx = np.linalg.eigh(brdms[bi] + 0.0022 * Bi.full_S2 )
            #print(vx.shape)

            crdm = np.dot(vx_old.T,np.dot(brdms[bi],vx_old))
            cs2 = np.dot(vx_old.T,np.dot(Bi.full_S2,vx_old))
            #print(crdm)
            #print(cs2)
            #print(Bi.full_S2)

            new_pt_brdm_idea = 0
            if new_pt_brdm_idea:
                print("NEW IDEa")
                #ltemp,vtemp = np.linalg.eigh(crdm[n_roots:,n_roots:] + 0.002 * cs2[n_roots:,n_roots:])
                ltemp,vtemp = np.linalg.eigh(brdm_curr[n_roots:,n_roots:] + Bi.full_S2[n_roots:,n_roots:])  #have to use it while doing PT correcrions and not the next one. why??

                sort_ind = np.argsort(ltemp)[::-1]
                ltemp = ltemp[sort_ind]
                vtemp = vtemp[:,sort_ind]

                print(crdm[0,0])
                print(ltemp)
                vx2 = np.eye(brdm_curr.shape[0])
                vx2[1:,1:] = vtemp

                vx = np.dot(vx,vx2)
                print("New v")
                print(np.dot(vx,vx2))
                #print(vx2)
                #vx = vx2
                #print(brdm_curr)
            else:
                print("NO NOEW IDEA")
                
                
            #print(vx.T.dot(brdms[bi]).dot(vx))
            lx = vx.T.dot(brdms[bi]).dot(vx).diagonal()
            #print(lx)
            #print(np.sum(lx))
           
            sort_ind = np.argsort(lx)[::-1]
            lx = lx[sort_ind]
            vx = vx[:,sort_ind]

            #vx_old = vx #vpt_new
            print("lx")
            print(lx)
            
            vp = vx[:,0:Bi.ss_dims[0]]
            vq = vx[:,Bi.ss_dims[0]:Bi.ss_dims[0]+Bi.ss_dims[1]]
           
            tmp, up = np.linalg.eigh(vp.T.dot(brdms[bi] + Bi.full_H + Bi.full_Sz + Bi.full_S2).dot(vp))
            vp = vp.dot(up)
            sort_ind = np.argsort(vp.T.dot(brdms[bi]).dot(vp).diagonal() )[::-1]
            vp = vp[:,sort_ind]
            v = vp
    
            if Bi.nq > 0: 
                tmp, uq = np.linalg.eigh(vq.T.dot(brdms[bi] + Bi.full_H + Bi.full_Sz + Bi.full_S2).dot(vq))
                vq = vq.dot(uq)
                sort_ind = np.argsort(vq.T.dot(brdms[bi]).dot(vq).diagonal() )[::-1]
                vq = vq[:,sort_ind]
                v = np.hstack( ( vp,vq) )
    
    
            sz = v.T.dot(Bi.full_Sz).dot(v).diagonal()
            s2 = v.T.dot(Bi.full_S2).dot(v).diagonal()
            lx = v.T.dot(brdms[bi]).dot(v).diagonal()
            h = v.T.dot(Bi.full_H).dot(v).diagonal()
    
            # compute transormation matrices to rotate ci vectors to new basis
            overlaps.append( Bi.vec().T.dot(v))
    
            #update Block vectors
            Bi.vecs[:,0:Bi.ss_dims[0]+Bi.ss_dims[1]] = v
            Bi.form_H()
            Bi.form_site_operators()
    
            print    
            print    
            print "   Eigenvalues of BRDM for ", lattice_blocks[bi]
            print "   -----------------------------------------------------------------------------"   
            print "   %-12s   %16s  %16s  %12s  %12s "%("Local State", "Occ. Number", "<H>", "<S2>", "<Sz>")
            for si,i in enumerate(lx):
                print "   %-12i   %16.8f  %16.8f  %12.4f  %12.4f "%(si,lx[si],h[si],abs(s2[si]),sz[si])
                #print "   %-4i   %16.8f  %16.8f  %16.4f "%(si,lx[si],h[si],sz[i])
        
        
        """
        # Rotate ci vectors to new basis
        """
        for si in range(0,last_vectors.shape[1]):
            for t in sorted(tucker_blocks):
                Tb = tucker_blocks[t]
                #print Tb.block_dims, last_vectors.shape
                v = last_vectors[Tb.start:Tb.stop, si]
                v.shape = Tb.block_dims
               
                for bi in range(0,n_blocks):
                    #print 
                    Bi = lattice_blocks[bi]
                    S = overlaps[bi]
                    
                    if Tb.address[bi] == 0:
                        S = S[0:Bi.np,:][:,0:Bi.np]
                    elif Tb.address[bi] == 1:
                        S = S[Bi.np:Bi.np+Bi.nq,:][:,Bi.np:Bi.np+Bi.nq]
                    #print ":: ", v.shape, S.shape
                    v = np.tensordot(v,S,axes=(0,0))
                v.shape = Tb.full_dim
                last_vectors[Tb.start:Tb.stop, si] = v
        #l,u = np.linalg.eigh(last_vectors.T.dot(last_vectors))
        #last_vectors = last_vectors.dot(u)
        last_vectors, tmp = np.linalg.qr(last_vectors)
    
    
    ref_norm = np.linalg.norm(v[tucker_blocks[0,-1].start:tucker_blocks[0,-1].stop])
    print
    print " Norm of Reference state %12.8f " %ref_norm
    
    if pt_order == 0:
        print
        print " %10s  %12s  %12s" %("Iteration", "Energy", "Delta")
        for ei,e in enumerate(energy_per_iter):
            if ei>0:
                print " %10i  %12.8f  %12.1e" %(ei,e,e-energy_per_iter[ei-1])
            else:
                print " %10i  %12.8f  %12s" %(ei,e,"")


    if pt_order >= 2:
        print
        print " %10s  %12s  %12s" %("Iteration", "Energy", "Delta")
        for ei,e in enumerate(energy_per_iter_lcc):
            if ei>0:
                print " %10i  %12.8f  %12.1e" %(ei,e,e-energy_per_iter_lcc[ei-1])
            else:
                print " %10i  %12.8f  %12s" %(ei,e,"")


if __name__== "__main__":

    size = (1,12)
    blocks = [[0,1,2,3],[4,5,6,7],[8,9,10,11]]
    n_p_states = [4,4,4]
    nbody_tucker(   j12 = hamiltonian_generator.make_2d_lattice(size=size,blocks=blocks),
                    blocks = blocks, 
                    n_p_states = n_p_states,
                    n_body_order =2,
                    pt_order =2,
                    #pt_mit =12,
                    n_roots = 1,
                    diis_start=1,
                    n_diis_vecs=6,      #max diis subspace size
                    pt_type = 'mp',
                )




