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

def printm(m):
    # {{{
    """ print matrix """
    for r in m:
        for ri in r:
            print "% 10.13f" %ri,
        print
    # }}}

def vibin_pt2(n_blocks,lattice_blocks, tucker_blocks, tucker_blocks_pt,n_body_order, l, v, j12, pt_type):
    """# {{{

        E(2) = v_sA H_AX [D_XX - E_s^0]^-1 H_XA v_As

             = v_sA H_AX

        pt_type: mp or en
                
                 mp uses H1 as zeroth-order Hamiltonian
                 en uses the diagonal of H for zeroth-order
    """
    do_2b_diag = 0

    if pt_type == "mp":
        do_2b_diag = 0
    elif pt_type == "en":
        do_2b_diag = 1
    else:
        print " Bad value for pt_type"
        exit(-1)

    n_roots = v.shape[1]
    e2 = np.zeros((n_roots))
    dim_tot_X = 0
    dim_tot_A = 0
    for t_l in sorted(tucker_blocks_pt):
        tb_l = tucker_blocks_pt[t_l]
        dim_tot_X += tb_l.full_dim
    for t_l in sorted(tucker_blocks):
        tb_l = tucker_blocks[t_l]
        dim_tot_A += tb_l.full_dim
    
    H_Xs = np.zeros((dim_tot_X, n_roots))
    D_X = np.zeros((dim_tot_X))
    
    for t_l in sorted(tucker_blocks_pt):
        tb_l = tucker_blocks_pt[t_l]
        D_X[tb_l.start:tb_l.stop] = build_H_diag(lattice_blocks, tb_l, tb_l, j12, do_2b_diag)

        for t_r in sorted(tucker_blocks):
            tb_r = tucker_blocks[t_r]
            hv,s2v = pt_build_H1v(lattice_blocks, tb_l, tb_r, j12,v[tb_r.start:tb_r.stop,:])
            H_Xs[tb_l.start:tb_l.stop,:] += hv

    DHv = np.array(())
    for s in range(0, n_roots):
        dx = 1/(l[s]-D_X)
        DHv = np.multiply(dx, H_Xs[:,s])
        e2[s] = H_Xs[:,s].T.dot(DHv)
    DHv = DHv.reshape(dim_tot_X,1)

    """
    Vibin adding stuff
    Insights:
    The variable 'l'  is the NB0 part. nroots is 1 for single referenece.
    tucker_blocks_pt always forms  +2 tuceker vectors from NBn.
    """

    v = np.append(v,H_Xs).reshape(dim_tot_X+dim_tot_A,1)
    H_Xv = np.zeros((dim_tot_X, n_roots))
    H_Xv,temp = H1_build_tucker_blocked_sigma(n_blocks,tucker_blocks_pt, lattice_blocks, n_body_order+2, j12,DHv)

    H_XA = np.zeros((dim_tot_X, n_roots))
    for t_l in sorted(tucker_blocks_pt):
        tb_l = tucker_blocks_pt[t_l]
        for t_r in sorted(tucker_blocks):
            tb_r = tucker_blocks[t_r]
            H_XA[tb_l.start:tb_l.stop, tb_r.start:tb_r.stop],tmp = build_H(lattice_blocks, tb_l, tb_r, j12)

    #####Vibin checking SZ for diagonal##########################
    #Szzz = np.zeros((dim_tot_X,dim_tot_X)) 
    #for t_l in sorted(tucker_blocks_pt):
    #    tb_l = tucker_blocks_pt[t_l]
    #    for t_r in sorted(tucker_blocks_pt):
    #        tb_r = tucker_blocks_pt[t_r]
    #        Szzz[tb_l.start:tb_l.stop, tb_r.start:tb_r.stop],tmp = pt_build_H0Sz(lattice_blocks, tb_l, tb_r, j12)
    #printm(Szzz)
    #Szz = np.zeros((dim_tot_X,1))
    #for t_l in sorted(tucker_blocks_pt):
    #    tb_l = tucker_blocks_pt[t_l]
    #    for t_r in sorted(tucker_blocks):
    #        tb_r = tucker_blocks[t_r]
    #        Szz[tb_l.start:tb_l.stop, tb_r.start:tb_r.stop],tmp = pt_build_H0Sz(lattice_blocks, tb_l, tb_r, j12)
    #printm(Szz)
    ################################################

    res2 = np.multiply(dx,H_XA[:,0])
    res2 = res2.reshape(dim_tot_X,1)
    res4 = np.multiply(dx,H_Xv[:,0]).reshape(dim_tot_X,1)
    H_Xv4,temp = H1_build_tucker_blocked_sigma(n_blocks,tucker_blocks_pt, lattice_blocks, n_body_order+2, j12,H_Xv)
    res5 = np.multiply(dx,H_Xv4[:,0]).reshape(dim_tot_X,1)

    #print res.shape
    Hvv,Svv = H1_build_tucker_blocked_sigma(n_blocks,tucker_blocks_pt, lattice_blocks, n_body_order+2, j12,res4)
    Hmp3,Smp3 = H1_build_tucker_blocked_sigma(n_blocks,tucker_blocks_pt, lattice_blocks, n_body_order+2, j12,res2)
    Hmp5,Smp5 = H1_build_tucker_blocked_sigma(n_blocks,tucker_blocks_pt, lattice_blocks, n_body_order+2, j12,res5)
    e4 = np.dot(DHv.T, Hvv)
    e3 = np.dot(DHv.T, Hmp3)
    e5 = np.dot(DHv.T, Hmp5)
    print("pt 2 % 10.15f " %e2)
    print("pt 3 % 10.15f " %e3)
    print("pt 4 % 10.15f " %e4)
    print("pt 5 % 10.15f " %e5)
    #print " %5s    %16s  %16s  %12s" %("State","Energy PT3","Relative","<S2>")
    return e2

def PT_nth_vector(n_blocks,lattice_blocks, tucker_blocks, tucker_blocks_pt,n_body_order, l, v, j12, pt_type):
    do_2b_diag = 0

    if pt_type == "lcc":
        do_2b_diag = 0
    else:
        print " Bad value for pt_type"
        exit(-1)
    """

    C(n) = R * V * C(n-1) - sum_k  R * E(k) * C(n-k)
    
    The second term is omitted coz it is the non renormalised term and makes the method non size consistent.

    """
    n = 40 #the order of wf

    dim_tot_X = 0 #Dim of orthogonal space
    dim_tot_A = 0 #Dim of model space
    for t_l in sorted(tucker_blocks_pt):
        tb_l = tucker_blocks_pt[t_l]
        dim_tot_X += tb_l.full_dim
    for t_l in sorted(tucker_blocks):
        tb_l = tucker_blocks[t_l]
        dim_tot_A += tb_l.full_dim

    D_X = np.zeros((dim_tot_X))     #diagonal of X space
    H_Xs = np.zeros((dim_tot_X, 1))
    E_mpn = np.zeros((n+2))         #PT energy
    v_n = np.zeros((dim_tot_X,n+1))   #list of PT vectors
       

    #Forming the resolvent

    for t_l in sorted(tucker_blocks_pt):
        tb_l = tucker_blocks_pt[t_l]
        D_X[tb_l.start:tb_l.stop] = build_H_diag(lattice_blocks, tb_l, tb_l, j12, do_2b_diag)
        for t_r in sorted(tucker_blocks):
            tb_r = tucker_blocks[t_r]
            hv,s2v = pt_build_H1v(lattice_blocks, tb_l, tb_r, j12,v[tb_r.start:tb_r.stop,:])
            H_Xs[tb_l.start:tb_l.stop,:] += hv

    res = 1/(l - D_X) 



    #First order wavefunction
    H_Xs = H_Xs.reshape(dim_tot_X)
    v_n[: ,0] = np.multiply(res, H_Xs)

    #Second order energy
    E_mpn[1] = np.dot(H_Xs.T,v_n[:,0])
    emp2 = E_mpn[1]
    

    vv_n = 0
    wigner = np.zeros((2*n+2))
    for i in range(1,n):
        h1,S1 = H1_build_tucker_blocked_sigma(n_blocks,tucker_blocks_pt, lattice_blocks, n_body_order+2, j12,v_n[:,i-1].reshape(dim_tot_X,1))
        vv_n1 = h1.reshape(dim_tot_X)

        ##RENORMALISED TERMS
        #for k in range(0,i):
        #   vv_n += np.multiply(E_mpn[k],v_n[:,i-k-2].reshape(dim_tot_X))

        v_n[:,i] = np.multiply(res,vv_n1-vv_n) 
        E_mpn[i+1] = np.dot(H_Xs.T, v_n[:,i])
        wigner[i] = E_mpn[i]
        
        if i >= n/2-1:
            wigner[2*i+1] = np.dot(vv_n1,v_n[:,i]) 
            h2,S2 = H1_build_tucker_blocked_sigma(n_blocks,tucker_blocks_pt, lattice_blocks, n_body_order+2, j12,v_n[:,i].reshape(dim_tot_X,1))
            vv_n2 = h2.reshape(dim_tot_X)
            wigner[2*i+2] = np.dot(vv_n2,v_n[:,i])

        ##EVEN: Wigner method to get 2n perturbation correction to energy
        #h2,S2 = H1_build_tucker_blocked_sigma(n_blocks,tucker_blocks_pt, lattice_blocks, n_body_order+2, j12,v_n[:,i].reshape(dim_tot_X,1))
        #
        #wigner[2*i] = np.dot(v_n[:,i-1],h2)
        #for k in range(1,2*i-2):
        #   mini = min(i,2*i-k-1) 
        #   maxi = max(1,i-k) 
        #   for m in range(maxi,mini):
        #       wigner[2*i] += wigner[k] * np.dot(v_n[:,m].T,v_n[:,2*i-m-k])
        #     
        ##ODD: Wigner method to get 2n+1 perturbation correction to energy
        #wigner[2*i+1] = np.dot(v_n[:,i],h2)
        #for k in range(1,2*i-1):
        #   mini = min(i+1,2*i-k) 
        #   maxi = max(1,i+1-k) 
        #   for m in range(maxi,mini):
        #       wigner[2*i+1] += wigner[k] * np.dot(v_n[:,m].T,v_n[:,2*i-m-k])


    print("Nth order pertubations:") 
    #for i in range(0,n):
    #    print("Order: %2d  % 10.15f " %(i+1, E_mpn[i]))
    #E_mp = l[0] 
    #for i in range(0,n):
    #    E_mp += E_mpn[i]
    #    print("Order: %d  % 10.15f " %(i+1, E_mp))
    for i in range(0,2*n):
        print("Order: %2d  % 10.15f " %(i+1, wigner[i]))


def pt_build_H1(blocks,tb_l, tb_r,j12):
  # {{{
    """
    Build the Hamiltonian between two tensor blocks, tb_l and tb_r, without ever constructing a full hilbert space

    Forming H1
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


def pt_build_H1v(blocks,tb_l, tb_r,j12,v):
  # {{{
    """
    Build the Hamiltonian vector product between two tensor blocks, tb_l and tb_r, without ever constructing a full hilbert space
    
    Form the H1.v 
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
    
    n_sig = v.shape[1]  # number of sigma vectors 

    Hv  = np.zeros((tb_l.full_dim,n_sig))
    S2v = np.zeros((tb_l.full_dim,n_sig))


    #   Add up all the one-body contributions, making sure that the results is properly dimensioned for the 
    #   target subspace

    if len(different) == 0:

        assert(tb_l.full_dim == tb_r.full_dim)
        full_dim = tb_l.full_dim
        #<abcd|H1+H2+H3+H4|abcd>
        #
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
        #Hv  += h1v.reshape(tb_l.full_dim, n_sig)
        #S2v += s1v.reshape(tb_l.full_dim, n_sig)
        
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



def H1_build_tucker_blocked_sigma(n_blocks,tucker_blocks, lattice_blocks, n_body_order, j12,v):
    #{{{

    """
        s   = <abcd| H | aBcD> v_aBcD
        
            = <bd|H24|BD>v_BD I1*v_a I3v_c
            
    A function to give part of the nth PT vector by doing H1.v
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

            #print " Here:", tb_l, tb_r
            #if tb_l.start == tb_r.start:
            #    continue 
            hv,s2v = pt_build_H1v(lattice_blocks, tb_l, tb_r, j12,v_r)
            #h,s2 = build_H(lattice_blocks, tb_l, tb_r, j12)
            
            Hv[tb_l.start:tb_l.stop,:] += hv
            S2v[tb_l.start:tb_l.stop,:] += s2v
            #H[tb_l.start:tb_l.stop, tb_r.start:tb_r.stop] = build_H(lattice_blocks, tb_l, tb_r, j12)
            #H[tb_r.start:tb_r.stop, tb_l.start:tb_l.stop] = H[tb_l.start:tb_l.stop, tb_r.start:tb_r.stop].T

    return Hv, S2v


def build_dimer_newH0Sz(tb_l, tb_r, Bi, Bj,j12):
    """
    A build dimer block to confirm if the Sz.Sz in between two blocks are diagonal or not.
    Not in use. Also Sz.Sz not diagonal.
    """
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
        szi = Bi.Szi_ss(si,space_i_l,space_i_r)
        
        for sj in Bj.sites:
            space_j_l = tb_l.address[Bj.index]
            space_j_r = tb_r.address[Bj.index]
            szj = Bj.Szi_ss(sj,space_j_l,space_j_r)
           
            #h12  -= j12[si,sj] * np.kron(spi, smj)
            #h12  -= j12[si,sj] * np.kron(smi, spj)
            #h12  -= j12[si,sj] * np.kron(szi, szj) * 2
           
            s1s2 = 2 * np.tensordot(szi,szj, axes=0)
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

def pt_build_H0Sz(blocks,tb_l, tb_r,j12):
  # {{{
    """
    Build the Hamiltonian between two tensor blocks, tb_l and tb_r, without ever constructing a full hilbert space

    A function formed to check if the Sz.Sz block is diagonal. Currently not in use. Also it is not diagonal
      
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
        
        
        #   <ab|H12|ab> Ic Id
        # + <ac|H13|ac> Ib Id
        # + Ia <bc|H23|bc> Id + etc
        
        for bi in range(0,n_blocks):
            for bj in range(bi+1,n_blocks):
                Bi = blocks[bi]
                Bj = blocks[bj]
                dim_e = full_dim / tb_l.block_dims[bi] / tb_l.block_dims[bj]

                #build full Hamiltonian on sublattice
                h2,s2 = build_dimer_newH0Sz(tb_l, tb_r, Bi, Bj, j12)
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
            h2,s2 = build_dimer_newH0Sz(tb_l, tb_r, Bi, Bj, j12)
          
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
            h2,s2 = build_dimer_newH0Sz(tb_l, tb_r, Bi, Bj, j12)
          
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
        h2,s2 = build_dimer_newH0Sz(tb_l, tb_r, Bi, Bj, j12)
       
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
