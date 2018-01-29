#! /usr/bin/env python
import numpy as np
import scipy
import scipy.linalg
import scipy.io
import copy as cp


def tucker_decompose(A,thresh,n_keep_max):
    """
    Perform tucker decomposition of input tensor
    
        Toss vectors with eigenvalues less than thresh 
        
        specify max number of vectors to keep (n=0: all)
    """
    dims = A.shape
    n_modes = len(dims)
    tucker_factors = []
    core = cp.deepcopy(A)
    n = n_keep_max

    if n_keep_max <= 0:
        n = max(dims)
    
    for sd,d in enumerate(dims):

        print " Contract A along index %4i" %(sd),
        print "   Dimension %4i" %(d)
            
        d_range = range(0,sd) 
        d_range.extend(range(sd+1,n_modes))
        
        AA = np.tensordot(A,A,axes=(d_range,d_range))
       
        l,U = np.linalg.eigh(AA)
        sort_ind = np.argsort(l)[::-1]
        l = l[sort_ind]
        U = U[:,sort_ind]


        keep = []
        trace_error = 0
        print "   Eigenvalues for mode %4i contraction:"%sd
        for si,i in enumerate(l):
            if(abs(i)>=thresh and si<n):
                print "   %-4i   %16.8f : Keep"%(si,i)
                keep.extend([si])
            else:
                #print "   %-4i   %16.8f : Toss"%(si,i)
                trace_error += i
        print "   ------------------------------------------" 
        print "   %5s  %16.8f : Error : %8.1e" %("Trace", AA.trace(),trace_error)
        print
        U = U[:,keep]

        #print U.shape, core.shape, sd
        core = np.tensordot(core,U,axes=(0,0))
        #print "core: ", core.shape
        #print "A   : ", A.shape
        tucker_factors.append(U)

    return core, tucker_factors


def tucker_decompose_proj(A,thresh,n_keep_max,proj):
    """
    Perform tucker decomposition of input tensor
    
        Toss vectors with eigenvalues less than thresh 
        
        specify max number of vectors to keep (n=0: all)
        
        proj is  a list of indices to not include in the gramian diagonalization. 
    """
    dims = A.shape
    n_modes = len(dims)
    tucker_factors = []
    core = cp.deepcopy(A)
    n = n_keep_max

    if n_keep_max <= 0:
        n = max(dims)
    
    for sd,d in enumerate(dims):

        print " Contract A along index %4i" %(sd),
        print "   Dimension %4i" %(d)
            
        d_range = range(0,sd) 
        d_range.extend(range(sd+1,n_modes))
        
        AA = np.tensordot(A,A,axes=(d_range,d_range))
        
        proj_d = proj[sd]

        AA = AA[ proj_d::,:][:,proj_d::]
        
        l,U = np.linalg.eigh(AA)
        sort_ind = np.argsort(l)[::-1]
        l = l[sort_ind]
        U = U[:,sort_ind]


        keep = []
        trace_error = 0
        print "   Eigenvalues for mode %4i contraction:"%sd
        for si,i in enumerate(l):
            if(abs(i)>=thresh and si<n):
                print "   %-4i   %16.8f : Keep"%(si,i)
                keep.extend([si])
            else:
                #print "   %-4i   %16.8f : Toss"%(si,i)
                trace_error += i
        print "   ------------------------------------------" 
        print "   %5s  %16.8f : Error : %8.1e" %("Trace", AA.trace(),trace_error)
        print

        padding = np.zeros((proj_d,len(keep)))
        
        U = np.vstack((padding,U[:,keep]))

        #print U.shape, core.shape, sd
        core = np.tensordot(core,U,axes=(0,0))
        #print "core: ", core.shape
        #print "A   : ", A.shape
        tucker_factors.append(U)

    return core, tucker_factors


def tucker_decompose_list(A,n_keep_list):
    """
    Perform tucker decomposition of input tensor
    
        n_keep_list [n_dim1, n_dim2, n_dim3, ...]

        if n_dimi < 0, it means to keep the complement to the n important components
    """
    dims = A.shape
    n_modes = len(dims)
    tucker_factors = []
    core = cp.deepcopy(A)
    n = n_keep_list
    print " Decompose code: ", n

    if len(n_keep_list) != len(dims):
        print " len(n_keep_list) != len(dims)"
        exit(-1)

    for sd,d in enumerate(dims):

        print " Contract A along index %4i" %(sd),
        print "   Dimension %4i" %(d)
            
        d_range = range(0,sd) 
        d_range.extend(range(sd+1,n_modes))
        
        AA = np.tensordot(A,A,axes=(d_range,d_range))
       
        l,U = np.linalg.eigh(AA)
        sort_ind = np.argsort(l)[::-1]
        l = l[sort_ind]
        U = U[:,sort_ind]


        keep = []
        print "   Eigenvalues for mode %4i contraction:"%sd
        for si,i in enumerate(l):

            if (n[sd]>0): #keep important
                if( si<n[sd]):
                    print "   %-4i   %16.8f : Keep"%(si,i)
                    keep.extend([si])
                else:
                    print "   %-4i   %16.8f : Toss"%(si,i)
            elif n[sd] < 0 : 
                if( si>=-n[sd]):
                    print "   %-4i   %16.8f : Keep"%(si,i)
                    keep.extend([si])
                else:
                    print "   %-4i   %16.8f : Toss"%(si,i)
            elif (n[sd]==0): #keep 
                print "   %-4i   %16.8f : Keep"%(si,i)
                keep.extend([si])
        print
        U = U[:,keep]

        #print U.shape, core.shape, sd
        core = np.tensordot(core,U,axes=(0,0))
        #print "core: ", core.shape
        #print "A   : ", A.shape
        tucker_factors.append(U)

    return core, tucker_factors



def tucker_recompose(core, tucker_factors, trans=0):
    """
    Recover original tensor from tucker decomposition 
    """
    print " Re-compose"
    dims = [] 
    A = cp.deepcopy(core)
    n_modes = len(dims)
  
    for sd,d in enumerate(tucker_factors):
        dims.append(d.shape[1])

    for sd,d in enumerate(dims):

        print " Contract A along index %4i " %(sd),
        print "   Dimension %4i" %(d)
            
        d_range = range(0,sd) 
        d_range.extend(range(sd+1,n_modes))
   
        
        if trans==0:
            A = np.tensordot(A,tucker_factors[sd],axes=(0,1))
        if trans==1:
            A = np.tensordot(A,tucker_factors[sd].T,axes=(0,1))

    return A


def transform_tensor(core, vectors, trans=0):
    """
    Perform a multi linear transformation on a tensor
    Core is the tensor to be transformed
    vectors is a list of matrices which define the transformation for each tensor index
    trans = 1: transpose the matrices in 'vectors'
                the resulting tensor will then be in the row space
    trans = 0: don't transpose the matrices in 'vectors'
                the resulting tensor will then be in the column space
    """
    dims = [] 
    A = cp.deepcopy(core)
    n_modes = len(dims)
  
    for sd,d in enumerate(vectors):
        dims.append(d.shape[1])

    for sd,d in enumerate(dims):

        print " Contract A along index %4i " %(sd),
        print "   Dimension %4i" %(d),
        print "   Operation: ", A.shape, " x(%i) "% sd,  vectors[sd].shape
            
        d_range = range(0,sd) 
        d_range.extend(range(sd+1,n_modes))
   
        
        if trans==0:
            A = np.tensordot(A,vectors[sd],axes=(0,1))
        if trans==1:
            A = np.tensordot(A,vectors[sd],axes=(0,0))

    print " %37s Final shape of tensor: " %"", A.shape
    return A


def form_gramian1(A, tuck_factors_A, B, tuck_factors_B, open_dims, trans1=0, trans2=0):
    """
    Form grammian tensor, where open_dims = 1
         __Ua--Ub__
        |          |
        |__      __|
        |          |
        A__Ua--Ub__B
        |          |
        |__Ua--Ub__|

    A is a ndarray numpy tensor
    B is a ndarray numpy tensor
    open_dims tells us which dimension to not contract over

    tuck_factors_A  :   a list of numpy matrices with row indices being the final basis, and column indices being the tucker
                        basis indices. trans1=1 switches this definition
    tuck_factors_B  :   a list of numpy matrices with row indices being the final basis, and column indices being the tucker
                        basis indices. trans1=1 switches this definition
    """
    assert(len(tuck_factors_A) == len(tuck_factors_B) )
    assert(len(A.shape) == len(tuck_factors_A) )
    assert(len(B.shape) == len(tuck_factors_B) )
    
    n_dims = len(A.shape)

    A_inds = range(n_dims) 
    B_inds = range(n_dims)
    
    tuck_factors_l = []
    tuck_factors_r = []
            
    for d in range(n_dims):
        print "   here: ", A.shape[d] ,tuck_factors_A[d].shape[1] 

    for d in range(n_dims):
        if (trans1,trans2) == (0,0):
           
            in_open = 0
            for dd in open_dims:
                if d==dd:
                    in_open = 1
            if in_open == 1:
                tuck_factors_l.extend([tuck_factors_A[d]])
                tuck_factors_r.extend([tuck_factors_B[d]])
                
                old_index = A_inds.index(d)
                A_inds.pop(old_index)
                
                old_index = B_inds.index(d)
                B_inds.pop(old_index)
                
                continue

            assert( tuck_factors_A[d].shape[0] == tuck_factors_B[d].shape[0]) 
            assert( A.shape[d] == tuck_factors_A[d].shape[1] )
            assert( B.shape[d] == tuck_factors_B[d].shape[1] )
            
            S_d = tuck_factors_A[d].T.dot(tuck_factors_B[d])

            #Contract this dimension's basis overlap with one of the two tensors, (which ever has the largest index)
            
            if S_d.shape[0] >= S_d.shape[1]:
                # A--S(Ai,Bi)--B  -->  AS(Bj)--B --> ASB
                
                A = np.tensordot(A,S_d,axes=(d,0))

                # keep info about moving current dimension to the end A(i,d,j,k) U(d,l) -> A(i,j,k,l)
                old_index = A_inds.index(d)
                A_inds.append( A_inds.pop(old_index) )

                print "A_inds:",  A_inds
            else:
                # A--S(Ai,Bi)--B  -->  A--S(Aj)B --> ASB
                
                B = np.tensordot(B,S_d,axes=(d,0))

                old_index = B_inds.index(d)
                B_inds.append( A_inds.pop(old_index) )
                print "B_inds:",  B_inds
        else:
            print "Transposes NYI"
            exit(-1)
    
    tuck_factors_C = []
    tuck_factors_C.extend( [tuck_factors_l] ) 
    tuck_factors_C.extend( [tuck_factors_r] ) 

    print " Forming the gramiam:  ", "A(",A_inds,") B(",B_inds,")"
    print "   ", A.shape
    print "   ", B.shape
    AA= np.tensordot(A,B,axes=(A_inds,B_inds))
    print " = ", AA.shape
    AA = tuck_factors_l[0].dot(AA).dot(tuck_factors_r[0].T)
    return AA
    #return AA, tuck_factors_C

           
def form_1fdm(A, B, open_dims):
    """
        This assumes that A and B share a Tucker basis, such that we don't need to actually contract with the tucker factors
    
    Form grammian tensor, where open_dims = 1
         __Ua--Ub__
        |          |
        |__      __|
        |          |
        A__Ua--Ub__B
        |          |
        |__Ua--Ub__|

    A is a ndarray numpy tensor
    B is a ndarray numpy tensor
    open_dims tells us which dimension to not contract over

    """
    n_dims = len(A.shape)

    A_inds = range(n_dims) 
    B_inds = range(n_dims)

    ind_contract_list = []
            
    for d in range(n_dims):
        in_open = 0
        for dd in open_dims:
            if d==dd:
                in_open = 1
        if in_open != 1:
            ind_contract_list.extend([d])

    #print A.shape, B.shape, ind_contract_list
    AB= np.tensordot(A,B,axes=(ind_contract_list,ind_contract_list))
    return AB

    
   


"""
def transform_tucker_block(self, A, tb_l, tb_r):
    ''' 
    Perform a multi linear transformation on a tensor
    tb_l is the current tucker_block basis 
    tb_r is the destination tucker_block basis 
    A is the tensor to be transformed

    This assumes that the lattice_blocks are the same
    '''

    for bi in range(0,self.n_blocks):
        Bi = self.blocks[bi]
        Bj = self.blocks[bi]
        S = Bi.vec().T.dot(Bj.vec())
        A = np.tensordot(A,S,axes=(0,0))
"""
    
#def form_tot_gramiam1(v,P_dims, Q_dims, QQ_dims, blocks, 
