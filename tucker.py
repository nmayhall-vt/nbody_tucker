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
        print "   Eigenvalues for mode %4i contraction:"%sd
        for si,i in enumerate(l):
            if(abs(i)>=thresh and si<n):
                print "   %-4i   %16.8f : Keep"%(si,i)
                keep.extend([si])
            else:
                print "   %-4i   %16.8f : Toss"%(si,i)
        print
        U = U[:,keep]

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

