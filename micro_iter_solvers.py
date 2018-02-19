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
from davidson import *
from pt import *


def do_variational_microiteration_update(args, n_blocks, tucker_blocks, lattice_blocks, n_order, j12, dav_thresh,it,last_vectors):

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
            hv, s2v = build_tucker_blocked_sigma(n_blocks, tucker_blocks, lattice_blocks, n_order, j12, dav.vec_curr) 
            dav.sig_curr = hv
    
        if args['dav_precond']:
            hv_diag = build_tucker_blocked_diagonal(n_blocks, tucker_blocks, lattice_blocks, n_order, j12, 0) 
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
        
    l = dav.eigenvalues()
    v = dav.eigenvectors()

    return l,v

