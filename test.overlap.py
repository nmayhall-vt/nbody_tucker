#! /usr/bin/env python

import numpy as np
import scipy
import scipy.linalg
import scipy.io
import copy as cp
import argparse
import scipy.sparse
import scipy.sparse.linalg

import tucker

np.random.seed(2)

block_dim = 4
n_blocks = 4
n_pa = (1,2,3,4)
n_pb = (1,3,1,3)
full_dim = np.power(block_dim,n_blocks)

A = scipy.linalg.orth(np.random.rand(full_dim,1))
B = scipy.linalg.orth(np.random.rand(full_dim,1))



dims = np.ones(n_blocks,dtype=int)*block_dim

A.shape = dims
B.shape = dims

a,ua = tucker.tucker_decompose_list(A,n_pa)
b,ub = tucker.tucker_decompose_list(B,n_pb)

A = tucker.tucker_recompose(a,ua)
B = tucker.tucker_recompose(b,ub)


A.shape = (full_dim,1)
B.shape = (full_dim,1)

#print A.T.dot(A)
print "   Overlap full       = %16.12f" % A.T.dot(B)[0,0]
print "   Overlap compressed = %16.12f" % tucker.form_overlap(a,ua,b,ub)
#print B.T.dot(B)
