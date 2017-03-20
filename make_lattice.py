#! /usr/bin/env python
import numpy as np
import scipy
import scipy.linalg
import scipy.io
import copy as cp

from hdvv import *
"""
Test forming HDVV Hamiltonian and projecting onto "many-body tucker basis"
"""

n_sites = 0
j12 = np.array([])
blocks = []
if 0:
    j12 = np.loadtxt("heis_j12.m")

if 0:
    i = -5
    j = -1
    j12 = np.zeros([n_sites,n_sites])+j*np.random.rand(n_sites,n_sites)
    j12 = np.zeros([n_sites,n_sites])+j

    j12[0,1]=i
    j12[0,2]=i
    j12[0,3]=i
    j12[1,2]=i
    j12[1,3]=i
    j12[2,3]=i
   
    j12[4,5]=i
    j12[4,6]=i
    j12[4,7]=i
    j12[5,6]=i
    j12[5,7]=i
    j12[6,7]=i
    
    j12 = .5*(j12 + j12.transpose())
    np.fill_diagonal(j12,0)
    print j12

if 0:
    i = -1
    j = -.5
    j12 = np.zeros([n_sites,n_sites])+j*(np.random.rand(n_sites,n_sites)-.5)
    j12 = np.zeros([n_sites,n_sites])+j

    j12[0,1]=i
    j12[0,2]=i
    j12[1,2]=i
    j12[3,4]=i
    j12[3,5]=i
    j12[4,5]=i
    j12[6,7]=i
    j12[6,8]=i
    j12[7,8]=i
    
    j12 = (j12 + j12.transpose())
    np.fill_diagonal(j12,0)
    print j12

if 1:
    b = 2 # block size
    n_blocks = 4
    n_sites = n_blocks*b
    
    i = -1
    j = -.1
    j12 = np.zeros([n_sites,n_sites])+j
    j12 = np.zeros([n_sites,n_sites])+j*(np.random.rand(n_sites,n_sites)-.5)*2
    bi = 0
    for f in range(n_blocks):
        a = range(bi,bi+b)
        blocks.extend([a])
        j12[bi:bi+b, bi:bi+b] = i*np.ones([b,b]) * np.power(-1,bi) 
        #print range(bi,bi+b)
        bi += b
    
    j12 = .5*(j12 + j12.transpose())
    np.fill_diagonal(j12,0)
    
    lattice = np.ones(n_blocks*b)
    
    for row in j12:
        for col in row:
            print "%5.2f" %col,
        print 
    print blocks

np.savetxt("heis_j12.m",j12)
np.savetxt("heis_lattice.m",lattice, fmt='%i')
np.savetxt("heis_blocks.m",blocks, fmt='%i')


