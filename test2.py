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

n_sites = 8 
lattice = np.ones(n_sites)

np.random.seed(2)
j12 = np.random.rand(n_sites,n_sites)
j12 = j12 - .5

j12 = j12 + j12.transpose()
np.fill_diagonal(j12,0)

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

if 1:
    i = -1
    j = -.1
    j12 = np.zeros([n_sites,n_sites])+j
    j12 = np.zeros([n_sites,n_sites])+j*(np.random.rand(n_sites,n_sites)*2-.5)

    j12[0,1]=i
    j12[2,3]=i
    j12[4,5]=i
    j12[6,7]=i
    
    j12 = .5*(j12 + j12.transpose())
    np.fill_diagonal(j12,0)
    print j12
    
H_tot, H_list, S2_tot = form_hdvv_H(lattice,j12)
   
    
l,v = np.linalg.eigh(H_tot)
S2_eig = np.dot(v.transpose(),np.dot(S2_tot,v))

au2ev = 27.21165;
au2cm = 219474.63;

convert = au2ev/au2cm;		# convert from wavenumbers to eV
convert = 1;			# 1 for wavenumbers
print " %5s    %12s  %12s  %12s" %("State","Energy","Relative","<S2>")
for si,i in enumerate(l):
    print " %5i =  %12.8f  %12.8f  %12.8f" %(si,i*convert,(i-l[0])*convert,S2_eig[si,si])
    if si>10:
        break

v0 = v[:,0]

v0 = np.reshape(v0,4*np.ones(n_sites/2).astype(int))
#v0 = np.reshape(v0,16*np.ones(n_sites/4).astype(int))
#print v[:,0]
#print
#print v0

Acore, Atfac = tucker_decompose(v0,0,1)
B = tucker_recompose(Acore,Atfac)
print "\n Norm of Error tensor due to compression:  %12.3e\n" %np.linalg.norm(B-v0)

#1-Body
if 0:
    for si,i in enumerate(v0.shape):
        n_modes = len(v0.shape)
        dims2 = np.ones(n_modes)
        dims2[si]=-1
        Bcore, Btfac = tucker_decompose_list(v0,dims2)
        B = B + tucker_recompose(Bcore,Btfac)

#2-Body
if 0:
    n_modes = len(v0.shape)
    dims = v0.shape
    for si,i in enumerate(dims):
        for sj,j in enumerate(dims):
            if si>sj:
                dims2 = np.ones(n_modes)
                dims2[si]=-1
                dims2[sj]=-1
                Bcore, Btfac = tucker_decompose_list(v0,dims2)
                B = B + tucker_recompose(Bcore,Btfac)

#3-Body
if 0:
    dims = v0.shape
    n_modes = len(dims)
    for si,i in enumerate(dims):
        for sj,j in enumerate(dims):
            if si>sj:
                for sk,k in enumerate(dims):
                    if sj>sk:
                        dims2 = np.ones(n_modes)
                        dims2[si]=-1
                        dims2[sj]=-1
                        dims2[sk]=-1
                        Bcore, Btfac = tucker_decompose_list(v0,dims2)
                        B = B + tucker_recompose(Bcore,Btfac)


B = np.reshape(B,[np.power(2,n_sites)])
BB = np.dot(B,B)

Bl = np.dot(B.transpose(),np.dot(H_tot,B))
Bs = np.dot(B.transpose(),np.dot(S2_tot,B))
print
print " Energy  Error due to compression    :  %12.8f - %12.8f = %12.8f" %(Bl,l[0],Bl-l[0])
print " Energy/vv  Error due to compression :  %12.8f - %12.8f = %12.8f" %(Bl/BB,l[0],Bl/BB-l[0])
print " Spin Error due to compression       :  %12.8f %12.8f" %(S2_eig[0,0],Bs/BB)
print " Norm of compressed vector           :  %12.8f"%(BB)



#for si,i in enumerate(
