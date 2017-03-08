#! /usr/bin/env python
import numpy as np
import scipy
import scipy.linalg
import scipy.io
import copy as cp
import argparse

from hdvv import *
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
parser.add_argument('-s','--save', default=False, action="store_true", help='Save the Hamiltonian and S2 matrices', required=False)
parser.add_argument('-r','--read', default=False, action="store_true", help='Read the Hamiltonian and S2 matrices', required=False)
parser.add_argument('-hdvv','--hamiltonian', type=str, default="heis_blocks.m", help='File containing matrix of Hamiltonian', required=False)
parser.add_argument('-s2','--s2', type=str, default="heis_s2.m", help='File containing matrix of s2', required=False)
args = vars(parser.parse_args())
#
#   Let minute specification of walltime override hour specification

j12 = np.loadtxt(args['j12'])
lattice = np.loadtxt(args['lattice']).astype(int)
blocks = np.loadtxt(args['blocks']).astype(int)
n_sites = len(lattice)
n_blocks = len(blocks)

print " j12:\n", j12
print " lattice:\n", lattice 
print " blocks:\n", blocks
print " n_blocks:\n", n_blocks

H_tot = np.array([])
S2_tot = np.array([])
H_dict = {}

if args['read']:
    H_tot = np.loadtxt(args['hamiltonian'])
    S2_tot = np.loadtxt(args['s2'])
else:
    H_tot, H_dict, S2_tot = form_hdvv_H(lattice,j12)

if args['save']==True:
    np.savetxt("heis_hamiltonian.m",H_tot)
    np.savetxt("heis_s2.m",S2_tot)


print " Diagonalize Hamiltonian (%ix%i):\n" %(H_tot.shape[0],H_tot.shape[0])
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

# reshape eigenvector into tensor
dims_0 = []
for bi,b in enumerate(blocks):
    block_dim = np.power(2,b.shape[0])
    dims_0.extend([block_dim])

v0 = np.reshape(v0,dims_0)

n_p_space = 1

Acore, Atfac = tucker_decompose(v0,0,n_p_space)
U0 = Atfac[0] 
site_states = []
for bi,b in enumerate(blocks):
    # form block hamiltonian
    j_b = np.zeros([len(b),len(b)])
    j_b = j12[:,b][b]
    lat_b = lattice[b]
    H_b, tmp, S2_b = form_hdvv_H(lat_b,j_b)
    
    l_b,v_b = np.linalg.eigh(H_b)
    site_states.extend([v_b])

#    ji[0,1] = j12[ii,jj]
#    ji[0,2] = j12[ii,kk]
#    ji[1,2] = j12[jj,kk]
#    ji = ji + ji.transpose()
#    lati = np.array([lattice[ii],lattice[jj],lattice[kk]])
#    Hi, tmp, S2i = form_hdvv_H(lati,ji)
#    li,vi = np.linalg.eigh(Hi)
#    site_states.extend([vi])

print
U0 = site_states[0]
print U0
print


exit(-1)


B = tucker_recompose(Acore,Atfac)
print "\n Norm of Error tensor due to compression:  %12.3e\n" %np.linalg.norm(B-v0)

#1-Body
if 0:
    for si,i in enumerate(v0.shape):
        n_modes = len(v0.shape)
        dims2 = np.ones(n_modes)*n_p_space
        dims2[si]=-n_p_space
        Bcore, Btfac = tucker_decompose_list(v0,dims2)
        B = B + tucker_recompose(Bcore,Btfac)

#2-Body
if 0:
    n_modes = len(v0.shape)
    dims = v0.shape
    for si,i in enumerate(dims):
        for sj,j in enumerate(dims):
            if si>sj:
                dims2 = np.ones(n_modes)*n_p_space
                dims2[si]=-n_p_space
                dims2[sj]=-n_p_space
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
                        dims2 = np.ones(n_modes)*n_p_space
                        dims2[si]=-n_p_space
                        dims2[sj]=-n_p_space
                        dims2[sk]=-n_p_space
                        Bcore, Btfac = tucker_decompose_list(v0,dims2)
                        B = B + tucker_recompose(Bcore,Btfac)


B = np.reshape(B,[np.power(2,n_sites)])
BB = np.dot(B,B)

Bl = np.dot(B.transpose(),np.dot(H_tot,B))
Bs = np.dot(B.transpose(),np.dot(S2_tot,B))
print
print " Energy  Error due to compression    :  %12.8f - %12.8f = %12.8f" %(Bl/BB,l[0],Bl/BB-l[0])
print " Spin Error due to compression       :  %12.8f %12.8f" %(S2_eig[0,0],Bs/BB)
print " Norm of compressed vector           :  %12.8f"%(BB)



#for si,i in enumerate(
