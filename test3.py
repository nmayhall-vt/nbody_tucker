#! /usr/bin/env python
import numpy as np
import scipy
import scipy.linalg
import scipy.io
import copy as cp
import argparse

from hdvv import *

def printm(m):
    # {{{
    """ print matrix """
    for r in m:
        for ri in r:
            print "%10.1e" %ri,
        print
    # }}}

def get_guess_vectors(lattice, j12, blocks, n_p_states):
    # {{{
    print " Generate initial guess vectors"
    p_states = []
    q_states = []
    for bi,b in enumerate(blocks):
	# form block hamiltonian
	j_b = np.zeros([len(b),len(b)])
	j_b = j12[:,b][b]
	lat_b = lattice[b]
	H_b, tmp, S2_b = form_hdvv_H(lat_b,j_b)
    
	l_b,v_b = np.linalg.eigh(H_b)
	p_states.extend([v_b[:,0:n_p_states]])
        q_states.extend([v_b[:,n_p_states::]])
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
    H_b, tmp, S2_b = form_hdvv_H(lat_b,j_b)
    return H_b
    # }}}


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
parser.add_argument('-np','--n_p_space', type=int, default="1", help='Number of vectors in block P space', required=False)
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

n_p_states = args['n_p_space'] 

Acore, Atfac = tucker_decompose(v0,0,n_p_states)
U0 = Atfac[0] 

# Get initial compression vectors 
p_states, q_states = get_guess_vectors(lattice, j12, blocks, n_p_states)

# |Ia,Ib,Ic> P(Ia,a) P(Ib,b) P(Ic,c) = |abc>    : |PPP>
#
# |Ia,Ib,Ic> Q(Ia,A) P(Ib,b) P(Ic,c) = |Abc>    : |QPP>
# |Ia,Ib,Ic> P(Ia,a) Q(Ib,B) P(Ic,c) = |aBc>    : |PQP>
#
# |Ia,Ib,Ic> Q(Ia,A) Q(Ib,B) P(Ic,c) = |ABc>    : |QQP>
#
#<abc|Ha+Hb+Hc+Hab+Hac+Hbc|abc>
#
#<a|Ha|a><bc|bc> = <a|Ha|a>
#<ab|Hab|ab><c|c> = <ab|Hab|ab>
#

Ha = form_superblock_hamiltonian(lattice, j12, blocks, [0])
Hb = form_superblock_hamiltonian(lattice, j12, blocks, [1])
Hc = form_superblock_hamiltonian(lattice, j12, blocks, [2])
Hab = form_superblock_hamiltonian(lattice, j12, blocks, [0,1])
Hac = form_superblock_hamiltonian(lattice, j12, blocks, [0,2])
Hbc = form_superblock_hamiltonian(lattice, j12, blocks, [1,2])
Habc = form_superblock_hamiltonian(lattice, j12, blocks, [0,1,2])

Ia = np.eye(Ha.shape[0])
Ib = np.eye(Hb.shape[0])
Ic = np.eye(Hc.shape[0])

a = p_states[0]
b = p_states[1]
c = p_states[2]
ab = np.kron(p_states[0],p_states[1])
ac = np.kron(p_states[0],p_states[2])
bc = np.kron(p_states[1],p_states[2])
AB = np.kron(q_states[0],q_states[1])

Hab = Hab - np.kron(Ha,np.eye(Hb.shape[0])) - np.kron(np.eye(Ha.shape[0]),Hb) 
Hac = Hac - np.kron(Ha,np.eye(Hc.shape[0])) - np.kron(np.eye(Ha.shape[0]),Hc) 
Hbc = Hbc - np.kron(Hb,np.eye(Hc.shape[0])) - np.kron(np.eye(Hb.shape[0]),Hc) 

Ia = np.dot(a.transpose(),np.dot(Ia,a))
Ib = np.dot(b.transpose(),np.dot(Ib,b))
Ic = np.dot(c.transpose(),np.dot(Ic,c))
Ha = np.dot(a.transpose(),np.dot(Ha,a))
Hb = np.dot(b.transpose(),np.dot(Hb,b))
Hc = np.dot(c.transpose(),np.dot(Hc,c))
Hab = np.dot(ab.transpose(),np.dot(Hab,ab))
Hac = np.dot(ac.transpose(),np.dot(Hac,ac))
Hbc = np.dot(bc.transpose(),np.dot(Hbc,bc))

H0 = np.kron(Ha,np.kron(Ib,Ic)) + np.kron(Ia,np.kron(Hb,Ic)) + np.kron(Ia,np.kron(Ib,Hc)) 
H0 += np.kron(Hab,Ic) 
H0 += np.kron(Ia,Hbc) 
tmp = np.kron(Hac,Ib)
tmp = tmp.reshape([2,2,2,2,2,2])
tmp = tmp.transpose((0,2,1,3,5,4))
tmp = tmp.reshape(8,8)
H0 += tmp

abc = np.kron(p_states[0],np.kron(p_states[1],p_states[2]))

printm(H0) 
Habc = np.dot(abc.transpose(),np.dot(Habc,abc))
print 
printm(Habc - H0)

l0,v0 = np.linalg.eigh(H0)
l1,v1 = np.linalg.eigh(Habc)
print np.array([l0,l1])
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
