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

def form_compressed_hamiltonian(vecs,Hi,Hij):
    dim = 1 # dimension of subspace
    dims = [] # list of mode dimensions (size of hilbert space on each fragment) 
    for vi,v in enumerate(vecs):
        dim = dim*v.shape[1]
        dims.extend([v.shape[1]])
    H = np.zeros((dim,dim))
    Htest = np.zeros((dim,dim))
    n_dims = len(dims)


    H1 = cp.deepcopy(Hi)
    H2 = cp.deepcopy(Hij)
    for vi,v in enumerate(vecs):
        H1[vi] = np.dot(v.transpose(),np.dot(Hi[vi],v))

    for vi,v in enumerate(vecs):
        for wi,w in enumerate(vecs):
            if wi>vi:
                vw = np.kron(v,w)
                H2[(vi,wi)] = np.dot(vw.transpose(),np.dot(Hij[(vi,wi)],vw))
    
    dimsdims = dims
    dimsdims = np.append(dims,dims)

    #Htest = Htest.reshape(dimsdims)
    #   Add up all the one-body contributions, making sure that the results is properly dimensioned for the 
    #   target subspace
    dim_i1=1 #   dimension of space to the left
    dim_i2=dim #   dimension of space to the right
    
    for vi,v in enumerate(vecs):
        i1 = np.eye(dim_i1)
        dim_i2 = dim_i2/v.shape[1]
        i2 = np.eye(dim_i2)
        
        #print "dim_i1  :  dim_i2", dim_i1, dim_i2, dim
        H += np.kron(i1,np.kron(H1[vi],i2))
        
        #nv = v.shape[1]
        #test = np.ones(len(dimsdims)).astype(int)
        #test[vi] = nv
        #test[vi+len(dims)] = nv
        
        #h = cp.deepcopy(H1[vi])
        #h = h.reshape(test)
        #Htest = np.einsum('ijkljk->ijklmn',Htest,h)
        
        dim_i1 = dim_i1 * v.shape[1]

     
#    print H.reshape(dimsdims)[:,:,:,:,:,:]
#    print "a"
#    print "a"
#    print "a"
#    print H.reshape(dimsdims)
#    Htest = Htest.reshape(dim,dim) 
#    exit(-1)
#    printm(Htest)
#    print
#    printm(H)


    #   Add up all the two-body contributions, making sure that the results is properly dimensioned for the 
    #   target subspace
    dim_i1=1 #   dimension of space to the left
    dim_i2=1 #   dimension of space in the middle 
    dim_i2=dim #   dimension of space to the right
  
    H = H.reshape(dimsdims)

    #print H.shape
    #print H[tuple([slice(0,3)])*len(H.shape)].shape
    ##print H[tuple([slice(0,3)])*len(H.shape)] - H
    #print np.diagonal(np.diagonal(H)).shape
    #print H[np.ones(len(H.shape)).astype(int)].shape
    
    #sliceij = []
    #for d in dimsdims:
    #    sliceij.extend([slice(0,d)])
    #print sliceij

    for vi,v in enumerate(vecs):
        for wi,w in enumerate(vecs):
            if wi>vi:

                nv = v.shape[1]
                nw = w.shape[1]
                dim_env = dim / nv / nw
                #print ": ", nv, nw, dim_env, dim
                
    
                i1 = np.eye(dim_env)
                h = np.kron(H2[(vi,wi)],i1)
                #print ": ", H2[(vi,wi)].shape, i1.shape, h.shape, H.shape
                
                #print H2[(vi,wi)].shape, " x ", i1.shape, " = ", h.shape
                
                tens_dims    = []
                tens_inds    = []
                tens_inds.extend([vi])
                tens_inds.extend([wi])
                tens_dims.extend([nv])
                tens_dims.extend([nw])
                for ti,t in enumerate(vecs):
                    if (ti != vi) and (ti != wi):
                        tens_dims.extend([t.shape[1]])
                        tens_inds.extend([ti])
                tens_dims = np.append(tens_dims, tens_dims) 
                #tens_inds = np.append(tens_inds, tens_inds) 

                sort_ind = np.argsort(tens_inds)
                
    
                #print "sort: ", sort_ind, np.array(tens_inds)[sort_ind]
                #print ":",vi,wi, tens_inds, tens_dims
                #swap indices since we have done kronecker product as H2xI
                #tens_dims[vi], tens_dims[] = tens_dims[0], tens_dims[vi] 
                #tens_dims[vi+n_dims], tens_dims[0+n_dims] = tens_dims[0+n_dims], tens_dims[vi+n_dims] 
                swap = np.append(sort_ind,sort_ind+n_dims) 
                H += h.reshape(tens_dims).transpose(swap)
                #print "swap ", swap
                #h = h.reshape(tens_dims)
                #h = h.transpose(swap)
                #print h.shape
                #print h.shape, dimsdims
                
  
                #sl = cp.deepcopy(sliceij)
                #sl
                #print H[sl].shape
                #H[] = Hij[(vi,wi)]
                #dim_i2 = 1
                #for i in range(vi,wi):
                #    dim_i2 = dim_i2 * vecs[i].shape[1]
                #dim_i2 = dim_i2/v.shape[1]
                #i2 = np.eye(dim_i2)
                
                #print "dim_i1  :  dim_i2", dim_i1, dim_i2, dim
                #H += np.kron(i1,np.kron(H1[vi],i2))
             
                #dim_i1 = dim_i1 * v.shape[1]

    H = H.reshape(dim,dim)
    #printm(H)
    return H

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
parser.add_argument('-hdvv','--hamiltonian', type=str, default="heis_hamiltonian.npy", help='File containing matrix of Hamiltonian', required=False)
parser.add_argument('-s2','--s2', type=str, default="heis_s2.npy", help='File containing matrix of s2', required=False)
parser.add_argument('--eigvals', type=str, default="heis_eigvals.npy", help='File of Hamiltonian eigvals', required=False)
parser.add_argument('--eigvecs', type=str, default="heis_eigvecs.npy", help='File of Hamiltonian eigvecs', required=False)
parser.add_argument('-np','--n_p_space', type=int, default="1", help='Number of vectors in block P space', required=False)
parser.add_argument('--use_exact_tucker_factors', action="store_true", default=False, help='Use compression vectors from tucker decomposition of exact ground states', required=False)
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

#get Hamiltonian and eigenstates 
if args['read']:
    print "Reading Hamiltonian and S2 from disk"
    H_tot = np.load(args['hamiltonian'])
    S2_tot = np.load(args['s2'])
    v = np.load(args['eigvecs'])
    l = np.load(args['eigvals'])
else:
    print "Building Hamiltonian"
    H_tot, H_dict, S2_tot = form_hdvv_H(lattice,j12)


    print " Diagonalize Hamiltonian (%ix%i):\n" %(H_tot.shape[0],H_tot.shape[0]), H_tot.shape
    #l,v = np.linalg.eigh(H_tot)
    l,v = scipy.sparse.linalg.eigsh(H_tot, k=min(100,H_tot.shape[0]))



if args['save']==True:
    np.save("heis_hamiltonian",H_tot)
    np.save("heis_s2",S2_tot)



print v.shape
print S2_tot.shape
au2ev = 27.21165;
au2cm = 219474.63;

convert = au2ev/au2cm;		# convert from wavenumbers to eV
convert = 1;			# 1 for wavenumbers
S2_eig = np.dot(v.transpose(),np.dot(S2_tot,v))
print " %5s    %12s  %12s  %12s" %("State","Energy","Relative","<S2>")
for si,i in enumerate(l):
    print " %5i =  %12.8f  %12.8f  %12.8f" %(si,i*convert,(i-l[0])*convert,S2_eig[si,si])
    if si>10:
        break

v0 = v[:,0]

if args['save']==True:
    np.save("heis_eigvecs",v)
    np.save("heis_eigvals",l)





# reshape eigenvector into tensor
dims_0 = []
for bi,b in enumerate(blocks):
    block_dim = np.power(2,b.shape[0])
    dims_0.extend([block_dim])

v0 = np.reshape(v0,dims_0)

n_p_states = args['n_p_space'] 

# Get initial compression vectors 
p_states, q_states = get_guess_vectors(lattice, j12, blocks, n_p_states)

if args['use_exact_tucker_factors']:
    p_states = []
    q_states = []
    Acore, Atfac = tucker_decompose(v0,0,0)
    for bi,b in enumerate(Atfac):
        #p_states.extend([scipy.linalg.orth(np.random.rand(b.shape[0],n_p_states))])
        p_states.extend([b[:,0:n_p_states]])

        q_states.extend([b[:,n_p_states::]])


#
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
#<Abc|Hab|Abc> = <Ab|Hab|Ab>


Hi = {}
Hij = {}
vecs0 = []
#1 body operators
for bi,b in enumerate(blocks):
    Hi[bi] = form_superblock_hamiltonian(lattice, j12, blocks, [bi])

#2 body operators
for bi,b in enumerate(blocks):
    for bj,bb in enumerate(blocks):
        if bj>bi:
            hi = Hi[bi]
            hj = Hi[bj]
            
            Hij[(bi,bj)] = form_superblock_hamiltonian(lattice, j12, blocks, [bi,bj])
            Hij[(bi,bj)] -= np.kron(hi,np.eye(hj.shape[0])) 
            Hij[(bi,bj)] -= np.kron(np.eye(hi.shape[0]),hi) 

# get vecs for PPP class
for bi,b in enumerate(blocks):
    vecs0.extend([p_states[bi]])

#vecs0[0] = np.hstack((p_states[0],q_states[0]))
#vecs0[1] = np.hstack((p_states[1],q_states[1]))
#vecs0[2] = np.hstack((p_states[2],q_states[2]))
#print vecs0[0].shape
Hp = form_compressed_hamiltonian(vecs0,Hi,Hij)
lp,vp = np.linalg.eigh(Hp)

print
print " Energy  Error due to compression    :  %12.8f - %12.8f = %12.8f" %(lp[0],l[0],lp[0]-l[0])


Ha = form_superblock_hamiltonian(lattice, j12, blocks, [0])
Hb = form_superblock_hamiltonian(lattice, j12, blocks, [1])
Hc = form_superblock_hamiltonian(lattice, j12, blocks, [2])
Hab = form_superblock_hamiltonian(lattice, j12, blocks, [0,1])
Hac = form_superblock_hamiltonian(lattice, j12, blocks, [0,2])
Hbc = form_superblock_hamiltonian(lattice, j12, blocks, [1,2])
#Habc = form_superblock_hamiltonian(lattice, j12, blocks, [0,1,2])

Ia = np.eye(Ha.shape[0])
Ib = np.eye(Hb.shape[0])
Ic = np.eye(Hc.shape[0])

a = p_states[0]
b = p_states[1]
c = p_states[2]
ab = np.kron(a,b)
ac = np.kron(a,c)
bc = np.kron(b,c)

A = q_states[0]
B = q_states[1]
C = q_states[2]
AB = np.kron(A,B)
AC = np.kron(A,C)
BC = np.kron(B,C)


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
#printm(H0)
#exit(-1)

H0 += np.kron(Hab,Ic) 
H0 += np.kron(Ia,Hbc) 
tmp = np.kron(Hac,Ib)
npv = n_p_states
tmp = tmp.reshape([npv,npv,npv,npv,npv,npv])
tmp = tmp.transpose((0,2,1,3,5,4))
tmp = tmp.reshape(np.power(npv,3),np.power(npv,3))
H0 += tmp

#abc = np.kron(p_states[0],np.kron(p_states[1],p_states[2]))
#Habc = np.dot(abc.transpose(),np.dot(Habc,abc))

l0,v0 = np.linalg.eigh(H0)
#l1,v1 = np.linalg.eigh(Habc)
#print np.array([l0,l1])

print
print " Energy  Error due to compression    :  %12.8f - %12.8f = %12.8f" %(l0[0],l[0],l0[0]-l[0])

exit(-1)












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
