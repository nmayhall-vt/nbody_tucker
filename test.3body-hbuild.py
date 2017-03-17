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
	p_states.extend([v_b[:,0:n_p_states[bi]]])
        q_states.extend([v_b[:,n_p_states[bi]::]])
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

def form_compressed_hamiltonian_diag(vecs,Hi,Hij):
    # {{{
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

                #todo and check
                #h.shape = (tens_dims)
                #H += h.transpose(swap)
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
    # }}}

def form_compressed_hamiltonian_offdiag_1block_diff(vecs_l,vecs_r,Hi,Hij,differences):
    # {{{
    """
        Find (1-site) Hamiltonian matrix in between differently compressed spaces:
            i.e., 
                <Abcd| H(1) |ab'c'd'> = <A|Ha|a> * del(bb') * del(cc')
            or like,
                <aBcd| H(1) |abcd> = <B|Hb|b> * del(aa') * del(cc')
        
        Notice that the full Hamiltonian will also have the two-body part:
            <Abcd| H(2) |abcd> = <Ab|Hab|ab'> del(cc') + <Ac|Hac|ac'> del(bb')

       
        differences = [blocks which are not diagonal]
            i.e., for <Abcd|H(1)|abcd>, differences = [1]
                  for <Abcd|H(1)|aBcd>, differences = [1,2], and all values are zero for H(1)
            
        vecs_l  [vecs_A, vecs_B, vecs_C, ...]
        vecs_r  [vecs_A, vecs_B, vecs_C, ...]
    """
    dim_l = 1 # dimension of left subspace
    dim_r = 1 # dimension of right subspace
    dim_same = 1 # dimension of space spanned by all those fragments is the same space (i.e., not the blocks inside differences)
    dims_l = [] # list of mode dimensions (size of hilbert space on each fragment) 
    dims_r = [] # list of mode dimensions (size of hilbert space on each fragment) 
  
    
    block_curr = differences[0] # current block which is offdiagonal

    #   vecs_l correspond to the bra states
    #   vecs_r correspond to the ket states


    assert(len(vecs_l) == len(vecs_r))
    

    for bi,b in enumerate(vecs_l):
        dim_l = dim_l*b.shape[1]
        dims_l.extend([b.shape[1]])
        if bi !=block_curr:
            dim_same *= b.shape[1]

    dim_same_check = 1
    for bi,b in enumerate(vecs_r):
        dim_r = dim_r*b.shape[1]
        dims_r.extend([b.shape[1]])
        if bi !=block_curr:
            dim_same_check *= b.shape[1]

    assert(dim_same == dim_same_check)

    H = np.zeros((dim_l,dim_r))
    print " Size of Hamitonian block: ", H.shape

    assert(len(dims_l) == len(dims_r))
    n_dims = len(dims_l)


    H1 = cp.deepcopy(Hi)
    H2 = cp.deepcopy(Hij)
    
    ## Rotate the single block Hamiltonians into the requested single site basis 
    #for vi in range(0,n_dims):
    #    l = vecs_l[vi]
    #    r = vecs_r[vi]
    #    H1[vi] = l.T.dot(Hi[vi]).dot(r)

    # Rotate the double block Hamiltonians into the appropriate single site basis 
#    for vi in range(0,n_dims):
#        for wi in range(0,n_dims):
#            if wi>vi:
#                vw_l = np.kron(vecs_l[vi],vecs_l[wi])
#                vw_r = np.kron(vecs_r[vi],vecs_r[wi])
#                H2[(vi,wi)] = vw_l.T.dot(Hij[(vi,wi)]).dot(vw_r)
   

    dimsdims = np.append(dims_l,dims_r) # this is the tensor layout for the many-body Hamiltonian in the current subspace
    
    vecs = vecs_l
    dims = dims_l
    dim = dim_l
    
    #   Add up all the one-body contributions, making sure that the results is properly dimensioned for the 
    #   target subspace
    dim_i1=1 # dimension of space for fragments to the left of the current 'different' fragment
    dim_i2=1 # dimension of space for fragments to the right of the current 'different' fragment

   
    # <abCdef|H1|abcdef> = eye(a) x eye(b) x <C|H1|c> x eye(d) x eye(e) x eye(f)
    for vi in range(n_dims):
        if vi<block_curr:
            dim_i1 *= dims_l[vi]
            assert(dims_l[vi]==dims_r[vi])
        elif vi>block_curr:
            dim_i2 *= dims_l[vi]
            assert(dims_l[vi]==dims_r[vi])
   
    # Rotate the current single block Hamiltonian into the requested single site basis 
    l = vecs_l[block_curr]
    r = vecs_r[block_curr]
    
    h1_block  = l.T.dot(Hi[block_curr]).dot(r)

    i1 = np.eye(dim_i1)
    i2 = np.eye(dim_i2)
    H += np.kron(i1,np.kron(h1_block,i2))

    # <abCdef|H2(0,2)|abcdef> = <aC|H2|ac> x eye(b) x eye(d) x eye(e) x eye(f) = <aCbdef|H2|acbdef>
    #
    #   then transpose:
    #       flip Cb and cb
    H.shape = (dims_l+dims_r)

    for bi in range(0,block_curr):
        #print "block_curr, bi", block_curr, bi
        vw_l = np.kron(vecs_l[bi],vecs_l[block_curr])
        vw_r = np.kron(vecs_r[bi],vecs_r[block_curr])
        h2 = vw_l.T.dot(Hij[(bi,block_curr)]).dot(vw_r) # i.e.  get reference to <aC|Hij[0,2]|a'c>, where block_curr = 2 and bi = 0

        dim_i = dim_same / dims[bi]
        #i1 = np.eye(dim_i)

        #tmp_h2 = np.kron( h2, i1)
        h2.shape = (dims_l[bi],dims_l[block_curr], dims_r[bi],dims_r[block_curr])
        
        tens_inds = []
        tens_inds.extend([bi,block_curr])
        tens_inds.extend([bi+n_dims,block_curr+n_dims])
        for bbi in range(0,n_dims):
            if bbi != bi and bbi != block_curr:
                tens_inds.extend([bbi])
                tens_inds.extend([bbi+n_dims])
                #print "h2.shape", h2.shape,
                h2 = np.tensordot(h2,np.eye(dims[bbi]),axes=0)
   
        #print "h2.shape", h2.shape,
        sort_ind = np.argsort(tens_inds)
        H += h2.transpose(sort_ind)
        #print "tens_inds", tens_inds
        #print "sort_ind", sort_ind 
        #print "h2", h2.transpose(sort_ind).shape
        #H += h2
    
    for bi in range(block_curr, n_dims):
        if bi == block_curr:
            continue
        print "block_curr, bi", block_curr, bi
        vw_l = np.kron(vecs_l[block_curr],vecs_l[bi])
        vw_r = np.kron(vecs_r[block_curr],vecs_r[bi])
        h2 = vw_l.T.dot(Hij[(block_curr,bi)]).dot(vw_r) # i.e.  get reference to <aC|Hij[0,2]|a'c>, where block_curr = 2 and bi = 0

        h2.shape = (dims_l[block_curr],dims_l[bi], dims_r[block_curr],dims_r[bi])
        
        tens_inds = []
        tens_inds.extend([block_curr,bi])
        tens_inds.extend([block_curr+n_dims,bi+n_dims])
        for bbi in range(0,n_dims):
            if bbi != bi and bbi != block_curr:
                tens_inds.extend([bbi])
                tens_inds.extend([bbi+n_dims])
                #print "h2.shape", h2.shape,
                h2 = np.tensordot(h2,np.eye(dims[bbi]),axes=0)
   
        #print "h2.shape", h2.shape,
        sort_ind = np.argsort(tens_inds)
        #print "h2", h2.transpose(sort_ind).shape
        #print "tens_inds", tens_inds
        #print "sort_ind", sort_ind 
        H += h2.transpose(sort_ind)
        #print "h2.shape", h2.shape
    
    H.shape = (dim_l,dim_r)

    
    return H
    # }}}

def form_compressed_hamiltonian_offdiag_2block_diff(vecs_l,vecs_r,Hi,Hij,differences):
    # {{{
    """
        Find Hamiltonian matrix in between differently compressed spaces:
            i.e., 
                <Abcd| H(0,2) |a'b'C'd'> = <Ac|Ha|a'C'> * del(bb')
        
       
        differences = [blocks which are not diagonal]
            i.e., for <Abcd|H(1)|abCd>, differences = [0,2]
            
        vecs_l  [vecs_A, vecs_B, vecs_C, ...]
        vecs_r  [vecs_A, vecs_B, vecs_C, ...]
    """
    dim_l = 1 # dimension of left subspace
    dim_r = 1 # dimension of right subspace
    dim_same = 1 # dimension of space spanned by all those fragments is the same space (i.e., not the blocks inside differences)
    dims_l = [] # list of mode dimensions (size of hilbert space on each fragment) 
    dims_r = [] # list of mode dimensions (size of hilbert space on each fragment) 
  
    assert( len(differences) == 2) # make sure we are not trying to get H(1) between states with multiple fragments orthogonal 
    
    block_curr1 = differences[0] # current block which is offdiagonal
    block_curr2 = differences[1] # current block which is offdiagonal

    #   vecs_l correspond to the bra states
    #   vecs_r correspond to the ket states


    assert(len(vecs_l) == len(vecs_r))
    

    for bi,b in enumerate(vecs_l):
        dim_l = dim_l*b.shape[1]
        dims_l.extend([b.shape[1]])
        if bi !=block_curr1 and bi != block_curr2:
            dim_same *= b.shape[1]

    dim_same_check = 1
    for bi,b in enumerate(vecs_r):
        dim_r = dim_r*b.shape[1]
        dims_r.extend([b.shape[1]])
        if bi !=block_curr1 and bi != block_curr2:
            dim_same_check *= b.shape[1]

    assert(dim_same == dim_same_check)

    H = np.zeros((dim_l,dim_r))
    print " Size of Hamitonian block: ", H.shape

    assert(len(dims_l) == len(dims_r))
    n_dims = len(dims_l)



    dimsdims = np.append(dims_l,dims_r) # this is the tensor layout for the many-body Hamiltonian in the current subspace
    
    #   Add up all the one-body contributions, making sure that the results is properly dimensioned for the 
    #   target subspace
    dim_i1=1 # dimension of space for fragments to the left of the current 'different' fragment
    dim_i2=1 # dimension of space for fragments to the right of the current 'different' fragment

   

    # <Abcdef|H2(0,2)|abCdef> = <Ac|H2|aC> x eye(b) x eye(d) x eye(e) x eye(f) = <aCbdef|H2|acbdef>
    #
    #   then transpose:
    #       flip Cb and cb
    H.shape = (dims_l+dims_r)

    assert(block_curr1 < block_curr2)

    print " block_curr1, block_curr2", block_curr1, block_curr2
    vw_l = np.kron(vecs_l[block_curr1],vecs_l[block_curr2])
    vw_r = np.kron(vecs_r[block_curr1],vecs_r[block_curr2])
    h2 = vw_l.T.dot(Hij[(block_curr1,block_curr2)]).dot(vw_r) # i.e.  get reference to <aC|Hij[0,2]|a'c>, where block_curr = 2 and bi = 0

    h2.shape = (dims_l[block_curr1],dims_l[block_curr2], dims_r[block_curr1],dims_r[block_curr2])
    
    tens_inds = []
    tens_inds.extend([block_curr1,block_curr2])
    tens_inds.extend([block_curr1+n_dims,block_curr2+n_dims])
    for bbi in range(0,n_dims):
        if bbi != block_curr1 and bbi != block_curr2:
            tens_inds.extend([bbi])
            tens_inds.extend([bbi+n_dims])
            #print "h2.shape", h2.shape,

            assert(dims_l[bbi] == dims_r[bbi])
            dims = dims_l
            h2 = np.tensordot(h2,np.eye(dims[bbi]),axes=0)
   
    #print "h2.shape", h2.shape,
    sort_ind = np.argsort(tens_inds)
    H += h2.transpose(sort_ind)
    #print "tens_inds", tens_inds
    #print "sort_ind", sort_ind 
    #print "h2", h2.transpose(sort_ind).shape
    #H += h2
    
    H.shape = (dim_l,dim_r)

    
    return H
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
parser.add_argument('-hdvv','--hamiltonian', type=str, default="heis_hamiltonian.npy", help='File containing matrix of Hamiltonian', required=False)
parser.add_argument('-s2','--s2', type=str, default="heis_s2.npy", help='File containing matrix of s2', required=False)
parser.add_argument('--eigvals', type=str, default="heis_eigvals.npy", help='File of Hamiltonian eigvals', required=False)
parser.add_argument('--eigvecs', type=str, default="heis_eigvecs.npy", help='File of Hamiltonian eigvecs', required=False)
parser.add_argument('-np','--n_p_space', type=int, nargs="+", help='Number of vectors in block P space', required=False)
parser.add_argument('--use_exact_tucker_factors', action="store_true", default=False, help='Use compression vectors from tucker decomposition of exact ground states', required=False)
args = vars(parser.parse_args())
#
#   Let minute specification of walltime override hour specification

j12 = np.loadtxt(args['j12'])
lattice = np.loadtxt(args['lattice']).astype(int)
blocks = np.loadtxt(args['blocks']).astype(int)
n_sites = len(lattice)
n_blocks = len(blocks)

np.random.seed(2)

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
        p_states.extend([b[:,0:n_p_states[bi]]])
        q_states.extend([b[:,n_p_states[bi]::]])

if 0:
    # do random guess
    p_states = []
    q_states = []
    for bi,b in enumerate(blocks):
        block_dim = np.power(2,b.shape[0])
        r = scipy.linalg.orth(np.random.rand(block_dim,block_dim))
        p_states.extend([r[:,0:n_p_states[bi]]])
        q_states.extend([r[:,n_p_states[bi]::]])

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

vecsQ = []
for bi in range(n_blocks):
    v = cp.deepcopy(vecs0)
    v[bi] = q_states[bi]
    vecsQ.extend([v])

vecsQQ = {}
for bi in range(n_blocks):
    for bj in range(bi+1,n_blocks):
        v = cp.deepcopy(vecs0)
        v[bi] = q_states[bi]
        v[bj] = q_states[bj]
        vecsQQ[bi,bj] = v

H0_0 = form_compressed_hamiltonian_diag(vecs0,Hi,Hij) # <PPP|H|PPP>
#

vecs1 = vecsQ[0]
vecs2 = vecsQ[1]
vecs3 = vecsQ[2]

H_sectors = {}
H_sectors[0,0] = H0_0

n_body_order = 2

if n_body_order >= 1:
    for bi in range(n_blocks):
        H_sectors[bi+1,bi+1]    = form_compressed_hamiltonian_diag(vecsQ[bi],Hi,Hij) # <QPP|H|QPP>
        
        H_sectors[0,bi+1]       = form_compressed_hamiltonian_offdiag_1block_diff(vecs0,vecsQ[bi],Hi,Hij,[bi]) # <PPP|H|QPP>
        H_sectors[bi+1,0]       = H_sectors[0,bi+1].T
        for bj in range(bi+1,n_blocks):
            H_sectors[bi+1,bj+1] = form_compressed_hamiltonian_offdiag_2block_diff(vecsQ[bi],vecsQ[bj],Hi,Hij,[bi,bj]) # <QPP|H|PQP>
            H_sectors[bj+1,bi+1] = H_sectors[bi+1,bj+1].T


if n_body_order >= 2:
    for bi in range(n_blocks):
        for bj in range(bi+1,n_blocks):
            bij = (bi+1,bj+1)
            H_sectors[bij,bij]  = form_compressed_hamiltonian_diag(vecsQQ[bi,bj],Hi,Hij) # <QPQ|H|QPQ>
            
            H_sectors[0,bij]    = np.zeros( (H0_0.shape[1], H_sectors[bij,bij].shape[1] ) ) # <PPP|H|QPQ>
            H_sectors[bij,0]    = H_sectors[0,bij].T
            
    for bi in range(n_blocks):
        for bj in range(bi+1,n_blocks):
            bij = (bi+1,bj+1)
            for bk in range(n_blocks):
                if bk == bi:
                    H_sectors[bk+1,bij] = form_compressed_hamiltonian_offdiag_1block_diff(vecsQ[bk],vecsQQ[bi,bj],Hi,Hij,[bj]) # <QPP|H|PQQ>
                    H_sectors[bij,bk+1] = H_sectors[bk+1,bij].T
                elif bk == bj:
                    H_sectors[bk+1,bij] = form_compressed_hamiltonian_offdiag_1block_diff(vecsQ[bk],vecsQQ[bi,bj],Hi,Hij,[bi]) # <QPP|H|PQQ>
                    H_sectors[bij,bk+1] = H_sectors[bk+1,bij].T
                else:
                    #H_sectors[bk+1,bij] = np.zeros(( len(vecsQ[bk]),len(vecsQQ[bi,bj]) )) 
                    H_sectors[bk+1,bij]    = np.zeros( (H_sectors[bk+1,bk+1].shape[1] , H_sectors[bij,bij].shape[1] ) ) # <PPP|H|QPQ>
                    H_sectors[bij,bk+1] = H_sectors[bk+1,bij].T
            
                for bl in range(bk+1,n_blocks):
                    bkl = (bk+1,bl+1)

                    #only compute upper triangular blocks
                    if bk < bi:
                        continue
                    if bk == bi and bl <= bj:
                        continue
                    
                    print 
                    print " Form Hamiltonian for <%s|H|%s>" %(bij, bkl)
                    diff = {}
                    diff[bi] = 1
                    diff[bj] = 1
                    diff[bk] = 1
                    diff[bl] = 1
                    for bbi in (bi,bj):
                        for bbj in (bk,bl):
                            if bbi == bbj:
                                diff[bbi] = 0
                    diff2 = []
                    for bbi in diff.keys():
                        if diff[bbi] == 1:
                            diff2.extend([bbi])
                   
                    if len(diff2) == 1:
                        H_sectors[bij,bkl] = form_compressed_hamiltonian_offdiag_2block_diff(vecsQQ[bi,bj],vecsQQ[bk,bl],Hi,Hij,diff2) # <QPQ|H|QQP>
                        H_sectors[bkl,bij] = H_sectors[bij,bkl].T
                    if len(diff2) > 1:
                        #H_sectors[bij,bkl] = np.zeros(( len(vecsQQ[bi,bj]),len(vecsQQ[bk,bl]) )) 
                        H_sectors[bij,bkl]    = np.zeros( (H_sectors[0,bij].shape[1] , H_sectors[0,bkl].shape[1] ) ) # <PPP|H|QPQ>
                        H_sectors[bkl,bij] = H_sectors[bij,bkl].T


#for k in H_sectors.keys():
#    print k
Htest = H0_0

if n_body_order == 1:
    # Singles
    for bi in range(n_blocks+1):
        row_i = np.array([])
        
        for bj in range(n_blocks+1):
            if bj == 0:
                row_i = H_sectors[bi,bj]
            else:
                row_i = np.hstack((row_i,H_sectors[bi,bj]))
        
        if bi == 0:
            Htest = row_i
        else:
            Htest = np.vstack((Htest,row_i))


if n_body_order == 2:
    # 0,0 
    row_0 = H_sectors[0,0]
    # 0,S
    for bi in range(1,n_blocks+1):
        row_0 = np.hstack( ( row_0, H_sectors[0,bi] ) )
    # 0,D
    for bi in range(1,n_blocks+1):
        for bj in range(bi+1, n_blocks+1):
            bij = (bi,bj)
            print row_0.shape, H_sectors[0,bij].shape
            
            row_0 = np.hstack( ( row_0, H_sectors[0,(bi,bj)] ) )

    Htest = row_0

    
    # Singles
    for bi in range(1,n_blocks+1):
        # bi,0
        row_i = H_sectors[bi,0] 
        # bi,bj 
        for bj in range(1,n_blocks+1):
            row_i = np.hstack((row_i,H_sectors[bi,bj]))
        # bi,bjk 
        for bj in range(1,n_blocks+1):
            for bk in range(bj+1,n_blocks+1):
                row_i = np.hstack((row_i,H_sectors[bi,(bj,bk)]))
        
        Htest = np.vstack((Htest,row_i))


    #Doubles 
    for bi in range(1,n_blocks+1):
        for bj in range(bi+1,n_blocks+1):
            # bij,0
            row_ij = H_sectors[(bi,bj),0]                                       #<ij|H|0>
            # bij,bk
            for bk in range(1,n_blocks+1):
                row_ij = np.hstack( (row_ij, H_sectors[(bi,bj),bk]) )           #<ij|H|k>
            # bij,bkl
            for bk in range(1,n_blocks+1):
                for bl in range(bk+1,n_blocks+1):
                    row_ij = np.hstack( (row_ij, H_sectors[(bi,bj),(bk,bl)]) )  #<ij|H|kl>
            
            Htest = np.vstack((Htest,row_ij))




lp,vp = np.linalg.eigh(Htest)
print 
print " Eigenvectors of compressed Hamiltonian"
print " %5s    %12s  %12s  %12s" %("State","Energy","Relative","<S2>")
for si,i in enumerate(lp):
    print " %5i =  %12.8f  %12.8f  %12.8s" %(si,i*convert,(i-lp[0])*convert,"--")
    if si>10:
        break
print 
print
print " Energy  Error due to compression    :  %12.8f - %12.8f = %12.8f" %(lp[0],l[0],lp[0]-l[0])
exit(-1)

#vecs0[1] = np.hstack((p_states[1],q_states[1]))
#vecs0[2] = np.hstack((p_states[2],q_states[2]))
#print vecs0[0].shape
H0_0 = form_compressed_hamiltonian_diag(vecs0,Hi,Hij) # <PPP|H|PPP>
H1_1 = form_compressed_hamiltonian_diag(vecs1,Hi,Hij) # <QPP|H|QPP>
H2_2 = form_compressed_hamiltonian_diag(vecs2,Hi,Hij) # <PQP|H|PQP>
H3_3 = form_compressed_hamiltonian_diag(vecs3,Hi,Hij) # <PPQ|H|PPQ>

H0_1 = form_compressed_hamiltonian_offdiag_1block_diff(vecs0,vecs1,Hi,Hij,[0]) # <PPP|H|QPP>
H0_2 = form_compressed_hamiltonian_offdiag_1block_diff(vecs0,vecs2,Hi,Hij,[1])
H0_3 = form_compressed_hamiltonian_offdiag_1block_diff(vecs0,vecs3,Hi,Hij,[2])

H1_2 = form_compressed_hamiltonian_offdiag_2block_diff(vecs1,vecs2,Hi,Hij,[0,1]) # <QPP|H|PQP>
H1_3 = form_compressed_hamiltonian_offdiag_2block_diff(vecs1,vecs3,Hi,Hij,[0,2])
H2_3 = form_compressed_hamiltonian_offdiag_2block_diff(vecs2,vecs3,Hi,Hij,[1,2])


# set all q-space contributions to zero
#H1_1 = np.zeros((H1_1.shape[0],H1_1.shape[0]))
#H2_2 = np.zeros((H2_2.shape[0],H2_2.shape[0]))
#H3_3 = np.zeros((H3_3.shape[0],H3_3.shape[0]))

H_ss = np.array([]) 



#print H0_1
n_body_order = 1
if n_body_order == 1:
    H0  = np.hstack((H0_0, H0_1, H0_2, H0_3))
    H1  = np.hstack((H0_1.T, H1_1, H1_2, H1_3))
    H2  = np.hstack((H0_2.T, H1_2.T, H2_2, H2_3))
    H3  = np.hstack((H0_3.T, H1_3.T, H2_3.T, H3_3))

    H_ss = np.vstack((H0,H1,H2,H3))

if n_body_order == 2:
    H0  = np.hstack((H0_0, H0_1, H0_2, H0_3, H0_12, H0_13, H0_23))
    H1  = np.hstack((H0_1.T, H1_1, H1_2, H1_3, H1_12, H1_13, H1_23))
    H2  = np.hstack((H0_2.T, H1_2.T, H2_2, H2_3, H2_12, H2_13, H2_23))
    H3  = np.hstack((H0_3.T, H1_3.T, H2_3.T, H3_3, H3_12, H3_13, H3_23))
    H12 = np.hstack((H0_12.T, H1_12.T, H2_12.T, H3_12.T, H12_12, H12_13, H12_23))
    H13 = np.hstack((H0_13.T, H1_13.T, H2_13.T, H3_13.T, H12_13.T, H13_13, H13_23))
    H23 = np.hstack((H0_23.T, H1_23.T, H2_23.T, H3_23.T, H12_23.T, H13_23.T, H23_23))

    H_ss = np.vstack((H0,H1,H2,H3,H12,H13,H23))


#H_ss = H0_0
#print H2_3
lp,vp = np.linalg.eigh(H_ss)
#np.save("a",H1_2)
#Hb = np.load('b.npy')
#print H1_2 - Hb
#exit(-1)

#re-compose
if 0:
    v0 = vp[0][0 : n0]
    v1 = vp[0][n0 : n0+n1]
    v2 = vp[0][n0+n1 : n0+n1+n2]
    v3 = vp[0][n0+n1+n2 : n0+n1+n2+n3]

    print " norm of PPP component %12.8f" %np.dot(v0,v0)
    np0 = p_states[0].shape[1]
    np1 = p_states[1].shape[1]
    np2 = p_states[2].shape[1]
    
    nq0 = q_states[0].shape[1]
    nq1 = q_states[1].shape[1]
    nq2 = q_states[2].shape[1]
    
    v0 = v0.reshape([np0,np1,np2])
    v1 = v1.reshape([nq0,np1,np2])
    v2 = v2.reshape([np0,nq1,np2])
    v3 = v3.reshape([np0,np1,nq2])

    vec = np.zeros(H_tot.shape[0])
    vec = vec.reshape(dims_0)
    
    vec += tucker_recompose(v0,vecs0)
    
    vec += tucker_recompose(v1,vecs1)
    vec += tucker_recompose(v2,vecs2)
    vec += tucker_recompose(v3,vecs3)

    vec = vec.reshape(H_tot.shape[0])
    vv = np.dot(vec,vec)
    print " Expectation value: %12.8f"% (np.dot(vec.transpose(),np.dot(H_tot,vec))/vv )
    print " norm:              %12.8f"% vv 

print H0_0.shape
print H_ss.shape
print 
print " Eigenvectors of compressed Hamiltonian"
print " %5s    %12s  %12s  %12s" %("State","Energy","Relative","<S2>")
for si,i in enumerate(lp):
    print " %5i =  %12.8f  %12.8f  %12.8s" %(si,i*convert,(i-lp[0])*convert,"--")
    if si>10:
        break
print 
print
print " Energy  Error due to compression    :  %12.8f - %12.8f = %12.8f" %(lp[0],l[0],lp[0]-l[0])

exit(-1)






















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












