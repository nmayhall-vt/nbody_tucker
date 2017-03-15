#! /usr/bin/env python
import numpy as np
import scipy
import scipy.linalg
import scipy.io
import copy as cp
import argparse
import scipy.sparse
import scipy.sparse.linalg

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
parser.add_argument('--max_iter', type=int, default=100, help='Max iterations for solving for the compression vectors', required=False)
parser.add_argument('-hdvv','--hamiltonian', type=str, default="heis_hamiltonian.npy", help='File containing matrix of Hamiltonian', required=False)
parser.add_argument('-s2','--s2', type=str, default="heis_s2.npy", help='File containing matrix of s2', required=False)
parser.add_argument('--eigvals', type=str, default="heis_eigvals.npy", help='File of Hamiltonian eigvals', required=False)
parser.add_argument('--eigvecs', type=str, default="heis_eigvecs.npy", help='File of Hamiltonian eigvecs', required=False)
parser.add_argument('-np','--n_p_space', type=int, default="1", nargs='+',  help='Number of vectors in block P space', required=False)
parser.add_argument('-nb','--n_body_order', type=int, default="0", help='n_body spaces', required=False)
parser.add_argument('-ts','--target_state', type=int, default="0", nargs='+', help='state(s) to target during (possibly state-averaged) optimization', required=False)
parser.add_argument('--use_exact_tucker_factors', action="store_true", default=False, help='Use compression vectors from tucker decomposition of exact ground states', required=False)
args = vars(parser.parse_args())
#
#   Let minute specification of walltime override hour specification

j12 = np.loadtxt(args['j12'])
lattice = np.loadtxt(args['lattice']).astype(int)
blocks = np.loadtxt(args['blocks']).astype(int)
n_sites = len(lattice)
n_blocks = len(blocks)

n_body_order = args['n_body_order']

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

dim_tot = 1 # dimension of uncontracted configuration space
for d in dims_0:
    dim_tot *= d
v0 = np.reshape(v0,dims_0)

n_p_states = args['n_p_space'] 
print "n_p_states",  n_p_states



if len(n_p_states) != len(blocks):
    print " Number of p states not equal to number of blocks"
    exit(-1)


# Get initial compression vectors 
p_states, q_states = get_guess_vectors(lattice, j12, blocks, n_p_states)

if args['use_exact_tucker_factors']:
    p_states = []
    q_states = []
    Acore, Atfac = tucker_decompose(v0,0,0)
    for bi,b in enumerate(Atfac):
        p_states.extend([b[:,0:n_p_states[bi] ]])
        q_states.extend([b[:,n_p_states[bi]::]])

if 0:
    # do random guess
    p_states = []
    q_states = []
    #U = scipy.linalg.orth(np.random.rand(n_p_states,n_p_states))
    for bi,b in enumerate(blocks):
        block_dim = np.power(2,b.shape[0])
        r = scipy.linalg.orth(np.random.rand(block_dim,block_dim))
        #p_states.extend([np.dot(r[:,0:n_p_states],U)]) # make sure invariant
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


# loop over compression vector iterations
energy_per_iter = []
maxiter = args['max_iter'] 
for it in range(0,maxiter):
    print " Tucker optimization: Iteration %4i" %it
    # get vecs for PPP class
    dim0 = 1
    vecs0 = []
    for bi,b in enumerate(blocks):
        vecs0.extend([p_states[bi]])
        dim0 *= p_states[bi].shape[1]
    #for bi,b in enumerate(blocks):
    #    vecs0.extend([p_states[bi].transpose()])
    
    vecs1 = cp.deepcopy(vecs0)
    vecs2 = cp.deepcopy(vecs0)
    vecs3 = cp.deepcopy(vecs0)
    
    vecs12 = cp.deepcopy(vecs0)
    vecs13 = cp.deepcopy(vecs0)
    vecs23 = cp.deepcopy(vecs0)
    
    vecs123 = cp.deepcopy(vecs0)
    
    
    vecs1[0] = q_states[0]
    vecs2[1] = q_states[1]
    vecs3[2] = q_states[2]
    
    vecs12[0] = q_states[0]
    vecs12[1] = q_states[1]
    
    vecs13[0] = q_states[0]
    vecs13[2] = q_states[2]
    
    vecs23[1] = q_states[1]
    vecs23[2] = q_states[2]

    vecs123[0] = q_states[0]
    vecs123[1] = q_states[1]
    vecs123[2] = q_states[2]
    
    dim1 = dim0/p_states[0].shape[1]*q_states[0].shape[1]
    dim2 = dim0/p_states[1].shape[1]*q_states[1].shape[1]
    dim3 = dim0/p_states[2].shape[1]*q_states[2].shape[1]
    
    dim12 = dim0 / p_states[0].shape[1] / p_states[1].shape[1] * q_states[0].shape[1]  * q_states[1].shape[1]
    dim13 = dim0 / p_states[0].shape[1] / p_states[2].shape[1] * q_states[0].shape[1]  * q_states[2].shape[1]
    dim23 = dim0 / p_states[1].shape[1] / p_states[2].shape[1] * q_states[1].shape[1]  * q_states[2].shape[1]
    
    dim123 = q_states[0].shape[1] * q_states[1].shape[1] * q_states[2].shape[1]
    
    v0v0 = vecs0 + vecs0

    if n_body_order >= 1: 
        v0v1 = vecs0 + vecs1
        v0v2 = vecs0 + vecs2
        v0v3 = vecs0 + vecs3
        v1v1  = vecs1 + vecs1
        v1v2  = vecs1 + vecs2
        v1v3  = vecs1 + vecs3
        v2v2  = vecs2 + vecs2
        v2v3  = vecs2 + vecs3
        v3v3  = vecs3 + vecs3
 
    if n_body_order >= 2: 
        v0v12 = vecs0 + vecs12
        v0v13 = vecs0 + vecs13
        v0v23 = vecs0 + vecs23
        v1v12 = vecs1 + vecs12
        v1v13 = vecs1 + vecs13
        v1v23 = vecs1 + vecs23
        v2v12 = vecs2 + vecs12
        v2v13 = vecs2 + vecs13
        v2v23 = vecs2 + vecs23
        v3v12 = vecs3 + vecs12
        v3v13 = vecs3 + vecs13
        v3v23 = vecs3 + vecs23
        v12v12 = vecs12 + vecs12
        v12v13 = vecs12 + vecs13
        v12v23 = vecs12 + vecs23
        v13v13 = vecs13 + vecs13
        v13v23 = vecs13 + vecs23
        v23v23 = vecs23 + vecs23
 
    if n_body_order >= 3: 
        v0v123 = vecs0 + vecs123
        v1v123 = vecs1 + vecs123
        v2v123 = vecs2 + vecs123
        v3v123 = vecs3 + vecs123
        v12v123 = vecs12 + vecs123
        v13v123 = vecs13 + vecs123
        v23v123 = vecs23 + vecs123
        v123v123 = vecs123 + vecs123
    


    dimsdims = np.append(dims_0, dims_0)
    
    H_tot = H_tot.reshape(dimsdims)


    H0_0   = transform_tensor(H_tot,v0v0,trans=1)
    
    if n_body_order >= 1:
        H1_1   = transform_tensor(H_tot,v1v1,trans=1)
        H2_2   = transform_tensor(H_tot,v2v2,trans=1)
        H3_3   = transform_tensor(H_tot,v3v3,trans=1)
        
        H0_1    = transform_tensor(H_tot,v0v1,trans=1)
        H0_2    = transform_tensor(H_tot,v0v2,trans=1)
        H0_3    = transform_tensor(H_tot,v0v3,trans=1)
        
        H1_2    = transform_tensor(H_tot,v1v2,trans=1)
        H1_3    = transform_tensor(H_tot,v1v3,trans=1)
        
        H2_3    = transform_tensor(H_tot,v2v3,trans=1)

    if n_body_order >= 2:
        H12_12 = transform_tensor(H_tot,v12v12,trans=1)
        H13_13 = transform_tensor(H_tot,v13v13,trans=1)
        H23_23  = transform_tensor(H_tot,v23v23,trans=1)
        H0_12   = transform_tensor(H_tot,v0v12,trans=1)
        H0_13   = transform_tensor(H_tot,v0v13,trans=1)
        H0_23   = transform_tensor(H_tot,v0v23,trans=1)
        H1_12   = transform_tensor(H_tot,v1v12,trans=1)
        H1_13   = transform_tensor(H_tot,v1v13,trans=1)
        H1_23   = transform_tensor(H_tot,v1v23,trans=1)
        H2_12   = transform_tensor(H_tot,v2v12,trans=1)
        H2_13   = transform_tensor(H_tot,v2v13,trans=1)
        H2_23   = transform_tensor(H_tot,v2v23,trans=1)
 
        H3_12   = transform_tensor(H_tot,v3v12,trans=1)
        H3_13   = transform_tensor(H_tot,v3v13,trans=1)
        H3_23   = transform_tensor(H_tot,v3v23,trans=1)
 
        H12_13  = transform_tensor(H_tot,v12v13,trans=1)
        H12_23  = transform_tensor(H_tot,v12v23,trans=1)
 
        H13_23  = transform_tensor(H_tot,v13v23,trans=1)
   
    if n_body_order >= 3:
        H123_123 = transform_tensor(H_tot,v123v123,trans=1)
        H0_123  = transform_tensor(H_tot,v0v123,trans=1)
        H1_123  = transform_tensor(H_tot,v1v123,trans=1)
        H2_123  = transform_tensor(H_tot,v2v123,trans=1)
        H3_123  = transform_tensor(H_tot,v3v123,trans=1)
        H12_123 = transform_tensor(H_tot,v12v123,trans=1)
        H13_123 = transform_tensor(H_tot,v13v123,trans=1)
        H23_123 = transform_tensor(H_tot,v23v123,trans=1)
    
    H_ss2   = transform_tensor(H0_0,v0v0)

    H0_0    = H0_0.reshape(dim0,dim0)

    if n_body_order >= 1:
        H1_1    = H1_1.reshape(dim1,dim1)
        H2_2    = H2_2.reshape(dim2,dim2)
        H3_3    = H3_3.reshape(dim3,dim3)
        
        H0_1    = H0_1.reshape(dim0,dim1)
        H0_2    = H0_2.reshape(dim0,dim2)
        H0_3    = H0_3.reshape(dim0,dim3)
        H1_2    = H1_2.reshape(dim1,dim2)
        H1_3    = H1_3.reshape(dim1,dim3)
        H2_3    = H2_3.reshape(dim2,dim3)
        
    if n_body_order >= 2:
        H12_12  = H12_12.reshape(dim12,dim12)
        H13_13  = H13_13.reshape(dim13,dim13)
        H23_23  = H23_23.reshape(dim23,dim23)
        H0_12   = H0_12.reshape(dim0,dim12)
        H0_13   = H0_13.reshape(dim0,dim13)
        H0_23   = H0_23.reshape(dim0,dim23)
        H1_12   = H1_12.reshape(dim1,dim12)
        H1_13   = H1_13.reshape(dim1,dim13)
        H1_23   = H1_23.reshape(dim1,dim23)
        H2_12   = H2_12.reshape(dim2,dim12)
        H2_13   = H2_13.reshape(dim2,dim13)
        H2_23   = H2_23.reshape(dim2,dim23)
        H3_12   = H3_12.reshape(dim3,dim12)
        H3_13   = H3_13.reshape(dim3,dim13)
        H3_23   = H3_23.reshape(dim3,dim23)
        H12_13  = H12_13.reshape(dim12,dim13)
        H12_23  = H12_23.reshape(dim12,dim23)
        H13_23  = H13_23.reshape(dim13,dim23)
   
    if n_body_order >= 3:
        H123_123 = H123_123.reshape(dim123,dim123)
        H0_123  = H0_123.reshape(dim0,dim123)
        H1_123  = H1_123.reshape(dim1,dim123)
        H2_123  = H2_123.reshape(dim2,dim123)
        H3_123  = H3_123.reshape(dim3,dim123)
        H12_123 = H12_123.reshape(dim12,dim123)
        H13_123 = H13_123.reshape(dim13,dim123)
        H23_123 = H23_123.reshape(dim23,dim123)
    
    
#    singles = 0
#    doubles = 0
#    triples = 0
    
#    n0 = H0_0.shape[0]
#    n1 = H1_1.shape[0]
#    n2 = H2_2.shape[0]
#    n3 = H3_3.shape[0]
#    n12 = H12_12.shape[0]
#    n13 = H13_13.shape[0]
#    n23 = H23_23.shape[0]
#    n123 = H123_123.shape[0]
    

    H_ss = H0_0

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

    if n_body_order == 3:
        
        H0  = np.hstack((H0_0, H0_1, H0_2, H0_3, H0_12, H0_13, H0_23, H0_123))
        H1  = np.hstack((H0_1.T, H1_1, H1_2, H1_3, H1_12, H1_13, H1_23, H1_123))
        H2  = np.hstack((H0_2.T, H1_2.T, H2_2, H2_3, H2_12, H2_13, H2_23, H2_123))
        H3  = np.hstack((H0_3.T, H1_3.T, H2_3.T, H3_3, H3_12, H3_13, H3_23, H3_123))
        H12 = np.hstack((H0_12.T, H1_12.T, H2_12.T, H3_12.T, H12_12, H12_13, H12_23, H12_123))
        H13 = np.hstack((H0_13.T, H1_13.T, H2_13.T, H3_13.T, H12_13.T, H13_13, H13_23, H13_123))
        H23 = np.hstack((H0_23.T, H1_23.T, H2_23.T, H3_23.T, H12_23.T, H13_23.T, H23_23, H23_123))
        H123= np.hstack((H0_123.T, H1_123.T, H2_123.T, H3_123.T, H12_123.T, H13_123.T, H23_123.T, H123_123))
        

        H_ss = np.vstack((H0,H1,H2,H3,H12,H13,H23,H123))
    
    #print " H_ss2.shape",  H_ss2.shape
    H_ss2 = H_ss2.reshape(dim_tot,dim_tot)
    print " Dimension of Hamiltonian: ", H_ss.shape
    
    l0,v0 = np.linalg.eigh(H_ss)

    if 0:
        va = cp.deepcopy(v0[0])
        ha = cp.deepcopy(H_ss)
        for a in l0:
            print a

        print 
        print np.dot(v0[:,0],np.dot(H_ss,v0[:,0]))
        print np.einsum('i,ij,j->',va, ha, va)
        
        ha = ha.reshape(n_p_states+n_p_states)
        va = va.reshape(n_p_states)
        
        print va.shape, ha.shape
        print np.einsum('ijk,ijklmn,lmn->',va, ha, va)
        #exit(-1)

    if 0 and n_body_order == 0:
        v_gs = v0[:,0]
 
        dim_list = []
        for bi, b in enumerate(blocks):
            dim_list.extend([p_states[bi].shape[1]])
        print dim_list
        print v_gs.shape
        v_gs = v_gs.reshape(dim_list)
        #Acore, Atfac = tucker_decompose(v_gs,0,n_p_states)
        v_gs = transform_tensor(v_gs,vecs0)
        v_gs = v_gs.reshape(dim_tot)
        print " H_ss2.shape, v_gs.shape", H_ss2.shape, v_gs.shape
        print " Expectation value: %12.8f" % np.dot(v_gs,np.dot(H_ss2,v_gs))
        print " Norm:              %12.8f" % np.dot(v_gs,v_gs)

        v_gs = v_gs.reshape(dims_0)

        v_gs = transform_tensor(v_gs, vecs0, trans=1)
        tmp = transform_tensor(H_tot, v0v0, trans=1) 
        tmp = np.einsum('ijk,ijklmn,lmn->',v_gs,tmp,v_gs)
        print tmp


    # stuff for iterating, re-compose
    if 1:
        n0 = H0_0.shape[1]
       
        P_dim = n0
        Q_dims = [] # dimension of Q space for each block i.e., Q_dims[3] is the dimension of this space |abcDef...> 
        QQ_dims = [] # dimension of Q space for each block-dimer i.e., QQ_dims[3] is the dimension of this space |AbcdEf...> 
        QQQ_dims = []

        for bi,b in enumerate(blocks):
            q_dim = n0 / p_states[bi].shape[1] * q_states[bi].shape[1]
            Q_dims.extend([q_dim])

        for bi,b in enumerate(blocks):
            for bbi,bb in enumerate(blocks):
                if bbi > bi:
                    q_dim = n0 / p_states[bi].shape[1] / p_states[bbi].shape[1] * q_states[bi].shape[1] * q_states[bbi].shape[1]
                    QQ_dims.extend([q_dim])

        for bi,b in enumerate(blocks):
            for bbi,bb in enumerate(blocks):
                if bbi > bi:
                    for bbbi,bbb in enumerate(blocks):
                        if bbbi > bbi:
                            q_dim = n0 
                            q_dim = q_dim / p_states[bi].shape[1]   * q_states[bi].shape[1]
                            q_dim = q_dim / p_states[bbi].shape[1]  * q_states[bbi].shape[1]
                            q_dim = q_dim / p_states[bbbi].shape[1] * q_states[bbbi].shape[1]
                            QQQ_dims.extend([q_dim])
        
        print P_dim
        print Q_dims
        print QQ_dims
        print QQQ_dims
   
        #   
        #   Pick the target state to optimize tucker factors for. todo: extend this for a state-averaged tucker
        target_state = args['target_state'] 
        v = cp.deepcopy(v0[:,target_state])
      
        v_0 = v[0:P_dim]
        print "v_0", v_0
        print " norm of PPP component %12.8f" %np.dot(v_0,v_0)
        
        v_0 = v_0.reshape(n_p_states)

        vec_curr = transform_tensor(v_0, vecs0)
        
        if n_body_order >= 1:
            
            start = P_dim
            for bi,b in enumerate(blocks):
                print bi
                stop = start + Q_dims[bi]
                v_tmp = cp.deepcopy(v0[start:stop,target_state])
        
                # copy all P space vectors, and replace current block with Q vectors
                vecs_b = cp.deepcopy(vecs0)
                vecs_b[bi] = q_states[bi]
                
                dim_b = cp.deepcopy(n_p_states)
                dim_b[bi] = q_states[bi].shape[1]
                v_tmp = v_tmp.reshape(dim_b)
               
                # add this recomposed portion of the CI vector
                vec_curr += transform_tensor(v_tmp, vecs_b)
                start = stop
        
        if n_body_order >= 2:
            
            block_dimer_index = 0
            for bi,b in enumerate(blocks):
                for bbi,bb in enumerate(blocks):
                    if bbi > bi:

                        stop = start + QQ_dims[block_dimer_index]
                        v_tmp = cp.deepcopy(v0[start:stop,target_state])
                     
                        # copy all P space vectors, and replace current block with Q vectors
                        vecs_b = cp.deepcopy(vecs0)
                        vecs_b[bi] = q_states[bi]
                        vecs_b[bbi] = q_states[bbi]
                        
                        dim_b       = cp.deepcopy(n_p_states)
                        dim_b[bi]   = q_states[bi].shape[1]
                        dim_b[bbi]  = q_states[bbi].shape[1]
                        
                        v_tmp = v_tmp.reshape(dim_b)
                        
                        # add this recomposed portion of the CI vector
                        vec_curr += transform_tensor(v_tmp, vecs_b)

                        block_dimer_index += 1
                        print "QQ_dims, start, stop", start, stop
                        start = stop
        
        #Acore, Atfac = tucker_decompose_list(vec_curr,n_p_states)
        Acore, Atfac = tucker_decompose(vec_curr,0,0)
        p_states = []
        q_states = []
        for bi,b in enumerate(Atfac):
            p_states.extend([b[:,0:n_p_states[bi]]])
            q_states.extend([b[:,n_p_states[bi]::]])
      
        vec_curr = vec_curr.reshape(dim_tot)
        H_tot = H_tot.reshape(dim_tot, dim_tot)
        
        #vv = np.dot(vec_curr, vec_curr)
        #print " Expectation value: %12.8f"% (np.dot(vec_curr,np.dot(H_tot,vec_curr))/vv )
        #print " norm:              %12.8f"% vv 
#        H_tot = H_tot.reshape(dim_tot, dim_tot)
#        vv = np.dot(vec,vec)
#        print " Expectation value: %12.8f"% (np.dot(vec.transpose(),np.dot(H_tot,vec))/vv )
#        print " norm:              %12.8f"% vv 
#        
#        if 0:
#            # decompose
#            dim_list = []
#            for bi, b in enumerate(blocks):
#                dim_list.extend([p_states[bi].shape[1]])
#            print dim_list
#            vec = vec.reshape(dims_0)
#            p_states = []
#            q_states = []
#            Acore, Atfac = tucker_decompose(vec,0,0)
#            for bi,b in enumerate(Atfac):
#                #p_states.extend([scipy.linalg.orth(np.random.rand(b.shape[0],n_p_states))])
#                p_states.extend([b[:,0:n_p_states]])
#                q_states.extend([b[:,n_p_states::]])

    print " %5s    %12s  %12s  %12s" %("State","Energy","Relative","Exact")
    for si,i in enumerate(l0):
        print " %5i =  %12.8f  %12.8f  %12.8f" %(si,i*convert,(i-l0[0])*convert,(l[si]-l[0])*convert)
        if si>10:
            break
    
    print
    print " Energy  Error due to compression    :  %12.8f - %12.8f = %12.8f" %(l0[0],l[0],l0[0]-l[0])


    energy_per_iter += [l0[0]]

    thresh = 1e-8
    if it > 0:
        if abs(l0[0]-energy_per_iter[it-1]) < thresh:
            break

print " %10s  %12s  %12s" %("Iteration", "Energy", "Delta")
for ei,e in enumerate(energy_per_iter):
    if ei>0:
        print " %10i  %12.8f  %12.8f" %(ei,e,e-energy_per_iter[ei-1])
    else:
        print " %10i  %12.8f" %(ei,e)



