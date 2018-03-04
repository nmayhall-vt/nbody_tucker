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


def printm(m):
    # {{{
    """ print matrix """
    for r in m:
        for ri in r:
            print "%10.1e" %ri,
        print
    # }}}

def get_guess_vectors(lattice, j12, blocks, n_p_states, n_q_states):
    # {{{
    print " Generate initial guess vectors"
    p_states = []
    q_states = []
    for bi,b in enumerate(blocks):
        # form block hamiltonian
        j_b = np.zeros([len(b),len(b)])
        j_b = j12[:,b][b]
        lat_b = lattice[b]
        H_b, tmp, S2_b, Sz_b = form_hdvv_H(lat_b,j_b)
    
        # Diagonalize an arbitrary linear combination of the quantum numbers we insist on preserving
        l_b,v_b = np.linalg.eigh(H_b + Sz_b + S2_b) 
        l_b = v_b.transpose().dot(H_b).dot(v_b).diagonal()
       
        Sz_b = v_b.transpose().dot(Sz_b).dot(v_b).diagonal()
        
        sort_ind = np.argsort(l_b)
        l_b = l_b[sort_ind]
        v_b = v_b[:,sort_ind]
            

        print " Guess eigenstates"
        for l in range(len(l_b)):
            print " %6i %12.8f %12.8f" %(l,l_b[l],Sz_b[l])
        p_states.extend([v_b[:,0:n_p_states[bi]]])
        #q_states.extend([v_b[:,n_p_states[bi]::]])
        q_states.extend([v_b[:,n_p_states[bi]: n_p_states[bi]+n_q_states[bi]]])
    return p_states, q_states
    # }}}






       


"""
Test forming HDVV Hamiltonian and projecting onto "many-body tucker basis"
"""
#   Setup input arguments
parser = argparse.ArgumentParser(description='Finds eigenstates of a spin lattice',
formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#parser.add_argument('-d','--dry_run', default=False, action="store_true", help='Run but don\'t submit.', required=False)
parser.add_argument('-ju','--j_unit', type=str, default="cm", help='What units are the J values in', choices=['cm','ev'],required=False)
parser.add_argument('-l','--lattice', type=str, default="heis_lattice.m", help='File containing vector of sizes number of electrons per lattice site', required=False)
parser.add_argument('-j','--j12', type=str, default="heis_j12.m", help='File containing matrix of exchange constants', required=False)
parser.add_argument('-b','--blocks', type=str, default="heis_blocks.m", help='File containing vector of block sizes', required=False)
#parser.add_argument('-s','--save', default=False, action="store_true", help='Save the Hamiltonian and S2 matrices', required=False)
#parser.add_argument('-r','--read', default=False, action="store_true", help='Read the Hamiltonian and S2 matrices', required=False)
#parser.add_argument('-hdvv','--hamiltonian', type=str, default="heis_hamiltonian.npy", help='File containing matrix of Hamiltonian', required=False)
#parser.add_argument('-s2','--s2', type=str, default="heis_s2.npy", help='File containing matrix of s2', required=False)
#parser.add_argument('--eigvals', type=str, default="heis_eigvals.npy", help='File of Hamiltonian eigvals', required=False)
#parser.add_argument('--eigvecs', type=str, default="heis_eigvecs.npy", help='File of Hamiltonian eigvecs', required=False)
parser.add_argument('-np','--n_p_space', type=int, nargs="+", help='Number of vectors in block P space', required=False)
parser.add_argument('-nq','--n_q_space', type=int, nargs="+", help='Number of vectors in block Q space', required=False)
parser.add_argument('-nb','--n_body_order', type=int, default="0", help='n_body spaces', required=False)
parser.add_argument('-nr','--n_roots', type=int, default="10", help='Number of eigenvectors to find in compressed space', required=False)
parser.add_argument('--n_print', type=int, default="10", help='number of states to print', required=False)
parser.add_argument('--use_exact_tucker_factors', action="store_true", default=False, help='Use compression vectors from tucker decomposition of exact ground states', required=False)
parser.add_argument('-ts','--target_state', type=int, default="0", nargs='+', help='state(s) to target during (possibly state-averaged) optimization', required=False)
parser.add_argument('-mit', '--max_iter', type=int, default=10, help='Max iterations for solving for the compression vectors', required=False)
parser.add_argument('--thresh', type=int, default=8, help='Threshold for pspace iterations', required=False)
parser.add_argument('-pt','--pt_order', type=int, default=2, help='PT correction order ?', required=False)
parser.add_argument('-pt_type','--pt_type', type=str, default='mp', choices=['mp','en'], help='PT correction denominator type', required=False)
parser.add_argument('-ms','--target_ms', type=float, default=0, help='Target ms space', required=False)
parser.add_argument('-opt','--optimization', type=str, default="None", help='Optimization algorithm for Tucker factors',choices=["none", "diis"], required=False)
args = vars(parser.parse_args())
#
#   Let minute specification of walltime override hour specification

j12 = np.loadtxt(args['j12'])
lattice = np.loadtxt(args['lattice']).astype(int)
blocks = np.loadtxt(args['blocks']).astype(int)
n_sites = len(lattice)
n_blocks = len(blocks)
    
if len(blocks.shape) == 1:
    print 'blocks',blocks
    
    blocks.shape = (1,len(blocks.transpose()))
    n_blocks = len(blocks)


n_p_states = args['n_p_space'] 
n_q_states = args['n_q_space'] 


if args['n_p_space'] == None:
    n_p_states = []
    for bi in range(n_blocks):
        n_p_states.extend([1])
    args['n_p_space'] = n_p_states

assert(len(args['n_p_space']) == n_blocks)

np.random.seed(2)


au2ev = 27.21165;
au2cm = 219474.63;
convert = au2ev/au2cm;	# convert from wavenumbers to eV
convert = 1;			# 1 for wavenumbers

if args['j_unit'] == 'cm':
    j12 = j12 * au2ev/au2cm

print " j12:\n", j12
print " lattice:\n", lattice 
print " blocks:\n", blocks
print " n_blocks:\n", n_blocks

H_tot = np.array([])
S2_tot = np.array([])
H_dict = {}




# calculate problem dimensions 
dims_tot = []
dim_tot = 1
for bi,b in enumerate(blocks):
    block_dim = np.power(2,b.shape[0])
    dims_tot.extend([block_dim])
    dim_tot *= block_dim

if args['n_q_space'] == None:
    n_q_states = []
    for bi in range(n_blocks):
        n_q_states.extend([dims_tot[bi]-n_p_states[bi]])
    args['n_q_space'] = n_q_states


# Get initial compression vectors 
p_states, q_states = get_guess_vectors(lattice, j12, blocks, n_p_states, n_q_states)



"""
H = -2 \sum_{ij} J_{ij} S_i\cdotS_j
  = - \sum_{ij} S^+_i S^-_j  +  S^-_i S^+_j  + 2 S^z_i S^z_j

<abc| H12 |def> = - \sum_{ij}  <a|S+i|d><b|S-j|e> <c|f> 
                             + <a|S-i|d><b|S+j|e> <c|f>
                             +2<a|Szi|d><b|Szj|e> <c|f> 

Form matrix representations of S+, S-, Sz in local basis

Sp = {}
Sp[bi] = Op(i,a,b)    # bi = block, i=site, a=bra, b=ket

<Abc| H12 | aBd> = -\sum_{ij\in 12}   <A|S+i|a><b|S-j|B><c|d>
                                    + <A|S-i|a><b|S+j|B><c|d>
                                    +2<A|Szi|a><b|Szj|B><c|d>


"""

#build operators
op_Sp = {}
op_Sm = {}
op_Sz = {}
local_states = {}   
                    

blocks_in = cp.deepcopy(blocks)
lattice_blocks = {}         # dictionary of block objects


#
#   Initialize Block objects
print 
print " Prepare Lattice Blocks:"
print n_p_states, n_q_states
for bi in range(0,n_blocks):
    lattice_blocks[bi] = Lattice_Block()
    lattice_blocks[bi].init(bi,blocks_in[bi,:],[n_p_states[bi], n_q_states[bi]])

    lattice_blocks[bi].np = n_p_states[bi] 
    lattice_blocks[bi].nq = n_q_states[bi] 
    lattice_blocks[bi].vecs = np.hstack((p_states[bi],q_states[bi]))
    
    lattice_blocks[bi].extract_lattice(lattice)
    lattice_blocks[bi].extract_j12(j12)

    lattice_blocks[bi].form_H()
    lattice_blocks[bi].form_site_operators()

    print lattice_blocks[bi]

n_body_order = args['n_body_order'] 
    
dim_tot = 0

tb_0 = Tucker_Block()
address_0 = np.zeros(n_blocks,dtype=int)
tb_0.init((-1), lattice_blocks,address_0, dim_tot)

dim_tot += tb_0.full_dim

tucker_blocks = {}
tucker_blocks[0,-1] = tb_0 

print 
print " Prepare Tucker blocks:"
if n_body_order >= 1:
    for bi in range(0,n_blocks):
        tb = Tucker_Block()
        address = np.zeros(n_blocks,dtype=int)
        address[bi] = 1
        tb.init((bi), lattice_blocks,address, dim_tot)
        tucker_blocks[1,bi] = tb
        dim_tot += tb.full_dim
if n_body_order >= 2:
    for bi in range(0,n_blocks):
        for bj in range(bi+1,n_blocks):
            tb = Tucker_Block()
            address = np.zeros(n_blocks,dtype=int)
            address[bi] = 1
            address[bj] = 1
            tb.init((bi,bj), lattice_blocks,address,dim_tot)
            tucker_blocks[2,bi,bj] = tb
            dim_tot += tb.full_dim
if n_body_order >= 3:
    for bi in range(0,n_blocks):
        for bj in range(bi+1,n_blocks):
            for bk in range(bj+1,n_blocks):
                tb = Tucker_Block()
                address = np.zeros(n_blocks,dtype=int)
                address[bi] = 1
                address[bj] = 1
                address[bk] = 1
                tb.init((bi,bj,bk), lattice_blocks,address,dim_tot)
                tucker_blocks[3,bi,bj,bk] = tb
                dim_tot += tb.full_dim
if n_body_order >= 4:
    for bi in range(0,n_blocks):
        for bj in range(bi+1,n_blocks):
            for bk in range(bj+1,n_blocks):
                for bl in range(bk+1,n_blocks):
                    tb = Tucker_Block()
                    address = np.zeros(n_blocks,dtype=int)
                    address[bi] = 1
                    address[bj] = 1
                    address[bk] = 1
                    address[bl] = 1
                    tb.init((bi,bj,bk,bl), lattice_blocks,address,dim_tot)
                    tucker_blocks[4,bi,bj,bk,bl] = tb
                    dim_tot += tb.full_dim
if n_body_order >= 5:
    for bi in range(0,n_blocks):
        for bj in range(bi+1,n_blocks):
            for bk in range(bj+1,n_blocks):
                for bl in range(bk+1,n_blocks):
                    for bm in range(bl+1,n_blocks):
                        tb = Tucker_Block()
                        address = np.zeros(n_blocks,dtype=int)
                        address[bi] = 1
                        address[bj] = 1
                        address[bk] = 1
                        address[bl] = 1
                        address[bm] = 1
                        tb.init((bi,bj,bk,bl,bm), lattice_blocks,address,dim_tot)
                        tucker_blocks[5,bi,bj,bk,bl,bm] = tb
                        dim_tot += tb.full_dim

for tb in sorted(tucker_blocks):
    print tucker_blocks[tb], " Range= %8i:%-8i" %( tucker_blocks[tb].start, tucker_blocks[tb].stop)







# loop over compression vector iterations
energy_per_iter = []
maxiter = args['max_iter'] 
last_vector = np.array([])  # used to detect root flipping

diis_err_vecs = {}
diis_frag_grams = {}

opt = args['optimization'] 
ts = args['target_state'] 

for bi in range(0,n_blocks):
    diis_frag_grams[bi] = []

for it in range(0,maxiter):
    print 
    print " Build Hamiltonian:"
    H,S2 = build_tucker_blocked_H(n_blocks, tucker_blocks, lattice_blocks, n_body_order, j12) 
   
    print " Make full system"
    H_tot, tmp, S2_tot, Sz_tot = form_hdvv_H(lattice,j12)
    H_tot.shape = (2,2,2,2,2,2,2,2)
    alpha = np.array([1,0])
    beta  = np.array([0,1])
    s0 = np.kron(alpha,alpha)
    s1 = (np.kron(alpha,beta)-np.kron(beta,alpha)) / np.sqrt(2)
    s0.shape = (4,1)
    s1.shape = (4,1)
    print s0
    print s1
    W = s1.dot(s0.T)
    U,s,V = np.linalg.svd(W,full_matrices=1)
    W = U.dot(V)
    print "W s0="
    print W.dot(s0)
    print "W' s1="
    print W.T.dot(s1)
    print 
    print W
    W.shape = (2,2,2,2)

    
    U1 = lattice_blocks[0].vecs[:,[0,1,2,3]]
    U2 = lattice_blocks[1].vecs[:,[0,1,2,3]]
    U1 = lattice_blocks[0].v_ss(0)
    U2 = lattice_blocks[1].v_ss(0)
   
    U1 = np.array([[0,1,-1,0],[0,1,1,0]])/np.sqrt(2)
    U1 = U1.T
    U2 = U1
    #U1 = scipy.linalg.orth(np.random.rand(4,2))
    #U2 = scipy.linalg.orth(np.random.rand(4,2))

    #U1 = np.eye(4)
    #U2 = np.eye(4)
    #U1.shape = (2,2) 
    #U2.shape = (2,2) 
   
    l0 = np.kron(np.eye(2),W)
    l0 = np.kron(l0,np.eye(2))
    l0.shape = (16,16)
    print  "l0.shape: ", l0.shape
    l1 = np.kron(U1,U2)
    print  "l1.shape: ", l1.shape
    print l1
    print H_tot.shape
    
    H = H_tot
    S2 = S2_tot
    Sz = Sz_tot
  
    U = np.eye(4)
    U = .1*np.random.rand(4,4)
    U,s,V = np.linalg.svd(U)
    U = U.dot(V)
    U.shape = (2,2,2,2)
    print H.shape
    print W.shape
    #H = np.einsum("imjnkolp,nqor->imjqkrlp",H,U)
    #H = np.einsum("imjnkolp,qnro->imjqkrlp",H,U)
    #H = np.einsum("iqrlmnop,jkqr->ijklmnop",H,U)
    #H = np.einsum("ijklmqrp,noqr->ijklmnop",H,U)
    #H = np.einsum("ijklmnop,jkqr->iqrlmnop",H,U)
    #H = np.einsum("ijklmnop,noqr->ijklmqrp",H,U)
  
    w1 = U1
    w2 = U2
    
    w1.shape = (2,2,2)
    w2.shape = (2,2,2)
    G_old = np.zeros((4,4))
    print " %10s %12s %12s"%("Iteration", "tr(GU)", "-sum(s)") 
    for it in range(100):
        G = np.einsum("ijklmnop,qrno->ijklmqrp",H,U)
        G = np.einsum("ijklmnop,mnq->ijklqop",G,w1)
        G = np.einsum("ijklmno,nop->ijklmp",G,w2)
        sigma = G
        G = np.einsum("ijklmn,opjk->ioplmn",G,U)
        G = np.einsum("ijklmn,ijo->oklmn",G,w1)
        G = np.einsum("ijklm,jkn->inlm",G,w2)
        
        G.shape = (4,4)
        l,v = np.linalg.eigh(G)
        v = v[:,0]
        v.shape = (2,2)
        G.shape = (2,2,2,2)
        
        G = np.einsum("ijkl,ij->kl",G,v)
        G = np.einsum("ij,ij->",G,v)
        E_curr = G
       
        sigma = np.einsum("ijklmn,mn->ijkl",sigma,v)

        # get environment for u 
        Gu = np.einsum("ijkl,imn,olp,np->jkmo",sigma,w1,w2,v)
      
        U_old = U
        Gu = Gu.reshape(4,4)
        u0,s0,v0 = np.linalg.svd(Gu)
        U = -v0.T.dot(u0.T)
        U.shape = (2,2,2,2)
        Gu.shape = (2,2,2,2)
        Eu = np.einsum("ijkl,klij->",Gu,U)
       
        for it2 in range(1):
            # get environment for w1 
            #G1 = np.einsum("ijkl,mnjk->imnl",sigma,U)
            #G1 = np.einsum("imnl,nlp->imp",G1,w2)
            #G1 = np.einsum("imp,op->imo",G1,v)
            G1 = np.einsum("ijkl,mnjk,nlp,op->imo",sigma,U,w2,v)
            w1_old = w1 
            
            G1 = G1.reshape(4,2)
            u1,s1,v1 = np.linalg.svd(G1)
            u1 = u1[:,0:2]
            v1 = v1[0:2,:]
            w1 = -u1.dot(v1)
            w1.shape = (2,2,2)
            G1.shape = (2,2,2)
            E1 = np.einsum("ijk,ijk->",G1,w1)
             
            G2 = np.einsum("ijkl,mnjk,imo,op->nlp",sigma,U,w1_old,v)
            G2 = G2.reshape(4,2)
            u2,s2,v2 = np.linalg.svd(G2)
            u2 = u2[:,0:2]
            v2 = v2[0:2,:]
            w2 = -u2.dot(v2)
            w2.shape = (2,2,2)
            G2.shape = (2,2,2)
            E2 = np.einsum("ijk,ijk->",G2,w2)
        
        
        print " Energy: %12.8f : %12.8f %12.8f %12.8f" %( E_curr, Eu, E1, E2)

    print
    print
    print

    W.shape = (4,4)
    H.shape = (16,16)
    S2.shape = (16,16)
    Sz.shape = (16,16)
    
    H  = l0.T.dot(H).dot(l0)
    S2 = l0.T.dot(S2).dot(l0)
    Sz = l0.T.dot(Sz).dot(l0)
    
    H = l1.T.dot(H).dot(l1)
    S2 = l1.T.dot(S2).dot(l1)
    Sz = l1.T.dot(Sz).dot(l1)
  
    print "Sz:"
    print Sz.diagonal()
    print " Diagonalize Hamiltonian: Size of H: ", H.shape
    l = np.array([])
    v = np.array([])
    if H.shape[0] > 3000:
        l,v = scipy.sparse.linalg.eigsh(H, k=args["n_roots"] )
    else:
        l,v = np.linalg.eigh(H)
    
    S2 = v.T.dot(S2).dot(v)
    print " %5s    %16s  %16s  %12s" %("State","Energy","Relative","<S2>")
    for si,i in enumerate(l):
        if si<args['n_print']:
            print " %5i =  %16.8f  %16.8f  %12.8f" %(si,i*convert,(i-l[0])*convert,abs(S2[si,si]))


    energy_per_iter.append(l[ts]) 
    thresh = 1.0*np.power(10.0,-float(args['thresh']))
    if it > 0:
        if abs(l[ts]-energy_per_iter[it-1]) < thresh:
            break


    brdms = {}   # block reduced density matrix
    for bi in range(0,n_blocks):
        Bi = lattice_blocks[bi]
        brdms[bi] = np.zeros(( Bi.full_dim, Bi.full_dim )) 

    print
    print " Compute Block Reduced Density Matrices (BRDM):"
    for tb1 in sorted(tucker_blocks):
        Tb1 = tucker_blocks[tb1]
        vb1 = cp.deepcopy(v[Tb1.start:Tb1.stop, ts])
        vb1.shape = Tb1.block_dims
        for tb2 in sorted(tucker_blocks):
            Tb2 = tucker_blocks[tb2]
            
            if Tb1.id <= Tb2.id:
                # How many blocks are different between left and right?
                different = []
                for bi in range(0,n_blocks):
                    if Tb2.address[bi] != Tb1.address[bi]:
                        different.append(bi)
                
                if len(different) == 0:
                    vb2 = cp.deepcopy(v[Tb2.start:Tb2.stop, ts])
                    vb2.shape = Tb2.block_dims
                    for bi in range(0,n_blocks):
                        brdm_tmp = form_1fdm(vb1,vb2,[bi])
                        Bi = lattice_blocks[bi]
                        u1 = Bi.v_ss(Tb1.address[bi])
                        u2 = Bi.v_ss(Tb2.address[bi])
                        brdms[bi] += u1.dot(brdm_tmp).dot(u2.T)
                if len(different) == 1:
                    vb2 = cp.deepcopy(v[Tb2.start:Tb2.stop, ts])
                    vb2.shape = Tb2.block_dims
                    bi = different[0]
                    brdm_tmp = form_1fdm(vb1,vb2,[bi])
                    Bi = lattice_blocks[bi]
                    u1 = Bi.v_ss(Tb1.address[bi])
                    u2 = Bi.v_ss(Tb2.address[bi])
                    brdm_tmp = u1.dot(brdm_tmp).dot(u2.T)
                    brdms[bi] += brdm_tmp + brdm_tmp.T 
    
    for b in brdms:
        print "trace(b): %12.8f" % np.trace(brdms[b])


    if 0:
        print
        print " Compute Block Reduced Density Matrices (BRDM) FULL!:"
# {{{
        dim_tot_list = []
        full_dim = 1
        sum_1 = 0
        for bi in range(0,n_blocks):
            dim_tot_list.append(lattice_blocks[bi].full_dim)
            full_dim *= lattice_blocks[bi].full_dim
        
        vec_curr = np.zeros(dim_tot_list)
        for tb1 in sorted(tucker_blocks):
            Tb1 = tucker_blocks[tb1]
            vb1 = cp.deepcopy(v[Tb1.start:Tb1.stop, ts])
            sum_1 += vb1.T.dot(vb1)
            vb1.shape = Tb1.block_dims
 
            vec = []
            for bi in range(0,n_blocks):
                Bi = lattice_blocks[bi]
                vec.append((Bi.v_ss(Tb1.address[bi])))

            #sign = 1
            #print " tb1: ", tb1
            #if tb2[0] != 0:
            #    sign = -1
            #vec_curr += sign * transform_tensor(vb1,vec)
            vec_curr += transform_tensor(vb1,vec)
        Acore, Atfac = tucker_decompose(vec_curr,0,0)
        vec_curr.shape = (full_dim,1)
        print "v'v:   %12.8f" % vec_curr.T.dot(vec_curr)
        print "sum_1: %12.8f" % sum_1 
 # }}}


    print
    print " Compute Eigenvalues of BRDMs:"
    for bi in range(0,n_blocks):
        Bi = lattice_blocks[bi]

        brdm_curr = brdms[bi] + Bi.full_S2
        if opt == "diis":
            n_diis_vecs = 8 
            proj_p = Bi.v_ss(0).dot(Bi.v_ss(0).T)
            error_vector = proj_p.dot(brdm_curr) - (brdm_curr).dot(proj_p)
            error_vector.shape = (error_vector.shape[0]*error_vector.shape[1],1)
           
            print " Dimension of Error Vector matrix: ", Bi.diis_vecs.shape
            if Bi.diis_vecs.shape[0] == 0:
                Bi.diis_vecs = error_vector
            else:
                Bi.diis_vecs = np.hstack( (Bi.diis_vecs, error_vector) ) 

            diis_frag_grams[bi].append( brdm_curr )
            
            n_evecs = Bi.diis_vecs.shape[1]
            
            if it>0:
                S = Bi.diis_vecs.T.dot(Bi.diis_vecs )
                
                collapse = 1
                if collapse:
                    sort_ind = np.argsort(S.diagonal())
                    sort_ind = [sort_ind[i] for i in range(0,min(len(sort_ind),n_diis_vecs))]
                    Bi.diis_vecs = Bi.diis_vecs[:,sort_ind]
                    tmp = []
                    for i in sort_ind:
                        tmp.append(diis_frag_grams[bi][i])
                    diis_frag_grams[bi] = cp.deepcopy(tmp)
                    print " Vector errors", S.diagonal()[sort_ind] 
                    n_evecs = Bi.diis_vecs.shape[1]
                    S = Bi.diis_vecs.T.dot(Bi.diis_vecs )
                print " Number of error vectors: %4i " %n_evecs
                B = np.ones( (n_evecs+1, n_evecs+1) )
                B[-1,-1] = 0
                B[0:-1,0:-1] = cp.deepcopy(S) 
                r = np.zeros( (n_evecs+1,1) )
                r[-1] = 1
                if n_evecs > 0: 
                    x = np.linalg.pinv(B).dot(r)
                    
                    extrap_err_vec = np.zeros((Bi.diis_vecs.shape[0]))
                    extrap_err_vec.shape = (extrap_err_vec.shape[0])

                    for i in range(0,x.shape[0]-1):
                        brdm_curr += x[i]*diis_frag_grams[bi][i]
                        extrap_err_vec += x[i]*Bi.diis_vecs[:,i]
                    
                    print " DIIS Coeffs"
                    for i in x:
                        print "  %12.8f" %i
                    #print x.T
                    print " CURRENT           error vector %12.2e " % error_vector.T.dot(error_vector)

        lx,vx = np.linalg.eigh(brdm_curr + Bi.full_S2)
            
        lx = vx.T.dot(brdms[bi]).dot(vx).diagonal()
       
        sort_ind = np.argsort(lx)[::-1]
        lx = lx[sort_ind]
        vx = vx[:,sort_ind]
        
        vp = vx[:,0:Bi.ss_dims[0]]
        vq = vx[:,Bi.ss_dims[0]:Bi.ss_dims[0]+Bi.ss_dims[1]]
       
        tmp, up = np.linalg.eigh(vp.T.dot(brdms[bi] + Bi.full_H + Bi.full_Sz + Bi.full_S2).dot(vp))
        tmp, uq = np.linalg.eigh(vq.T.dot(brdms[bi] + Bi.full_H + Bi.full_Sz + Bi.full_S2).dot(vq))
        
        vp = vp.dot(up)
        vq = vq.dot(uq)

        sort_ind = np.argsort(vp.T.dot(brdms[bi]).dot(vp).diagonal() )[::-1]
        vp = vp[:,sort_ind]
        sort_ind = np.argsort(vq.T.dot(brdms[bi]).dot(vq).diagonal() )[::-1]
        vq = vq[:,sort_ind]
        
        v = np.hstack( ( vp,vq) )

        sz = v.T.dot(Bi.full_Sz).dot(v).diagonal()
        s2 = v.T.dot(Bi.full_S2).dot(v).diagonal()
        lx = v.T.dot(brdms[bi]).dot(v).diagonal()
        h = v.T.dot(Bi.full_H).dot(v).diagonal()
        
        #update Block vectors
        Bi.vecs[:,0:Bi.ss_dims[0]+Bi.ss_dims[1]] = v
        Bi.form_H()
        Bi.form_site_operators()

        print    
        print    
        print "   Eigenvalues of BRDM for ", lattice_blocks[bi]
        print "   -----------------------------------------------------------------------------"   
        print "   %-12s   %16s  %16s  %12s  %12s "%("Local State", "Occ. Number", "<H>", "<S2>", "<Sz>")
        for si,i in enumerate(lx):
            print "   %-12i   %16.8f  %16.8f  %12.4f  %12.4f "%(si,lx[si],h[si],abs(s2[si]),sz[si])
            #print "   %-4i   %16.8f  %16.8f  %16.4f "%(si,lx[si],h[si],sz[i])
        """
        print "   %-12s " %("----")
        print "   %-12s   %16.8f  %16.8f  %12.4f  %12.4f" %(
                "Trace"
                ,(grams[fi]).trace()
                ,Hi[fi].dot(grams[fi]).trace()
                ,S2i[fi].dot(grams[fi]).trace()
                ,Szi[fi].dot(grams[fi]).trace()
                )
                """

print " %10s  %12s  %12s" %("Iteration", "Energy", "Delta")
for ei,e in enumerate(energy_per_iter):
    if ei>0:
        print " %10i  %12.8f  %12.1e" %(ei,e,e-energy_per_iter[ei-1])
    else:
        print " %10i  %12.8f  %12s" %(ei,e,"")


