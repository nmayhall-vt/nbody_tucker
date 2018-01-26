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
import block3 


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
        
        sort_ind = np.argsort(l_b)
        l_b = l_b[sort_ind]
        v_b = v_b[:,sort_ind]
            

        print " Guess eigenstates"
        for l in l_b:
            print "%12.8f" %l
        p_states.extend([v_b[:,0:n_p_states[bi]]])
        #q_states.extend([v_b[:,n_p_states[bi]::]])
        q_states.extend([v_b[:,n_p_states[bi]: n_p_states[bi]+n_q_states[bi]]])
    return p_states, q_states
    # }}}

def get_H_compressed_block_states(block_basis_list,j12):
    """
    Build Hamiltonian for this sublattice, get ground state and Tucker 
        decompose this to get a compressed set of block states
    
    lattice_blocks is a list of Lattice_Block objects
    """
   
    lb_curr = block3.Lattice_Block()
    sites = []
    for lb in lattice_blocks:
        sites.extend(lb.sites)
    
    lb_curr.init(-1,sites,j12)

    lattice = [1]*lb_curr.n_sites
    H, tmp, S2i, Szi = form_hdvv_H(lattice,j12)
    
    return block_basis_list 




       


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
parser.add_argument('-pns','--pns_thresh', type=float, default=1e-8, help='Threshold for pair-natural-state (pns) truncations iterations', required=False)
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
#p_states, q_states = get_guess_vectors(lattice, j12, blocks, n_p_states, n_q_states)



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


##
##   Initialize Block objects
#print 
#print " Prepare Lattice Blocks:"
#print n_p_states, n_q_states
#for bi in range(0,n_blocks):
#    lattice_blocks[bi] = Lattice_Block()
#    lattice_blocks[bi].init(bi,blocks_in[bi,:],[n_p_states[bi], n_q_states[bi]])
#
#    lattice_blocks[bi].np = n_p_states[bi] 
#    lattice_blocks[bi].nq = n_q_states[bi] 
#    lattice_blocks[bi].vecs = np.hstack((p_states[bi],q_states[bi]))
#    
#    lattice_blocks[bi].extract_lattice(lattice)
#    lattice_blocks[bi].extract_j12(j12)
#
#    lattice_blocks[bi].form_H()
#    lattice_blocks[bi].form_site_operators()
#
#    print lattice_blocks[bi]

n_body_order = args['n_body_order'] 
    
dim_tot = 0

tb_0 = Tucker_Block()
address_0 = np.zeros(n_blocks,dtype=int)
tb_0.init((-1), lattice_blocks,address_0, dim_tot)

dim_tot += tb_0.full_dim



# New lattice blocks
#PPP states
print("\n Set up Lattice_Blocks")
lattice_blocks2 = []
for bi in range(0,n_blocks):
    lb = block3.Lattice_Block()
    lb.init(bi,blocks_in[bi,:],j12)
    lb.form_site_operators()
    lb.form_H()
    lattice_blocks2.append(lb)
    print(lb)

block_basis = {}
print("\n Set up P states and 1b Q states")
for bi in range(0,n_blocks):
    lb = lattice_blocks2[bi]
    
    # Form full Hamiltonian for Lattice_Block bi
    lattice = [1]*lb.n_sites  # assume spin-1/2 lattice for now 
    Hi, tmp, S2i, Szi = form_hdvv_H(lattice, lb.j12)  # rewrite this
   
    e, v = np.linalg.eigh(Hi + .1*S2i + .01*Szi)

    e = v.T.dot(Hi).dot(v).diagonal()
    
    sort_ind = np.argsort(e)
    e = e[sort_ind]
    v = v[:,sort_ind]
    
    S2i = v.T.dot(S2i).dot(v)
    print "\n Single Block Eigenstates:"
    for idx,ei in enumerate(e):
        print" %12i %12.8f %8.4f"%(idx,ei,S2i[idx,idx])
   
    # 
    # For now, just choose n lowest energy states for P space, but later 
    #   we may which to choose n lowest energy of specify ms value
    #
    p = v[:,0:n_p_states[bi]]
    q = v[:,n_p_states[bi]:n_q_states[bi]+1]

    bb_p = block3.Block_Basis(lb,"P")
    bb_q = block3.Block_Basis(lb,"Q")

    bb_p.set_vecs(p)
    bb_q.set_vecs(q)

    #
    #   address the block_basis by a tuple (lb.index,label) 
    block_basis[(lb.index,bb_p.label)] = bb_p
    block_basis[(lb.index,bb_q.label)] = bb_q



tucker_blocks = {}

print("\n Set up Tucker_Block objects")

tb_0 = block3.Tucker_Block()
for bi in range(0,n_blocks):
    tb_0.add_block(block_basis[(bi,"P")])

tucker_blocks[0] = tb_0 

if n_body_order >= 1:
    for bi in range(0,n_blocks):
        
        tb = cp.deepcopy(tb_0)
        tb.label = (1,bi)
        tb.set_start(dim_tot)
        tb.set_block(block_basis[(bi,"Q")]) 

        tucker_blocks[tb.label] = tb
        
        dim_tot += tb.full_dim

if n_body_order >= 2:
    for bi in range(0,n_blocks):
        for bj in range(bi+1,n_blocks):
    
            print "\n Form basis for the following pair of blocks: ", bi, bj
            
            #
            #   create a small tucker block object involving only Blocks bi and bj 
            #   to create a small dimer hamiltonian for the full hilbert space on bi and bj
            bbi = cp.deepcopy(block_basis[(bi,"P")])
            bbj = cp.deepcopy(block_basis[(bj,"P")])
            
            bbi.append(block_basis[(bi,"Q")])
            bbj.append(block_basis[(bj,"Q")])

            bbi.label = "Q(%i|%i)"%(bi,bj)
            bbj.label = "Q(%i|%i)"%(bi,bj)

            tb_ij = block3.Tucker_Block()
            tb_ij.add_block(bbi)
            tb_ij.add_block(bbj)
          

            #
            #   Build and diagonalize dimer Hamiltonian
            H,S2 = block3.build_H(tb_ij, tb_ij, j12)
    
            e, v_curr = scipy.sparse.linalg.eigsh(H,1)
                
            s = v_curr.T.dot(S2).dot(v_curr)
            print " Energy of GS: %12.8f %12.8f <S2>: "%(e[0],s[0,0])
          
           
            #
            #   Tucker decompose the ground state to get 
            #   a new set of Q vectors for this dimer
            v_curr = v_curr[:,0]
            v_curr.shape = (tb_ij.block_dims)
           
            n_p_i = block_basis[(bi,"P")].n_vecs
            n_p_j = block_basis[(bj,"P")].n_vecs

            v_curr = v_curr[n_p_i::,n_p_j::]

            pns_thresh = args["pns_thresh"]
            v_comp, U = tucker_decompose(v_curr,pns_thresh,0)
           
            vi = tb_ij.blocks[0].vecs[:,n_p_i::].dot(U[0])
            vj = tb_ij.blocks[1].vecs[:,n_p_j::].dot(U[1])

            #test_1 = 0
            #if test_1:
            #    """ 
            #    use this to check how much benefit we get from using 
            #    pair specific states
            #    """
            #    nkeep_i = U[0].shape[1]
            #    nkeep_j = U[1].shape[1]
            #    vi = block_basis[(bi,"Q")].vecs[:,0:nkeep_i]
            #    vj = block_basis[(bj,"Q")].vecs[:,0:nkeep_j]

            #
            #   Create a new Tucker_Block instance with these vectors and 
            #   add it to the active list
            tb = block3.Tucker_Block()
            tb = cp.deepcopy(tb_0)
            tb.set_start(dim_tot)
             
            bbi = cp.deepcopy(block_basis[(bi,"Q")])
            bbj = cp.deepcopy(block_basis[(bj,"Q")])
            bbi.label = "Q(%i|%i)"%(bi,bj)
            bbj.label = "Q(%i|%i)"%(bi,bj)

            bbi.set_vecs(vi)
            bbj.set_vecs(vj)
            
            tb.set_block(bbi)
            tb.set_block(bbj)
            
            tb.refresh_dims()
            tb.update_label()
            
            tucker_blocks[tb.label] = tb
            
            dim_tot += tb.full_dim
            
if n_body_order >= 3:
    for bi in range(0,n_blocks):
        for bj in range(bi+1,n_blocks):
            for bk in range(bj+1,n_blocks):
    
                print "\n\n Form basis for the following tuple of blocks: ", bi, bj, bk
                
                #
                #   create a small tucker block object involving only Blocks bi and bj and bk 
                #   to create a small dimer hamiltonian 
                #   
                #   Unlike for the dimer case, we want to restrict the space in which we represent
                #   the trimer hamiltonian.  We take the span of all the constituent dimer q spaces

                # Start by collecting all the P spaces for bi bj bk
                bbi = cp.deepcopy(tb_0.blocks[bi])
                bbj = cp.deepcopy(tb_0.blocks[bj])
                bbk = cp.deepcopy(tb_0.blocks[bk])
              
                # From the appropriate initial Q spaces
                tb_bij = tucker_blocks[(2,bi,bj)]
                tb_bik = tucker_blocks[(2,bi,bk)]
                tb_bjk = tucker_blocks[(2,bj,bk)]
                
                bbi_q = cp.deepcopy(tb_bij.blocks[bi])  # ij:i
                bbj_q = cp.deepcopy(tb_bij.blocks[bj])  # ij:j
                bbk_q = cp.deepcopy(tb_bik.blocks[bk])  # ik:k
                
                bbi_q = cp.deepcopy(tb_bik.blocks[bi])  # ij:i
                bbj_q = cp.deepcopy(tb_bjk.blocks[bj])  # ij:j
                bbk_q = cp.deepcopy(tb_bjk.blocks[bk])  # ik:k
                
                bbi_q.append(tb_bik.blocks[bi])           # ik:i
                bbj_q.append(tb_bjk.blocks[bj])           # jk:j
                bbk_q.append(tb_bjk.blocks[bk])           # jk:k
    
                va = tb_bik.blocks[bi].vecs*1 
                vb = tb_bij.blocks[bi].vecs*1 
    
                bbi_q.orthogonalize()
                bbj_q.orthogonalize()
                bbk_q.orthogonalize()
              
                #print bbi_q.vecs.T.dot(bbi.vecs)
                #print bbj_q.vecs.T.dot(bbj.vecs)
                #print bbk_q.vecs.T.dot(bbk.vecs)
                # Combine P and Q to create trimer Hamiltonian
                bbi.append(bbi_q)
                bbj.append(bbj_q)
                bbk.append(bbk_q)
               
                bbi.label = "Q(%i|%i|%i)"%(bi,bj,bk)
                bbj.label = "Q(%i|%i|%i)"%(bi,bj,bk)
                bbk.label = "Q(%i|%i|%i)"%(bi,bj,bk)
    
                tb_curr = block3.Tucker_Block()
                tb_curr.add_block(bbi)
                tb_curr.add_block(bbj)
                tb_curr.add_block(bbk)
             
                #print "Nick:",(bbi_q.n_vecs, bbj_q.n_vecs,bbk_q.n_vecs)
                if min(bbi_q.n_vecs, bbj_q.n_vecs,bbk_q.n_vecs) == 0:
                    print " No trimer term needed"
                    # create a zero dimensional Tucker_Block as a place holder for the 
                    # dimension of the space to grow with iterations.
                    tb = block3.Tucker_Block()
                    tb = cp.deepcopy(tb_0)
                    tb.set_start(dim_tot)
                
                    bbi_q.label = bbi.label
                    bbj_q.label = bbj.label
                    bbk_q.label = bbk.label
                    
                    tb.set_block(bbi_q)
                    tb.set_block(bbj_q)
                    tb.set_block(bbk_q)
                    
                    tb.refresh_dims()
                    tb.update_label()
                    
                    tucker_blocks[tb.label] = tb
                    continue

    
                #
                #   Build and diagonalize dimer Hamiltonian
                print " ", tb_curr
                print " Build local Hamiltonian"
                H,S2 = block3.build_H(tb_curr, tb_curr, j12)
        
                print " Diagonalize local H" 
                e = np.array([])
                v_curr = np.array([])
                if H.shape[0]> 5000:
                    e, v_curr = scipy.sparse.linalg.eigsh(H,1)
                else:
                    e, v_curr = np.linalg.eigh(H)
                
                s = v_curr.T.dot(S2).dot(v_curr)
                print " Energy of GS: %12.8f %12.8f <S2>: "%(e[0],s[0,0])
             
               
                #
                #   Tucker decompose the ground state to get 
                #   a new set of Q vectors for this dimer
                v_curr = v_curr[:,0]
                v_curr.shape = (tb_curr.block_dims)
             
                n_p_i = block_basis[(bi,"P")].n_vecs
                n_p_j = block_basis[(bj,"P")].n_vecs
                n_p_k = block_basis[(bk,"P")].n_vecs
    
                v_curr = v_curr[n_p_i::,n_p_j::,n_p_k::]
    
                pns_thresh = args["pns_thresh"]
                v_comp, U = tucker_decompose(v_curr,pns_thresh,0)
               
                vi = tb_curr.blocks[0].vecs[:,n_p_i::].dot(U[0])
                vj = tb_curr.blocks[1].vecs[:,n_p_j::].dot(U[1])
                vk = tb_curr.blocks[2].vecs[:,n_p_k::].dot(U[2])
    
                #
                #   Create a new Tucker_Block instance with these vectors and 
                #   add it to the active list
                tb = block3.Tucker_Block()
                tb = cp.deepcopy(tb_0)
                tb.set_start(dim_tot)
                 
                bbi = cp.deepcopy(block_basis[(bi,"Q")])
                bbj = cp.deepcopy(block_basis[(bj,"Q")])
                bbk = cp.deepcopy(block_basis[(bk,"Q")])

                bbi.label = "Q(%i|%i|%i)"%(bi,bj,bk)
                bbj.label = "Q(%i|%i|%i)"%(bi,bj,bk)
                bbk.label = "Q(%i|%i|%i)"%(bi,bj,bk)
    
                bbi.set_vecs(vi)
                bbj.set_vecs(vj)
                bbk.set_vecs(vk)
                
                tb.set_block(bbi)
                tb.set_block(bbj)
                tb.set_block(bbk)
                
                tb.refresh_dims()
                tb.update_label()
                
                tucker_blocks[tb.label] = tb
                
                dim_tot += tb.full_dim


if n_body_order >= 4:
    for bi in range(0,n_blocks):
        for bj in range(bi+1,n_blocks):
            for bk in range(bj+1,n_blocks):
                for bl in range(bk+1,n_blocks):
    
                    print "\n Form basis for the following tuple of blocks: ", bi, bj, bk, bl
                    
                    #
                    #   create a small tucker block object involving only Blocks bi and bj and bk 
                    #   to create a small dimer hamiltonian 
                    #   
                    #   Unlike for the dimer case, we want to restrict the space in which we represent
                    #   the trimer hamiltonian.  We take the span of all the constituent dimer q spaces
                    
                    # Start by collecting all the P spaces for bi bj bk
                    
                    bbi = cp.deepcopy(tb_0.blocks[bi])
                    bbj = cp.deepcopy(tb_0.blocks[bj])
                    bbk = cp.deepcopy(tb_0.blocks[bk])
                    bbl = cp.deepcopy(tb_0.blocks[bl])
                    
                    # From the appropriate initial Q spaces
                    tb_bijk = tucker_blocks[(3,bi,bj,bk)]
                    tb_bijl = tucker_blocks[(3,bi,bj,bl)]
                    tb_bikl = tucker_blocks[(3,bi,bk,bl)]
                    tb_bjkl = tucker_blocks[(3,bj,bk,bl)]
                   
                    bbi_q = cp.deepcopy(tb_bijk.blocks[bi])  # ijk:i
                    bbj_q = cp.deepcopy(tb_bijk.blocks[bj])  # ijk:j
                    bbk_q = cp.deepcopy(tb_bijk.blocks[bk])  # ijk:k
                    bbl_q = cp.deepcopy(tb_bijl.blocks[bl])  # ijl:l
                    
                    bbi_q.append(tb_bijl.blocks[bi])         # ijl:i
                    bbi_q.append(tb_bikl.blocks[bi])         # ikl:i
                    bbj_q.append(tb_bijl.blocks[bj])         # ijl:j
                    bbj_q.append(tb_bjkl.blocks[bj])         # jkl:j
                    bbk_q.append(tb_bikl.blocks[bk])         # ikl:k
                    bbk_q.append(tb_bjkl.blocks[bk])         # jkl:k
                    bbl_q.append(tb_bikl.blocks[bl])         # ikl:l
                    bbl_q.append(tb_bjkl.blocks[bl])         # jkl:l
                   
                    bbi_q.orthogonalize()
                    bbj_q.orthogonalize()
                    bbk_q.orthogonalize()
                    bbl_q.orthogonalize()
                    
                    # Combine P and Q to create trimer Hamiltonian
                    bbi.append(bbi_q)
                    bbj.append(bbj_q)
                    bbk.append(bbk_q)
                    bbl.append(bbl_q)
                   
                    bbi.label = "Q(%i|%i|%i|%i)"%(bi,bj,bk,bl)
                    bbj.label = "Q(%i|%i|%i|%i)"%(bi,bj,bk,bl)
                    bbk.label = "Q(%i|%i|%i|%i)"%(bi,bj,bk,bl)
                    bbl.label = "Q(%i|%i|%i|%i)"%(bi,bj,bk,bl)
                    
                    tb_curr = block3.Tucker_Block()
                    tb_curr.add_block(bbi)
                    tb_curr.add_block(bbj)
                    tb_curr.add_block(bbk)
                    tb_curr.add_block(bbl)
                    
                    if min(bbi_q.n_vecs, bbj_q.n_vecs,bbk_q.n_vecs,bbl_q.n_vecs) == 0:
                        print " No term needed"
                        # create a zero dimensional Tucker_Block as a place holder for the 
                        # dimension of the space to grow with iterations.
                        tb = block3.Tucker_Block()
                        tb = cp.deepcopy(tb_0)
                        tb.set_start(dim_tot)
                        
                        bbi_q.label = bbi.label
                        bbj_q.label = bbj.label
                        bbk_q.label = bbk.label
                        bbl_q.label = bbl.label
                        
                        tb.set_block(bbi_q)
                        tb.set_block(bbj_q)
                        tb.set_block(bbk_q)
                        tb.set_block(bbl_q)
                        
                        tb.refresh_dims()
                        tb.update_label()
                        
                        tucker_blocks[tb.label] = tb
                        continue
                    
                    print bbi, bbi.lb
                    print bbj, bbj.lb
                    print bbk, bbk.lb
                    print bbl, bbl.lb
                    
                    #
                    #   Build and diagonalize dimer Hamiltonian
                    print " ", tb_curr
                    print " Build local Hamiltonian"
                    H,S2 = block3.build_H(tb_curr, tb_curr, j12)
                    
                    print " Diagonalize local H" 
                    e = np.array([])
                    v_curr = np.array([])
                    if H.shape[0]> 5000:
                        e, v_curr = scipy.sparse.linalg.eigsh(H,1)
                    else:
                        e, v_curr = np.linalg.eigh(H)
                   
                    s = v_curr.T.dot(S2).dot(v_curr)
                    print " Energy of GS: %12.8f %12.8f <S2>: "%(e[0],s[0,0])
                    
                   
                    #
                    #   Tucker decompose the ground state to get 
                    #   a new set of Q vectors for this dimer
                    v_curr = v_curr[:,0]
                    v_curr.shape = (tb_curr.block_dims)
                    
                    n_p_i = block_basis[(bi,"P")].n_vecs
                    n_p_j = block_basis[(bj,"P")].n_vecs
                    n_p_k = block_basis[(bk,"P")].n_vecs
                    n_p_l = block_basis[(bl,"P")].n_vecs
                    
                    v_curr = v_curr[n_p_i::,n_p_j::,n_p_k::,n_p_l::]
                    
                    pns_thresh = args["pns_thresh"]
                    v_comp, U = tucker_decompose(v_curr,pns_thresh,0)
                   
                    vi = tb_curr.blocks[0].vecs[:,n_p_i::].dot(U[0])
                    vj = tb_curr.blocks[1].vecs[:,n_p_j::].dot(U[1])
                    vk = tb_curr.blocks[2].vecs[:,n_p_k::].dot(U[2])
                    vl = tb_curr.blocks[3].vecs[:,n_p_l::].dot(U[3])
                    
                    #
                    #   Create a new Tucker_Block instance with these vectors and 
                    #   add it to the active list
                    tb = block3.Tucker_Block()
                    tb = cp.deepcopy(tb_0)
                    tb.set_start(dim_tot)
                     
                    bbi = cp.deepcopy(block_basis[(bi,"Q")])
                    bbj = cp.deepcopy(block_basis[(bj,"Q")])
                    bbk = cp.deepcopy(block_basis[(bk,"Q")])
                    bbl = cp.deepcopy(block_basis[(bl,"Q")])
                        
                    bbi.label = "Q(%i|%i|%i|%i)"%(bi,bj,bk,bl)
                    bbj.label = "Q(%i|%i|%i|%i)"%(bi,bj,bk,bl)
                    bbk.label = "Q(%i|%i|%i|%i)"%(bi,bj,bk,bl)
                    bbl.label = "Q(%i|%i|%i|%i)"%(bi,bj,bk,bl)
                    
                    
                    bbi.set_vecs(vi)
                    bbj.set_vecs(vj)
                    bbk.set_vecs(vk)
                    bbl.set_vecs(vl)
                    
                    tb.set_block(bbi)
                    tb.set_block(bbj)
                    tb.set_block(bbk)
                    tb.set_block(bbl)
                    
                    tb.refresh_dims()
                    tb.update_label()
                    
                    tucker_blocks[tb.label] = tb
                    
                    dim_tot += tb.full_dim
            


for tb in sorted(tucker_blocks):
    t = tucker_blocks[tb]
    if t.full_dim == 0:
        continue
    print "%20s ::"%str(t.label), t
print " Full dimension of all Tucker Blocks: ", dim_tot
    #print "%20s ::"%str(t.label), t, " Range= %8i:%-8i" %( t.start, t.stop)


H  = np.zeros([dim_tot,dim_tot])
S2 = np.zeros([dim_tot,dim_tot])
#O  = np.zeros([dim_tot,dim_tot])    # overlap

#for i in sorted(tucker_blocks):
#    for j in sorted(tucker_blocks):
#        tb_i = tucker_blocks[i]
#        tb_j = tucker_blocks[j]
#
#        o = block3.tucker_block_overlap(tb_i,tb_j)
#

H,S2 = block3.build_tucker_blocked_H(tucker_blocks, j12) 

print(" Diagonalize Full H")
l,v = scipy.sparse.linalg.eigsh(H,args["n_roots"])

S2 = v.T.dot(S2).dot(v)
print " %5s    %16s  %16s  %12s" %("State","Energy","Relative","<S2>")
for si,i in enumerate(l):
    if si<args['n_print']:
        print " %5i =  %16.8f  %16.8f  %12.8f" %(si,i*convert,(i-l[0])*convert,abs(S2[si,si]))


