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


def form_all_brdms(tucker_blocks,v):
   
    n_blocks = tucker_blocks[tucker_blocks.keys()[0]].n_blocks
    for bi in range(0,n_blocks):
        print bi


       


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


H,S2 = block3.build_tucker_blocked_H(tucker_blocks, j12) 
l,v = np.linalg.eigh(H + S2*.01)
l = v.T.dot(H).dot(v).diagonal()
S2 = v.T.dot(S2).dot(v)
        
nb_order = 0      
tucker_blocks_pt = {}

n_roots = args['n_roots']
pns_thresh = args["pns_thresh"]

dim_tot_X = 0

e_curr = l[0]
for bi in range(n_blocks):
    for bj in range(bi+1,n_blocks):
        tb = cp.deepcopy(tucker_blocks[0])
        tb.set_block(block_basis[(bi,"Q")])
        tb.set_block(block_basis[(bj,"Q")])
        
        #tb.set_start(dim_tot_X)
        #dim_tot_X += tb.full_dim
        
        tb.refresh_dims()
        tb.update_label()
        print " PT: ",tb
        tucker_blocks_pt = {}
        tucker_blocks_pt[tb.label] = tb
       
        lpt,vpt = block3.form_pt2_v(tucker_blocks, tucker_blocks_pt, l[0:n_roots], v[:,0:n_roots], j12)
      
        #v.shape = (tb.block_dims)
        #v_comp, U = tucker_decompose(v,pns_thresh,0)

        e_curr += lpt[0]


print " Final energy: %12.8f" %e_curr
#e2 = compute_pt2(lattice_blocks, tucker_blocks, tucker_blocks_pt, l[0:n_roots], v[:,0:n_roots], j12, pt_type)
exit(-1)

if n_body_order >= 1:
    for bi in range(0,n_blocks):
        
        tb = cp.deepcopy(tb_0)
        tb.label = (1,bi)
        tb.set_start(dim_tot)
        tb.set_block(block_basis[(bi,"Q")]) 

        tucker_blocks[tb.label] = tb
        
        dim_tot += tb.full_dim


print "\n NB0 Eigenstates:"
for idx,ei in enumerate(l):
    print" %12i %12.8f %8.4f"%(idx,ei,S2[idx,idx])

exit(-1)

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
           
            #if tb.full_dim == 0:
                #for b in tb.blocks:
                    #b.set_vecs(np.zeros((b.vecs.shape[0],0)))
                    #tb.refresh_dims()
                    #tb.update_label()

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
               
                vi_1 = cp.deepcopy(tb_bij.blocks[bi].vecs)
                vi_2 = cp.deepcopy(tb_bik.blocks[bi].vecs)
                
                vj_1 = cp.deepcopy(tb_bij.blocks[bj].vecs)
                vj_2 = cp.deepcopy(tb_bjk.blocks[bj].vecs)
                
                vk_1 = cp.deepcopy(tb_bik.blocks[bk].vecs)
                vk_2 = cp.deepcopy(tb_bjk.blocks[bk].vecs)
             
                print tb_bij
                print tb_bik
                print tb_bjk
                
#                Oi = vi_1.T.dot(vi_2)
#                Oj = vj_1.T.dot(vj_2)
#                Ok = vk_1.T.dot(vk_2)
#                if min(Oi.shape) > 0:
#                    U,s,V = np.linalg.svd(Oi)
#                    for si in s:
#                        print "   %12.8f " %si
#                if min(Oj.shape) > 0:
#                    U,s,V = np.linalg.svd(Oj,full_matrices=True)
#                    V = V.T
#                    keep = []
#                    for sj in range(len(s)):
#                        if abs(s[sj]-1) > 1e-8:
#                            keep.append(sj)
#                            print "   %12.8f : New" %s[sj]
#                        else:
#                            print "   %12.8f : Same" %s[sj]
#                    #U = U[:,keep]
#                    V = V[:,keep]
#                    #vj_1 = vj_1.dot(U)
#                    #vj_2 = vj_2.dot(V)
#                if min(Ok.shape) > 0:
#                    U,s,V = np.linalg.svd(Ok)
#                    for sk in s:
#                        print "   %12.8f " %sk
                
                vi = np.hstack((vi_1,vi_2))
                vj = np.hstack((vj_1,vj_2))
                vk = np.hstack((vk_1,vk_2))
              
                if min(vi.shape) > 0:
                    U,s,V = np.linalg.svd(vi,full_matrices=True)
                    keep = []
                    for si in range(len(s)):
                        if abs(s[si])> 1e-12:
                            keep.append(si)
                    vi = U[:,keep]
                
                if min(vj.shape) > 0:
                    U,s,V = np.linalg.svd(vj,full_matrices=True)
                    keep = []
                    for si in range(len(s)):
                        if abs(s[si])> 1e-12:
                            keep.append(si)
                    vj = U[:,keep]

                if min(vk.shape) > 0:
                    U,s,V = np.linalg.svd(vk,full_matrices=True)
                    keep = []
                    for si in range(len(s)):
                        if abs(s[si])> 1e-12:
                            keep.append(si)
                    vk = U[:,keep]


                bbi_q = cp.deepcopy(tb_bij.blocks[bi])  # ij:i
                bbj_q = cp.deepcopy(tb_bij.blocks[bj])  # ij:j
                bbk_q = cp.deepcopy(tb_bik.blocks[bk])  # ik:k
                
                bbi_q.set_vecs(vi) 
                bbj_q.set_vecs(vj) 
                bbk_q.set_vecs(vk) 
              
                # make sure that new eigenvectors are orthogonal to p spaces

                #print bbi_q.vecs.T.dot(bbi.vecs)
                #print bbj_q.vecs.T.dot(bbj.vecs)
                #print bbk_q.vecs.T.dot(bbk.vecs)
                # Combine P and Q to create trimer Hamiltonian
                
                # From the appropriate initial Q spaces
                #tb_bi = tucker_blocks[(1,bi)]
                #tb_bj = tucker_blocks[(1,bj)]
                #tb_bk = tucker_blocks[(1,bk)]
                
                #bbi_q = cp.deepcopy(tb_bi.blocks[bi])  # ij:i
                #bbj_q = cp.deepcopy(tb_bj.blocks[bj])  # ij:j
                #bbk_q = cp.deepcopy(tb_bk.blocks[bk])  # ik:k
                
                
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
                
                    tb.blocks[bi].label = bbi.label
                    tb.blocks[bj].label = bbj.label
                    tb.blocks[bk].label = bbk.label
    
                    tb.blocks[bi].clear()
                    tb.blocks[bj].clear()
                    tb.blocks[bk].clear()
                    
                    tb.refresh_dims()
                    tb.update_label()
                    
                    tucker_blocks[tb.label] = tb
                    continue

    
                #
                #   Build and diagonalize dimer Hamiltonian
                print " Current subsystem tb", tb_curr
                print " Build local Hamiltonian"
                H,S2 = block3.build_H(tb_curr, tb_curr, j12)
        
                print " Diagonalize local H" 
                e = np.array([])
                v_curr = np.array([])
                if H.shape[0]> 3000:
                    e, v_curr = scipy.sparse.linalg.eigsh(H,1)
                else:
                    e, v_curr = np.linalg.eigh(H)
                
                s = v_curr.T.dot(S2).dot(v_curr)
                print " Energy of GS: %12.8f %12.8f <S2>: "%(e[0],s[0,0])
                for si,i in enumerate(e):
                    if si<args['n_print']:
                        print " %5i =  %16.8f  %16.8f  %12.8f" %(si,i*convert,(i-e[0])*convert,abs(s[si,si]))
               
                #
                #   Tucker decompose the ground state to get 
                #   a new set of Q vectors for this dimer
                v_curr = v_curr[:,0]
                v_curr.shape = (tb_curr.block_dims)
            
                n_p_i = block_basis[(bi,"P")].n_vecs
                n_p_j = block_basis[(bj,"P")].n_vecs
                n_p_k = block_basis[(bk,"P")].n_vecs
    
                #v_curr = v_curr[n_p_i::,n_p_j::,n_p_k::]
                #print v_curr
                    
                proj = [n_p_i,n_p_j,n_p_k]
   
        
                pns_thresh = args["pns_thresh"]
                v_comp, U = tucker_decompose_proj(v_curr,pns_thresh,0,proj)
               
                vi = tb_curr.blocks[0].vecs.dot(U[0])
                vj = tb_curr.blocks[1].vecs.dot(U[1])
                vk = tb_curr.blocks[2].vecs.dot(U[2])
   
                #vi_p = tb_curr.blocks[0].vecs[:,0:n_p_i]
                #vj_p = tb_curr.blocks[1].vecs[:,0:n_p_j]
                #vk_p = tb_curr.blocks[2].vecs[:,0:n_p_k]

                #vi = vi - vi_p.dot(vi_p.T.dot(vi))
                #vj = vj - vj_p.dot(vj_p.T.dot(vj))
                #vk = vk - vk_p.dot(vk_p.T.dot(vk))

                #vi = scipy.linalg.orth(vi)
                #vj = scipy.linalg.orth(vj)
                #vk = scipy.linalg.orth(vk)

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

                if tb.full_dim == 0:
                    tb.blocks[bi].clear()
                    tb.blocks[bj].clear()
                    tb.blocks[bk].clear()
                    tb.refresh_dims()

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
                  
                    vi =   cp.deepcopy(tb_bijk.blocks[bi].vecs)
                    vi = np.hstack((vi,tb_bijl.blocks[bi].vecs))
                    vi = np.hstack((vi,tb_bikl.blocks[bi].vecs))

                    vj =   cp.deepcopy(tb_bijk.blocks[bj].vecs)
                    vj = np.hstack((vj,tb_bijl.blocks[bj].vecs))
                    vj = np.hstack((vj,tb_bjkl.blocks[bj].vecs))

                    vk =   cp.deepcopy(tb_bijk.blocks[bk].vecs)
                    vk = np.hstack((vk,tb_bikl.blocks[bk].vecs))
                    vk = np.hstack((vk,tb_bjkl.blocks[bk].vecs))

                    vl =   cp.deepcopy(tb_bijl.blocks[bl].vecs)
                    vl = np.hstack((vl,tb_bikl.blocks[bl].vecs))
                    vl = np.hstack((vl,tb_bjkl.blocks[bl].vecs))
                
                    if min(vi.shape) > 0:
                        U,s,V = np.linalg.svd(vi,full_matrices=True)
                        keep = []
                        for si in range(len(s)):
                            if abs(s[si])> 1e-12:
                                keep.append(si)
                        vi = U[:,keep]
                    
                    if min(vj.shape) > 0:
                        U,s,V = np.linalg.svd(vj,full_matrices=True)
                        keep = []
                        for si in range(len(s)):
                            if abs(s[si])> 1e-12:
                                keep.append(si)
                        vj = U[:,keep]
                    
                    if min(vk.shape) > 0:
                        U,s,V = np.linalg.svd(vk,full_matrices=True)
                        keep = []
                        for si in range(len(s)):
                            if abs(s[si])> 1e-12:
                                keep.append(si)
                        vk = U[:,keep]
                    
                    if min(vl.shape) > 0:
                        U,s,V = np.linalg.svd(vl,full_matrices=True)
                        keep = []
                        for si in range(len(s)):
                            if abs(s[si])> 1e-12:
                                keep.append(si)
                        vl = U[:,keep]


                    bbi_q = cp.deepcopy(tb_bijk.blocks[bi])  # ijk:i
                    bbj_q = cp.deepcopy(tb_bijk.blocks[bj])  # ijk:j
                    bbk_q = cp.deepcopy(tb_bijk.blocks[bk])  # ijk:k
                    bbl_q = cp.deepcopy(tb_bijl.blocks[bl])  # ijl:l
                
                    bbi_q.set_vecs(vi) 
                    bbj_q.set_vecs(vj) 
                    bbk_q.set_vecs(vk) 
                    bbl_q.set_vecs(vl) 

                    
#                    bbi_q.append(tb_bijl.blocks[bi])         # ijl:i
#                    bbi_q.append(tb_bikl.blocks[bi])         # ikl:i
#                    bbj_q.append(tb_bijl.blocks[bj])         # ijl:j
#                    bbj_q.append(tb_bjkl.blocks[bj])         # jkl:j
#                    bbk_q.append(tb_bikl.blocks[bk])         # ikl:k
#                    bbk_q.append(tb_bjkl.blocks[bk])         # jkl:k
#                    bbl_q.append(tb_bikl.blocks[bl])         # ikl:l
#                    bbl_q.append(tb_bjkl.blocks[bl])         # jkl:l
#                    
#                    bbi_q.orthogonalize()
#                    bbj_q.orthogonalize()
#                    bbk_q.orthogonalize()
#                    bbl_q.orthogonalize()
                    
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
                    print tb_bijk 
                    print tb_bijl 
                    print tb_bikl 
                    print tb_bjkl 
                    print tb_curr
                    if min(bbi_q.n_vecs, bbj_q.n_vecs,bbk_q.n_vecs,bbl_q.n_vecs) == 0:
                        print " No term needed"
                        # create a zero dimensional Tucker_Block as a place holder for the 
                        # dimension of the space to grow with iterations.
                        tb = block3.Tucker_Block()
                        tb = cp.deepcopy(tb_0)
                        tb.set_start(dim_tot)
                        
                        tb.blocks[bi].label = bbi.label
                        tb.blocks[bj].label = bbj.label
                        tb.blocks[bk].label = bbk.label
                        tb.blocks[bl].label = bbl.label
                        
                        tb.blocks[bi].clear()
                        tb.blocks[bj].clear()
                        tb.blocks[bk].clear()
                        tb.blocks[bl].clear()
                    
                        
                        tb.refresh_dims()
                        tb.update_label()
                        
                        tucker_blocks[tb.label] = tb
                        continue
                    
                    #print bbi, bbi.lb
                    #print bbj, bbj.lb
                    #print bbk, bbk.lb
                    #print bbl, bbl.lb
                    
                    #
                    #   Build and diagonalize dimer Hamiltonian
                    print " Build local Hamiltonian"
                    H,S2 = block3.build_H(tb_curr, tb_curr, j12)
                    
                    print " Diagonalize local H" 
                    e = np.array([])
                    v_curr = np.array([])
                    if H.shape[0]> 3000:
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
                    
                    proj = [n_p_i,n_p_j,n_p_k,n_p_l]

                    pns_thresh = args["pns_thresh"]
                    v_comp, U = tucker_decompose_proj(v_curr,pns_thresh,0,proj)
                 

                    vi = tb_curr.blocks[0].vecs.dot(U[0])
                    vj = tb_curr.blocks[1].vecs.dot(U[1])
                    vk = tb_curr.blocks[2].vecs.dot(U[2])
                    vl = tb_curr.blocks[3].vecs.dot(U[3])
  
                    
                    vi = scipy.linalg.orth(vi)
                    vj = scipy.linalg.orth(vj)
                    vk = scipy.linalg.orth(vk)
                    vl = scipy.linalg.orth(vl)

                    
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
    
print " Check for further compression:"
ts = args['target_state']
for tbi in sorted(tucker_blocks):

    t = tucker_blocks[tbi]
    print " TB: ", t
    vi = v[t.start:t.stop,ts]

    #print vi.shape, t.block_dims

    vi.shape = t.block_dims
    v_comp, U = tucker_decompose(vi,0,0)


exit(-1)

for pno_it in range(1):
    print
    print "   Compute Block Reduced Density Matrices (BRDM) FULL!:"
    

    dim_tot_list = []
    config_basis_dim = 1
    sum_1 = 0
    ts = args['target_state']
    
    form_all_brdms(tucker_blocks,v[:,ts])
    
    for bi in range(0,n_blocks):
         
        bbi = cp.deepcopy(block_basis[(bi,"P")])
        dim_tot_list.append(bbi.lb.full_dim)
        config_basis_dim *= bbi.lb.full_dim
    
    vec_curr = np.zeros(dim_tot_list)
    for tb1 in sorted(tucker_blocks):
        Tb1 = tucker_blocks[tb1]
        if Tb1.full_dim == 0:
            continue
        vb1 = cp.deepcopy(v[Tb1.start:Tb1.stop, ts])
        sum_1 += vb1.T.dot(vb1)
        vb1.shape = Tb1.block_dims

        vec = []
        for bi in range(0,n_blocks):
            vec.append(Tb1.blocks[bi].vecs)

        vec_curr += transform_tensor(vb1,vec)

    
    
    dim_tot = tb_0.full_dim

    if n_body_order >= 1:
        for bi in range(0,n_blocks):
            dim_tot += tucker_blocks[(1,bi)].full_dim

    if n_body_order >= 2:
        for bi in range(0,n_blocks):
            for bj in range(bi+1,n_blocks):
                print "\n Form RDM for the following pair of blocks: ", bi, bj
             
                tb = cp.deepcopy(tucker_blocks[(2,bi,bj)])
                
                tb.set_start(dim_tot)
                
                ts = args['target_state']
                #
                #   create a small tucker block object involving only Blocks bi and bj 
                #   to create a small dimer hamiltonian for the full hilbert space on bi and bj
                
                bbi = cp.deepcopy(block_basis[(bi,"P")])
                bbj = cp.deepcopy(block_basis[(bj,"P")])
                
                bbi.append(block_basis[(bi,"Q")])
                bbj.append(block_basis[(bj,"Q")])
            
                vq_i = bbi.vecs * 1.0
                vq_j = bbj.vecs * 1.0
            
                contract_list = []
                for bk in range(n_blocks):
                    if bk != bi and bk != bj:
                        contract_list.append(bk)
                #print vec_curr.shape
                #print contract_list
                contract_list = range(n_blocks-2)
                vij = np.tensordot(vec_curr,vq_i, axes=(bi,0))
                vij = np.tensordot(vij,vq_j, axes=(bj-1,0))
                Dij = np.tensordot(vij,vij, axes=(contract_list,contract_list))
               
                n_p_i = block_basis[(bi,"P")].n_vecs
                n_p_j = block_basis[(bj,"P")].n_vecs
              
                #proj = [0]*n_blocks
                #proj[n_blocks-2] = n_p_i
                #proj[n_blocks-1] = n_p_j
               
                proj = [n_p_i,n_p_j,n_p_i,n_p_j]
                #print proj
                #print vij.shape
                v_comp, U = tucker_decompose_proj(Dij,pns_thresh,0,proj)
               
            
                #vq_i = vq_i.dot(U[n_blocks-2])
                #vq_j = vq_j.dot(U[n_blocks-1])
                vq_i = vq_i.dot(U[0])
                vq_j = vq_j.dot(U[1])

                
                bbi = cp.deepcopy(block_basis[(bi,"Q")])
                bbj = cp.deepcopy(block_basis[(bj,"Q")])
                bbi.label = "Q(%i|%i)"%(bi,bj)
                bbj.label = "Q(%i|%i)"%(bi,bj)
                
                bbi.set_vecs(vq_i)
                bbj.set_vecs(vq_j)
                
                tb.set_block(bbi)
                tb.set_block(bbj)
                
                tb.refresh_dims()
                
                #tb.update_label()
               
                if tb.full_dim == 0:
                    for b in tb.blocks:
                        #b.vecs = np.zeros((b.vecs.shape[0],0))
                        b.clear()
                        tb.refresh_dims()
                        #tb.update_label()
                
                tucker_blocks[tb.label] = tb
             
                print tb
                #print tb.blocks[bi].vecs.shape
                #print tb.blocks[bj].vecs.shape
                #print tb.full_dim, tb.block_dims
                dim_tot += tb.full_dim


	for tb in sorted(tucker_blocks):
	    t = tucker_blocks[tb]
	    if t.full_dim == 0:
	        continue
	    print "%20s ::"%str(t.label), "%6i"%t.start, t
	print " Full dimension of all Tucker Blocks: ", dim_tot
	    #print "%20s ::"%str(t.label), t, " Range= %8i:%-8i" %( t.start, t.stop)
	
	
	H  = np.zeros([dim_tot,dim_tot])
	S2 = np.zeros([dim_tot,dim_tot])
	H,S2 = block3.build_tucker_blocked_H(tucker_blocks, j12) 
	
	print(" Diagonalize Full H")
	l,v = scipy.sparse.linalg.eigsh(H,args["n_roots"])
	
	
	S2 = v.T.dot(S2).dot(v)
	print " %5s    %16s  %16s  %12s" %("State","Energy","Relative","<S2>")
	for si,i in enumerate(l):
	    if si<args['n_print']:
	        print " %5i =  %16.8f  %16.8f  %12.8f" %(si,i*convert,(i-l[0])*convert,abs(S2[si,si]))

                

                  

