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

dim_tot = 0
dim_tot += tb_0.full_dim
tb_0.update_label()
tucker_blocks[0] = tb_0 

for bi in range(0,n_blocks):
    tb = cp.deepcopy(tucker_blocks[0])
    tb.set_block(block_basis[(bi,"Q")])
    tb.refresh_dims()
    tb.update_label()
    
    tb.set_start(dim_tot)
    dim_tot += tb.full_dim
    tucker_blocks[tb.label] = tb
    
    print tb

n_roots = args['n_roots']

H,S2 = block3.build_tucker_blocked_H(tucker_blocks, j12) 
l,v = np.linalg.eigh(H + S2*.01)
v = v[:,0:n_roots]
l = v.T.dot(H).dot(v).diagonal()
S2 = v.T.dot(S2).dot(v)
for si,i in enumerate(l):
    print " %5i =  %16.8f  %16.8f  %12.8f" %(si,i*convert,(i-l[0])*convert,abs(S2[si,si]))
        
nb_order = 0      
tucker_blocks_pt = {}

pns_thresh = args["pns_thresh"]

e_curr = l[0]
tucker_blocks_0 = cp.deepcopy(tucker_blocks)
for bi in range(n_blocks):
    for bj in range(bi+1,n_blocks):
        tb = cp.deepcopy(tucker_blocks[0])
        tb.set_block(block_basis[(bi,"Q")])
        tb.set_block(block_basis[(bj,"Q")])
        
        
        tb.refresh_dims()
        tb.update_label()
        print " PT: ",tb
        tucker_blocks_pt_curr = {}
        tucker_blocks_pt_curr[tb.label] = tb
       
        lpt,vpt = block3.form_pt2_v(tucker_blocks_0, tucker_blocks_pt_curr, l[0:n_roots], v[:,0:n_roots], j12)
     
        vpt = vpt[:,args["target_state"]]
        print vpt.shape
        print tb.block_dims
        vpt.shape = (tb.block_dims)
        v_comp, U, V = tucker_decompose(vpt,pns_thresh,0)
        
        tb.blocks[bi].set_vecs(tb.blocks[bi].vecs.dot(U[bi]))
        tb.blocks[bj].set_vecs(tb.blocks[bj].vecs.dot(U[bj]))
        tb.refresh_dims()
        tb.update_label()
        
        tb.set_start(dim_tot)
        dim_tot += tb.full_dim
        
        e_curr += lpt[0]
        tucker_blocks[tb.label] = tb
   
        tb = cp.deepcopy(tucker_blocks[0])
        tb.set_block(block_basis[(bi,"Q")])
        tb.set_block(block_basis[(bj,"Q")])
        tb.blocks[bi].set_vecs(tb.blocks[bi].vecs.dot(V[bi]))
        tb.blocks[bj].set_vecs(tb.blocks[bj].vecs.dot(V[bj]))
        tb.blocks[bi].label = "R"
        tb.blocks[bj].label = "R"
        tb.refresh_dims()
        tb.update_label()
        tucker_blocks_pt[tb.label] = tb
   
        tb = cp.deepcopy(tucker_blocks[0])
        tb.set_block(block_basis[(bi,"Q")])
        tb.set_block(block_basis[(bj,"Q")])
        tb.blocks[bi].set_vecs(tb.blocks[bi].vecs.dot(U[bi]))
        tb.blocks[bj].set_vecs(tb.blocks[bj].vecs.dot(V[bj]))
        tb.blocks[bi].label = "Q"
        tb.blocks[bj].label = "R"
        tb.refresh_dims()
        tb.update_label()
        tucker_blocks_pt[tb.label] = tb
   
        tb = cp.deepcopy(tucker_blocks[0])
        tb.set_block(block_basis[(bi,"Q")])
        tb.set_block(block_basis[(bj,"Q")])
        tb.blocks[bi].set_vecs(tb.blocks[bi].vecs.dot(V[bi]))
        tb.blocks[bj].set_vecs(tb.blocks[bj].vecs.dot(U[bj]))
        tb.blocks[bi].label = "R"
        tb.blocks[bj].label = "Q"
        tb.refresh_dims()
        tb.update_label()
        tucker_blocks_pt[tb.label] = tb


dim_tot = 0
dim_tot_X = 0
print " Final energy: %12.8f" %e_curr
#e2 = compute_pt2(lattice_blocks, tucker_blocks, tucker_blocks_pt, l[0:n_roots], v[:,0:n_roots], j12, pt_type)
for tbi in sorted(tucker_blocks):
    t = tucker_blocks[tbi]
    print " TB: ", t
    dim_tot += t.full_dim
for tbi in sorted(tucker_blocks_pt):
    t = tucker_blocks_pt[tbi]
    t.set_start(dim_tot_X)
    print " TB: ", t
    dim_tot_X += t.full_dim

print " Full dimension of all VAR Tucker Blocks: ", dim_tot
print " Full dimension of all PT  Tucker Blocks: ", dim_tot_X

H,S2 = block3.build_tucker_blocked_H(tucker_blocks, j12) 

print(" Diagonalize Full H")
l,v = scipy.sparse.linalg.eigsh(H,args["n_roots"])

S2 = v.T.dot(S2).dot(v)
print " %5s    %16s  %16s  %12s" %("State","Energy","Relative","<S2>")
for si,i in enumerate(l):
    print " %5i =  %16.8f  %16.8f  %12.8f" %(si,i*convert,(i-l[0])*convert,abs(S2[si,si]))
    #if si<args['n_print']:
    #    print " %5i =  %16.8f  %16.8f  %12.8f" %(si,i*convert,(i-l[0])*convert,abs(S2[si,si]))

lpt,vpt = block3.form_pt2_v(tucker_blocks, tucker_blocks_pt, l[0:n_roots], v[:,0:n_roots], j12)

print " %5s    %16s  %16s  %12s" %("State","Energy","Relative","<S2>")
for si,i in enumerate(lpt):
    e = i+l[si]
    e0 = lpt[0]+l[0]
    print " %5i =  %16.8f  %16.8f  %12.8f" %(si,(e)*convert,(e-e0)*convert,abs(S2[si,si]))


for t_0 in sorted(tucker_blocks):
    tb_0 = tucker_blocks[t_0]
    if tb_0.label[0] == 2:

        for bi in range(n_blocks):
            for bj in range(bi+1,n_blocks):
                if tb_0.blocks[bi].label == "P" and tb_0.blocks[bj].label == "P": 

                    tb = cp.deepcopy(tb_0)
                    tb.set_block(block_basis[(bi,"Q")])
                    tb.set_block(block_basis[(bj,"Q")])
                    tb.set_start(dim_tot_X)
                    
                    tb.refresh_dims()
                    tb.update_label()
                    print " PT: ",tb
                    tucker_blocks_pt_curr = {}
                    tucker_blocks_pt_curr[tb.label] = tb
                  
                    lpt,vpt = block3.form_pt2_v(tucker_blocks, tucker_blocks_pt_curr, l[0:1], v[:,0:1], j12)
                    
                    #vpt.shape = (tb.block_dims)
                    #v_comp, U, V = tucker_decompose(vpt,pns_thresh,0)
