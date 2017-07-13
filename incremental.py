#! /usr/bin/env python
import numpy as np
import scipy
import scipy.linalg
import scipy.io
import copy as cp
import argparse
import scipy.sparse
import scipy.sparse.linalg
#import subprocess

import sys
import os
sys.path.insert(0, './')

from hdvv import *
from block import *
from davidson import *


       


"""
 Incremental tool
"""
#   Setup input arguments
parser = argparse.ArgumentParser(description='Finds eigenstates of a spin lattice',
formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#parser.add_argument('-d','--dry_run', default=False, action="store_true", help='Run but don\'t submit.', required=False)
parser.add_argument('-ju','--j_unit', type=str, default="cm", help='What units are the J values in', choices=['cm','ev'],required=False)
#parser.add_argument('-l','--lattice', type=str, default="heis_lattice.m", help='File containing vector of sizes number of electrons per lattice site', required=False)
parser.add_argument('-j','--j12', type=str, default="heis_j12.m", help='File containing matrix of exchange constants', required=False)
parser.add_argument('-b','--blocks', type=str, default="heis_blocks.m", help='File containing vector of block sizes', required=False)
parser.add_argument('-np','--n_p_space', type=int, nargs="+", help='Number of vectors in block P space', required=False)
parser.add_argument('-nq','--n_q_space', type=int, nargs="+", help='Number of vectors in block Q space', required=False)
parser.add_argument('-nb','--n_body_order', type=int, default="0", help='n_body spaces', required=False)
parser.add_argument('-nr','--n_roots', type=int, default="10", help='Number of eigenvectors to find in compressed space', required=False)
parser.add_argument('--n_print', type=int, default="10", help='number of states to print', required=False)
parser.add_argument('-ts','--target_state', type=int, default="0", nargs='+', help='state(s) to target during (possibly state-averaged) optimization', required=False)
parser.add_argument('-mit', '--max_iter', type=int, default=30, help='Max iterations for solving for the compression vectors', required=False)
parser.add_argument('-diis_thresh','--diis_thresh', type=int, default=8, help='Threshold for pspace diis iterations', required=False)
parser.add_argument('-dav_thresh','--dav_thresh', type=int, default=8, help='Threshold for supersystem davidson iterations', required=False)
parser.add_argument('-pt','--pt_order', type=int, default=0, help='PT correction order ?', required=False)
parser.add_argument('-pt_type','--pt_type', type=str, default='mp', choices=['mp','en'], help='PT correction denominator type', required=False)
parser.add_argument('-ms','--target_ms', type=float, default=0, help='Target ms space', required=False)
parser.add_argument('-opt','--optimization', type=str, default="diis", help='Optimization algorithm for Tucker factors',choices=["none", "diis"], required=False)
parser.add_argument('-direct','--direct', type=int, default=1, help='Evaluate the matrix on the fly?',choices=[0,1], required=False)
parser.add_argument('-dmit', '--dav_max_iter', type=int, default=20, help='Max iterations for solving for the CI-type coefficients', required=False)
parser.add_argument('-precond', '--dav_precond', type=int, default=1, help='Use preconditioner?', required=False)
args = vars(parser.parse_args())
#
#   Let minute specification of walltime override hour specification

j12 = np.loadtxt(args['j12'])
#lattice = np.loadtxt(args['lattice']).astype(int)
lattice = np.ones((j12.shape[0],1))
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

cmd = "../../nbtuck_davidson.py -ju ev -mit 1 -j %s -nr 1 -direct 0 -nb %s -pt %s -b %s" %(args['j12'],args['n_body_order'], args['pt_order'], args['blocks'])
#p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True
#print p
os.system("echo 'Do 0Body' > 0b.txt") 
pvec = [1]*n_blocks
pvec = " -np " + str(pvec)[1:-1].replace(",","")
os.system(cmd + pvec + " >> 0b.txt") 
print " Done: block "

nb_order = 2

os.system("echo 'Do 1Body' > 1b.txt") 
for bi in range(0,n_blocks):
    pvec = [1]*n_blocks
    pvec[bi] = 4
    pvec = " -np " + str(pvec)[1:-1].replace(",","")
    os.system(cmd + pvec + " >> 1b.txt") 
    print " Done: block ", bi

os.system("echo 'Do 2Body' > 2b.txt") 
for bi in range(0,n_blocks):
    for bj in range(bi+1,n_blocks):
        pvec = [1]*n_blocks
        pvec[bi] = 4
        pvec[bj] = 4
        pvec = " -np " + str(pvec)[1:-1].replace(",","")
        os.system(cmd + pvec + " >> 2b.txt") 
        print " Done: block ", bi, bj

os.system("echo 'do 3body' > 3b.txt") 
for bi in range(0,n_blocks):
    for bj in range(bi+1,n_blocks):
        for bk in range(bj+1,n_blocks):
            pvec = [1]*n_blocks
            pvec[bi] = 4
            pvec[bj] = 4
            pvec[bk] = 4
            pvec = " -np " + str(pvec)[1:-1].replace(",","")
            os.system(cmd + pvec + " >> 3b.txt") 
            print " done: block ", bi, bj, bk

os.system("echo 'do 3body' > 4b.txt") 
if nb_order > 3:
    for bi in range(0,n_blocks):
        for bj in range(bi+1,n_blocks):
            for bk in range(bj+1,n_blocks):
                for bl in range(bk+1,n_blocks):
                    pvec = [1]*n_blocks
                    pvec[bi] = 4
                    pvec[bj] = 4
                    pvec[bk] = 4
                    pvec[bl] = 4
                    pvec = " -np " + str(pvec)[1:-1].replace(",","")
                    os.system(cmd + pvec + " >> 4b.txt") 
                    print " done: block ", bi, bj, bk, bl

#
#   analyze
if args['pt_order'] > 0:
    os.system(" grep 'PT2' -A2 0b.txt | grep ' 0 =' | sed 's/.*=[ ]\+//' | sed 's/ .*//' > 0b_a.txt ")
    os.system(" grep 'PT2' -A2 1b.txt | grep ' 0 =' | sed 's/.*=[ ]\+//' | sed 's/ .*//' > 1b_a.txt ")
    os.system(" grep 'PT2' -A2 2b.txt | grep ' 0 =' | sed 's/.*=[ ]\+//' | sed 's/ .*//' > 2b_a.txt ")
    os.system(" grep 'PT2' -A2 3b.txt | grep ' 0 =' | sed 's/.*=[ ]\+//' | sed 's/ .*//' > 3b_a.txt ")
    if nb_order > 3:
        os.system(" grep 'PT2' -A2 4b.txt | grep ' 0 =' | sed 's/.*=[ ]\+//' | sed 's/ .*//' > 4b_a.txt ")

else:
    os.system(" grep ' 0 =' 0b.txt | sed 's/.*=[ ]\+//' | sed 's/ .*//' > 0b_a.txt ")
    os.system(" grep ' 0 =' 1b.txt | sed 's/.*=[ ]\+//' | sed 's/ .*//' > 1b_a.txt ")
    os.system(" grep ' 0 =' 2b.txt | sed 's/.*=[ ]\+//' | sed 's/ .*//' > 2b_a.txt ")
    os.system(" grep ' 0 =' 3b.txt | sed 's/.*=[ ]\+//' | sed 's/ .*//' > 3b_a.txt ")
    if nb_order > 3:
        os.system(" grep ' 0 =' 4b.txt | sed 's/.*=[ ]\+//' | sed 's/ .*//' > 4b_a.txt ")

e0_b = np.loadtxt('0b_a.txt').astype(float)
e1b = np.loadtxt('1b_a.txt').astype(float)
e2b = np.loadtxt('2b_a.txt').astype(float)
e3b = np.loadtxt('3b_a.txt').astype(float)
if nb_order > 3:
    e4b = np.loadtxt('4b_a.txt').astype(float)

e1_b = np.zeros((n_blocks))
e2_b = np.zeros((n_blocks,n_blocks))
e3_b = np.zeros((n_blocks,n_blocks,n_blocks))
if nb_order > 3:
    e4_b = np.zeros((n_blocks,n_blocks,n_blocks,n_blocks))

for bi in range(0,n_blocks):
    e1_b[bi] = e1b[bi] - e0_b

count = 0
for bi in range(0,n_blocks):
    for bj in range(bi+1,n_blocks):
        e2_b[bi,bj] = e2b[count] - e0_b - e1_b[bi] - e1_b[bj]
        count += 1

count = 0
for bi in range(0,n_blocks):
    for bj in range(bi+1,n_blocks):
        for bk in range(bj+1,n_blocks):
            e3_b[bi,bj,bk] = e3b[count] - e0_b - e2_b[bi,bj] - e2_b[bi,bk] - e2_b[bj,bk] - e1_b[bi] - e1_b[bj] - e1_b[bk]
            count += 1

if nb_order > 3:
    count = 0
    for bi in range(0,n_blocks):
        for bj in range(bi+1,n_blocks):
            for bk in range(bj+1,n_blocks):
                for bl in range(bk+1,n_blocks):
                    term = e4b[count] - e0_b
                    term -= e3_b[bi,bj,bk] + e3_b[bi,bj,bl] + e3_b[bi,bk,bl] + e3_b[bj,bk,bl] 
                    term -= e2_b[bi,bj] + e2_b[bi,bk] + e2_b[bi,bl] + e2_b[bj,bk] + e2_b[bj,bl] + e2_b[bk,bl]
                    term -= e1_b[bi] + e1_b[bj] + e1_b[bk] + e1_b[bl]
                    e4_b[bi,bj,bk,bl] = term 
                    count += 1

print
print " n-Body Expansion:"
print "  0-Body = %12.8f" % e0_b
print "  1-Body = %12.8f" % np.sum(e1_b)
print "  2-Body = %12.8f" % np.sum(e2_b)
print "  3-Body = %12.8f" % np.sum(e3_b)
if nb_order > 3:
    print "  4-Body = %12.8f" % np.sum(e4_b)

#print "  sum    = %12.8f" % (e0_b + np.sum(e1_b) + np.sum(e2_b))
if nb_order > 3:
    print "  sum    = %12.8f" % (e0_b + np.sum(e1_b) + np.sum(e2_b) + np.sum(e3_b) + np.sum(e4_b))
else:
    print "  sum    = %12.8f" % (e0_b + np.sum(e1_b) + np.sum(e2_b) + np.sum(e3_b))


