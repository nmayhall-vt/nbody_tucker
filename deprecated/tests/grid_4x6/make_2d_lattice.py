#!/usr/bin/env python
import argparse,os,errno,re
import csv
import numpy as np
#from colour import Color


#   Setup input arguments
parser = argparse.ArgumentParser(description='Create an mxn grid lattice',
formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-size', type=int, default=(2,2), nargs=2, help='size of grid', required=False)
parser.add_argument('-b','--blocks', type=str, default="heis_blocks.m", help='File containing vector of block sizes', required=False)
args = vars(parser.parse_args())

blocks = np.loadtxt(args['blocks']).astype(int)
n_blocks = len(blocks)


blocks = {}
with open(args['blocks']) as f:
    count = 0
    for line in f:
        l = [int(x) for x in line.split()]
        blocks[count] = l
        count += 1

n = args['size'][0]
m = args['size'][1]

n_sites = m*n
intra_block_bonds = np.zeros((n_sites, n_sites)) 
for bi in range(0,n_blocks):
    for si in blocks[bi]:
        for sj in blocks[bi]:
            intra_block_bonds[si,sj] = 1


scale = 1.0/64.0 
for s in range(0,12):
    J = np.zeros((m*n,m*n))
    for i in range(0,m):
        for j in range(0,n):
            ij = j+i*n
            for k in range(0,m):
                for l in range(0,n):
                    kl = l+k*n
                    if abs(i-k) == 1 and j==l or abs(j-l) == 1 and i==k:
                        J[ij,kl] = -scale
                        
                        if intra_block_bonds[ij,kl] == 1:
                            J[ij,kl] = -1



    scale *= 2
    
    filename = "j12.%02i.m"%s, scale
    print filename
    np.savetxt("j12.%02i.m"%s,J)
   
    

