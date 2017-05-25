#!/usr/bin/env python
import argparse,os,errno,re
import csv
import numpy as np
#from colour import Color


#   Setup input arguments
parser = argparse.ArgumentParser(description='Create an mxn grid lattice',
formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-size', type=int, default=(2,2), nargs=2, help='size of grid', required=False)
args = vars(parser.parse_args())

n = args['size'][0]
m = args['size'][1]

J = np.zeros((m*n,m*n))
for i in range(0,m):
    for j in range(0,n):
        ij = j+i*n
        for k in range(0,m):
            for l in range(0,n):
                kl = l+k*n
                if abs(i-k) == 1 and j==l or abs(j-l) == 1 and i==k:
                    J[ij,kl] = -1
                    if ij%2==0 and kl%2==1:
                        J[ij,kl] = -1*3
print J
np.savetxt("j12.m",J)
   
    

