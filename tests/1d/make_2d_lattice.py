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
                        if ij%2==0 and kl%2==1:
                            J[ij,kl] = -1
                        else:
                            J[ij,kl] = -scale
    filename = "j12.%02i.m"%s, scale
    print filename
    np.savetxt("j12.%02i.m"%s,J)
    
    scale *= 2
    
   
    

