#!/usr/bin/env python
import argparse,os,errno,re
import csv
import numpy as np
#from colour import Color


#   Setup input arguments
parser = argparse.ArgumentParser(description='fill this out',
formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('infiles', nargs='+', help='List of input files to process')
parser.add_argument('-v','--vector', type=int, default=1, help='Which vector to sort by?', required=False)
args = vars(parser.parse_args())

fv_i = args['vector']

for fileName in args['infiles']:
    print 
    print " Creating Laplacian matrix from exchange coupling constants:"

    Ain = np.loadtxt(fileName)
    A = abs(Ain)


    n_sites = A.shape[0]
   
    edges = {}
    counted = {}
    for i in range(0,n_sites):
        counted[i] = 0
        for j in range(i+1,n_sites):
            edges[A[i,j]] = (i,j)

    sort_ind = []
    for e in sorted(edges, reverse=True):
        i = edges[e][0]
        j = edges[e][1]
        if counted[i] == 0 and counted[j] == 0:
            print e
            sort_ind.append(i)
            sort_ind.append(j)
            counted[i] = 1
            counted[j] = 1

    print sort_ind

    Asorted = Ain[sort_ind,:][:,sort_ind]
    np.savetxt(fileName+".sorted.m",Asorted)
   
    

