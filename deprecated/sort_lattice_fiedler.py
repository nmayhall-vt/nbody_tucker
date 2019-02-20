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

#fv_e = np.zeros((24,24)) 
#ifile_count = 0
fv_e = []
for fileName in args['infiles']:
    print 
    print " Creating Laplacian matrix from exchange coupling constants:"

    Ain = np.loadtxt(fileName)
    A = abs(Ain)
    Amax = np.max(A)
    for i in range(0,A.shape[0]):
        for j in range(0,A.shape[0]):
            #A[i,j] = pow(10,A[i,j]-Amax) 
            pass

    n_sites = A.shape[0]

    D = np.zeros((n_sites))

    for i in range(0,n_sites):
        for j in range(0,n_sites):
            D[i] += A[i,j]
    L = np.diag(D)-A

    l,v = np.linalg.eigh(L)
    
    sort_ind = np.argsort(l)

    l = l[sort_ind]
    v = v[:,sort_ind]

    fv = v[:,fv_i]

    sort_ind = np.argsort(fv)

    fv = fv[sort_ind]

    print
    print sort_ind+1
    
    Asorted = Ain[sort_ind,:][:,sort_ind]
    #np.savetxt(fileName+".sorted.m",Asorted)
   
    
    print
    print " Eigenvalues: "
    for i in range(l.shape[0]):
        print " %4i = %18.8f" %(i,l[i])

   
    #for i in range(l.shape[0]):
    #    fv_e[i,file_count]
    #file_count += 1
    fv_e.append(l)

    print
    print " Fiedler Vector: ", l[fv_i]
    for i in range(fv.shape[0]):
        print " %4i = %18.8f" %(i,fv[i])


np.savetxt("graph_eigenvalues.m",np.array(fv_e))

