#! /usr/bin/env python
import numpy as np
import scipy
import scipy.linalg
import scipy.io
import copy as cp


from tucker import *


np.random.seed(2)
A = np.random.rand(8,8,8,8,8)
dims = A.shape
n_modes = len(A.shape)

#Acore, Atfac = tucker_decompose(A,0,1)
Acore, Atfac = tucker_decompose_list(A,(1,1,1,1,1))

B = tucker_recompose(Acore,Atfac)

print "\n Norm of Error tensor due to compression: %12.3e\n" %np.linalg.norm(B-A)

C = B

#1-Body
for si,i in enumerate(dims):
    dims2 = np.ones(n_modes)
    dims2[si]=-1
    Bcore, Btfac = tucker_decompose_list(A,dims2)
    C = C + tucker_recompose(Bcore,Btfac)

#2-Body
for si,i in enumerate(dims):
    for sj,j in enumerate(dims):
        if si>sj:
            dims2 = np.ones(n_modes)
            dims2[si]=-1
            dims2[sj]=-1
            Bcore, Btfac = tucker_decompose_list(A,dims2)
            C = C + tucker_recompose(Bcore,Btfac)


#3-Body
for si,i in enumerate(dims):
    for sj,j in enumerate(dims):
        if si>sj:
            for sk,k in enumerate(dims):
                if sj>sk:
                    dims2 = np.ones(n_modes)
                    dims2[si]=-1
                    dims2[sj]=-1
                    dims2[sk]=-1
                    Bcore, Btfac = tucker_decompose_list(A,dims2)
                    C = C + tucker_recompose(Bcore,Btfac)

#4-Body
if 1:
		for si,i in enumerate(dims):
		    for sj,j in enumerate(dims):
		        if si>sj:
		            for sk,k in enumerate(dims):
		                if sj>sk:
		                    for sl,l in enumerate(dims):
		                        if sk>sl:
				                    dims2 = np.ones(n_modes)
				                    dims2[si]=-1
				                    dims2[sj]=-1
				                    dims2[sk]=-1
				                    dims2[sl]=-1
				                    Bcore, Btfac = tucker_decompose_list(A,dims2)
				                    C = C + tucker_recompose(Bcore,Btfac)


print "\n Norm of Error tensor due to 111   compression: %12.3e\n" %np.linalg.norm(B-A)
print "\n Norm of Error tensor due to 3body compression: %12.3e\n" %np.linalg.norm(C-A)
