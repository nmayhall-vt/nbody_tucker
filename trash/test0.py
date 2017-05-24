#! /usr/bin/env python
import numpy as np
import scipy
import scipy.linalg
import scipy.io
import copy as cp


from tucker import *


np.random.seed(2)
A = np.random.rand(8,8,8,8,9)
n_modes = len(A.shape)

Acore, Atfac = tucker_decompose(A,0,1)

B = tucker_recompose(Acore,Atfac)

print "\n Norm of Error tensor due to compression: %12.3e\n" %np.linalg.norm(B-A)


Bcore, Btfac = tucker_decompose(B,0,1)
C = tucker_recompose(Bcore,Btfac)
print "\n Norm of Error tensor due to compression: %12.3e\n" %np.linalg.norm(C-B)
