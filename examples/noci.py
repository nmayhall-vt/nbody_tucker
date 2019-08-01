import numpy as np
import nbtucker
import hamiltonian_generator 
import itertools as it
from nbtucker import *

n_blocks = 4
iepa_order = 2



#blocks = [[0,1,6,7],[2,3,8,9],[4,5,10,11],[12,13,18,19],[14,15,20,21],[16,17,22,23]]
#blocks = [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15],[16,17,18,19],[20,21,22,23]]
#blocks = [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15],[16,17,18,19]]
blocks = [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]]
#blocks = [[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13],[14,15]]
#blocks = [[0,1,2,3],[4,5,6,7],[8,9,10,11]]

#j12 = np.loadtxt('j12.04.1d.period.afero.24site.m')
#j12 = np.loadtxt('jtemp_12.m')
j12 = np.loadtxt('j12_grid4x4.m')
#j12 = np.loadtxt('block4/j12.06.m')
#j12 = np.loadtxt('j12.05.1d.linear.afero.12site.m')

assert(np.allclose(j12.T,j12))


enb0, tb, lb, v = nbtucker.nbody_tucker(j12 = j12, blocks = blocks, n_body_order = 0)
enb1, tb, lb, v = nbtucker.nbody_tucker(j12 = j12, blocks = blocks1, n_body_order = 0)
#print(E0+ecorr)
print "NB0   : %16.12f"%enb0 
print "NB0   : %16.12f"%enb0 

