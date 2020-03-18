import numpy as np
import nbtucker
import hamiltonian_generator 
import itertools as it
from nbtucker import *

n_blocks = 4

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
n_p_states = [1,1,1,1]

e, tbs0, lbs0, v,brdm  = nbtucker.nbody_tucker(j12 = j12, blocks = blocks, n_p_states = n_p_states,n_body_order = 2)

#nbtucker.nbody_tucker(   j12 = j12,
#                    blocks = blocks, 
#                    n_p_states = n_p_states,
#                    n_body_order =2,
#                    pt_order =2,
#                    pt_mit =20,
#                    n_roots = 1,
#                    diis_start=0,
#                    n_diis_vecs=4,      #max diis subspace size
#                    max_iter = 20,
#                    pt_type = 'lcc',
#                    #lattice_blocks = lbs0,
#            )
