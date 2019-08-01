import numpy as np
import nbtucker
import hamiltonian_generator 

n_blocks = 4

n_p_states = [4,4,4,4]
blocks = [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]]

j12 = np.loadtxt('j12_grid4x4.m')

nbtucker.nbody_tucker(   j12 = j12,
                    blocks = blocks, 
                    n_p_states = n_p_states,
                    n_body_order =2,
                    #pt_mit =12,
            )

