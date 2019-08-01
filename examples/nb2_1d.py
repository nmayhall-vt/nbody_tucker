import numpy as np
import nbtucker
import hamiltonian_generator 

blocks = [[0,1,2,3],[4,5,6,7],[8,9,10,11]]
n_p_states = [4,4,4]

nbtucker.nbody_tucker(   j12 = np.loadtxt('j12.04.1d.linear.afero.12site.m'),
                    blocks = blocks, 
                    n_p_states = n_p_states,
                    n_body_order = 2,
                    pt_order = 0,
                    pt_mit = 2,
                    n_roots = 1,
                    diis_start=0,
                    pt_type = 'mp'
            )

