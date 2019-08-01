import numpy as np
import nbtucker
import hamiltonian_generator 

n_p_states = [4,4,4,4,4,4]
blocks = [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15],[16,17,18,19],[20,21,22,23]]

n_p_states = [4,4,4]
blocks = [[0,1,2,3],[4,5,6,7],[8,9,10,11]]

nbtucker.nbody_tucker(   j12 = np.loadtxt('j12.05.1d.linear.afero.12site.m'),
                    blocks = blocks, 
                    n_p_states = n_p_states,
                    n_body_order =2,
                    pt_order =2,
                    #pt_mit =12,
                    n_roots = 1,
                    diis_start=0,
                    n_diis_vecs=4,      #max diis subspace size
                    pt_type = 'mp',
            )

"""
for i in range(2,6):
    nbtucker.nbody_tucker(   j12 = np.loadtxt('j12.0'+str(i)+'.1d.period.afero.24site.m'),
                        blocks = blocks, 
                        n_p_states = n_p_states,
                        n_body_order =2,
                        pt_order =2,
                        #pt_mit =12,
                        n_roots = 1,
                        diis_start=0,
                        pt_type = 'mp',
                )

"""
