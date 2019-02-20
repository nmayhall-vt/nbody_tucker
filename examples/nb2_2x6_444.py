import nbtucker
import hamiltonian_generator 

size = (2,6)
blocks = [[0,1,2,3],[4,5,6,7],[8,9,10,11]]
n_p_states = [4,4,4]

nbtucker.nbody_tucker(   j12 = hamiltonian_generator.make_2d_lattice(size=size,blocks=blocks),
                blocks = blocks, 
                n_p_states = n_p_states
            )




