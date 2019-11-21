import numpy as np
import scipy
import scipy.linalg
import scipy.io
import copy as cp
import argparse
import scipy.sparse
import scipy.sparse.linalg

from hdvv import *
from block import *
from davidson import *

import hamiltonian_generator 
from nbtucker import *

def test1():
    size = (2,6)
    blocks = [[0,1,2,3],[4,5,6,7],[8,9,10,11]]
    n_p_states = [4,4,4]
    e = nbody_tucker(   j12 = hamiltonian_generator.make_2d_lattice(size=size,blocks=blocks),
                    blocks = blocks, 
                    n_p_states = n_p_states
                )
    assert(abs(e+12.26873176) < 1e-7)
