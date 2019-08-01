import numpy as np
import nbtucker
import hamiltonian_generator 
import itertools as it
from nbtucker import *

n_blocks = 3
iepa_order = 2



#blocks = [[0,1,6,7],[2,3,8,9],[4,5,10,11],[12,13,18,19],[14,15,20,21],[16,17,22,23]]
#blocks = [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15],[16,17,18,19],[20,21,22,23]]
#blocks = [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15],[16,17,18,19]]
blocks = [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]]
#blocks = [[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13],[14,15]]
blocks = [[0,1,2,3],[4,5,6,7],[8,9,10,11]]

#j12 = np.loadtxt('j12.04.1d.period.afero.24site.m')
#j12 = np.loadtxt('jtemp_12.m')
j12 = np.loadtxt('j12_grid4x4_ec.m')
#j12 = np.loadtxt('block4/j12.06.m')
#j12 = np.loadtxt('j12.05.1d.linear.afero.12site.m')

assert(np.allclose(j12.T,j12))


#enb0, tb, lb, v = nbtucker.nbody_tucker(j12 = j12, blocks = blocks, n_body_order = 0)
#enb2, tb, lb, v = nbtucker.nbody_tucker(j12 = j12, blocks = blocks, n_body_order = 2)

#print(v)

def transform_Jinter(J,jold,jnew):
# {{{
    J1 = np.zeros((n,n))
    for i in range(0,J.shape[0]):
        for j in range(0,J.shape[0]):
            if J[i,j] == jold:
                print("haan")
                J1[i,j] = jnew
            else:
                J1[i,j] = J[i,j]

    return J1
# }}}

def evec_continuation(n,H,e1,e2,e3,e4):
# {{{
    #right now implemented for 4 vecs
    sub_evec = np.zeros((H.shape[0],4))

    sub_evec[:,0] = e1.reshape(H.shape[0])
    sub_evec[:,1] = e2.reshape(H.shape[0])
    sub_evec[:,2] = e3.reshape(H.shape[0])
    sub_evec[:,3] = e4.reshape(H.shape[0])

    sub_H = sub_evec.T.dot(H.dot(sub_evec))

    print(sub_H)

    print(np.linalg.eigvals(sub_H))
    print(sub_evec.T.dot(sub_evec))
    S = sub_evec.T.dot(sub_evec)

    #forming S^-1/2 to transform to A and B block.
    sal, svec = np.linalg.eigh(S)
    idx = sal.argsort()[::-1]
    sal = sal[idx]
    svec = svec[:, idx]
    sal = sal**-0.5
    sal = np.diagflat(sal)
    X = svec.dot(sal.dot(svec.T))

    sub_H2 = X.T.dot(sub_H.dot(X))
    print(sub_H2)

    print(np.linalg.eigvalsh(sub_H2))
    e_new = np.linalg.eigvalsh(sub_H2) 
    return e_new 
# }}}

n = j12.shape[0]
J1 = transform_Jinter(j12,-0.5,-0.010)
J2 = transform_Jinter(j12,-0.5,-0.020)
J3 = transform_Jinter(j12,-0.5,-0.030)
J4 = transform_Jinter(j12,-0.5,-0.040)
J = transform_Jinter(j12,-0.5,-10.0)
 
enb1, tb, lb, v1 = nbtucker.nbody_tucker(j12 = J1, blocks = blocks, n_body_order = 2)
enb2, tb, lb, v2 = nbtucker.nbody_tucker(j12 = J2, blocks = blocks, n_body_order = 2)
enb3, tb, lb, v3 = nbtucker.nbody_tucker(j12 = J3, blocks = blocks, n_body_order = 2)
enb4, tb, lb, v4 = nbtucker.nbody_tucker(j12 = J4, blocks = blocks, n_body_order = 2)



enb2, tb, lb, v4 = nbtucker.nbody_tucker(j12 = J, blocks = blocks, n_body_order = 2)
H,S2 = build_tucker_blocked_H(n_blocks,tb, lb, n_body_order=2, j12=J)
print(H)


e_new = evec_continuation(n,H,v1,v2,v3,v4)
efci, tb, lb, v4 = nbtucker.nbody_tucker(j12 = J, blocks = blocks, n_body_order = 4)
print("fci  ",efci)
print("nb2  ",enb2)
print("pred ",e_new[0])


