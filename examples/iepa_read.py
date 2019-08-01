import numpy as np
import nbtucker
import hamiltonian_generator 
import itertools as it
from nbtucker import *

n_blocks = 4
iepa_order = 2



#blocks = [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15],[16,17,18,19],[20,21,22,23]]
#blocks = [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15],[16,17,18,19]]
blocks = [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]]
#blocks = [[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13],[14,15]]
#blocks = [[0,1,2,3],[4,5,6,7],[8,9,10,11]]

#j12 = np.loadtxt('j12.04.1d.period.afero.24site.m')
#j12 = np.loadtxt('jtemp_12.m')
j12 = np.loadtxt('j12_grid4x4.m')
#j12 = np.loadtxt('block4/j12.05.m')
j12 = np.loadtxt('j12.05.1d.linear.afero.16site.m')

assert(np.allclose(j12.T,j12))

nps0 = [1,1,1,1]
e0, tbs0, lbs0, v0  = nbtucker.nbody_tucker(j12 = j12, blocks = blocks, n_p_states = nps0,n_body_order = 0)
brdm0 = nbtucker.form_brdm(n_blocks, tbs0, lbs0, j12,v0,ts=0)
nps1 = [1,4,4,1]
e, tbs, lbs, v  = nbtucker.nbody_tucker(j12 = j12, blocks = blocks, n_p_states = nps1,n_body_order = 0)
brdm1 = nbtucker.form_brdm(n_blocks, tbs, lbs, j12,v,ts=0)

print(np.dot(lbs0[0].vecs.T,np.dot(brdm0[0],lbs0[0].vecs)))
print(np.dot(lbs0[0].vecs.T,np.dot(brdm1[0],lbs0[0].vecs)))

crdm1 = np.dot(lbs0[0].vecs.T,np.dot(brdm1[0],lbs0[0].vecs))
print(-np.linalg.eigvalsh(-crdm1))

et, tbs, lbs, v  = nbtucker.nbody_tucker(j12 = j12, blocks = blocks, n_p_states = nps1,n_body_order = 0, lattice_blocks = lbs)


brdms = nbtucker.form_brdm(n_blocks, tbs, lbs, j12,v,ts=0)

overlaps = []
bi = 0
for bi in range(0,n_blocks):
    Bi = lbs[bi]
    lx,vx = np.linalg.eigh(brdms[bi] + 0.0022 * Bi.full_S2 )

    lx = vx.T.dot(brdms[bi]).dot(vx).diagonal()
    sort_ind = np.argsort(lx)[::-1]
    lx = lx[sort_ind]
    vx = vx[:,sort_ind]
    print lx

    vp = vx[:,0:Bi.ss_dims[0]]
    vq = vx[:,Bi.ss_dims[0]:Bi.ss_dims[0]+Bi.ss_dims[1]]

    tmp, up = np.linalg.eigh(vp.T.dot(brdms[bi] + Bi.full_H + Bi.full_Sz + Bi.full_S2).dot(vp))
    vp = vp.dot(up)
    sort_ind = np.argsort(vp.T.dot(brdms[bi]).dot(vp).diagonal() )[::-1]
    vp = vp[:,sort_ind]
    v = vp

    if Bi.nq > 0: 
        tmp, uq = np.linalg.eigh(vq.T.dot(brdms[bi] + Bi.full_H + Bi.full_Sz + Bi.full_S2).dot(vq))
        vq = vq.dot(uq)
        sort_ind = np.argsort(vq.T.dot(brdms[bi]).dot(vq).diagonal() )[::-1]
        vq = vq[:,sort_ind]
        v = np.hstack( ( vp,vq) )


    sz = v.T.dot(Bi.full_Sz).dot(v).diagonal()
    s2 = v.T.dot(Bi.full_S2).dot(v).diagonal()
    lx = v.T.dot(brdms[bi]).dot(v).diagonal()
    h = v.T.dot(Bi.full_H).dot(v).diagonal()

    # compute transormation matrices to rotate ci vectors to new basis
    overlaps.append( Bi.vec().T.dot(v))     ############################NOT BEING USED NOW

    #update Block vectors
    Bi.vecs[:,0:Bi.ss_dims[0]+Bi.ss_dims[1]] = v
    Bi.form_H()
    Bi.form_site_operators()


e, tbs, lbs, v  = nbtucker.nbody_tucker(j12 = j12, blocks = blocks, n_p_states = nps0,n_body_order = 0, lattice_blocks = lbs)
print(e0,et,e)
