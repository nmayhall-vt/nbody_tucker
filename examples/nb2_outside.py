import numpy as np
import nbtucker
import hamiltonian_generator 
import itertools as it
from nbtucker import *

n_blocks = 4
iepa_order = 2



blocks = [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]]

j12 = np.loadtxt('j12_grid4x4.m')
#j12 = np.loadtxt('j12.05.1d.linear.afero.16site.m')

nps0 = [4,4,4,4]

miter = 20
energy_per_iter = []
ts = 0

e, tbs, lbs, v,brdms  = nbtucker.nbody_tucker(j12 = j12, blocks = blocks, n_p_states = nps0,n_body_order = 2,diis_start=100,max_iter=1)
energy_per_iter.append(e)
e_old = e

for i in range(0,miter):

    print("outer shape",v.shape)

    #brdms = nbtucker.form_brdm(n_blocks, tbs, lbs, j12,v,ts)
    print(brdms)

    overlaps = []
    lbs_new = {}
    for bi in range(0,n_blocks):
        Bi = lbs[bi]
        lx,vx = np.linalg.eigh(brdms[bi] + 0.0022 * Bi.full_S2 )
        #lx,vx = np.linalg.eigh(brdms[bi])

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
        overlaps.append( Bi.vec().T.dot(v))

        #update Block vectors
        print(v)
        Bi.vecs[:,0:Bi.ss_dims[0]+Bi.ss_dims[1]] = v
        Bi.form_H()
        Bi.form_site_operators()
        #lbs[()][bi] = Bi
        lbs_new[bi] = Bi
    print("lbsnew")
    print(lbs_new[0].vecs)

    e, tbs, lbs, v,brdms  = nbtucker.nbody_tucker(j12 = j12, blocks = blocks, n_p_states = nps0,n_body_order = 2, lattice_blocks = lbs_new,max_iter=1)
    energy_per_iter.append(e)
    if abs(e - e_old) < 1e-9:
        break
    e_old = e


for i in range(0,len(energy_per_iter)):
    print(energy_per_iter[i])

