import numpy as np
import nbtucker
import hamiltonian_generator 
import itertools as it
from nbtucker import *

n_blocks = 3
iepa_order = 2



#blocks = [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15],[16,17,18,19],[20,21,22,23]]
#blocks = [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15],[16,17,18,19]]
blocks = [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]]
#blocks = [[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13],[14,15]]
blocks = [[0,1,2,3],[4,5,6,7],[8,9,10,11]]

#j12 = np.loadtxt('j12.04.1d.period.afero.24site.m')
#j12 = np.loadtxt('jtemp_12.m')
j12 = np.loadtxt('j12_grid4x4.m')
#j12 = np.loadtxt('j12.04.1d.linear.afero.12site.m')
#j12 = np.loadtxt('block4/j12.05.m')
#j12 = np.loadtxt('j12.05.1d.linear.afero.16site.m')
#j12 = np.loadtxt('j12.05.1d.period.afero.16site.m')
j12 = np.loadtxt('j12.05.lin.12.m')

assert(np.allclose(j12.T,j12))


vec_key = []
frag = [i for i in range(0,n_blocks)]
for l in range(0,iepa_order+1):
    for k in it.combinations(frag,l):
        vec_key.append(k)

nps0 = [1,1,1,1]
e0, tbs0, lbs0, v,brdmn  = nbtucker.nbody_tucker(j12 = j12, blocks = blocks, n_p_states = nps0,n_body_order = 0)

miter = 5
energy_per_iter = []
e_icpa = []
d_old = {}
d_new = {}
for i in range(0,miter):
    e = {}
    tbs = {}
    lbs = {}
    core = {}
    brdm = {}
    for key in vec_key:
        nps_iepa=[]
        for a in range(0,n_blocks):
            if a in key:
                nps_iepa.append(16)
            else:
                nps_iepa.append(1)
        print(nps_iepa)
        e[key], tbs[key], lbs[key], core[key],brdm[key]  = nbtucker.nbody_tucker(j12 = j12, blocks = blocks, n_p_states = nps_iepa,n_body_order = 0,max_iter=1,lattice_blocks=lbs0)

        #brdm[key] = nbtucker.form_brdm(n_blocks, tbs[key], lbs[key], j12,core[key],ts=0)


    for key in vec_key:
        print " Key    :%16s" %(key,),"   E:%16.10f   "%(e[key])

    ###
    # generate the new RDM using the IEPA formalism
    ###

    crdm = {}
    new_rdm = {}
    corr_rdm = {}
    for bi in frag:
        ii = 0
        for key in vec_key:
            #crdm[key,bi] = np.dot(lbs[key][bi].vecs.T,np.dot(brdm[key][bi],lbs[key][bi].vecs))
            #print key
            #print("OLD")
            #print crdm[key,bi]
            #print brdm[key][bi]
            #print(-np.linalg.eigvalsh(-crdm[key,bi]))
            if len(key) == 0:
                #new_rdm[bi] = crdm[key,bi]
                new_rdm[bi] = cp.deepcopy(brdm[key][bi])
            if len(key) == 1:
                ii += 1     #imp
                #new_rdm[bi] -= (n_blocks-ii)* crdm[key,bi]
                new_rdm[bi] -= (n_blocks-ii)* brdm[key][bi]
                corr_rdm[key,bi] = brdm[key][bi] - brdm[()][bi] 
            if len(key) == 2:
                #new_rdm[bi] +=  crdm[key,bi]
                new_rdm[bi] +=  brdm[key][bi]
                #corr_rdm[key,bi] = brdm[key][bi] - brdm[(key[0]),][bi] - brdm[(key[1]),][bi]
                corr_rdm[key,bi] = brdm[key][bi] - brdm[()][bi]

        if iepa_order ==1: 
            for key in vec_key:
                if len(key) == 0:
                    new_rdm[bi] = cp.deepcopy(brdm[key][bi])
                if len(key) == 1:
                    new_rdm[bi] += corr_rdm[key,bi] 
                if len(key) == 2:
                    new_rdm[bi] += corr_rdm[key,bi] - corr_rdm[(key[0],),bi] - corr_rdm[(key[1],),bi]

        #print("CNEW")
        #print(new_rdm[bi])
        #print(np.dot(lbs[key][bi].vecs.T,np.dot(new_rdm[bi],lbs[key][bi].vecs)))

    ii = 0
    for key in vec_key:
        if len(key) == 0:
            icpa = e[key]
        if len(key) == 1:
            ii += 1     #imp
            if iepa_order ==2: 
                icpa -= (n_blocks-ii)* e[key]
            if iepa_order ==1: 
                icpa += e[key] - e[()]
        if len(key) == 2:
            icpa +=  e[key]
    e_icpa.append(icpa)

    ###
    #   diagonalize the new BRDM
    ###
    overlaps = []
    lbs_new = {}
    for bi in frag:
        Bi = lbs[()][bi]
        #lx,vx = np.linalg.eigh(new_rdm[bi] + 0.0022 * Bi.full_S2  + 0.0032 * Bi.full_Sz )
        lx,vx = np.linalg.eigh(new_rdm[bi])

        lx = vx.T.dot(new_rdm[bi]).dot(vx).diagonal()
        sort_ind = np.argsort(lx)[::-1]
        lx = lx[sort_ind]
        vx = vx[:,sort_ind]
        print("BRDM eigenvalues     ")
        print lx

        vp = vx[:,0:Bi.ss_dims[0]]
        vq = vx[:,Bi.ss_dims[0]:Bi.ss_dims[0]+Bi.ss_dims[1]]

        tmp, up = np.linalg.eigh(vp.T.dot(new_rdm[bi] + Bi.full_H + Bi.full_Sz + Bi.full_S2).dot(vp))
        vp = vp.dot(up)
        sort_ind = np.argsort(vp.T.dot(new_rdm[bi]).dot(vp).diagonal() )[::-1]
        vp = vp[:,sort_ind]
        v = vp

        if Bi.nq > 0: 
            tmp, uq = np.linalg.eigh(vq.T.dot(new_rdm[bi] + Bi.full_H + Bi.full_Sz + Bi.full_S2).dot(vq))
            vq = vq.dot(uq)
            sort_ind = np.argsort(vq.T.dot(new_rdm[bi]).dot(vq).diagonal() )[::-1]
            vq = vq[:,sort_ind]
            v = np.hstack( ( vp,vq) )


        sz = v.T.dot(Bi.full_Sz).dot(v).diagonal()
        s2 = v.T.dot(Bi.full_S2).dot(v).diagonal()
        lx = v.T.dot(new_rdm[bi]).dot(v).diagonal()
        h = v.T.dot(Bi.full_H).dot(v).diagonal()

        # compute transormation matrices to rotate ci vectors to new basis
        overlaps.append( Bi.vec().T.dot(v))

        #update Block vectors
        #print(v)
        Bi.vecs[:,0:Bi.ss_dims[0]+Bi.ss_dims[1]] = v
        Bi.form_H()
        Bi.form_site_operators()
        #lbs[()][bi] = Bi
        lbs_new[bi] = cp.deepcopy(Bi)
        print("--------------")
        print("CRDM for block",bi)
        print("--------------")
        print(np.dot(lbs0[bi].vecs.T,np.dot(new_rdm[bi],lbs0[bi].vecs)))


    e, tbs0, lbs0, v,brdmm  = nbtucker.nbody_tucker(j12 = j12, blocks = blocks, n_p_states = nps0,n_body_order = 0, lattice_blocks = lbs_new,max_iter=1)

    d_new[i] = brdmm[0]

    print(lbs0[0].vecs)
    #energy_per_iter.append(eh1+eh2+eh3+eh4)
    energy_per_iter.append(e)

nps0 = [16,16,16,16]
efci, tbs0, lbs0, v,brdm  = nbtucker.nbody_tucker(j12 = j12, blocks = blocks, n_p_states = nps0,n_body_order = 0,lattice_blocks=lbs_new)
e, tbs0, lbs0, v,brdm  = nbtucker.nbody_tucker(j12 = j12, blocks = blocks, n_p_states = nps0,n_body_order = 0)

print(e0)
for i in range(0,len(energy_per_iter)):
    print(energy_per_iter[i],e_icpa[i])

for i in range(0,len(energy_per_iter)):
    print(d_new[i])


print("old")
print(brdmn[0])
print(brdm[0])
print(e)
print(efci)
