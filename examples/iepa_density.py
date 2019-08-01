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
#j12 = np.loadtxt('j12.05.1d.linear.afero.12site.m')

assert(np.allclose(j12.T,j12))

"""
nps0 = [1,1,1,1]
e0, tbs0, lbs0, v0  = nbtucker.nbody_tucker(j12 = j12, blocks = blocks, n_p_states = nps0,n_body_order = 0)
brdm0 = nbtucker.form_brdm(n_blocks, tbs0, lbs0, j12,v0,ts=0)
nps1 = [16,16,1,1]
e, tbs, lbs, v  = nbtucker.nbody_tucker(j12 = j12, blocks = blocks, n_p_states = nps1,n_body_order = 0)
brdm1 = nbtucker.form_brdm(n_blocks, tbs, lbs, j12,v,ts=0)

print(np.dot(lbs0[0].vecs.T,np.dot(brdm0[0],lbs0[0].vecs)))
print(np.dot(lbs0[0].vecs.T,np.dot(brdm1[0],lbs0[0].vecs)))

crdm1 = np.dot(lbs0[0].vecs.T,np.dot(brdm1[0],lbs0[0].vecs))
print(-np.linalg.eigvalsh(-crdm1))
"""


vec_key = []
frag = [i for i in range(0,n_blocks)]
for l in range(0,iepa_order+1):
    for k in it.combinations(frag,l):
        vec_key.append(k)

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
    e[key], tbs[key], lbs[key], core[key]  = nbtucker.nbody_tucker(j12 = j12, blocks = blocks, n_p_states = nps_iepa,n_body_order = 0)

    brdm[key] = nbtucker.form_brdm(n_blocks, tbs[key], lbs[key], j12,core[key],ts=0)


for key in vec_key:
    print " Key    :%16s" %(key,),"   E:%16.10f   "%(e[key])

crdm = {}
new_rdm = {}
for bi in frag:
    ii = 0
    for key in vec_key:
        crdm[key,bi] = np.dot(lbs[key][bi].vecs.T,np.dot(brdm[key][bi],lbs[key][bi].vecs))
        print key
        print("OLD")
        print crdm[key,bi]
        #print(-np.linalg.eigvalsh(-crdm[key,bi]))
        if len(key) == 0:
            #new_rdm[bi] = crdm[key,bi]
            new_rdm[bi] = brdm[key][bi]
        if len(key) == 1:
            ii += 1     #imp
            #new_rdm[bi] -= (n_blocks-ii)* crdm[key,bi]
            new_rdm[bi] -= (n_blocks-ii)* brdm[key][bi]
        if len(key) == 2:
            #new_rdm[bi] +=  crdm[key,bi]
            new_rdm[bi] +=  brdm[key][bi]
    print("NEW")
    print(new_rdm[bi])
    print(np.dot(lbs[key][bi].vecs.T,np.dot(new_rdm[bi],lbs[key][bi].vecs)))
            



