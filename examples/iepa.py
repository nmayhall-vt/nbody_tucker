import numpy as np
import nbtucker
import hamiltonian_generator 
import itertools as it
from nbtucker import *

n_blocks = 4
iepa_order = 2



#blocks = [[0,1,6,7],[2,3,8,9],[4,5,10,11],[12,13,18,19],[14,15,20,21],[16,17,22,23]]
#blocks = [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15],[16,17,18,19],[20,21,22,23]]
#blocks = [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15],[16,17,18,19]]
blocks = [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]]
#blocks = [[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13],[14,15]]
#blocks = [[0,1,2,3],[4,5,6,7],[8,9,10,11]]

#j12 = np.loadtxt('j12.04.1d.period.afero.24site.m')
#j12 = np.loadtxt('jtemp_12.m')
j12 = np.loadtxt('j12_grid4x4.m')
#j12 = np.loadtxt('block4/j12.06.m')
#j12 = np.loadtxt('j12.05.1d.linear.afero.12site.m')
n_p_states = [4,4,4,4]

nbtucker.nbody_tucker(   j12 = j12,
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


new_e = 0 
ii = 0
for key in vec_key:
    #print(-np.linalg.eigvalsh(-crdm[key,bi]))
    if len(key) == 0:
        #new_rdm[bi] = crdm[key,bi]
        new_e = e[key]
    if len(key) == 1:
        ii += 1     #imp
        #new_rdm[] -= (n_blocks-ii)* crdm[key,]
        new_e -= (n_blocks-ii)* e[key]
    if len(key) == 2:
        #new_rdm[] +=  crdm[key,]
        new_e +=  e[key]
print("OLD")
print(e[()])
print("NEW")
print(new_e)
            
