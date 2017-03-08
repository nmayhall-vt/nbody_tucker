#! /usr/bin/env python
import numpy as np
import scipy
import scipy.linalg
import scipy.io
import copy as cp


from tucker import *


def form_hdvv_H(lattice,j12):
    """
    Form Heisenberg-Dirac van-Vleck Hamiltonian
    """
    n_sites = len(lattice)
    n_configs = np.power(2,n_sites) 
    
    print " Lattice: ", lattice
    print
    print " J12"
    print j12
    print


    sx = .5*np.array([[0,1.],[1.,0]])
    sy = .5*np.array([[0,(0-1j)],[(0+1j),0]])
    sz = .5*np.array([[1,0],[0,-1]])
    s2 = .75*np.array([[1,0],[0,1]])
    s1 = sx + sy + sz
    I1 = np.eye(2)

    sp = sx + sy*(0+1j)
    sm = sx - sy*(0+1j)

    sp = sp.real
    sm = sm.real


    H_dict = {}
    H_tot = np.zeros([n_configs,n_configs])
    S2_tot = np.zeros([n_configs,n_configs])

    
    for si,i in enumerate(lattice):
        i1 = np.eye(np.power(2,si))
        i2 = np.eye(np.power(2,n_sites-si-1))
        S2_tot += np.kron(i1,np.kron(s2,i2))

        for sj,j in enumerate(lattice):
            if sj>si:
                h = np.kron(sp,sm) 
                h += np.kron(sm,sp) 
                h += 2*np.kron(sz,sz)
                h = -j12[si,sj]*h
                H_dict[(si,sj)] = h

                j = j12[si,sj]
                i1 = np.eye(np.power(2,si))
                i2 = np.eye(np.power(2,sj-si-1))
                i3 = np.eye(np.power(2,n_sites-sj-1))
                print si,sj
                #print
                #print "i1\n", i1 
                #print "i2\n", i2 
                #print "i3\n", i3 
                #print "h\n", h
                S1S2   = np.kron(i1,np.kron(sp,np.kron(i2,np.kron(sm,i3))))
                S1S2 += np.kron(i1,np.kron(sm,np.kron(i2,np.kron(sp,i3)))) 
                S1S2 += 2*np.kron(i1,np.kron(sz,np.kron(i2,np.kron(sz,i3))))
                
                H_tot -= j*S1S2
                S2_tot += S1S2

    return H_tot, H_dict, S2_tot



