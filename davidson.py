import numpy as np
import scipy
import scipy.linalg
import scipy.io
import copy as cp
import argparse
import scipy.sparse
import scipy.sparse.linalg

class Davidson:
    def __init__(self,dim,n_roots):
        self.iter           = 0     # Current iteration
        self.max_iter       = 20    # Max iterations
        self.precondition   = 1     # Use preconditioning?
        self.n_vecs         = 0     # Current number of vectors in subspace
        self.max_vecs       = 10    # Max number of vectors in subspace
        self.thresh         = 1e-8  # Convergence threshold
        self.n_roots        = n_roots  # Number of roots to optimize
        self.dim            = dim      # Dimension of matrix to diagonalize

        self.res_vals       = []    # Residual values
        self.ritz_vals      = []    # Ritz values
 
        self.sigma          = np.array([])  # current sigma vectors
        self.vecs           = np.array([])  # current subspace vectors
        self.AV_j           = np.array([])

    def form_sigma(self):
        pass
    
    def form_rand_guess(self):
        self.vecs   = np.random.rand(self.dim, self.n_roots)
    
    def update(self):
        if self.iter == 0:
            self.AV_j = self.sigma
        else:
            self.AV_j = np.hstack(( self.AV_j, self.sigma ))
        T = self.vecs.T.dot(self.AV_j)
        print self.AV_j.shape
        print self.sigma.shape
        T = .5*(T+T.T)
        l,v = np.linalg.eigh(T)
        
        sort_ind = np.argsort(l)
        l = l[sort_ind]
        v = v[:,sort_ind]

        V_k = cp.deepcopy(self.vecs)

        res_vals = []
        ritz_vals = []
        V_new = np.empty((self.dim, 0))
        for n in range(0,self.n_roots):
            l_n = l[n]
            ritz_vals.append(l_n)
            v_n = v[:,n]
	    r_n = (self.AV_j - l_n*self.vecs).dot(v_n);
            b_n = np.linalg.norm(r_n)
            r_n = r_n/b_n
            res_vals.append(b_n)
            
            r_n.shape = (r_n.shape[0],1)

            if b_n > self.thresh:
                print V_k.shape, r_n.shape
                r_n = r_n - self.vecs.dot(np.dot(self.vecs.T,r_n))
                
                if (V_new.shape[1] > 0):
                    r_n = r_n - V_new.dot(np.dot(V_new.T,r_n))

		b_n_p = np.linalg.norm(r_n)
                if (b_n / b_n_p > 1e-3):
                    r_n = r_n / b_n_p
                    V_new = np.hstack((V_new, r_n))

        self.vecs = np.hstack(( self.vecs, V_new ))
        self.n_vecs = self.vecs
        self.iter += 1

def print_iteration(self):
    print "  Iteration %4i " %self.iter, 
    print "|",
    print " Vecs:%4li : "% self.n_vecs ;
    print "|",
    for r in range(0,self.n_roots):
        print " %16.8f "%self.ritz_vals[r],
    print "|",
    for r in range(0,self.n_roots):
        print " %6.1e " % self.res_vals(r), 
    print

def converged(self):
    """
    Check for convergence
    """
    done = 1
    for k in range(0,self.res_vals):
        if abs(self.res_vals[k] > self.thresh):
            done = 0
    return done

def precondition(self):
    pass
