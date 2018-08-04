from __future__ import division

import sys
sys.path.insert(0, '../../../pylib')
sys.path.insert(0, '../GPy')
from pulp.utils.datalog import dlog

import numpy as np
#from sklearn.gaussian_process import GaussianProcess

import GPy
from GPy.util.linalg import pdinv

from scipy.stats import bernoulli
from pulp.utils.parallel import pprint
import pylab as plt

from mpi4py import MPI
comm = MPI.COMM_WORLD

#%load_ext autoreload
#%autoreload 2

def gp_selection(y, s, candidates, kern, opt_time, sigma2, gp_iter, gp_noise_variance, pca_proj_y, rand_proj_y):
    """
    input:
        - data y: (N, D) entire set of y^(1)...y^(D) \in R^D ... (N,D) ... to be considered as a lump per s_h^(n) dimension
        - latents s: (N, H) entire set of s... (N,H) ... to be considered as an N-length-vector for each s_h
        - kern: kernel to use for y, predefined in model_params, get's optimized/updated every gp_kern_reset iterations
        - opt_time: says when it's time to optimize/update hyperparams 
    output:
        - candidates: (N, Hprime) entire set of current "best" Hprime variables to choose from s for each n'th data point
                       ... n,h'th entry is the h'th variable (s_h)^(n) to use from s per n'th data point, y^(n) 
    """

    N, D = y.shape
    N_, H = s.shape
    _, Hprime = candidates.shape
    assert N == N_
   
    # define GP parameters and kernel
    # -------------------------------

    # TODO: make kernels a parameter to set 
    #regularize_kern = GPy.kern.white(input_dim=D)
    #var_est = y.var()#(axis=0) #* np.eye(D)
    ###pprint("var estimate = %f" % sigma2)
    #if gp_iter == 1:
    #regularize_kern = GPy.kern.linear(input_dim=D, variances = gp_iter*1.5)
        #regularize_kern = y.var(axis=1) * np.eye(D) 
    #kern += regularize_kern

    # create simple GP model
    # ----------------------
    TN = comm.allreduce(y.shape[0])
    data = comm.allgather({'y' : y,  's' : s})
    Y = np.zeros( (TN, y.shape[1]) )
    S = np.zeros( (TN, s.shape[1]) )
    c = 0
    for d in (data):
        n = d['y'].shape[0]
        Y[c:c+n] = d['y']
        S[c:c+n] = d['s']
        c+=n
    
    pprint('TN =%d' % TN)
    pprint('N = %d, N_ = %d' % (N, N_) )
    pprint('Y.shape = ')
    pprint(Y.shape)
    pprint('S.shape = ')
    pprint(S.shape)

    m = GPy.models.GPRegression(Y,S, kern)
    m["noise_variance"] = gp_noise_variance 

  

    pprint("Pre-op kernel parameters:")
    pprint(kern.parts[0]._get_params()) 
    pprint('noise variance = %f' % m["noise_variance"])

    # only optimize GP parameters every so few EM iterations
    # ------------------------------------------------------
    #opt_time = False        # XXX
    if opt_time == True:
        pprint("Re-optimizing GP")
        m.optimize()
        opt_time = False
        m = comm.bcast({'m' : m})['m']
        
    
    # print kernel params
    if len(kern.parts[0]._get_params()) > 1:
        kernel_type = 'rbf'
        pprint('Kernel type is %s' % kernel_type)
        rbf_var, rbf_lengthscale = kern.parts[0]._get_params()   
    else:    
        kernel_type = 'linear'
        pprint('Kernel type is %s' % kernel_type)
        rbf_var = 1.
        rbf_lengthscale = 1.    
        lin_ker_variance = kern.parts[0]._get_params() 

    pprint("post-op kernel parameters:")
    pprint(kern.parts[0]._get_params()) 
    pprint('post-op noise variance = %f' % m["noise_variance"])

    # Do efficient LOO-cross-validation and GP learning (Rasmussen ch. 5, eq. (5.12))
    # -------------------------------------------------------------------------------
    curr_kern = m.kern.K(Y)
    curr_kern += np.eye(TN) * m["noise_variance"]    
    invkern = np.linalg.inv(curr_kern)
    invSigma, sigmaL, _, _ = pdinv(curr_kern)
    pred_s_all = S - ( np.inner(invSigma, S.T) ) / np.diag(invkern)[:,None] # same as with np.dot(invkern, s)
    cov_s_all  = 1/ np.diag(invkern)[:,None]
    pred_err = pred_s_all - S
    ms_pred_err = (pred_err**2).mean()
    mean_cov = cov_s_all.mean()

    pprint("s predictions")
    pprint(pred_s_all[:10,:])
    pprint("s")
    pprint(S[:10,:])
    pprint("predictive covariance:")
    pprint(cov_s_all[:10,:])
    pprint("mean MSE of prediction")
    pprint(ms_pred_err)
    #attempt = pred_s_all + mse_s_all # still an idea


    # --------- select candidates ------------
    norm_s_pred_all = np.zeros( shape=(TN, H) )
    
    for n in xrange(N):
        abs_pred_s_all = np.abs(pred_s_all[n,:])  
        pred_sort = np.argsort(abs_pred_s_all)[::-1]
        candidates[n] = np.argsort(abs_pred_s_all)[-Hprime:]
        norm_s_pred_all[n, :] = (pred_s_all[n] - pred_s_all.mean()) / (2*cov_s_all[n])  
    


    # ---------- make some GP-prediction plots ------------

        if gp_iter == 1:
            # prepare projections of y
            rand_proj = np.random.normal( size=(D) )        
            rand_proj_y = np.inner(rand_proj, Y)

            # PCA projection
            cov_y = np.inner(Y.T, Y.T)
            eig_vals, eig_vects = np.linalg.eig(cov_y)
            first_eig_val_idx = np.argsort(eig_vals)[-1:] 
            first_eig_vec = eig_vects[first_eig_val_idx]  
            pca_proj_y = np.inner(first_eig_vec, Y)
       

   
    return candidates, m.kern.copy(), m["noise_variance"], rbf_var, rbf_lengthscale, S, pred_s_all, cov_s_all, mean_cov, ms_pred_err, pca_proj_y, rand_proj_y, Y, y




