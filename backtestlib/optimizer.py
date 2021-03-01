import logging as log
import sys
log.basicConfig(level = log.INFO, stream=sys.stdout,format='backtest-%(asctime)s-%(funcName)s: %(message)s', datefmt='%Y-%b-%d %H:%M:%S')

import numpy as np
import pandas as pd
import numpy.linalg as LA
import scipy.optimize
from typing import Union

class Optimizer:
    """ An optimizer converts a signal to positions
    """
    
    def __init__(self, **kw):
        """ Pass into parameters (e.g. max position), if any.
        """
        pass

    def optimize(self, signal, **kw):
        """ 
        Pass into signals and other parames, if any.
        Returns position of each symbol.
        """
        pass
        
class EqualWeighted(Optimizer):
    """
    This optimizer longs 1 dollar, with equal position for each stock.
    """

    def __init__(self, **kw):
        log.info(f"BenchMark initialized")
        
    def optimize(self, signal=None, **kw):
        """ Builds equal position for all stocks.
        """
        
        pos = 1 / len(signal)*np.ones_like(signal)
        return np.nan_to_num(pos, 0)
        
        
class SampleBased(Optimizer):
    """
    This optimizer given postion as variance inverse signal
    """

    def __init__(self, **kw):
        log.info(f"SampleBased initialized")
        
    def optimize(self, signal=None, sample_var: np.ndarray = None, **kw):
        """ Builds variance inverse signal position for all stock  sample_variance is n*n square matrix, with n = len(signal)
        """
        assert sum(np.isnan(signal)) == 0
        pos = LA.inv(sample_var)@np.nan_to_num(signal.values, 0)
        pos/=np.sum(pos)
        return np.nan_to_num(pos, 0)
        
        
        
class SpectralCut(Optimizer):
    """
    This optimizer uses SpectralCut
    """

    def __init__(self,delta = 0.1, **kw):
        log.info(f"SpectralCut initialized with  delta {delta}")
        self.delta = delta
    def optimize(self, signal=None, sample_var: np.ndarray = None, **kw):
        """ Builds variance inverse signal position for all stock  sample_variance is n*n square matrix, with n = len(signal)
        """
        assert sum(np.isnan(signal)) == 0
        n = len(sample_var)
        u,s, vh = LA.svd(sample_var)
        mu = signal.values
        delta = self.delta

        def get_appx_error(K):
            cut = np.zeros(n)
            cut[:K] = 1.
            A = u@np.diag(cut)@vh
            mu_K = A@mu
            appx_error = LA.norm(mu- mu_K)/LA.norm(mu)
            return appx_error
        appx_errors = np.array([get_appx_error(K) for K in range(n+1)])
        K_opt = np.argmax(appx_errors < delta)

        log.info(f"Optimal K is {K_opt}")
        if K_opt == 0:
            log.info(f"delta too big")
        
        cut = np.zeros(n)
        cut[:K_opt] = 1.
        
        a = cut * (u.T@mu)
        log.info(f" return component {a}")
        B = u@np.diag(1/(s+1e-8)*cut)@vh
        pos =  B@mu
        pos /= np.sum(pos)
        return np.nan_to_num(pos, 0)
        
class SpectralSelection(Optimizer):
    """
    This optimizer uses SpectralSelection
    """

    def __init__(self,delta = 0.1, c = 1, **kw):
        log.info(f"SpectralCut initialized with  delta {delta}, c {c}")
        self.delta = delta
        self.c = c
        
    def optimize(self, signal=None, sample_var: np.ndarray = None, **kw):
        """ Builds SpectralSelection 
        """
        assert sum(np.isnan(signal)) == 0
        n = len(sample_var)
        u,s, vh = LA.svd(sample_var)
        mu = signal.values
        delta = self.delta
        c = self.c

        a_ls = u.T@mu

        def get_appx_error(gamma):
            
            a_sel = np.sign(a_ls)*np.maximum(np.abs(a_ls) - gamma *np.power(s,-c),0)
            mu_sel = u@a_sel
            appx_error = LA.norm(mu- mu_sel)/LA.norm(mu)
            return abs(delta - appx_error)
        sol = scipy.optimize.minimize(get_appx_error, x0 = [0])
        
        gamma_opt = sol.x[0]
        log.info(f"Optimal gamma is {round(gamma_opt,3)}")
        a_sel = np.sign(a_ls)*np.maximum(np.abs(a_ls) - gamma_opt *np.power(s,-c),0)
        log.info(f" return component {a_sel}")
        pos = u@np.diag(1/(s+1e-8))@a_sel
        pos /= np.sum(pos)
        return np.nan_to_num(pos, 0)
        
class ShrinkToIdentity(Optimizer):
    """
    This optimizer uses identity shrinkage method
    - suggested in 2004 Ledoit and Wolf A well-conditioned estimator forlarge-dimensional covariance matrices
    - https://www.sciencedirect.com/science/article/pii/S0047259X03000964 formula (14)
    """

    def __init__(self, **kw):
        log.info(f"ShrinkToIdentity initialized")
        
    def optimize(self, signal: pd.Series = None, returns: np.ndarray = None, **kw):
        """ Builds ShrinkToIdentity 
        signal: proxy of return 
        returns: p by n matrix of returns 
        """
        assert sum(np.isnan(signal)) == 0
        mu  = signal.values
        n = returns.shape[1]
        returns_demean = returns - np.mean(returns,axis = 1).reshape((-1,1))
        sample_var = (returns_demean@returns_demean.T)/n

        p = len(sample_var)
        ident = np.diag(np.ones(p))
        def f_dot(A,B):
            p = len(A)
            return np.sum(np.diag(A@B))/p

        def f_norm(A):
            return f_dot(A,A)**.5

        m_n =  f_dot(sample_var,ident)
        d_n =  f_norm(sample_var - m_n)
        b_n_bar = np.mean([f_norm(returns_demean[:,[i]]@returns_demean[:,[i]].T - sample_var)**2 for i in range(n)])/n
        b_n_bar = b_n_bar**.5 
        b_n = min(d_n, b_n_bar)
        a_n = (d_n*d_n - b_n*b_n)**.5
        log.info(f"shrink coefficients: a {round(a_n,3)}, b {round(b_n,3)}, d {round(d_n,3)}, m {round(m_n,3)}")

        var_opt = b_n**2/d_n**2*m_n + a_n**2/d_n**2*sample_var
        pos = LA.inv(var_opt)@np.nan_to_num(mu, 0)
        pos/=np.sum(pos)
        return np.nan_to_num(pos, 0)
        
class POET(Optimizer):
    """
    This optimizer uses principal orthogonal complement thresholding method
    - suggested in 2013 J Fan Large covariance estimation by thresholding principal orthogonal complements
    - choose soft-thresholding and optimal K
    """

    def __init__(self,C = 0.2, **kw):
        log.info(f"ShrinkToIdentity initialized")
        self.C = C
        
    def optimize(self, signal: pd.Series = None, returns: np.ndarray = None, **kw):
        assert sum(np.isnan(signal)) == 0
        C = self.C
        p,n = returns.shape
        returns_demean = returns - np.mean(returns,axis = 1).reshape((-1,1))
        sample_var = (returns_demean@returns_demean.T)/n

        # first part: determine K
        def IC1(n,p):
            return (p+n)/n/p*np.log(p*n/(p+n))
        u = LA.svd(returns_demean.T@returns_demean)[0]

        ans = []
        for K in range(max(int((p+1)**.5),4)):
            F = u[:,:K]*n**.5
            loss = np.log(LA.norm(returns_demean - returns_demean@(F@F.T)/n )**2/p/n) 
            loss += K*IC1(n,p)
            ans.append(loss)
        K_opt = np.argmin(ans)
        log.info(f"Optimal K is {K_opt}")

        #  part 2 : determine factor variance and residue threshholds
        F = u[:,:K_opt]*n**.5
        B = returns_demean@F/n
        U = returns_demean - B@F.T
        Sigma_R= U@U.T/n

        Theta = (U@U.T -Sigma_R)**2/n
        Tau = C*np.log(p)/np.sqrt(n)*np.sqrt(Theta)

        off_diag = np.sign(Sigma_R) * np.maximum(0, np.abs(Sigma_R) - Tau)
        off_diag -=np.diag(np.diag(off_diag))

        log.info(f"offdiagnal f-norm under C={C} is {round(LA.norm(off_diag),4)}")
        Sigma_RT = off_diag + np.diag(np.diag(Sigma_R))
        Sigma = B@B.T + Sigma_RT 

        #  part 3 : essemnle
        pos = LA.inv(Sigma)@np.nan_to_num(signal.values, 0)
        pos/=np.sum(pos)
        return np.nan_to_num(pos, 0)
 
