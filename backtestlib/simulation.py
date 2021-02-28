import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .optimizer import BenchMark, SampleBased, SpectralCut, SpectralSelection
from numpy.linalg import inv

class Simulator:
    """Simulator with multivariate normal returns
    """
    
    def __init__(self, n, p, mu_real, sigma_real, seed=0):
        self.p = p
        self.n = n
        self.mu_real = mu_real
        self.sigma_real = sigma_real
        self.seed = seed
        self.sharpe_ratios = {}
        
    def gen_sample(self):
        return np.random.multivariate_normal(mean=self.mu_real, 
                                             cov=self.sigma_real, 
                                             size=self.n*2)
    
    @property
    def max_sharpe(self):
        if self.mu_real is not None and self.sigma_real is not None:
            return np.sqrt(self.mu_real.T @ inv(self.sigma_real) @ self.mu_real)
        return 
    
    @property
    def msr_weight(self):
        if self.mu_real and self.sigma_real:
            return inv(self.sigma_real) @ self.mu_real / (np.ones(len(self.mu_real)).T @ inv(self.sigma_real) @ self.mu_real)
        return 
        
    @staticmethod
    def _parse_opt(opt, *args, **kwargs):
        if opt == 'sp':
            return SampleBased()
        elif opt == 'sc':
            return SpectralCut(*args, **kwargs)
        elif opt == 'ss':
            return SpectralSelection(*args, **kwargs)
        else:
            raise ValueError("opt not supported yet")
        
    def run_sim(self, opt, mu_pred, num_times=500, *args, **kwargs):
        sharpe_ratios = []
        _opt = self._parse_opt(opt, *args, **kwargs)
        np.random.seed(self.seed)
        for _ in range(num_times):
            sample = self.gen_sample()
            if isinstance(sample, pd.DataFrame):
                sample = sample.values
            sample_corr = np.corrcoef(sample[:self.n,:].T)
            weights = _opt.optimize(signal=pd.Series(mu_pred), 
                                   sample_var=sample_corr)
            os_pnl = (sample[self.n:,:] @ weights)
            sharpe_ratios.append(os_pnl.mean() / os_pnl.std())
        self.sharpe_ratios[opt] = sharpe_ratios
        return sharpe_ratios
    
    
    def plot_dist(self, opts, names=None):
        df_all = []
        if not names: names = [None] * len(opts)
        for opt, name in zip(opts, names):
            df = pd.DataFrame(self.sharpe_ratios[opt], columns=['OS Sharpe Ratio'])
            df['Optimizer'] = name if name else opt
            df_all.append(df)
        df_all = pd.concat(df_all, axis=0)
        g = sns.displot(data=df_all, x='OS Sharpe Ratio', kind="kde", 
                        fill=True, hue='Optimizer')
        if self.max_sharpe:
            g.ax.axvline(self.max_sharpe)
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle(f"n = {self.n}; p = {self.p}", fontsize=20)
        

class SimulatorSB(Simulator):
    """Simulator with Structral Breaks
    """
    
    def __init__(self, n, p, mu_real, sigma_real, sigma_real_st, seed=0):
        super().__init__(n, p, mu_real, sigma_real, seed=seed)
        self.sigma_real_st = sigma_real_st
    
    def gen_sample(self, n_splits=4):
        sigmas = np.linspace(self.sigma_real_st.flatten(), self.sigma_real.flatten(), num=n_splits)
        sigma_all = [sigmas[i,:].reshape(self.p, self.p) for i in range(n_splits)]
        sample = []
        for sigma in sigma_all:
            sample.append(np.random.multivariate_normal(mean=self.mu_real, 
                                                        cov=sigma, 
                                                        size=self.n * 2 // n_splits))
        return np.vstack(sample)
        
        
class SimulatorBS(Simulator):
    
    def __init__(self, n, p, df_returns, seed=0):
        mu_real, sigma_real = None, None
        super().__init__(n, p, mu_real, sigma_real, seed=seed)
        self.df = df_returns
        
    def gen_sample(self):
        start_index = np.random.randint(0, len(self.df)-self.n*2)
        return self.df.iloc[start_index:start_index+self.n*2,:self.p]
        
