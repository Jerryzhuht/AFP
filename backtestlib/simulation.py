import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .optimizer import EqualWeighted, SampleBased, SpectralCut, SpectralSelection, POET, ShrinkToIdentity
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
            return np.sqrt(self.mu_real.T @ inv(self.sigma_real) @ self.mu_real) * np.sqrt(252)
        return 
    
    @property
    def msr_weight(self):
        if self.mu_real is not None and self.sigma_real is not None:
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
        
    def run_sim(self, opt, mu_pred, num_times=100, *args, **kwargs):
        sharpe_ratios = []
        _opt = self._parse_opt(opt, *args, **kwargs)
        np.random.seed(self.seed)
        for _ in range(num_times):
            sample = self.gen_sample()
            if isinstance(sample, pd.DataFrame):
                sample = sample.values
            sample_var = np.cov(sample[:self.n,:].T)
            weights = _opt.optimize(signal=pd.Series(mu_pred), 
                                   sample_var=sample_var)
            os_pnl = (sample[self.n:,:] @ weights)
            sharpe_ratios.append(os_pnl.mean() * np.sqrt(252) / os_pnl.std())
        self.sharpe_ratios[opt] = sharpe_ratios
        return sharpe_ratios, os_pnl
    
    
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
        return g
        

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
        

class BacktestEngine:
    
    def __init__(self, df_returns, p, lookback, rebalance_window):
        self.df = df_returns
        self.p = p
        self.l = lookback
        self.w = rebalance_window
        
        self.pnls = {}
        self.weights = {}
        self.turnovers = {}
    
    @staticmethod    
    def _parse_opt(opt, *args, **kwargs):
        if opt == 'sp':
            return SampleBased()
        elif opt == 'sc':
            return SpectralCut(*args, **kwargs)
        elif opt == 'ss':
            return SpectralSelection(*args, **kwargs)
        elif opt == 'ew':
            return EqualWeighted(*args, **kwargs)
        elif opt == 'poet':
            return POET(*args, **kwargs)
        elif opt == 's2i':
            return ShrinkToIdentity(*args, **kwargs)
        else:
            raise ValueError("opt not supported yet")
        
    def run(self, opt, mu_pred, *args, **kwargs):
        _opt = self._parse_opt(opt,*args, **kwargs)
        os_pnl_all = []
        weights_all = []
        turnover_all = []
        for i in range(self.l, len(self.df)-1, self.w):
            sample = self.df.iloc[i-self.l:i,:self.p]
            sample_var = sample.cov()
            weights = _opt.optimize(signal = pd.Series(mu_pred), 
                                    sample_var = sample_var,
                                    returns = sample.values.T)
            os_sample = self.df.iloc[i:i+self.w,:self.p]
            os_pnl = os_sample @ weights
            
            if i > self.l:
                target_weights = weights * prev_weights.sum()
                trv = np.abs(target_weights-prev_weights).sum()/prev_weights.sum()
                turnover_all.append(trv)
            
            os_sample_ret = (1+os_sample).cumprod(axis=0).iloc[-1,:].values
            prev_weights = weights * os_sample_ret
            
            os_pnl_all.append(os_pnl)
            weights_all.append(weights)
        self.pnls[opt] = pd.concat(os_pnl_all, axis=0)
        self.weights[opt] = np.vstack(weights_all)
        self.turnovers[opt] = np.array(turnover_all)
        
    def calc_stats(self, opts, names=None):
        df_stats = []
        if not names: names = opts[:]
        for opt, name in zip(opts, names):
            tmp = {}
            tmp['Method'] = name
            ret = 252 * self.pnls[opt].mean()
            std = np.sqrt(252) * self.pnls[opt].std()
            tmp['Return'] = "{:.2f}%".format(ret*100)
            tmp['Std'] = "{:.2f}%".format(std*100)
            tmp['Sharpe'] = "{:.2f}".format(ret/std)
            
            ww = self.weights[opt]
            short_pct = ((abs(ww).sum(axis=1)-1)/2/abs(ww).sum(axis=1)).mean()
            gross_expo = abs(ww).sum(axis=1).mean()
            tmp['% of Short'] = "{:.2f}%".format(abs(short_pct*100))
            tmp['Gross Exposure'] = "{:.2f}".format(gross_expo)
            
            mdd = (self.pnls[opt]+1).cumprod().diff().min()
            tmp['Max Drawdown'] = "{:.2f}%".format(mdd*100)
            
            trv = self.turnovers[opt].mean()
            tmp['Turnover'] = "{:.2f}".format(trv)
            
            df_stats.append(tmp)
        return pd.DataFrame(df_stats)

            
            