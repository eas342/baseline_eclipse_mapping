import matplotlib.pyplot as plt
import numpy as np
import starry
from copy import deepcopy
import astropy.units as u
import astropy.constants as const
from astropy.io import fits, ascii
from astropy.table import Table
from corner import corner
import pymc3 as pm
from celerite2.theano import terms, GaussianProcess
import pymc3_ext as pmx
import os

starry.config.lazy = True
starry.config.quiet = True

class starry_basemodel():
    def __init__(self,dataPath='sim_data/sim_data_baseline.ecsv',
                 descrip='newrho_001',
                 map_type='variable',amp_type='variable'):
        """
        Set up a starry model
        
        """
        self.data_path = dataPath
        self.get_data(path=self.data_path)
        self.mask = None
        self.descrip = "{}_maptype_{}_amp_type_{}".format(descrip,map_type,amp_type)
        self.mxap_name = 'mxap_{}.npz'.format(self.descrip)
        self.mxap_path = os.path.join('fit_data','mxap_soln',self.mxap_name)
        
        self.map_type = map_type
        self.amp_type = amp_type
        
    
    def get_data(self,path):
        """ Gather the data
        """
        self.dat = ascii.read(path)
        #self.tref = np.round(np.min(self.dat['Time (days)']))
        self.x = np.ascontiguousarray(self.dat['Time (days)'])# - self.tref)
        self.y = np.ascontiguousarray(self.dat['Flux'])
        self.yerr = np.ascontiguousarray(self.dat['Flux err'])
        self.meta = self.dat.meta
    
    def build_model(self):
        if self.mask is None:
            self.mask = np.ones(len(self.x), dtype=bool)
        
        with pm.Model() as model:
            
            A_map = starry.Map(ydeg=0,udeg=2,amp=1.0)
            A = starry.Primary(A_map,m=self.meta['M_star'],
                               r=self.meta['R_star'],
                               prot=self.meta['prot_star'])
            
            b_map = starry.Map(ydeg=3)
            if self.amp_type == 'variable':
                b_map.amp = pm.Normal("amp", mu=1e-3, sd=1e-3)
            elif 'fixedAt' in self.amp_type:
                b_map.amp = float(self.amp_type.split("fixedAt")[1])
            else:
                b_map.amp = 1e-3
            
            ncoeff = b_map.Ny - 1
            sec_mu = np.zeros(ncoeff)
            sec_cov = 0.5**2 * np.eye(ncoeff)
            if self.map_type == 'variable':
                b_map[1:,:] = pm.MvNormal("sec_y",sec_mu,sec_cov,shape=(ncoeff,))
            # sec_mu = np.zeros(b_map.Ny)
            # sec_mu[0] = 1e-3
            # sec_L = np.zeros(b_map.Ny)
            # sec_L[0] = (0.2 * sec_mu[0])**2 ## covariance is squared
            # sec_L[1:] = (0.5 * sec_mu[0])**2
            
            # b_map.set_prior(mu=sec_mu, L=sec_L)
            b = starry.kepler.Secondary(b_map,
                                        m=0.0,
                                        r=self.meta['rp'],
                                        prot=self.meta['Period'],
                                        porb=self.meta['Period'],
                                        t0=self.meta['t0'],
                                        inc=self.meta['inc'])
            b.theta0 = 180.0
            sys = starry.System(A,b)
            
            self.sys = sys
            
            lc_eval = pm.Deterministic('lc_eval',sys.flux(t=self.x[self.mask]))
            resid = self.y[self.mask] - lc_eval
            
            ## estimate the standard deviation
            sigma_lc = pm.Lognormal("sigma_lc", mu=np.log(np.std(self.y[self.mask])), sigma=0.5)
            
            ## estimate GP error as std
            sigma_gp = pm.Lognormal("sigma_gp", mu=np.log(np.std(self.y[self.mask]) * 1.0), sigma=0.5)
            ## Estimate GP error near the photon error
            #sigma_gp = pm.Lognormal("sigma_gp", mu=np.log(15e-6), sigma=0.5)
            rho_gp = pm.Lognormal("rho_gp", mu=np.log(2.5), sigma=0.5)
            #tau_gp = pm.Lognormal("tau_gp",mu=np.log(5e-2), sigma=0.5)
            #kernel = terms.SHOTerm(sigma=sigma_gp, rho=rho_gp, tau=tau_gp)
            
            ## non-periodic
            kernel = terms.SHOTerm(sigma=sigma_gp, rho=rho_gp, Q=0.25)
            
            gp = GaussianProcess(kernel, t=self.x[self.mask], yerr=sigma_lc,quiet=True)
            gp.marginal("gp", observed=resid)
            #gp_pred = pm.Deterministic("gp_pred", gp.predict(resid))
            final_lc = pm.Deterministic("final_lc",lc_eval + gp.predict(resid))
            
            # Save our initial guess w/ just the astrophysical component
            self.lc_guess_astroph = pmx.eval_in_model(lc_eval)
            self.lc_guess = pmx.eval_in_model(final_lc)
            self.resid_guess = pmx.eval_in_model(resid)
        self.model = model
    
    def plot_lc(self,point='guess'):
        if point == "guess":
            if hasattr(self,'lc_guess') == False:
                self.build_model()
            #f =  self.sys.flux(self.x).eval()
            f = self.lc_guess
            f_astroph = self.lc_guess_astroph
        elif point == "mxap":
            self.find_mxap()
            f = self.mxap_soln['final_lc']
            f_astroph = self.mxap_soln['lc_eval'] 
        else:
            raise Exception("Un-recognized test point {}".format(point))
        
        fig, (ax,ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
        ax.plot(self.x,self.y,'.',label='data')
        ax.plot(self.x,f,label='GP model')
        ax.plot(self.x,f_astroph,label='Astrophysical Component')
        resid = self.y - f
        ax.legend()
        
        ax2.errorbar(self.x,resid * 1e6,self.yerr,fmt='.')
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Resid (ppm)")
        fig.savefig('plots/lc_{}_{}.pdf'.format(self.descrip,point))
        plt.close(fig)
        
    def find_mxap(self,recalculate=False):
        """
        Find the Maximum a Priori solution
        If it has already been found, it is read from a file
        
        Parameters
        -----------
        recalculate: bool
            If true, re-caclulate the solution, even one exists. If False, and
            a previous one is found, it reads the previous solution.
        
        """
        if (os.path.exists(self.mxap_path) == True) & (recalculate == False):
            self.read_mxap()
        else:
            if hasattr(self,'model') == False:
                self.build_model()
        
            with self.model:
                self.mxap_soln =pmx.optimize()
            
            np.savez(self.mxap_path,**self.mxap_soln)
        
    def read_mxap(self):
        npzfiles  = np.load(self.mxap_path)#,allow_pickle=True)
        self.mxap_soln = dict(npzfiles)


def check_lognorm_prior(variable='rho'):
    
    if variable == "sigma_gp":
        x = np.linspace(-12, -1, 500)
        sb = starry_basemodel()
        ref_val = 15e-6#np.std(sb.y[sb.mask])
        mu=np.log(np.std(sb.y[sb.mask])) * 1.0
        sigma = 0.5
        #mu = np.log(ref_val)
        
        xLabel = r"$\sigma_{GP}$"
    elif variable == "rho":
        x = np.linspace(-5, 10, 500)
        xLabel = r"Log($\rho $)"
        mu=np.log(2.5)
        sigma = 0.5
        ref_val = 0.3
    else:
        raise Exception("Unrecognized variable {}".format(variable))
    
    logp = pm.Lognormal.dist(mu=mu, sigma=sigma).logp(np.exp(x)).eval()
    fig, ax = plt.subplots()
    ax.plot(x,np.exp(logp))
    ax.set_xlabel(xLabel)
    ax.set_ylabel("Probability")
    ax.axvline(np.log(ref_val),color="green")
    fig.savefig('plots/prior_checks/{}_check_001.pdf'.format(variable))
    
    
    
    