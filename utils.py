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
import corner
import os

starry.config.lazy = True
starry.config.quiet = True

class starry_basemodel():
    def __init__(self,dataPath='sim_data/sim_data_baseline.ecsv',
                 descrip='Orig_006_newrho_smallGP',
                 map_type='variable',amp_type='variable',
                 systematics='Cubic',use_gp=True,
                 degree=3):
        """
        Set up a starry model
        
        """
        
        
        self.mask = None
        self.systematics = systematics
        if self.systematics == 'Cubic':
            baselineType = ''
        else:
            baselineType = self.systematics
        
        self.descrip = "{}_maptype_{}_amp_type_{}{}".format(descrip,map_type,amp_type,baselineType)
        self.mxap_name = 'mxap_{}.npz'.format(self.descrip)
        self.mxap_path = os.path.join('fit_data','mxap_soln',self.mxap_name)
        
        self.map_type = map_type
        self.amp_type = amp_type
        self.degree = degree
        
        self.use_gp = use_gp
        
        self.data_path = dataPath
        self.get_data(path=self.data_path)
    
    def get_data(self,path):
        """ Gather the data
        """
        self.dat = ascii.read(path)
        #self.tref = np.round(np.min(self.dat['Time (days)']))
        self.x = np.ascontiguousarray(self.dat['Time (days)'])# - self.tref)
        
        if self.systematics == 'Cubic':
            self.y = np.ascontiguousarray(self.dat['Flux'])
        elif self.systematics == 'Flat':
            self.y = np.ascontiguousarray(self.dat['Flux before Baseline'])
        else:
            raise Exception("Unrecognized Lightcurve {}".format())
        
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
            
            b_map = starry.Map(ydeg=self.degree)
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
            #sigma_gp = pm.Lognormal("sigma_gp", mu=np.log(np.std(self.y[self.mask]) * 1.0), sigma=0.5)
            ## Estimate GP error near the photon error
            if self.use_gp == True:
                sigma_gp = pm.Lognormal("sigma_gp", mu=np.log(15e-6), sigma=0.5)
                rho_gp = pm.Lognormal("rho_gp", mu=np.log(2.5), sigma=0.5)
                #tau_gp = pm.Lognormal("tau_gp",mu=np.log(5e-2), sigma=0.5)
                #kernel = terms.SHOTerm(sigma=sigma_gp, rho=rho_gp, tau=tau_gp)
            
                ## non-periodic
                kernel = terms.SHOTerm(sigma=sigma_gp, rho=rho_gp, Q=0.25)
            
                gp = GaussianProcess(kernel, t=self.x[self.mask], yerr=sigma_lc,quiet=True)
                gp.marginal("gp", observed=resid)
                #gp_pred = pm.Deterministic("gp_pred", gp.predict(resid))
                final_lc = pm.Deterministic("final_lc",lc_eval + gp.predict(resid))
            else:
                final_lc = pm.Deterministic("final_lc",lc_eval)
                pm.Normal("obs", mu=final_lc, sd=sigma_lc, observed=self.y)
            
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
        fig.savefig('plots/lc_checks/lc_{}_{}.pdf'.format(self.descrip,point))
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

    def find_posterior(self):
        if hasattr(self,'model') == False:
            self.build_model()
        
        if hasattr(self,'mxap_soln') == False:
            self.find_mxap()
        
        outDir = os.path.join('fit_data','traces','trace_{}'.format(self.descrip))
        if os.path.exists(outDir) == False:
            os.mkdir(outDir)
            np.random.seed(42)
            with self.model: 
                trace = pm.sample( 
                    tune=3000, 
                    draws=3000, 
                    start=self.mxap_soln, 
                    cores=2, 
                    chains=2, 
                    init="adapt_full", 
                    target_accept=0.9)
            pm.save_trace(trace, directory =outDir, overwrite = True)
        else:
            with self.model:
                trace = pm.load_trace(directory=outDir)
        
        self.trace = trace
    
    def select_plot_variables(self,sph_harmonics='none',
                              include_GP=False):
        """
        Select a list of variables to plot
        This automatically skips over things like the lightcurve
                              
        sph_harmonics: str
            What to do with spherical harmonics?
                "none" excludes all spherical harmonics
                "all" will choose all spherical harmonics
        include_GP: bool
            Include the Gaussian process parameters?
        """
        if hasattr(self,'mxap_soln') == False:
            self.find_mxap()
        
        gp_vars = ['sigma_gp','rho_gp']
        
        mxap_vars = list(self.mxap_soln.keys())
        ## make a list of variables to keep
        keep_list = []
        for one_var in mxap_vars:
            mxap_val = self.mxap_soln[one_var]
            if one_var == 'sec_y':
                if sph_harmonics == 'none':
                    pass ## skip
                else:
                    keep_list.append(one_var)
            elif one_var[-5:] == 'log__':
                ## Skip variables that have both log and linear versions
                pass
            elif one_var == 'final_lc':
                ## skip the lighcurve deterministic variable
                pass
            elif one_var == 'lc_eval':
                ## skip the lighcurve deterministic variable
                pass
            
            elif (one_var in gp_vars):
                if include_GP == True:
                    keep_list.append(one_var)
                else:
                    pass
            else:
                keep_list.append(one_var)
                
        return keep_list
    
    def get_truths(self,varnames):
        truths = []
        for oneVar in varnames:
            if oneVar == 'amp':
                truths.append(self.meta['Amplitude'])
            elif oneVar == 'sigma_lc':
                truths.append(np.median(self.yerr))
            elif 'sec_y' in oneVar:
                ind = int(oneVar.split('sec_y__')[-1])
                truths.append(self.meta['y_input'][ind])
            else:
                truths.append(None)
        
        return truths
    
    def prep_corner(self):
        
        if hasattr(self,'trace') == False:
            self.find_posterior()
        
        varnames = self.select_plot_variables(sph_harmonics='all')
        
        samples = pm.trace_to_dataframe(self.trace, varnames=varnames)
        
        ## the trace_to_dataframe splits up arrays of variables
        final_varlist = list(samples.keys())
        
        truths = self.get_truths(final_varlist)
        labels = label_converter(final_varlist)
        
        return samples,truths,labels
    
    def plot_corner(self):
        samples, truths,labels = self.prep_corner()
        _ = corner.corner(samples,truths=truths,labels=labels)
        plt.savefig('plots/corner/{}'.format(self.descrip))
        plt.close()
    
    
    def find_design_matrix(self):
        if hasattr(self,'model') == False:
            self.build_model()
        
        with self.model:
            self.A = pmx.eval_in_model(self.sys.design_matrix(self.x))
        
    
def ylm_labels(degree):
    """
    Make a list of the Ylm variables for a given degree
    These are designed to be rendered in LaTeX
    
    The Y0,0 term is skipped because that is always 1.0 b/c the flux is
    Y0,0 * amplitude
    """
    labels = []
    for l in range(1, degree+1):
        for m in range(-l,1 + l):
            oneLabel = r"$Y_{%d,%d}$" % (l, m)
            labels.append(oneLabel)
    return labels

def label_converter(varList):
    """
    Convert the labels from the trace into ones for plotting
    """
    outList = []
    ylm_full = ylm_labels(10)
    if len(ylm_full) < len(varList) - 2:
        raise Exception("Need a bigger degree in labels list")
    
    for oneLabel in varList:
        if 'sec_y' in oneLabel:
            y_ind =  int(oneLabel.split('sec_y__')[-1])
            outLabel = ylm_full[y_ind]
        elif oneLabel == 'sigmal_lc':
            outLabel = "$\sigma_{lc}$"
        else:
            outLabel = oneLabel
        
        outList.append(outLabel)
    return outList
    

def compare_corners(sb1,sb2):
    
    samples1, truths1, labels1 = sb1.prep_corner()
    samples2, truths2, labels2 = sb2.prep_corner()
    fig1 = corner.corner(samples1,truths=truths1,
                        color='green')
    fig2 = corner.corner(samples2,truths=truths2,
                        color='red',fig=fig1,labels=labels2)
    fig1.savefig('plots/corner/comparison_{}_vs_{}.png'.format(sb1.descrip[0:20],sb2.descrip[0:20]))
    plt.close(fig1)

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
    
    
def plot_sph_harm_lc():
    sb = starry_basemodel(dataPath='sim_data/sim_data_baseline.ecsv',
                          descrip='Orig_006_newrho_smallGP',
                          map_type='variable',amp_type='variable',
                          systematics='Cubic',use_gp=False,degree=3)
    sb.find_design_matrix()
    
    titles = ylm_labels(sb.degree)
    titles.insert(0,'amp')
    titles.insert(0,'flat')
    
    nterms = len(titles)
    for ind in range(nterms):
        fig, ax = plt.subplots(figsize=(4,4))
        ax.plot(sb.x,sb.A[:,ind])
        ax.set_xlabel("Time")
        ax.set_ylabel("Flux")
        ax.set_title(titles[ind])
        outPath = os.path.join('plots','sph_harm_lc','lc_{:03d}.pdf'.format(ind),)
        fig.savefig(outPath,bbox_inches='tight')
        plt.close(fig)
        