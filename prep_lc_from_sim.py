from astropy.io import fits, ascii
from astropy.table import Table
import batman
import numpy as np
import matplotlib.pyplot as plt
from tshirt.pipeline import phot_pipeline, spec_pipeline
import os
import pdb
from scipy.optimize import curve_fit
from scipy import stats

tshirt_dir = spec_pipeline.baseDir

params = batman.TransitParams()
params.t0 = 10424./(3600. * 24.)     #time of inferior conjunction
params.per = 191684.6208 / (3600. * 24.) #orbital period
params.rp = 0.1504                   #planet radius (in units of stellar radii)
params.a = 8.73                      #semi-major axis (in units of stellar radii)
params.inc = 85.690                  #orbital inclination (in degrees)
params.ecc = 0.                      #eccentricity
params.w = 90.                       #longitude of periastron (in degrees)
params.u = [0.295,0.033,-0.079,0.034]  #limb darkening coefficients [u1, u2]
params.limb_dark = "nonlinear"       #limb darkening model

#def b_model(x,rp, inc, a):
#def b_model(x,rp,A,stretch1, stretch2,stretch3, stretch4):
def b_model(x, rp, A, B):
    params.rp = rp
    #params.u = [u1, u2, u3, u4]
    #params.inc = inc
    #params.a = a
    m = batman.TransitModel(params, x)    #initializes model
    m_flux = m.light_curve(params)          #calculates light curve
    
    
    ## I think Mirage has a bug with the in-transit versus out-of-transit grism dispersion calculation
    oot_pts = (x > 0.15903) | (x < 0.08220)
    in_transit_pts = oot_pts == False
    
    out_model = np.zeros_like(x)
    out_model[oot_pts] = B
    out_model[in_transit_pts] = m_flux[in_transit_pts] * A
    
    return out_model
    #dip = (m_flux - 1.0)
    #return A + stretch1 * dip + stretch2 * dip**2 + stretch3 * dip**3 + stretch4 * dip**4
    #return m_flux * A

def fit_lc_from_sim():
    param_path = os.path.join('parameters','spec_params','jwst','sim_mirage_029_hd189733b_transit')
    full_param_path = os.path.join(tshirt_dir,param_path,"spec_mirage_029_hd189_transit_p001.yaml")
    spec = spec_pipeline.spec(full_param_path)
    
    #spec.plot_wavebin_series(showPlot=True,nbins=1)
    
    t1, t2 = spec.get_wavebin_series(nbins=1)
    
    trel = t1['Time'] - np.min(t1['Time'])
    y = t1[t1.colnames[1]]
    
    fig, (ax,ax2) = plt.subplots(2,1,sharex=True)
    ax.plot(trel,y,'.')
    
    mask = np.ones_like(trel,dtype=bool)
    #p0 = [0.15, 0.295, 0.033, -0.079, 0.034]
    #p0 = [0.15, 85.69]
    #p0 = [0.15, 85.69,8.73]
    
    #p0 = [0.15,1.0, 1.0, 0.0, 0.0,0.0]
    #p0 = [0.15, 1.0]
    p0 = [0.15, 1.0, 1.0]
    
    for i in range(2):
        popt, pcov = curve_fit(b_model,trel[mask],y[mask],p0=p0)
        #popt = p0
        
        m_flux = b_model(trel,*popt)
        resid = y - m_flux
        
        mad = np.median(np.abs(resid))
        mask = np.abs(resid) < 10. * mad
        
        nbad = np.sum(mask == False)
        p0 = popt
        #pdb.set_trace()
    
    
    print('Rp/R* = {} +/- {} in fit'.format(popt[0],np.sqrt(pcov[0,0])))
    
    ax.plot(trel,m_flux)
    
    ax.set_ylabel("Normalized Flux")
    
    
    ax2.plot(trel[mask],resid[mask] * 1000.,'.')
    ax2.set_ylabel("Resid (ppt)")
    
    ax2.set_xlabel("Time (days from start)")
    
    print("Optimal params= {}".format(p0))
    
    fig.savefig('plots/specific_saves/lc_fit_from_mirage_sim.pdf')
    
    ## Save the residual data
    outDat = Table()
    outDat['Time (d)'] = trel[mask]
    outDat['Resid (ppt)'] = resid[mask] * 1000.
    
    outDat.meta['Planet'] = "HD 189733 b"
    outDat.meta['Filter'] = "F444W"
    
    yerr = t2[t2.colnames[1]]
    outDat['Theo Err (ppt)'] = yerr[mask] * 1000.
    outDat.write('sim_data/resids_from_mirage_sim.ecsv',format='ascii.ecsv',overwrite=True)
    
    plt.close(fig)

def gaussian(x,mu,sigma):
    coeff = 1./(sigma * np.sqrt(2. * np.pi)) 
    return coeff * np.exp(-0.5 * (x - mu)**2/sigma**2)

def examine_resid():
    residDat = ascii.read('sim_data/resids_from_mirage_sim.ecsv')
    resid = residDat['Resid (ppt)']
    
    x_mod = np.linspace(-1.,1.,1024)
    dx = np.ones_like(x_mod) * np.median(np.diff(x_mod))
    
    #sigma_est =np.median(residDat['Theo Err (ppt)'])
    sigma_est = np.std(resid)
    
    theo_gauss = gaussian(x_mod,mu=0.0,sigma=sigma_est)# * dx
    
    fig, ax = plt.subplots()
    ax.hist(resid,bins=50,normed=True)
    ax.plot(x_mod,theo_gauss)
    
    ax.set_yscale('log')
    ax.set_ylim(1e-3,5)
    fig.show()
    
    stat, critv, siglevel = stats.anderson(resid)
    print('A$^2$={:.3f}'.format(stat))
    
    print("Stdev = {} ppt".format(np.std(resid)))
    print("Cadence = {} sec".format(np.median(np.diff(residDat['Time (d)'])) * 24. * 3600.))