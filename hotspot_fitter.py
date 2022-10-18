import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from astropy.table import Table
import os
from astropy.modeling import models, fitting
import warnings
import pdb

class hotspot_fitter(object):
    
    def __init__(self,map2D,lon,lat,err2D=None,
                 xstart=50,xend=80,ystart=50,yend=99,
                 guess_x=50,guess_y=40):
        """
        Class for fitting a hotspot

        Parameters
        -----------
        map2D: numpy array
            The 2D full map
        lon: numpy array
            The 2D full longitude coordinate map
        lat: numpy aray
            The 2D full latitude coordinate map
        err2D: numpy array
            The 2D full error map
        xstart: int or float
            Minimum x index coordinate to fit (unitless)
        xend: int or float
            Maximum x index coordinate to fit (unitless)
        ystart: int or float
            Minimum y index coordinate to fit (unitless)
        yend: int or float
            Maximum y index coordinate to fit (unitless)
        guess_x: float
            Guess longitude (in degrees)
        guess_y: float
            Guess latitude (in degrees)
        """
        self.p_init = models.Gaussian2D(amplitude=0.7,
                                        x_mean=guess_x,
                                        y_mean=guess_y,
                                        x_stddev=30,y_stddev=30)
        
        self.p_init.amplitude.min = 0 ## don't fit a cold spot
        self.p_init.x_mean.bounds = (lon[0,xstart],lon[0,xend])
        self.p_init.y_mean.bounds = (lat[ystart,0],lat[yend,0])
        self.p_init.x_stddev.bounds = (1,(lon[0,xend] - lon[0,xstart]) * 2)
        self.p_init.y_stddev.bounds = (1,(lat[yend,0] - lat[ystart,0]) * 2)
        
        self.fit_p = fitting.LevMarLSQFitter()
        
        self.lat = lat
        self.lon = lon
        self.map2D = map2D
        self.err2D = err2D
        
        self.ystart, self.yend = ystart, yend
        self.xstart, self.xend = xstart, xend
    
    def fit_model(self):
        """
        Fit the map
        """
        with warnings.catch_warnings():
            # Ignore model linearity warning from the fitter
            warnings.simplefilter('ignore')
            
            ystart, yend = self.ystart, self.yend
            xstart, xend = self.xstart, self.xend
            
            if self.err2D is None:
                weights = None
                calc_uncertainties = False
            else:
                weights = 1./self.err2D[ystart:yend,xstart:xend]
                calc_uncertainties = True
            
            self.p_fit = self.fit_p(self.p_init,
                                    self.lon[ystart:yend,xstart:xend],
                                    self.lat[ystart:yend,xstart:xend],
                                    self.map2D[ystart:yend,xstart:xend],
                                    weights=weights)
            
            if self.err2D is not None:
                nparam = len(self.p_fit.param_names)
                npoints = self.map2D.size
                model2D = self.p_fit(self.lon,self.lat)
                resid2D = self.map2D - model2D
                
                red_chisq = ((resid2D/self.err2D)**2).sum()/(npoints - nparam)
                
                self.red_chisq = red_chisq
            else:
                self.red_chisq = 1.0
        
    def check_for_fit(self):
        """
        Check for fit
        
        """
        if hasattr(self,"p_fit") == False:
            self.fit_model()
    
    
    def plot_fits(self):
        """
        Plot the guess, data and best fit
        """
        self.check_for_fit()
        
        guess2D = self.p_init(self.lon,self.lat)
        model2D = self.p_fit(self.lon,self.lat)
        
        ystart, yend = self.ystart, self.yend
        xstart, xend = self.xstart, self.xend
        
        fig, axArr = plt.subplots(1,3,sharey=True,sharex=True)
        mapList = [guess2D,self.map2D,model2D]
        labelList = ['guess','data','fit']
        for ind,oneMap in enumerate(mapList):
            ax = axArr[ind]
            ax.imshow(oneMap,origin='lower')
            ax.set_title(labelList[ind])
            ax.plot([xstart,xend,xend,xstart,xstart],
                    [ystart,ystart,yend,yend,ystart],color='red')
            
            
        #plt.imshow(p_init(lat,lon))
    
    def return_peak(self):
        self.check_for_fit()
        lat_fit = self.p_fit.y_mean.value
        lon_fit = self.p_fit.x_mean.value
        
        return lon_fit, lat_fit
    
    def return_peak_cov(self):
        """
        Return the estimate for the covariance matrix of the fit
        
        Returns
        --------
        cov_matrix: 2D numpy array
            Covariance matrix of the positions in the format
            [var(x), cov(x,y)]
            [var(x,y), var(y)]
        """
        self.check_for_fit()
        
        ## make sure the indices are right
        assert self.p_fit.param_names[1] == 'x_mean'
        assert self.p_fit.param_names[2] == 'y_mean'
        
        cov_pos = self.fit_p.fit_info['param_cov'][1:2+1,1:2+1]
        
        
        return cov_pos * self.red_chisq
    
    def get_projected_hostpot(self):
        """
        Get the projected values onto a unit sphere
        """
        self.check_for_fit()
        latdeg = self.p_fit.y_mean.value
        londeg = self.p_fit.x_mean.value
        
        x_proj, y_proj = find_unit_circ_projection(londeg,latdeg)
        return x_proj, y_proj


def find_unit_circ_projection(londeg,latdeg):
    """
    Find the projection of lat/lon onto a unit circle
    
    Parameters
    ----------
    londeg: float or numpy array
        Longitude in degrees
    latdeg: float or numpy array
        Latitude in degrees
    
    Returns
    --------
    x_proj: float or numpy array
        Projected X
    y_proj: float or numpy array
        Projected Y
    """
    
    theta = latdeg * np.pi/180.
    phi = londeg * np.pi / 180.
    x_proj = np.sin(phi) * np.cos(theta)
    y_proj = np.sin(theta)
    return x_proj, y_proj
    