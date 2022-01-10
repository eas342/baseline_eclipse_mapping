import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from astropy.table import Table
import os
from astropy.modeling import models, fitting
import warnings


class hotspot_fitter(object):
    
    def __init__(self,map2D,lon,lat,
                 xstart=50,xend=80,ystart=50,yend=90):
        """
        Class for fitting a hotspot
        """
        self.p_init = models.Gaussian2D(amplitude=0.7,x_mean=50,y_mean=40,
                                        x_stddev=30,y_stddev=30)
        
        self.fit_p = fitting.LevMarLSQFitter()
        
        self.lat = lat
        self.lon = lon
        self.map2D = map2D
        
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
            
            self.p_fit = self.fit_p(self.p_init,
                                    self.lon[ystart:yend,xstart:xend],
                                    self.lat[ystart:yend,xstart:xend],
                                    self.map2D[ystart:yend,xstart:xend])
        
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
        
        fig, axArr = plt.subplots(1,3,sharey=True)
        mapList = [guess2D,self.map2D,model2D]
        labelList = ['guess','data','fit']
        for ind,oneMap in enumerate(mapList):
            ax = axArr[ind]
            ax.imshow(oneMap,origin='lower')
            ax.set_title(labelList[ind])
        
            
        #plt.imshow(p_init(lat,lon))
    
    def return_peak(self):
        self.check_for_fit()
        lon_fit = self.p_fit.y_mean.value
        lat_fit = self.p_fit.x_mean.value
        
        return lon_fit, lat_fit
    
    def get_projected_hostpot(self):
        """
        Get the projected values onto a unit sphere
        """
        self.check_for_fit()
        
        theta = self.p_fit.y_mean.value * np.pi/180.
        phi = self.p_fit.x_mean.value * np.pi / 180.
        x_proj = np.sin(phi) * np.cos(theta)
        y_proj = np.sin(theta)
        return y_proj, x_proj
    