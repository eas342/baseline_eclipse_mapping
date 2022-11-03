import utils
from astropy.io import fits, ascii
import os
import numpy as np


def set_up_sb_obj(map_type='variable',use_gp=True,
                  lc='F444W'):
    """ Set up a Starry Object """
    
    
    if lc=='F444W':
        dataPath='real_data/lc_hd189_f444w_005_with_variable_incl.ecsv'
        ampGuess=1.8e-3
        descrip = 'hd189F444WrealLC05varIncl'
        hotspotGuess_param = {'xstart':40,'xend':95,
                    'ystart':30,'yend':70,
                    'guess_x':30,'guess_y':0}
        cores=1
    elif lc=='F444W_old':
        dataPath='real_data/lc_hd189_f444w_004_renorm_BJDTDB.ecsv'
        ampGuess=1.8e-3
        descrip = 'hd189F444WrealLC04'
        hotspotGuess_param = {'xstart':40,'xend':95,
                    'ystart':30,'yend':70,
                    'guess_x':30,'guess_y':0}
        cores=2
    elif lc == 'F322W2':
        dataPath='real_data/lc_hd189_f322ww_001_BJDTDB.ecsv'
        ampGuess=1.1e-3
        descrip = 'hd189F322W2realLC01'
        hotspotGuess_param = {'xstart':40,'xend':95,
                    'ystart':10,'yend':70,
                    'guess_x':30,'guess_y':0}
        cores=2
    else:
        raise Exception("Unrecognized lc {}".format(lc))
    
    if use_gp == False:
        descrip = descrip + '_noGP'
    
    sb = utils.starry_basemodel(dataPath=dataPath,
                                descrip=descrip,
                                map_type=map_type,amp_type='variable',
                                systematics='Real',use_gp=use_gp,degree=2,
                                map_prior='physicalVisible',
                                hotspotGuess_param=hotspotGuess_param,
                                ampGuess=ampGuess,
                                cores=cores)
    return sb

def set_up_f322w2():
    sb = set_up_sb_obj(lc='F322W2')
    return sb