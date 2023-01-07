import utils
from astropy.io import fits, ascii
import os
import numpy as np


def set_up_sb_obj(map_type='variable',use_gp=True,
                  lc='F444W',overrideDegree=None,
                  map_prior='physicalVisible'):
    """ Set up a Starry Object """
    
    broadHotspotGuess_param = {'xstart':10,'xend':95,
                              'ystart':10,'yend':90,
                              'guess_x':30,'guess_y':0}
    ampPrior = (1.5e-3,0.4e-3)

    if lc == 'F444W':
        dataPath='lc_hd189_wPCA_v008_trunc_refpix_A10orb_1min_cad_F444W.ecsv'
        ampGuess=1.8e-3
        descrip = 'PCAc8refpx_hd189F444WrealLC_trunc'
        hotspotGuess_param = broadHotspotGuess_param
        cores=2
        degree=1        
    elif lc == 'F444W_old4':
        dataPath='real_data/lc_hd189_wPCA_v006_refpix_A10orb_1min_cad_F444W.ecsv'
        ampGuess=1.8e-3
        descrip = 'PCAc6refpx_hd189F444WrealLC'
        hotspotGuess_param = broadHotspotGuess_param
        cores=2
        degree=1
    elif lc == 'F444W_ing_egr':
        dataPath='real_data/lc_hd189_wPCA_v007_refpix_ing_egr_only_A10orb_1min_cad_F444W.ecsv'
        ampGuess=1.8e-3
        descrip = 'PCAc7refpx_ingegr_hd189F444WrealLC'
        hotspotGuess_param = broadHotspotGuess_param
        cores=2
        degree=1
    elif lc=='F444W_old3':
        #dataPath='real_data/lc_hd189_wPCA_v003_agol2010orb_1min_cad_F444W.ecsv'
        dataPath='real_data/lc_hd189_wPCA_v004_A10orb_1min_cad_F444W.ecsv'
        ampGuess=1.8e-3
        descrip = 'PCAc4_hd189F444WrealLC'
        hotspotGuess_param = broadHotspotGuess_param
        cores=2
        degree=1
    elif lc=='F444W_old2':
        dataPath='real_data/lc_hd189_f444w_005_with_variable_incl.ecsv'
        ampGuess=1.8e-3
        descrip = 'hd189F444WrealLC05varIncl'
        hotspotGuess_param = {'xstart':40,'xend':95,
                    'ystart':30,'yend':70,
                    'guess_x':30,'guess_y':0}
        cores=1
        degree=2
    elif lc=='F444W_old':
        dataPath='real_data/lc_hd189_f444w_004_renorm_BJDTDB.ecsv'
        ampGuess=1.8e-3
        descrip = 'hd189F444WrealLC04'
        hotspotGuess_param = {'xstart':40,'xend':95,
                    'ystart':30,'yend':70,
                    'guess_x':30,'guess_y':0}
        cores=2
        degree=2
    elif lc == 'F322W2':
        dataPath='real_data/lc_hd189_f322ww_002_variable_incl.ecsv'
        ampGuess=1.1e-3
        descrip = 'hd189F322W2realLC03varIncldeg1'
        hotspotGuess_param = {'xstart':20,'xend':95,
                    'ystart':10,'yend':90,
                    'guess_x':50,'guess_y':0}
        cores=1
        degree=1
    elif lc == 'F322W2deg2':
        dataPath='real_data/lc_hd189_f322ww_002_variable_incl.ecsv'
        ampGuess=1.1e-3
        descrip = 'hd189F322W2realLC02varIncl'
        hotspotGuess_param = {'xstart':40,'xend':95,
                    'ystart':10,'yend':70,
                    'guess_x':30,'guess_y':0}
        cores=1
        degree=2
    elif lc == 'PCAc_F322W2':
        #dataPath='real_data/lc_hd189_wPCA_v001_F322W2.ecsv'
        dataPath='real_data/lc_hd189_wPCA_v002_agol2010orb_F322W2.ecsv'
        ampGuess=1.35e-3
        #descrip = 'hd189_wPCA01_F322W2real'
        descrip = 'hd189_wPCA02_F322W2real'
        hotspotGuess_param = broadHotspotGuess_param
        cores=2
        degree=2
    elif lc == 'PCAc_F444W_inCO2':
        #dataPath='real_data/lc_hd189_wPCA_v001_F444W_inCO2.ecsv'
        dataPath='real_data/lc_hd189_wPCA_v002_agol2010orb_F444W_inCO2.ecsv'
        ampGuess=1.4e-3
        ampPrior
        #descrip = 'hd189_wPCA0_F444Wreal_inCO2'
        descrip = 'hd189_wPCA02_F444Wreal_inCO2'
        hotspotGuess_param = broadHotspotGuess_param
        cores=2
        degree=2
    elif lc == 'PCAc_F444W_outCO2':
        #dataPath='real_data/lc_hd189_wPCA_v001_F444W_outCO2.ecsv'
        dataPath='real_data/lc_hd189_wPCA_v002_agol2010orb_F444W_outCO2.ecsv'
        ampGuess=1.6e-3
        #ampGuess=1.1e-3 ## to make sure it optimizes
        #descrip = 'hd189_wPCA01_F444Wreal_outCO2'
        descrip = 'hd189_wPCA02_F444Wreal_outCO2'
        hotspotGuess_param = broadHotspotGuess_param
        cores=2
        degree=2
    else:
        raise Exception("Unrecognized lc {}".format(lc))
    
    if use_gp == False:
        descrip = descrip + '_noGP'
    
    if map_prior == 'physicalVisible':
        pass
    else:
        descrip = descrip + '_pxSamp'

    if overrideDegree is None:
        degree=degree
    else:
        degree=overrideDegree
        descrip = descrip + '_ovrdeg{}'.format(overrideDegree)

    sb = utils.starry_basemodel(dataPath=dataPath,
                                descrip=descrip,
                                map_type=map_type,amp_type='variable',
                                systematics='Real',use_gp=use_gp,degree=degree,
                                map_prior=map_prior,
                                hotspotGuess_param=hotspotGuess_param,
                                ampGuess=ampGuess,
                                ampPrior=ampPrior,
                                cores=cores,
                                nuts_init='auto',
                                t_subtracted=True)
    return sb

def run_pca_round02(use_gp=True,overrideDegree=None,
                    map_type='variable',map_prior='physicalVisible'):
    lcList = ['PCAc_F444W_outCO2','PCAc_F444W_inCO2','PCAc_F322W2']
    sbList = []
    for oneLCName in lcList:
        sbr = set_up_sb_obj(lc=oneLCName,use_gp=use_gp,overrideDegree=overrideDegree)
        sbr.run_all()
        sbList.append(sbr)
    utils.compare_histos(sbList[0],sbList[1],dataDescrips=['Out of CO2','In CO2'])

def run_pca_round02_simplest():
    run_pca_round02(use_gp=False,overrideDegree=1)


def set_up_f322w2():
    sb = set_up_sb_obj(lc='F322W2')
    return sb