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

    fix_Y10 = None

    if lc == 'F444W':
        dataPath = 'real_data/lc_hd189_wPCA_v009_refpix_FPAH_A10orb_1min_cad_F444W.ecsv'
        ampGuess=1.8e-3
        descrip = 'PCAc9refpix_hd189F444WreaLC_fixY10'
        hotspotGuess_param = broadHotspotGuess_param
        cores=2
        degree=1
        fix_Y10 = 0.5
    elif lc == 'F444W_old5':
        dataPath='real_data/lc_hd189_wPCA_v008_trunc_refpix_A10orb_1min_cad_F444W.ecsv'
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
        #dataPath='real_data/lc_hd189_wPCA_v002_agol2010orb_F322W2.ecsv'
        dataPath='real_data/lc_hd189_wPCA_v009_refpix_FPAH_A10orb_1min_cad_F322W2.ecsv'
        ampGuess=1.35e-3
        #descrip = 'hd189_wPCA01_F322W2real'
        #descrip = 'hd189_wPCA02_F322W2real'
        descrip = 'hd189_wPCA09_F322W2real'
        hotspotGuess_param = broadHotspotGuess_param
        fix_Y10 = 0.735 ## 1- min/max Spitzer 3.6 um flux
        cores=2
        degree=1
    elif lc == 'PCAc_F444W_inCO2':
        #dataPath='real_data/lc_hd189_wPCA_v001_F444W_inCO2.ecsv'
        #dataPath='real_data/lc_hd189_wPCA_v002_agol2010orb_F444W_inCO2.ecsv'
        dataPath='real_data/lc_hd189_wPCA_v009_refpix_FPAH_A10orb_1min_cad_F444W_inCO2.ecsv'
        ampGuess=1.4e-3
        ampPrior
        #descrip = 'hd189_wPCA0_F444Wreal_inCO2'
        #descrip = 'hd189_wPCA02_F444Wreal_inCO2'
        descrip = 'hd189_wPCA09_F444Wreal_inCO2'
        hotspotGuess_param = broadHotspotGuess_param
        fix_Y10 = 0.379 ## 1 - min/max Spitzer at 4.5 um, but only approx for in CO2
        cores=2
        degree=1
    elif lc == 'PCAc_F444W_outCO2':
        #dataPath='real_data/lc_hd189_wPCA_v001_F444W_outCO2.ecsv'
        #dataPath='real_data/lc_hd189_wPCA_v002_agol2010orb_F444W_outCO2.ecsv'
        dataPath='real_data/lc_hd189_wPCA_v009_refpix_FPAH_A10orb_1min_cad_F444W_outCO2.ecsv'
        ampGuess=1.6e-3
        #ampGuess=1.1e-3 ## to make sure it optimizes
        #descrip = 'hd189_wPCA01_F444Wreal_outCO2'
        descrip = 'hd189_wPCA09_F444Wreal_outCO2'
        hotspotGuess_param = broadHotspotGuess_param
        fix_Y10 = 0.379  ## 1 - min/max Spitzer at 4.5 um, but only approx for in CO2
        cores=2
        degree=1
    # elif lc == 'PCAc_F444W_outCO2':
    #     #dataPath='real_data/lc_hd189_wPCA_v001_F444W_outCO2.ecsv'
    #     #dataPath='real_data/lc_hd189_wPCA_v002_agol2010orb_F444W_outCO2.ecsv'
    #     dataPath='real_data/lc_hd189_wPCA_v009_refpix_FPAH_A10orb_1min_cad_F444W_outCO2.ecsv'
    #     ampGuess=1.6e-3
    #     #ampGuess=1.1e-3 ## to make sure it optimizes
    #     #descrip = 'hd189_wPCA01_F444Wreal_outCO2'
    #     descrip = 'hd189_wPCA09_F444Wreal_outCO2'
    #     hotspotGuess_param = broadHotspotGuess_param
    #     fix_Y10 = 0.379  ## 1 - min/max Spitzer at 4.5 um, but only approx for in CO2
    #     cores=2
    #     degree=1
 
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
                                t_subtracted=True,
                                fix_Y10=fix_Y10)
    return sb

def run_pca_round02(use_gp=True,overrideDegree=None,
                    map_type='variable',map_prior='physicalVisible'):
    lcList = ['PCAc_F444W_outCO2','PCAc_F444W_inCO2','PCAc_F322W2']
    sbList = []
    for oneLCName in lcList:
        sbr = set_up_sb_obj(lc=oneLCName,use_gp=use_gp,overrideDegree=overrideDegree)
        sbr.run_all(super_giant_corner=True)
        sbList.append(sbr)
        statDict2 = sbr.get_random_draws(calcStats=True,n_draws=40,projection='ortho',
                                         res=1024,save_hotspot_stats=False)
        sbr.plot_map_statistics(statDict=statDict2,projection='ortho',saveFitsFile=True)
    
    utils.compare_histos(sbList[0],sbList[1],dataDescrips=['Out of CO2','In CO2'])
    utils.compare_histos(sbList[0],sbList[2],dataDescrips=['Out of CO2 F444W','F322W2'])
    

def run_pca_round02_simplest():
    run_pca_round02(use_gp=False,overrideDegree=1)


def set_up_f322w2():
    sb = set_up_sb_obj(lc='F322W2')
    return sb

def test_if_nonuniform_is_significant():
    for mapType in ['variable','fixed']:
        sb = set_up_sb_obj(lc='PCAc_F444W_outCO2',map_type=mapType)
        sb.run_all(super_giant_corner=True)
    

def set_up_wasp69b(map_type='variable',
                   degree=1,map_prior='physicalVisible',
                   ephem='kexVar',
                   fix_Y1m1=None,
                   poly_baseline=None,
                   exp_trend=None,
                   light_delay=True):
    """
    Decoder
    V           variable map (F for fixed)
     Z          Sph harmonic degree (Z for 1, else #)
      M         Kokori et al. exoclock ephemeris w/
                propagating all orb variable uncertainties
                K for Kokori et al. ephemeris
                Z for Ivshina & WInnn
       Z        Free Y(1,-1). F for fixed
        1       Polynomial baseline. Z for GP
         E      Exponential trend
          _T     travel delay
    """

    if ephem == 'IW':
        ## Ivshina & Winn ephemeris, fixed
        dataPath = 'real_data/WASP69b_BB_MIRI_lc.ecsv'
        bit2 = 'Z'
    elif ephem == 'kex':
        dataPath = 'real_data/WASP69b_BB_MIRI_lc_002_new_ephem.ecsv'
        bit2 = 'K'
    elif ephem == 'kexVar':
        ## variable stellar mass and radius for additiona propagation
        dataPath = 'real_data/WASP69b_BB_MIRI_lc_003_new_ephem_var_MRs.ecsv'
        bit2 = 'M'
    else:
        raise Exception("Unrecognized ephemeris {}".format(ephem))
    
    if map_type == 'fixed':
        bit0 = 'F' ## fixed
    else:
        bit0 = 'V' ## variable
    if degree==1:
        bit1 = 'Z' ## degree 1
    else:
        bit1 = '{}'.format(degree)

    if fix_Y1m1 is None:
        bit3 = 'Z' ## free
    else:
        bit3 = 'F' ## fixed
    
    if poly_baseline is None:
        bit4 = 'Z' ## unused
        use_gp=True
    else:
        bit4 = '{}'.format(poly_baseline)
        use_gp=False

    if exp_trend == True:
        bit5 = 'E'
    else:
        bit5 = ''
    
    if light_delay == True:
        bit6 = '_T' ## new bit now that light travel delay is on
    else:
        bit6 = ''
    
    descrip = 'WASP69b_MIRI_BB_{}{}{}{}{}{}{}'.format(bit0,
                                               bit1,
                                               bit2,
                                               bit3,
                                               bit4,
                                               bit5,
                                               bit6)
    hotspotGuess_param = {'xstart':30,'xend':70,
            'ystart':15,'yend':85,
            'guess_x':0,'guess_y':0}
    ampGuess = 1.1e-3
    ampPrior = (1.1e-3,0.4)
    cores=2

    sb = utils.starry_basemodel(dataPath=dataPath,
                                descrip=descrip,
                                map_type=map_type,amp_type='variable',
                                systematics='Real',use_gp=use_gp,
                                poly_baseline=poly_baseline,
                                exp_trend=exp_trend,
                                degree=degree,
                                map_prior=map_prior,
                                hotspotGuess_param=hotspotGuess_param,
                                ampGuess=ampGuess,
                                ampPrior=ampPrior,
                                cores=cores,
                                nuts_init='auto',
                                t_subtracted=True,
                                fix_Y1m1=fix_Y1m1,
                                light_delay=light_delay)
    return sb

def wasp69_test_if_nonuniform_is_significant(poly_baseline=1,
                                             exp_trend=True,
                                             degree=1):
    for mapType in ['variable','fixed']:
        sb = set_up_wasp69b(map_type=mapType,poly_baseline=poly_baseline,
                            exp_trend=exp_trend,degree=degree)
        sb.run_all(super_giant_corner=True)

def plot_resid_wasp69b(degree=1,binResid=None):
    sb1 = set_up_wasp69b(map_type='fixed',exp_trend=True,poly_baseline=1,degree=degree)
    sb2 = set_up_wasp69b(map_type='variable',exp_trend=True,poly_baseline=1,degree=degree)
    utils.compare_residuals([sb1,sb2],
                            labels=['from Uniform Model',
                                    'Spherical Degree 1 - Uniform Model'],
                            binResid=binResid)
    
def check_hotspot_offset_fixY1m1(degree=1):
    """
    Check the hotspot offset if I fix Y1_-1
    (actually a strong prior at zero b/c I need to update code)
    """
    sb = set_up_wasp69b(map_type='variable',fix_Y1m1=0.0,poly_baseline=1,
                        exp_trend=True,degree=degree,light_delay=False)
    sb.run_all()
