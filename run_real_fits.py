import utils
from astropy.io import fits, ascii
from astropy.table import Table, vstack
import os
import numpy as np
import matplotlib.pyplot as plt
import corner
from copy import deepcopy

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
                   light_delay=True,
                   specificHarmonics=False,
                   fix_Y1m1_truly=False):
    """
    Decoder
    V           variable map (F for fixed)
     Z          Sph harmonic degree (Z for 1, else #)
      M         Kokori et al. exoclock ephemeris w/
                propagating all orb variable uncertainties
                K for Kokori et al. ephemeris
                Z for Ivshina & WInnn
       Z        Free Y(1,-1). F for "fixed", T for truly fixed.
        1       Polynomial baseline. Z for GP
         E      Exponential trend
          _T     travel delay
            -bo boolean mask to variable harmonics  
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

    if fix_Y1m1_truly == True:
        bit3 = 'T'
    elif fix_Y1m1 is None:
        bit3 = 'Z' ## free
    else:
        bit3 = 'F' ## "fixed" by using a tight prior
    
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

    if fix_Y1m1_truly == True:
        var_harmonic_mask = [True] * ((degree+1)**2 -1)
        var_harmonic_mask[0] = False
        bit7 = ''
    elif specificHarmonics == True:
        assert (degree==2), "degree must be 2 for this mask"
        bit7 = '-bo'
        var_harmonic_mask = [False, True, True, False, False, True, True, False]

        # '$Y_{1,-1}$',  False
        # '$Y_{1,0}$',  True
        # '$Y_{1,1}$',  True
        # '$Y_{2,-2}$',  False
        # '$Y_{2,-1}$', False
        # '$Y_{2,0}$',   True
        # '$Y_{2,1}$',   True
        # '$Y_{2,2}$']   False
    else:
        var_harmonic_mask = None
        bit7 = ''
 
 

    
    descrip = 'WASP69b_MIRI_BB_{}{}{}{}{}{}{}{}'.format(bit0,
                                               bit1,
                                               bit2,
                                               bit3,
                                               bit4,
                                               bit5,
                                               bit6,
                                               bit7)
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
                                light_delay=light_delay,
                                var_harmonic_mask=var_harmonic_mask)
    return sb

def wasp69_test_if_nonuniform_is_significant(poly_baseline=1,
                                             exp_trend=True,
                                             degree=1,
                                             light_delay=False):
    for mapType in ['variable','fixed']:
        sb = set_up_wasp69b(map_type=mapType,poly_baseline=poly_baseline,
                            exp_trend=exp_trend,degree=degree,
                            light_delay=light_delay)
        sb.run_all(super_giant_corner=True)

def plot_resid_wasp69b(degree=1,binResid=None,
                       light_delay=False):
    sb1 = set_up_wasp69b(map_type='fixed',exp_trend=True,poly_baseline=1,degree=degree,
                         light_delay=light_delay)
    sb2 = set_up_wasp69b(map_type='variable',exp_trend=True,poly_baseline=1,degree=degree,
                         light_delay=light_delay)
    utils.compare_residuals([sb1,sb2],
                            labels=['from Uniform Model',
                                    'Spherical Degree {} - Uniform Model'.format(degree)],
                            binResid=binResid)
    
def check_hotspot_offset_fixY1m1(degree=1):
    """
    Check the hotspot offset if I fix Y1_-1
    (actually a strong prior at zero b/c I need to update code)
    """
    sb = set_up_wasp69b(map_type='variable',fix_Y1m1=0.0,poly_baseline=1,
                        exp_trend=True,degree=degree,light_delay=False)
    sb.run_all()

def check_hotspot_offset_fixY1m1_truly(degree=1):
    """
    Check the hotspot offset if I fix Y1_-1
    Actually fixed the Y(1,-1) to zero
    """
    sb = set_up_wasp69b(map_type='variable',fix_Y1m1_truly=True,poly_baseline=1,
                        exp_trend=True,degree=degree,light_delay=False)
    sb.run_all()

def set_up_specific_harmonics_w69():
    sb = set_up_wasp69b(map_type='variable',poly_baseline=1,
                    exp_trend=True,degree=2,light_delay=False,
                    specificHarmonics=True)
    return sb

def examine_corner_specific_harmonics():
    sb = set_up_specific_harmonics_w69()
    samples, truths, labels = sb.prep_corner(include_sigma_lc=False,include_GP=False)
    cols = samples.columns
    use_labels = [labels[5],labels[9],labels[10],labels[11],labels[12]]
    use_cols = [cols[5],cols[9],cols[10],cols[11],cols[12]]
    plt.close('all')
    #fig, ax = plt.subplots(figsize=(5,5))
    fig = plt.figure(figsize=(7,7))
    _ = corner.corner(samples,labels=use_labels,var_names=use_cols,fig=fig)
    fig.savefig('plots/corner/'+sb.descrip+'_selected.png',
                dpi=150,bbox_inches='tight',facecolor='white')
    #use_samples = [samples[5],samples[9],samples[10],samples[11],samples[12]]
    

def compare_residuals_deg1_deg2_selected_harm():
    sb1 = set_up_wasp69b(map_type='variable',fix_Y1m1_truly=True,poly_baseline=1,
                        exp_trend=True,degree=1,light_delay=False)
    sb2 = set_up_specific_harmonics_w69()
    utils.compare_residuals([sb1,sb2],
                            labels=['Degree=1, selected terms',
                                    'Degree=2 selected - Degree=1 selected'],
                            binResid=None)

def compare_BIC_wasp69b_uniform_deg2():
    """
    Compare the BICs for lightcurve fits of WASP-69 b
    but take into account error inflation
    Use a common error inflation between the two fits
    """
    stats_f_file = ('fit_data/lc_BIC/'+
                    'WASP69b_MIRI_BB_F2MZ1E_maptype_fixed_amp_type_variableReal_lc_BIC.csv')
    stats_v_file = ('fit_data/lc_BIC/'+
                     'WASP69b_MIRI_BB_V2MZ1E_maptype_variable_amp_type_variableReal_lc_BIC.csv')
    stats_out_file = ('fit_data/lc_BIC/'+
                     'WASP69b_MIRI_BB_F2MZ1E_maptype_variable_amp_type_variableReal_BC_vs_variable.csv')
    stats_f = ascii.read(stats_f_file)
    stats_v = ascii.read(stats_v_file)

    ## use a common inflated error between them
    sigma_lc_c = stats_f['sigma_lc'][0]
    stats_f_c = redo_BIC(stats_f,sigma_lc_c)
    stats_v_c = redo_BIC(stats_v,sigma_lc_c)
    stats_common = vstack([stats_f_c,stats_v_c])
    stats_common['map_type'] = ['Fixed','Variable']
    stats_common.write(stats_out_file,overwrite=True)

def redo_BIC(stats,sigma_lc):
    """
    Re-do the BIC with a new sigma
    """
    stats_new = deepcopy(stats)
    old_sigma = stats['sigma_lc'][0]
    stats_new['sigma_lc'] = sigma_lc
    stats_new['chisq sigma_lc']  = stats['chisq sigma_lc'] * (old_sigma/sigma_lc)**2
    npoints = stats['npoints'][0]
    nvar = stats['nvar'][0]
    stats_new['red chisq sigma_lc'] = stats_new['chisq sigma_lc'] / float(npoints - nvar)
    stats_new['BIC sigma_lc'] = stats_new['chisq sigma_lc'] + + nvar * np.log(npoints)
    return stats_new