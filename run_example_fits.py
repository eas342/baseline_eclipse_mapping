import utils
from astropy.io import fits, ascii
import os
import numpy as np

def zero_eclipse_GP(descrip="Orig_006_newrho_smallGP"):
    """
    This shows how you can fit the lightcurve well with a GP
    even though the eclipse depth here is 0.0  !!
    
    """
    sb = utils.starry_basemodel(descrip=descrip,map_type='fixed',amp_type='fixedAt0')
    sb.plot_lc(point='mxap')
    
    sb = utils.starry_basemodel(descrip=descrip,map_type='fixed',amp_type='fixedAt1e-3')
    sb.plot_lc(point='mxap')

def check_flat_vs_curved_baseline(map_type='variable',find_posterior=False,
                                  super_giant_corner=False,lc_name='NC_HD189',
                                  map_prior='physical',degree=3):
    """
    Try fitting a lightcurve with a flat baseline vs curved
    
    map_type: str
        Map type "fixed" fixes it at uniform. "variable" solves for sph harmonics
    find_posterior: bool
        Find the posterior of the distribution? If True, the Bayesian sampler
        is employed to find it. If False, it just fins the max a priori solution
    super_giant_corner: bool
        Save the super giant corner plots?
        These can take a long time, so they can be skipped by changing to False
    lc_name: str
        Name of the lightcurve ('NC_HD189' uses a NIRCam lightcurve for HD 189733)
                                 'orig' is the original lightcurve at high precision
    map_prior: str
        What priors are put on the plot? 'phys' ensures physical (non-negative priors)
        and 'uniformPixels' does pixel sampling to ensure the pixels are non-negative
        'non-physical' allows the spherical harmonics to be postive or negative
    
    degree: int
        The spherical harmonic degree
    """
    if degree == 3:
        degree_descrip = ''
    else:
        degree_descrip = '_deg{}'.format(degree)
    
    if map_prior == 'physical':
        phys_descrip = 'phys'
    elif map_prior == 'uniformPixels':
        phys_descrip = 'pxSamp'
    else:
        phys_descrip = 'nophys'
    
    if lc_name == 'NC_HD189':
        systematics=['Flat','Quadratic']
        systematics_abbrev = ['flat_no_gp','quad']
        
        descrips = []
        for systematics_descrip in systematics_abbrev:
            thisDescrip = '{}_HD189NC{}PMass{}'.format(systematics_descrip,phys_descrip,degree_descrip)
            descrips.append(thisDescrip)
        
        dataPath='sim_data/sim_data_baseline_hd189_ncF444W.ecsv'
    elif lc_name == 'NC_HD189cubic':
        systematics=['Flat','Cubic']
        systematics_abbrev = ['flat_no_gp','cubic']
        
        descrips = []
        for systematics_descrip in systematics_abbrev:
            thisDescrip = '{}_HD189NC{}PMass{}'.format(systematics_descrip,phys_descrip,degree_descrip)
            descrips.append(thisDescrip)
        
        dataPath='sim_data/sim_data_cubic_baseline_hd189_ncF444W.ecsv'
    else:
        systematics=['Flat','Cubic']
        
        descrips_basenames = ['Flat_001_no_gp','Orig_006_newrho_smallGP']
        
        descrips = []
        for descrip_basename in descrips_basenames:
            descrips.append("{}{}".format(descrip_basename,phys_descrip))
        
        dataPath='sim_data/sim_data_baseline.ecsv'
    
    use_gp_list = [False,True]
    
    sb_list = []
    for ind,sys_type in enumerate(systematics):
        sb = utils.starry_basemodel(dataPath=dataPath,
                                    descrip=descrips[ind],
                                    map_type=map_type,amp_type='variable',
                                    systematics=sys_type,degree=degree,
                                    map_prior=map_prior,use_gp=use_gp_list[ind])
        sb.plot_lc(point='mxap')
        if find_posterior == True:
            sb.find_posterior()
            if super_giant_corner == True:
                sb.plot_corner()
        sb.get_random_draws()
        sb.plot_map_statistics()
        sb_list.append(sb)
    
    if super_giant_corner == True:
        utils.compare_corners(sb_list[0],sb_list[1])
    utils.compare_corners(sb_list[0],sb_list[1],
                          sph_harmonics='m=1',
                          include_sigma_lc=False,
                          extra_descrip='m_eq_1_')
    utils.compare_histos(sb_list[0],sb_list[1])
    
def run_inference(lc_name='NC_HD189',map_prior='physical',degree=3):
    check_flat_vs_curved_baseline(map_type='variable',find_posterior=True,
                                  lc_name=lc_name,map_prior=map_prior,
                                  degree=degree)
    

def run_all_inference_tests(map_prior='physical'):
    """
    Run all inference tests
    map
    """
    for lc_name in ['orig','NC_HD189']:
        run_inference(lc_name=lc_name,map_prior=map_prior)

def run_all_inference_px_sampling():
    run_all_inference_tests(map_prior='uniformPixels')

def test_map_plots():
    dataPath='sim_data/sim_data_baseline_hd189_ncF444W.ecsv'
    descrip = 'quad_HD189NCphysPMass'
    sb = utils.starry_basemodel(dataPath=dataPath,
                                descrip=descrip,
                                map_type='variable',amp_type='variable',
                                systematics='Quadratic',degree=3,
                                map_prior='physical',use_gp=True)
    return sb

def sb_px_sampling_obj():
    dataPath='sim_data/sim_data_baseline_hd189_ncF444W.ecsv'
    descrip = 'flat_no_gp_HD189NCpxSampPMass'
    sb = utils.starry_basemodel(dataPath=dataPath,
                                descrip=descrip,
                                map_type='variable',amp_type='variable',
                                systematics='Flat',degree=3,
                                map_prior='uniformPixels',use_gp=False)
    return sb

if __name__ == "__main__":
    run_all_inference_tests()
    
def posterior_ratios():
    compare_idealized = ['Orig_006_newrho_smallGPphys_maptype_variable_amp_type_variable_stats.csv',
                         'Flat_001_no_gpphys_maptype_variable_amp_type_variableFlat_stats.csv']
    compare_HD189 = ['quad_HD189NCphysPMass_maptype_variable_amp_type_variableQuadratic_stats.csv',
                     'flat_no_gp_HD189NCphysPMass_maptype_variable_amp_type_variableFlat_stats.csv']
    compare_pairs = [compare_idealized,compare_HD189]
    compare_planet_names = ['Idealized','HD_189733_b']
    ratio_name = ['systematics over flat ratio']
    
    for onePlanet, onePair in zip(compare_planet_names,compare_pairs):
        dat1 = ascii.read(os.path.join('fit_data/posterior_stats/',onePair[0]))
        dat2 = ascii.read(os.path.join('fit_data/posterior_stats/',onePair[1]))
        
        assert np.array_equal(dat1['Variable'],dat2['Variable'])
        
        ratio_table = dat1[['Variable','Label']]
        ratio_table['68 perc {}'.format(ratio_name)] = dat1['68 perc width'] / dat2['68 perc width']
        ratio_table['95 perc {}'.format(ratio_name)] = dat1['95 perc width'] / dat2['95 perc width']
        outName = "{}_ratios.csv".format(onePlanet)
        outPath = os.path.join('fit_data','posterior_stats',outName)
        
        ratio_table.write(outPath,overwrite=True)