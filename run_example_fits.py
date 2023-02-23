import utils
from astropy.io import fits, ascii
import os
import numpy as np
import arviz

def zero_eclipse_GP(descrip="Orig_006_newrho_smallGP"):
    """
    This shows how you can fit the lightcurve well with a GP
    even though the eclipse depth here is 0.0  !!
    
    """
    sb = utils.starry_basemodel(descrip=descrip,map_type='fixed',amp_type='fixedAt0')
    sb.plot_lc(point='mxap')
    
    sb = utils.starry_basemodel(descrip=descrip,map_type='fixed',amp_type='fixedAt1e-3')
    sb.plot_lc(point='mxap')


def make_plots_for_no_gp_fit_for_paper():
    for lc_name, degree in zip(['orig','NC_HD189'],[3,2]):
        sb0, sb1 = make_sb_obj_with_no_gp(degree=degree,lc_name=lc_name)
        sb1.run_all()
        sb0_with_gp, sb1_with_gp = make_sb_objects(degree=degree,use_gp_list=[False, True],
                                                   lc_name=lc_name)
        utils.compare_histos(sb1,sb1_with_gp,dataDescrips=['No GP, planet only','GP for systematics'])
    

def make_sb_obj_with_no_gp(map_type='variable',lc_name='NC_HD189',
                           map_prior='physical',degree=2,
                           widerSphHarmonicPriors=True):
    """
    Make a pair of starry basemodel objects with no GP applied to the flat 
    nor the baseline trend model
    """
    
    sb0, sb1 = make_sb_objects(map_type=map_type,lc_name=lc_name,
                               map_prior=map_prior,degree=degree,
                               use_gp_list = [False,False],
                               widerSphHarmonicPriors=widerSphHarmonicPriors)
    
    return sb0, sb1
    

def make_sb_obj_with_physical_visible(map_type='variable',lc_name='NC_HD189',
                                      degree=2):
    """
    Make a pair of starry basemodel objects with a physical (non-negative)
    map but only require that over visible longitudes
    """
    
    sb0, sb1 = make_sb_objects(map_type=map_type,lc_name=lc_name,
                               map_prior='physicalVisible',degree=degree,
                               use_gp_list = [False,True])
    
    return sb0, sb1

def make_sb_objects(map_type='variable',lc_name='NC_HD189',
                    map_prior='physical',degree=3,
                    use_gp_list = [False,True],
                    widerSphHarmonicPriors=False,
                    cores=2):
    """
    Make a pair of starry basemodel objects in utils.
    Generally, used to compare data with and without baselines
    
    Parameters
    -----------
    
    map_type: str
        Map type "fixed" fixes it at uniform. "variable" solves for sph harmonics
    lc_name: str
        Name of the lightcurve ('NC_HD189' uses a NIRCam lightcurve for HD 189733)
                                 'orig' is the original lightcurve at high precision
    map_prior: str
        What priors are put on the plot? 'phys' ensures physical (non-negative priors)
        and 'uniformPixels' does pixel sampling to ensure the pixels are non-negative
        'non-physical' allows the spherical harmonics to be postive or negative
    degree: int
        The spherical harmonic degree
    use_gp_list: 2 element list of booleans
        FOr the 
    cores: int
        Number of CPU cores passed to utils.starry_basemodel
    """
    if degree == 3:
        degree_descrip = ''
    else:
        degree_descrip = '_deg{}'.format(degree)
    
    if map_prior == 'physical':
        phys_descrip = 'phys'
    elif map_prior == 'uniformPixels':
        phys_descrip = 'pxSamp'
    elif map_prior == 'physicalVisible':
        phys_descrip = 'physVis'
    else:
        phys_descrip = 'nophys'
    
    if lc_name == 'NC_HD189':
        systematics=['Flat','Quadratic']
        systematics_abbrev = ['flat','quad']
        
        descrips = []
        for systematics_descrip in systematics_abbrev:
            thisDescrip = '{}_HD189NC{}PMass{}'.format(systematics_descrip,phys_descrip,degree_descrip)
            descrips.append(thisDescrip)
        
        dataPath='sim_data/sim_data_baseline_hd189_ncF444W.ecsv'
        vminmaxDef = [0.0,1.1]
    elif lc_name == 'NC_HD189cubic':
        systematics=['Flat','Cubic']
        systematics_abbrev = ['flat','cubic']
        
        descrips = []
        for systematics_descrip in systematics_abbrev:
            thisDescrip = '{}_HD189NC{}PMass{}'.format(systematics_descrip,phys_descrip,degree_descrip)
            descrips.append(thisDescrip)
        
        dataPath='sim_data/sim_data_cubic_baseline_hd189_ncF444W.ecsv'
        vminmaxDef = [0.0,1.1]
    elif lc_name == 'orig':
        systematics=['Flat','Cubic']
        
        descrips_basenames = ['Flat_001','Orig_006_newrho_smallGP']
        
        descrips = []
        for descrip_basename in descrips_basenames:
            descrips.append("{}{}{}".format(descrip_basename,phys_descrip,degree_descrip))
        
        dataPath='sim_data/sim_data_baseline.ecsv'
        vminmaxDef = [0.0,0.7]
    elif lc_name == 'GCM01_HD189':
        systematics = ['Flat','Cubic']
        systematics_abbrev = ['flat','cubic']

        descrips = []
        for systematics_descrip in systematics_abbrev:
            thisDescrip = '{}_GCM01_HD189NC{}PMass{}'.format(systematics_descrip,phys_descrip,degree_descrip)
            descrips.append(thisDescrip)
        
        dataPath = 'sim_data/gcm01_sim_data_cubic_baseline_hd189_ncF444W.ecsv'
        vminmaxDef = [0.0,0.7]
    elif lc_name == 'NC_HD189variableOrbit':
        systematics=['Flat','Quadratic']
        systematics_abbrev = ['flat','quad']
        
        descrips = []
        for systematics_descrip in systematics_abbrev:
            thisDescrip = '{}_HD189NCvarOrbit{}PMass{}'.format(systematics_descrip,phys_descrip,degree_descrip)
            descrips.append(thisDescrip)
        
        dataPath='sim_data/sim_data_baseline_hd189_ncF444W_variable_orbParam.ecsv'
        vminmaxDef = [0.0,1.1]
    elif lc_name == 'NC_HD189zeroImpact':
        systematics=['Flat','Quadratic']
        systematics_abbrev = ['flat','quad']
        
        descrips = []
        for systematics_descrip in systematics_abbrev:
            thisDescrip = '{}_HD189NCzeroImpact{}PMass{}'.format(systematics_descrip,phys_descrip,degree_descrip)
            descrips.append(thisDescrip)
        
        dataPath='sim_data/sim_data_baseline_hd189_ncF444W_zero_impact_param.ecsv'
        vminmaxDef = [0.0,1.1]
    elif lc_name == 'NC_HD189GPforward':
        systematics=['Flat','GPbaseline']
        systematics_abbrev = ['flat','GPbase']

        descrips = []
        for systematics_descrip in systematics_abbrev:
            thisDescrip = '{}_HD189NCgpSim{}PMass{}'.format(systematics_descrip,phys_descrip,degree_descrip)
            descrips.append(thisDescrip)
        
        dataPath='sim_data/sim_data_GP_baseline_hd189_ncF444W.ecsv'
        vminmaxDef = [0.0,1.1]
    else:
        raise Exception("Unrecognized lightcurve name {}".format(lc_name))
    
    if lc_name == 'GCM01_HD189':
        hotspotGuess_param = {'xstart':40,'xend':95,
                              'ystart':0,'yend':99,
                              'guess_x':30,'guess_y':0}
        inputLonLat=[52.447,1.3631]
    elif lc_name == 'NC_HD189zeroImpact':
        hotspotGuess_param = {'xstart':45,'xend':85,
                              'ystart':0,'yend':99,
                              'guess_x':50,'guess_y':40}
        inputLonLat=[58.05,45.33]
    else:
        hotspotGuess_param = {}
        inputLonLat=[58.05,45.33]
    
    sb_list = []
    for ind,sys_type in enumerate(systematics):
        thisDescription = descrips[ind]
        if use_gp_list[ind] == True:
            pass
        else:
            thisDescription = thisDescription + '_no_gp'
        
        if widerSphHarmonicPriors == True:
            thisDescription = thisDescription + '_widerSphHarmPriors'
        
        sb = utils.starry_basemodel(dataPath=dataPath,
                                    descrip=thisDescription,
                                    map_type=map_type,amp_type='variable',
                                    systematics=sys_type,degree=degree,
                                    map_prior=map_prior,use_gp=use_gp_list[ind],
                                    widerSphHarmonicPriors=widerSphHarmonicPriors,
                                    hotspotGuess_param=hotspotGuess_param,
                                    inputLonLat=inputLonLat,
                                    vminmaxDef=vminmaxDef,
                                    cores=cores)
        sb_list.append(sb)
    
    return sb_list

def check_flat_vs_curved_baseline(map_type='variable',find_posterior=False,
                                  super_giant_corner=False,lc_name='NC_HD189',
                                  map_prior='physical',degree=3,
                                  cores=2):
    """
    Try fitting a lightcurve with a flat baseline vs curved
    
    map_type: str
        Map type "fixed" fixes it at uniform. "variable" solves for sph harmonics
    find_posterior: bool
        Find the posterior of the distribution? If True, the Bayesian sampler
        is employed to find it. If False, it just finds the max a priori solution
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

    cores: int
        Number of CPU cores to pass to utils.starry_basemodel
    """
    
    
    sb_list = make_sb_objects(map_type=map_type,lc_name=lc_name,
                              map_prior=map_prior,degree=degree,
                              cores=cores)
    
    for sb in sb_list:
        sb.run_all(find_posterior=find_posterior,
                   super_giant_corner=super_giant_corner)
        
    
    if super_giant_corner == True:
        utils.compare_corners(sb_list[0],sb_list[1])
    utils.compare_corners(sb_list[0],sb_list[1],
                          sph_harmonics='m=1',
                          include_sigma_lc=False,
                          extra_descrip='m_eq_1_')
    utils.compare_histos(sb_list[0],sb_list[1])
    
def run_inference(lc_name='NC_HD189',map_prior='physical',degree=3,
                  cores=2):
    check_flat_vs_curved_baseline(map_type='variable',find_posterior=True,
                                  lc_name=lc_name,map_prior=map_prior,
                                  degree=degree,cores=cores)
    

def run_gcm01_inference():
    """
    Run the inference tests with the GCM01 as the forward model
    """
    run_inference(lc_name='GCM01_HD189',degree=2)


def run_all_inference_tests(map_prior='physicalVisible',degrees=[3,2]):
    """
    Run all inference tests for both planets for flat versus non-flat baselines
    
    Parameters
    ----------
    map_prior: str
        What priors are put on the plot? 'phys' ensures physical (non-negative priors)
        and 'uniformPixels' does pixel sampling to ensure the pixels are non-negative
        'non-physical' allows the spherical harmonics to be postive or negative
    
    degrees: 2 element list of ints
        List of spherical harmonics to use for the Idealized and HD 189733 b planets
    """
    
    for ind,lc_name in enumerate(['orig','NC_HD189']):
        run_inference(lc_name=lc_name,map_prior=map_prior,degree=degrees[ind])
    run_gcm01_inference()
    

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

def plot_px_sampling(xdata=None,sb=None):
    """
    Plot the result of pixel sampling to to see those results
    """
    if sb == None:
        sb = sb_px_sampling_obj()
        sb.find_posterior()
        xdata = arviz.convert_to_dataset(sb.trace)
    
    sb.plot_px_sampling()


if __name__ == "__main__":
    run_all_inference_tests()

def get_posterior_comparables(postName):
    """
    Get the posterior info
    Remove the GP parameters because they might not be comparable
    """
    full_dat = ascii.read(os.path.join('fit_data/posterior_stats/',postName))
    keep_pts = np.ones_like(full_dat['Variable'],dtype=bool)
    for ind,oneVar in enumerate(full_dat['Variable']):
        if oneVar == 'sigma_gp':
            keep_pts[ind] = False
        elif oneVar == 'rho_gp':
            keep_pts[ind] = False
        else:
            keep_pts[ind] = True
    
    return full_dat[keep_pts]


def posterior_ratios():
    """
    Calculate how much the posterior precisions compare (numerically) for models with/without baselines
    
    """
    compare_idealized = ['Orig_006_newrho_smallGPphys_maptype_variable_amp_type_variable_stats.csv',
                         'Flat_001_no_gpphys_maptype_variable_amp_type_variableFlat_stats.csv']
    compare_HD189 = ['quad_HD189NCphysPMass_deg2_maptype_variable_amp_type_variableQuadratic_stats.csv',
                     'flat_no_gp_HD189NCphysPMass_deg2_maptype_variable_amp_type_variableFlat_stats.csv']
    compare_HD189zeroImpact = ['quad_HD189NCzeroImpactphysVisPMass_deg2_maptype_variable_amp_type_variableQuadratic_stats.csv',
                                'flat_HD189NCzeroImpactphysVisPMass_deg2_no_gp_maptype_variable_amp_type_variableFlat_stats.csv']
    compare_HD189GCM = ['cubic_GCM01_HD189NCphysPMass_deg2_maptype_variable_amp_type_variable_stats.csv',
                        'flat_GCM01_HD189NCphysPMass_deg2_no_gp_maptype_variable_amp_type_variableFlat_stats.csv']
    compare_HD189GP = ['GPbase_HD189NCgpSimphysVisPMass_deg2_maptype_variable_amp_type_variableGPbaseline_stats.csv',
                       'flat_HD189NCgpSimphysVisPMass_deg2_no_gp_maptype_variable_amp_type_variableFlat_stats.csv']

    compare_pairs = [compare_idealized,compare_HD189,compare_HD189zeroImpact,
                    compare_HD189GCM,compare_HD189GP]
    compare_planet_names = ['Idealized','HD_189733_b','HD_189733b_b_w_zeroImpact',
                            'HD_189733b_GCM','HD_189733b_GPforward']
    ratio_name = ['systematics over flat ratio']
    
    for onePlanet, onePair in zip(compare_planet_names,compare_pairs):
        dat1 = get_posterior_comparables(onePair[0])
        dat2 = get_posterior_comparables(onePair[1])
        
        assert np.array_equal(dat1['Variable'],dat2['Variable'])
        
        ratio_table = dat1[['Variable','Label']]
        ratio_table['68 perc {}'.format(ratio_name)] = dat1['68 perc width'] / dat2['68 perc width']
        ratio_table['95 perc {}'.format(ratio_name)] = dat1['95 perc width'] / dat2['95 perc width']
        outName = "{}_ratios.csv".format(onePlanet)
        outPath = os.path.join('fit_data','posterior_stats',outName)
        
        ratio_table.write(outPath,overwrite=True)

def kelt9sim():
    hotspotGuess_param = {'xstart':40,'xend':60,
                            'ystart':40,'yend':60,
                            'guess_x':0,'guess_y':0}


    dataPath='sim_data/sim_data_kelt9.ecsv'
    descrip = 'flat_kelt9sim'
    sb = utils.starry_basemodel(dataPath=dataPath,
                                descrip=descrip,
                                map_type='variable',amp_type='variable',
                                systematics='Flat',degree=2,
                                hotspotGuess_param=hotspotGuess_param,
                                map_prior='physicalVisible',use_gp=False)
    return sb


def variable_orb():
    """
    Allow for a variable orbit
    """
    
    #lc_name = 'sim_data/sim_data_baseline_hd189_ncF444W_variable_orbParam.ecsv'
    lc_name = 'NC_HD189variableOrbit'
    map_prior='physicalVisible'

    run_inference(lc_name=lc_name,map_prior=map_prior,degree=2,
                  cores=1)

    compare_fixed_vs_variable_orb()

def compare_fixed_vs_variable_orb():
    lc_name_fixed = 'NC_HD189'
    map_prior='physicalVisible'
    cores=1
    sb_list_fix = make_sb_objects(lc_name=lc_name_fixed,
                                    map_prior=map_prior,degree=2,
                                    cores=cores)

    lc_name_var = 'NC_HD189variableOrbit'
    
    sb_list_var = make_sb_objects(lc_name=lc_name_var,
                                    map_prior=map_prior,degree=2,
                                    cores=cores)
    
    
    utils.compare_histos(sb_list_fix[0],sb_list_var[0],
                         dataDescrips=['Fixed Orb, Flat','Variable Orb, Flat'])


def zero_impact_param_test():
    """
    Test with zero impact parameter
    """
    lc_name='NC_HD189zeroImpact'
    check_flat_vs_curved_baseline(map_type='variable',find_posterior=True,
                                lc_name=lc_name,map_prior='physicalVisible',
                                degree=2,cores=2)

def GPforward_model():
    lc_name ='NC_HD189GPforward'
    check_flat_vs_curved_baseline(map_type='variable',find_posterior=True,
                                lc_name=lc_name,map_prior='physicalVisible',
                                degree=2,cores=2)
    