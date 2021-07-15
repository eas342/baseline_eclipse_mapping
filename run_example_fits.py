import utils

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
                                  super_giant_corner=False):
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
    """
    systematics=['Flat','Cubic']
    descrips = ['Flat_001','Orig_006_newrho_smallGP']
    
    sb_list = []
    for ind,sys_type in enumerate(systematics):
        sb = utils.starry_basemodel(descrip=descrips[ind],
                                    map_type=map_type,amp_type='variable',
                                    systematics=sys_type)
        sb.plot_lc(point='mxap')
        if find_posterior == True:
            sb.find_posterior()
            if super_giant_corner == True:
                sb.plot_corner()
        sb_list.append(sb)
    
    if super_giant_corner == True:
        utils.compare_corners(sb_list[0],sb_list[1])
    utils.compare_corners(sb_list[0],sb_list[1],
                          sph_harmonics='m=1',
                          include_sigma_lc=False,
                          extra_descrip='m_eq_1_')
    utils.compare_histos(sb_list[0],sb_list[1])
    
def run_inference():
    check_flat_vs_curved_baseline(map_type='variable',find_posterior=True)


    

if __name__ == "__main__":
    run_inference()
    