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

def check_flat_vs_curved_baseline(map_type='fixed',find_posterior=False):
    """
    Try fitting a lightcurve with a flat baseline vs curved
    """
    systematics=['Cubic','Flat']
    descrips = ['Orig_006_newrho_smallGP','Flat_001']
    for ind,sys_type in enumerate(systematics):
        sb = utils.starry_basemodel(descrip=descrips[ind],
                                    map_type=map_type,amp_type='variable',
                                    systematics=sys_type)
        sb.plot_lc(point='mxap')
        if find_posterior == True:
            sb.find_posterior()
    
def run_inference():
    check_flat_vs_curved_baseline(map_type='variable',find_posterior=True)
    
if __name__ == "__main__":
    run_inference()
    